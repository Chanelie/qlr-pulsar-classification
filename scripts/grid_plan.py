#!/usr/bin/env python3
"""
Job-list generator and scheduling planner for the full grid re-run.

Single source of truth for *what* runs, *in which order*, and *on which core*.
The launcher (run_grid.sh) consumes `--emit jobs`; humans read `--emit dry-run`.

Nothing here executes a job or modifies experiment outputs.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- core assignment used for the recorded runtime environment ----------------
# i9-14900KF: P-cores are logical 0-15 as 8 SMT pairs; E-cores are logical 16-31.
# One job per *physical* P-core -> take the even member of each sibling pair.
P_CORES = [0, 2, 4, 6, 8, 10, 12, 14]
E_CORES = list(range(16, 32))

ENCODINGS = ["amplitude", "angle", "data_reuploading"]
SEEDS = [0, 1, 2]
QUBITS = {"amplitude": 3, "angle": 8, "data_reuploading": 8}

BENCH_N, BENCH_L = 1000, 3
DEPTH_SWEEP_L = [1, 2, 5, 10]          # L=3 already covered by the benchmark tier
SIZE_SWEEP_N = [200, 500]
ALL_L = [1, 2, 3, 5, 10]


@dataclass
class Job:
    tier: int
    kind: str          # "qlr" | "baselines"
    n: int
    layers: int | None
    encoding: str | None
    seed: int
    core_class: str    # "P" | "E"
    est_s: float

    @property
    def session(self) -> str:
        if self.kind == "qlr":
            return f"qlr_N{self.n}_L{self.layers}_{self.encoding}_s{self.seed}"
        return f"bl_N{self.n}_s{self.seed}"

    @property
    def cmd(self) -> str:
        """Arguments for scripts/run_one.py. The launcher supplies the
        interpreter, the taskset pin, and the thread environment."""
        if self.kind == "qlr":
            return (
                f"--kind qlr --samples {self.n} --layers {self.layers} "
                f"--encoding {self.encoding} --seed {self.seed}"
            )
        return f"--kind baselines --samples {self.n} --seed {self.seed}"


# --- recorded runtime data ----------------------------------------------------
def load_v1_runtimes(tables_dir: Path) -> dict[tuple, float]:
    """Parse tables/runtime_N*_all_layers.csv -> {key: train_s + pred_s} per seed."""
    out: dict[tuple, float] = {}
    pat = re.compile(r"^QLR_(\w+) \(L=(\d+)\)$")

    def mean_value(cell: str) -> float:
        """Read the leading mean from a formatted mean-plus-minus-SE value."""
        match = re.match(r"\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", cell)
        if not match:
            raise ValueError(f"could not read numeric mean from {cell!r}")
        return float(match.group(1))

    for n in (200, 500, 1000):
        f = tables_dir / f"runtime_N{n}_all_layers.csv"
        if not f.exists():
            raise SystemExit(f"missing runtime provenance: {f}")
        with f.open() as fh:
            for row in csv.DictReader(fh):
                # cells are "mean ± se"
                tr = mean_value(row["Train_s"])
                pr = mean_value(row["Pred_s"])
                total = tr + pr
                m = pat.match(row["Model"])
                if m:
                    out[("qlr", n, m.group(1), int(m.group(2)))] = total
                else:
                    out[("model", n, row["Model"])] = total
    return out


def build_jobs(rt: dict[tuple, float]) -> list[Job]:
    jobs: list[Job] = []

    def qlr(tier, n, L, enc, sd):
        return Job(tier, "qlr", n, L, enc, sd, "P", rt[("qlr", n, enc, L)])

    def baselines(tier, n, sd):
        # one process trains LogReg + CSVM + XGBoost + QSVM and writes one cache file
        est = sum(rt[("model", n, m)] for m in ("LogReg", "CSVM", "XGBoost", "QSVM"))
        return Job(tier, "baselines", n, None, None, sd, "P", est)

    # Tier 1 - benchmark configuration: N=1000, L=3, all encodings + baselines/QSVM
    for enc in ENCODINGS:
        for sd in SEEDS:
            jobs.append(qlr(1, BENCH_N, BENCH_L, enc, sd))
    for sd in SEEDS:
        jobs.append(baselines(1, BENCH_N, sd))

    # Tier 2 - depth sweep at N=1000
    for L in DEPTH_SWEEP_L:
        for enc in ENCODINGS:
            for sd in SEEDS:
                jobs.append(qlr(2, BENCH_N, L, enc, sd))

    # Tier 3 - training-size sweep at N in {200, 500}
    for n in SIZE_SWEEP_N:
        for sd in SEEDS:
            jobs.append(baselines(3, n, sd))
        for L in ALL_L:
            for enc in ENCODINGS:
                for sd in SEEDS:
                    jobs.append(qlr(3, n, L, enc, sd))
    return jobs


def schedule(jobs: list[Job], slots: int, barrier: bool) -> tuple[float, list[tuple]]:
    """Greedy list-scheduling in queue order. Returns (makespan_s, timeline)."""
    free = [0.0] * slots
    timeline = []
    tier_end = 0.0
    cur_tier = jobs[0].tier if jobs else 0
    for j in jobs:
        if barrier and j.tier != cur_tier:
            free = [max(f, tier_end) for f in free]
            cur_tier = j.tier
        i = min(range(slots), key=lambda k: free[k])
        start = free[i]
        end = start + j.est_s
        free[i] = end
        tier_end = max(tier_end, end)
        timeline.append((j, i, start, end))
    return max(free), timeline


def hms(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def days(s: float) -> str:
    return f"{s/86400:.2f} d"


def main() -> None:
    ap = argparse.ArgumentParser()
    # Reference runtime tables are used solely for scheduling
    # estimates. They are read-only and do not affect model results.
    ap.add_argument("--tables", default=str(ROOT / "tables"),
                    help="directory holding runtime_N*_all_layers.csv")
    ap.add_argument("--emit", choices=["dry-run", "jobs"], default="dry-run")
    ap.add_argument("--slots", type=int, default=len(P_CORES))
    args = ap.parse_args()

    rt = load_v1_runtimes(Path(args.tables))
    jobs = build_jobs(rt)

    if args.emit == "jobs":
        for j in jobs:
            print(f"{j.tier}\t{j.session}\t{j.core_class}\t{j.est_s:.0f}\t{j.cmd}")
        return

    mk_open, tl = schedule(jobs, args.slots, barrier=False)
    mk_bar, _ = schedule(jobs, args.slots, barrier=True)

    print("=" * 100)
    print("DRY RUN - full experimental grid.  NOTHING LAUNCHED.")
    print("=" * 100)
    print(f"runtime provenance : {args.tables}/runtime_N{{200,500,1000}}_all_layers.csv")
    print(f"P-core slots       : {args.slots}  -> logical CPUs {P_CORES}")
    print(f"E-core pool        : logical CPUs {E_CORES[0]}-{E_CORES[-1]}")
    print()

    print("-" * 100)
    print("1. JOB LIST IN QUEUE ORDER")
    print("-" * 100)
    print(f"{'#':>4}  {'tier':<5} {'screen session':<40} {'core':<5} {'est (v1)':>10}")
    last = None
    for i, j in enumerate(jobs, 1):
        if j.tier != last:
            names = {1: "BENCHMARK  N=1000 L=3 + baselines/QSVM",
                     2: "DEPTH SWEEP  N=1000, L in {1,2,5,10}",
                     3: "SIZE SWEEP  N in {200,500}, all L"}
            print(f"\n  --- TIER {j.tier}: {names[j.tier]} ---")
            last = j.tier
        print(f"{i:>4}  {j.tier:<5} {j.session:<40} {j.core_class:<5} {hms(j.est_s):>10}")

    print()
    print("-" * 100)
    print("2. JOB COUNTS")
    print("-" * 100)
    tot = 0
    for t in (1, 2, 3):
        sub = [j for j in jobs if j.tier == t]
        q = sum(1 for j in sub if j.kind == "qlr")
        b = sum(1 for j in sub if j.kind == "baselines")
        print(f"  tier {t}: {len(sub):>3} jobs  ({q} QLR, {b} baselines+QSVM)"
              f"   serial CPU-time {days(sum(j.est_s for j in sub))}")
        tot += len(sub)
    print(f"  TOTAL : {tot:>3} jobs                       serial CPU-time "
          f"{days(sum(j.est_s for j in jobs))}")

    print()
    print("-" * 100)
    print(f"3. ESTIMATED WALL-CLOCK ON {args.slots} SLOTS")
    print("-" * 100)
    for t in (1, 2, 3):
        sub = [(j, s, e) for (j, _c, s, e) in tl if j.tier == t]
        t0 = min(s for _j, s, _e in sub)
        t1 = max(e for _j, _s, e in sub)
        print(f"  tier {t} completes at  T+{hms(t1):<10} ({days(t1)})   "
              f"[tier spans {hms(t1 - t0)}]")
    print()
    print(f"  MAKESPAN, no tier barrier : {hms(mk_open)}  ({days(mk_open)})")
    print(f"  MAKESPAN, tier barriers   : {hms(mk_bar)}  ({days(mk_bar)})")
    eff = sum(j.est_s for j in jobs) / (mk_open * args.slots) * 100
    print(f"  slot utilisation (no barrier): {eff:.1f}%")

    longest = max(jobs, key=lambda j: j.est_s)
    print()
    print(f"  LONGEST POLE: {longest.session}")
    print(f"    {hms(longest.est_s)} ({days(longest.est_s)}) single-threaded on one core.")
    print(f"    This one job alone sets a hard floor on the total wall-clock time.")
    top = sorted(jobs, key=lambda j: -j.est_s)[:5]
    print("    top 5 by cost:")
    for j in top:
        print(f"      {j.session:<40} {hms(j.est_s):>10}")

    print()
    print("-" * 100)
    print("4. SESSION NAMING + PROGRESS COMMAND")
    print("-" * 100)
    print("  QLR       : qlr_N<samples>_L<layers>_<encoding>_s<seed>")
    print("  baselines : bl_N<samples>_s<seed>")
    print("  example   : qlr_N1000_L3_angle_s0")
    print()
    print("  progress:")
    print("    column -t -s'|' logs/status.log | tail -40 ; screen -ls | grep -c Detached")


if __name__ == "__main__":
    main()
