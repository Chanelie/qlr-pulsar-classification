#!/usr/bin/env python3
"""Public command-line entry point for the pulsar QLR workflow."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "HTRU2.csv"


USAGE = """\
Pulsar QLR workflow

Usage:
  python run.py paths
  python run.py status [--samples 200 500 1000]
  python run.py plan [grid_plan options]
  python run.py run-one --kind qlr --samples 200 --layers 3 --encoding angle --seed 0
  python run.py figures [--set thesis|diagnostics|terminology|all]
  python run.py aggregate [--samples 200 500 1000]
  python run.py run-grid --dry-run

The status and figure commands require the saved result caches. The repository
does not silently start training when a cache is missing.
"""


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["QLR_DATA_PATH"] = str(DATASET)
    return env


def require_dataset() -> None:
    if not DATASET.exists():
        raise SystemExit(
            f"Dataset not found: {DATASET}\n"
            "See data/README.md for the UCI HTRU2 source."
        )


def run_python(script: str, args: list[str]) -> int:
    command = [sys.executable, str(ROOT / "scripts" / script), *args]
    return subprocess.run(command, cwd=ROOT, env=environment()).returncode


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0

    command, args = sys.argv[1], sys.argv[2:]

    if command == "paths":
        print(f"Repository : {ROOT}")
        print(f"Dataset    : {DATASET}")
        print(f"Code       : {ROOT / 'qlr'}")
        print(f"Scripts    : {ROOT / 'scripts'}")
        print(f"Tables     : {ROOT / 'tables'}")
        print(f"Results    : {ROOT / 'results'}")
        return 0

    if command in {"run-one", "run-grid"}:
        require_dataset()

    if command == "status":
        return run_python("aggregate_all.py", ["--check", *args])
    if command == "plan":
        return run_python("grid_plan.py", args)
    if command == "run-one":
        if not args:
            print("run-one requires the arguments shown in --help", file=sys.stderr)
            return 2
        return run_python("run_one.py", args)
    if command == "figures":
        return run_python("generate_figures.py", args)
    if command == "aggregate":
        stages = [
            ("coverage check", "aggregate_all.py", ["--check", *args]),
            ("aggregation pass 1", "aggregate_all.py", args),
            ("summary tables", "build_all_layer_summary_tables.py", []),
            ("aggregation pass 2", "aggregate_all.py", args),
            ("figures", "generate_figures.py", ["--set", "thesis"]),
        ]
        for label, script, stage_args in stages:
            print(f"\n== {label} ==", flush=True)
            code = run_python(script, stage_args)
            if code:
                return code
        return 0
    if command == "run-grid":
        if os.name == "nt":
            print("run-grid requires Linux, GNU screen, and taskset.", file=sys.stderr)
            return 2
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / "run_grid.sh"), *args],
            cwd=ROOT,
            env=environment(),
        ).returncode

    print(f"Unknown command: {command}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
