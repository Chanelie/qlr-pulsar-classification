#!/usr/bin/env bash
#
# Full-grid launcher: one config per independent single-threaded process,
# each in its own detached screen session, each pinned to one physical
# P-core. Jobs use separate processes to avoid backend conflicts and share
# nothing but the filesystem.
#
#   ./scripts/run_grid.sh --dry-run     show the plan, launch nothing
#   ./scripts/run_grid.sh               launch the grid
#   ./scripts/run_grid.sh --limit 1     launch only the first job (smoke test)
#   ./scripts/run_grid.sh --status      one-shot progress summary
#   ./scripts/run_grid.sh --stop        kill every job session
#
# Progress at a glance:
#   column -t -s'|' logs/status.log | sort -k2
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QLR_PYTHON:-python}"

# Physical P-cores only: logical 0-15 are 8 SMT sibling pairs, so one job
# per pair. Logical 16-31 are E-cores (max 4.4 GHz vs 5.7-6.0) and are
# deliberately left idle to avoid changing the power budget during runtime
# measurements. Inspect and adapt this list for the target machine.
CORES=(0 2 4 6 8 10 12 14)
SLOTS=${#CORES[@]}
MAX_ATTEMPTS=3          # worker retries on a non-zero exit
POLL_SECONDS=20

LOGDIR="$ROOT/logs"
JOBDIR="$LOGDIR/jobs"
SLOTDIR="$LOGDIR/slots"
STATUS="$LOGDIR/status.log"

ts()      { date +%Y-%m-%dT%H:%M:%S; }
now()     { date +%s; }
live()    { screen -ls 2>/dev/null | grep -cE "[0-9]+\.(qlr|bl)_" || true; }
log_row() { printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' "$@" >> "$STATUS"; }

# ---------------------------------------------------------------------------
# Worker: runs inside a screen session and owns exactly one core.
# Retries the job up to MAX_ATTEMPTS on non-zero exit, then records the
# outcome. Releases its slot lock on any exit path.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "__supervise" ]]; then
    core="$2"; session="$3"; concurrency="$4"; shift 4
    lock="$SLOTDIR/core${core}.lock"
    trap 'rm -f "$lock"' EXIT

    attempt=1
    rc=1
    while (( attempt <= MAX_ATTEMPTS )); do
        start_s=$(now); start_t=$(ts)
        log_row "$session" "START" "$core" "$start_t" "-" "-" "-" "$attempt" "$concurrency" "$(hostname)"

        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        QLR_CONCURRENCY="$concurrency" \
        QLR_ATTEMPT="$attempt" \
        taskset -c "$core" "$PYTHON" "$ROOT/scripts/run_one.py" \
            "$@" --session "$session" --attempt "$attempt" \
            >> "$JOBDIR/${session}.attempt${attempt}.log" 2>&1
        rc=$?

        end_s=$(now); wall=$(( end_s - start_s ))
        if (( rc == 0 )); then
            log_row "$session" "DONE" "$core" "$start_t" "$(ts)" "$wall" "0" "$attempt" "$concurrency" "$(hostname)"
            break
        fi

        if (( attempt < MAX_ATTEMPTS )); then
            log_row "$session" "RETRY" "$core" "$start_t" "$(ts)" "$wall" "$rc" "$attempt" "$concurrency" "$(hostname)"
        else
            log_row "$session" "FAIL" "$core" "$start_t" "$(ts)" "$wall" "$rc" "$attempt" "$concurrency" "$(hostname)"
        fi
        (( attempt++ ))
    done

    # A hard crash (SIGSEGV) never reaches run_one.py's own record writer,
    # so guarantee a record exists either way.
    "$PYTHON" - "$ROOT" "$session" "$core" "$rc" "$((attempt > MAX_ATTEMPTS ? MAX_ATTEMPTS : attempt))" "$concurrency" <<'PYEOF'
import json, pathlib, socket, sys
root, session, core, rc, attempts, conc = sys.argv[1:7]
p = pathlib.Path(root) / "results" / "run_records" / f"{session}.json"
p.parent.mkdir(parents=True, exist_ok=True)
rec = {}
if p.exists():
    try:
        rec = json.loads(p.read_text())
    except Exception:
        rec = {}
rec.setdefault("session", session)
rec.setdefault("hostname", socket.gethostname())
rec.setdefault("pinned_core", int(core))
rec.setdefault("concurrency_at_launch", int(conc))
rec["exit_status"] = int(rc)
rec["attempts"] = int(attempts)
rec["retries"] = int(attempts) - 1
rec["status"] = rec.get("status", "ok" if int(rc) == 0 else "failed")
if int(rc) != 0:
    rec["status"] = "failed"
p.write_text(json.dumps(rec, indent=2, default=str))
PYEOF
    exit "$rc"
fi

# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
DRY=0; LIMIT=0; ONLY=""; MODE=run
while (( $# )); do
    case "$1" in
        --dry-run) DRY=1 ;;
        --limit)   LIMIT="$2"; shift ;;
        --only)    ONLY="$2"; shift ;;   # ERE on session name; re-run one config
        --status)  MODE=status ;;
        --stop)    MODE=stop ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ "$MODE" == "status" ]]; then
    [[ -f "$STATUS" ]] || { echo "no status log yet ($STATUS)"; exit 0; }
    echo "== per-job latest state =="
    awk -F'|' '{s[$1]=$2"  core="$3"  wall="$6"s  exit="$7"  attempt="$8}
               END{for (k in s) printf "%-42s %s\n", k, s[k]}' "$STATUS" | sort
    echo
    awk -F'|' '{s[$1]=$2} END{for (k in s) c[s[k]]++;
               for (t in c) printf "%-6s %d\n", t, c[t]}' "$STATUS"
    echo "live sessions: $(live)"
    exit 0
fi

if [[ "$MODE" == "stop" ]]; then
    mapfile -t sess < <(screen -ls 2>/dev/null | grep -oE "[0-9]+\.(qlr|bl)_[A-Za-z0-9_]+" || true)
    rm -f "$SLOTDIR"/core*.lock   # remove stale locks after interrupted jobs
    (( ${#sess[@]} )) || { echo "no job sessions running"; exit 0; }
    for s in "${sess[@]}"; do screen -S "$s" -X quit && echo "killed $s"; done
    exit 0
fi

command -v screen  >/dev/null || { echo "screen not found"  >&2; exit 1; }
command -v taskset >/dev/null || { echo "taskset not found" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "interpreter not found: $PYTHON" >&2; exit 1; }

mkdir -p "$JOBDIR" "$SLOTDIR" "$ROOT/results/run_records"

# Job list: tier <tab> session <tab> core_class <tab> est_s <tab> args
mapfile -t JOBS < <("$PYTHON" "$ROOT/scripts/grid_plan.py" --emit jobs)
(( ${#JOBS[@]} )) || { echo "empty job list" >&2; exit 1; }
if [[ -n "$ONLY" ]]; then
    mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | awk -F'\t' -v p="$ONLY" '$2 ~ p')
    (( ${#JOBS[@]} )) || { echo "no job matches --only '$ONLY'" >&2; exit 1; }
fi

# Resume: drop configs that already have a non-empty cache file, so a restarted
# dispatcher continues the queue instead of re-running finished work from idx=0.
# Paths come from the same helpers run_one.py uses, so there is one definition of
# where a config's artifact lives. The size test matters because a crash during
# pickle.dump leaves a 0-byte file that must not count as done.
mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | "$PYTHON" -c "
import sys; sys.path.insert(0, '$ROOT')
from qlr.cache import ArtifactManager
from qlr.config import Config
from qlr.paths import baseline_cache_path
AM = ArtifactManager()
for row in sys.stdin.read().splitlines():
    a = row.split('\t')[4].split()
    kw = dict(zip(a[::2], a[1::2]))
    n, seed = int(kw['--samples']), int(kw['--seed'])
    if kw['--kind'] == 'qlr':
        enc = kw['--encoding']
        path = AM._path({'regime': f'N{n}', 'encoding': enc,
                         'n_qubits': 3 if enc == 'amplitude' else 8,
                         'n_layers': int(kw['--layers']),
                         'epochs': Config.EPOCHS, 'seed': seed})
    else:
        path = baseline_cache_path(n, seed)
    if not (path.exists() and path.stat().st_size):
        print(row)
")
(( ${#JOBS[@]} )) || { echo "[grid] nothing to do -- every selected config already has a cache"; exit 0; }

(( LIMIT > 0 )) && JOBS=("${JOBS[@]:0:LIMIT}")

if (( DRY )); then
    echo "DRY RUN -- nothing will be launched"
    echo "interpreter : $PYTHON"
    echo "cores       : ${CORES[*]}  (${SLOTS} concurrent slots)"
    echo "retries     : up to $MAX_ATTEMPTS attempts per job"
    echo "jobs        : ${#JOBS[@]}"
    echo
    i=0
    for row in "${JOBS[@]}"; do
        IFS=$'\t' read -r tier session cls est args <<< "$row"
        printf '%3d  tier%s  %-42s core=%-3s  %s\n' \
            "$((++i))" "$tier" "$session" "${CORES[$(( (i-1) % SLOTS ))]}" \
            "run_one.py $args"
    done
    exit 0
fi

# Write environment metadata once, before any job can race on it. The thread
# vars must be exported here too -- write_environment_meta_once() snapshots
# os.environ, and the dispatcher's own shell would otherwise record nulls for
# settings that every job actually runs under.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
"$PYTHON" -c "
import sys; sys.path.insert(0, '$ROOT')
from qlr.config import Config, write_environment_meta_once
Config.setup_directories(); write_environment_meta_once()
print('[grid] environment_meta.json ready')
"

echo "[grid] $(ts) launching ${#JOBS[@]} jobs across $SLOTS cores: ${CORES[*]}"
touch "$STATUS"

idx=0
while (( idx < ${#JOBS[@]} )); do
    # find a core whose slot lock is free
    placed=0
    for core in "${CORES[@]}"; do
        lock="$SLOTDIR/core${core}.lock"
        [[ -e "$lock" ]] && continue

        IFS=$'\t' read -r tier session cls est args <<< "${JOBS[$idx]}"
        : > "$lock"                      # claim before launching
        concurrency=$(( $(live) + 1 ))   # jobs running once this one starts

        screen -dmS "$session" bash "$ROOT/scripts/run_grid.sh" \
            __supervise "$core" "$session" "$concurrency" $args

        printf '[grid] %s  slot core=%-3s  %-42s (tier %s, est %ss, conc %s)\n' \
            "$(ts)" "$core" "$session" "$tier" "$est" "$concurrency"
        (( idx++ )); placed=1
        sleep 1
        (( idx < ${#JOBS[@]} )) || break
    done
    (( placed )) || sleep "$POLL_SECONDS"
done

echo "[grid] all ${#JOBS[@]} jobs dispatched; $(live) still running"
echo "[grid] progress:  column -t -s'|' $STATUS | sort -k2"
