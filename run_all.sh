#!/usr/bin/env bash
# run_all.sh - build (if needed), validate, simulate and analyse in one go.
#
#   ./run_all.sh                       # uses config.yaml
#   ./run_all.sh --config my.yaml      # another configuration file
#   ./run_all.sh --events 500 --seed 7 # override events / seed from the command line
#   ./run_all.sh --mode screening      # gamma screening only
#   ./run_all.sh --archive e356        # also copy the finished run to results/archive/e356
#                                      # (analyze.py uses archived runs of the same material and
#                                      #  geometry at different gamma energies for linearity plots)
#   ./run_all.sh --no-analysis | --rebuild | --validate-only
#
# Geant4 must be findable by CMake (source geant4.sh, or export CMAKE_PREFIX_PATH / Geant4_DIR).
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=config.yaml; ARCHIVE=""; ANALYSIS=1; REBUILD=0; VALONLY=0; EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --archive) ARCHIVE="$2"; shift 2;;
    --events|--seed|--mode|--outdir) EXTRA+=("$1" "$2"); shift 2;;
    --no-analysis) ANALYSIS=0; shift;;
    --rebuild) REBUILD=1; shift;;
    --validate-only) VALONLY=1; shift;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done

if [[ -n "${GEANT4_SH:-}" ]]; then source "$GEANT4_SH"; fi

# ---- build ---------------------------------------------------------------------------------
if [[ $REBUILD -eq 1 || ! -x build/CsPbBr3Sim || CsPbBr3Sim.cc -nt build/CsPbBr3Sim ]]; then
  echo "== Building =="
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS:-} > build/cmake.log 2>&1 || { mkdir -p build; cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS:-}; }
  cmake --build build -j "$(nproc 2>/dev/null || echo 2)"
fi

# ---- validate ---------------------------------------------------------------------------
echo "== Validating $CONFIG =="
./build/CsPbBr3Sim "$CONFIG" --validate ${EXTRA[@]+"${EXTRA[@]}"}
[[ $VALONLY -eq 1 ]] && exit 0

# ---- simulate ---------------------------------------------------------------------------
echo "== Simulating =="
./build/CsPbBr3Sim "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}

# results directory = output.directory of the config (or --outdir)
RESULTS=$(python3 - "$CONFIG" "${EXTRA[@]+"${EXTRA[@]}"}" <<'PY'
import sys
args=sys.argv[1:]; out=None
if "--outdir" in args: out=args[args.index("--outdir")+1]
else:
    try:
        import yaml; out=yaml.safe_load(open(args[0]))["output"]["directory"]
    except Exception: out="results"
print(out)
PY
)

if [[ -n "$ARCHIVE" ]]; then
  mkdir -p "$RESULTS/archive/$ARCHIVE"
  cp "$RESULTS"/events.csv "$RESULTS"/summary.json "$RESULTS/archive/$ARCHIVE/"
  echo "archived run -> $RESULTS/archive/$ARCHIVE"
fi

# ---- analyse ------------------------------------------------------------------------------
if [[ $ANALYSIS -eq 1 ]]; then
  echo "== Analysing =="
  python3 analyze.py --results "$RESULTS"
fi
