#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run_experiment.sh
#  Full-pipeline runner for the CSE-CIC-IDS2018 IDS research project.
#
#  Requirements:
#    - Python 3.10+
#    - CSV files present in ./data/  (Wednesday-14-02, Thursday-15-02,
#      Friday-02-03-2018 TrafficForML_CICFlowMeter.csv)
#
#  Usage:
#    ./run_experiment.sh                  # full run
#    ./run_experiment.sh --skip-setup     # skip virtualenv + deps
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  IDS Research Pipeline — CSE-CIC-IDS2018"
echo "  Project root: ${PROJECT_ROOT}"
echo "═══════════════════════════════════════════════════════════════════════"

# ── Setup ────────────────────────────────────────────────────────────────────
if [[ "${1:-}" != "--skip-setup" ]]; then
    if [[ ! -d ".venv" ]]; then
        echo "[1/3] Creating virtual environment..."
        python3 -m venv .venv
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate

    echo "[2/3] Installing dependencies..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
else
    echo "[1-2/3] Skipping setup (--skip-setup passed)."
    [[ -d ".venv" ]] && source .venv/bin/activate
fi

# ── Sanity check: data present ───────────────────────────────────────────────
echo "[3/3] Checking dataset..."
for f in \
    "data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv" \
    "data/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv" \
    "data/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"; do
    if [[ ! -f "${f}" ]]; then
        echo "  ERROR: missing ${f}"
        echo "  Download the CSVs from https://www.unb.ca/cic/datasets/ids-2018.html"
        exit 1
    fi
done
echo "  ✓ All CSVs present"

# ── Run pipeline ─────────────────────────────────────────────────────────────
echo
echo "──────────────────────────────────────────────────────────────────────"
echo "  Running main.py  (full pipeline — this will take time)"
echo "──────────────────────────────────────────────────────────────────────"
python main.py

echo
echo "═══════════════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "  Results are in ./results/"
echo "═══════════════════════════════════════════════════════════════════════"
