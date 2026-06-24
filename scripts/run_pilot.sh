#!/usr/bin/env bash
# run_pilot.sh -- end-to-end smoke test of the whole pipeline on the MOCK
# simulator. Run this first; it should finish in well under a minute and prove
# every stage works before you touch ParaEMT.
set -euo pipefail
cd "$(dirname "$0")/.."

# activate venv if present
[ -d .venv ] && source .venv/bin/activate

echo "== 1/4 label validation (swing-equation cross-check) =="
python -m src.validate_labels

echo; echo "== 2/4 generate pilot dataset (40 per disturbance) =="
python -m src.generate_dataset --n 40 --jobs "${JOBS:-8}" \
    --out data/processed/pilot.csv

echo; echo "== 3/4 train surrogates (per disturbance) =="
python -m src.train --data data/processed/pilot.csv --target all --per-disturbance

echo; echo "== 4/4 active-learning demo (islanding / RoCoF) =="
python -m src.active_learning --disturbance islanding --target rocof_hz_s \
    --seed-n 15 --queries 30 --pool 300

echo; echo "Pilot complete. If this all ran, the pipeline is sound --"
echo "now wire in ParaEMT (see README) and flip USE_MOCK=False."
