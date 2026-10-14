#!/usr/bin/env bash
# Usage:  nohup bash run_all.sh heavy > run.log 2>&1 &
#         (profile: standard or heavy; default heavy)
# Safe to rerun after a stop: every step continues where it stopped.
set -u
cd "$(dirname "$0")"
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
PROFILE=${1:-heavy}
mkdir -p results
if [ ! -s results/mc_batches.csv ] && [ -z "${SKIP_SMOKE:-}" ]; then
  echo "== smoke test"; $PY campaign/smoke_test.py || { echo "smoke test failed; stopping"; exit 1; }
fi
echo "== Part A: Monte Carlo ($PROFILE)"; $PY campaign/run_mc.py --profile "$PROFILE"
[ $? -eq 2 ] && exit 2
echo "== analysis and bundle after Part A"; $PY campaign/analyze.py > /dev/null; $PY make_results_zip.py
if [ "$PROFILE" = "heavy" ]; then
  echo "== Part D: paired decoder check"; $PY campaign/decoder_check.py --shots 2000
  echo "== Part C: circuit fault distances"; $PY campaign/run_distances.py fault --spacings 2 3 --time-limit 1800
  echo "== Part B: Steane-input code distance"; $PY campaign/run_distances.py code --spacings 2 --time-limit 7200
  echo "== final analysis and bundle"; $PY campaign/analyze.py > /dev/null; $PY make_results_zip.py
fi
echo "== all done"
