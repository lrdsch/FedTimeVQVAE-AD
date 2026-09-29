#!/usr/bin/env bash
# run_tests.sh — does this checkout work? One command, four layers.
#
#   bash scripts/run_tests.sh              # static + unit tests + the shipped results (CPU)
#   bash scripts/run_tests.sh --smoke      # ... plus every arm trained end to end on a GPU
#   PY=/path/to/python bash scripts/run_tests.sh
#
# Layer 1  static     every file byte-compiles, every module imports, the shell drivers parse
# Layer 2  unit       the aggregation math: CPU, seconds, no GPU
# Layer 3  results    the paper's tables re-derived from artifacts/runs/ (skipped if absent)
# Layer 4  smoke      --smoke only: every reporting arm trains, saves and scores on one GPU at
#                     collapsed budgets. Minutes, and the numbers it produces are meaningless
#                     by construction -- what it proves is that no arm raises or silently
#                     fails to write its json.
#
# Layers 2 and 4 need the synthetic dataset; it is generated here if missing (under a second).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PY:-python3}"
SMOKE=0; [[ "${1:-}" == "--smoke" ]] && SMOKE=1

PASS=0; FAIL=0; SKIP=0; LOG="$REPO/artifacts/_tests"; mkdir -p "$LOG"
ok()   { PASS=$((PASS+1)); printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m  %s   %s\n' "$1" "${2:-}"; }
skip() { SKIP=$((SKIP+1)); printf '  ----  %s   %s\n' "$1" "${2:-}"; }
run()  {   # run <label> <logfile> <cmd...>
  local label="$1" lf="$LOG/$2"; shift 2
  if "$@" > "$lf" 2>&1; then ok "$label"; else bad "$label" "-> $lf"; fi
}

echo "== 1. static =="
run "every file byte-compiles" compileall.log "$PY" -m compileall -q "$REPO/config.py" \
    "$REPO/data.py" "$REPO/utils.py" "$REPO/metrics_core.py" "$REPO/model" "$REPO/pipeline" \
    "$REPO/metrics" "$REPO/lib" "$REPO/scripts" "$REPO/preprocessing"
run "the pipeline imports" imports.log "$PY" -c "
import sys; sys.path[:0] = ['$REPO', '$REPO/pipeline', '$REPO/scripts', '$REPO/model']
import config, data, utils, metrics_core, federated, federated_eval, detect, cf_eval
assert federated_eval.PAPER_ARMS"
for s in launch.sh smoke_arms.sh paper_runs.sh; do
  if bash -n "scripts/$s" 2>"$LOG/syntax_$s.log"; then ok "scripts/$s parses"
  else bad "scripts/$s parses" "-> $LOG/syntax_$s.log"; fi
done

echo "== 2. unit (the aggregation math) =="
[[ -d data/raw/toy_fed_uni ]] || "$PY" scripts/build_toy_fed_uni.py > "$LOG/build_toy.log" 2>&1
for t in fed_codebook_unittest fed_cb_server_ema_unittest fed_enc_algo_unittest \
         fed_regression_unittest fed_val_selection_unittest; do
  CUDA_VISIBLE_DEVICES="" run "$t" "$t.log" "$PY" "scripts/$t.py"
done

echo "== 3. the paper's numbers, from the shipped results =="
if [[ -d artifacts/runs/zn_a2/ucr_split_w2p ]]; then
  run "paper_numbers.py (Sections VI-A..VI-E)" paper_numbers.log "$PY" scripts/paper_numbers.py
  run "c50_table.py (Table IV)" c50_table.log "$PY" scripts/c50_table.py
  run "fusion_probe.py (configuration (n))" fusion_probe.log \
      "$PY" scripts/fusion_probe.py --run artifacts/runs/zn_main --all-series
else
  skip "the shipped results" "artifacts/runs/ is empty in this checkout"
fi

echo "== 4. the launcher =="
if [[ -d data/raw/ucr_split_w2p ]]; then
  run "launch.sh --dry over the development cohort" launch_dry.log \
      bash scripts/launch.sh --cohort ucr2p_10 --arms local,centralized --tag _selftest --dry
else
  skip "launch.sh --dry" "build the benchmark first (docs/REPRODUCE.md §3)"
fi

if [[ $SMOKE -eq 1 ]]; then
  echo "== 5. every arm trains, saves and scores (GPU, minutes) =="
  if bash scripts/smoke_arms.sh > "$LOG/smoke_arms.log" 2>&1; then
    ok "smoke_arms.sh ($(grep -c '^PASS' "$LOG/smoke_arms.log") cases)"
  else
    bad "smoke_arms.sh" "-> $LOG/smoke_arms.log"
    grep '^FAIL' "$LOG/smoke_arms.log" | sed 's/^/        /'
  fi
else
  skip "smoke_arms.sh" "pass --smoke to train every arm on a GPU"
fi

echo
echo "$PASS passed · $FAIL failed · $SKIP skipped   (logs: $LOG)"
exit $(( FAIL > 0 ))
