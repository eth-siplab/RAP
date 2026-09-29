#!/usr/bin/env bash
# --------------------------------------------
# End-to-end test runner
# --------------------------------------------
# Restore Anything Pipeline: Segment Anything Meets Image Restoration (arXiv 2023)
# https://github.com/eth-siplab/RAP
# Jiaxi Jiang (https://jiaxi-jiang.com/)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich

# End-to-end checks against a running server (python -m rap).
# Each script prints what it measured and saves screenshots/images to tests/out/;
# a script fails if it exits with an error or reports page/console errors.
#   PYTHON=...          interpreter with playwright (default: python)
#   RAP_URL=...         server to test (default: http://127.0.0.1:7860)
#   RAP_EXTRA_LIBS=...  extra LD_LIBRARY_PATH for headless Chromium (e.g. a dir with libasound.so.2)
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
[ -n "$RAP_EXTRA_LIBS" ] && export LD_LIBRARY_PATH=$RAP_EXTRA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
mkdir -p tests/out/ui tests/out/logs
failed=()

run() {  # run <name> <tail lines>
  local log=tests/out/logs/$1.log
  timeout 900 "$P" "tests/$1.py" > "$log" 2>&1
  local code=$?
  echo "== $1"
  grep -v Warning "$log" | tail -"$2"
  if [ $code -ne 0 ] || grep -qE "Traceback|pageerror|^errors: \[.+\]|^errors: [^n\[]" "$log"; then
    echo "   FAILED (exit $code, see $log)"
    failed+=("$1")
  fi
}

run test_kair 6
run test_kair_tile 2
run test_api 3
run test_api2 5
run test_edit_layers 4
for t in test_ui test_ui_regress test_faces_check test_adjust test_adjust2 test_ui_brush test_ui_remove test_ui_auto \
         test_ui_flows test_ui_crop test_history_export test_ui_bg test_ui_genfill test_ui_human test_nav test_ui_kair; do
  run $t 3
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "ALL PASSED"
else
  echo "FAILED: ${failed[*]}"
  exit 1
fi
