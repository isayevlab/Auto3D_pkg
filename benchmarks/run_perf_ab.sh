#!/usr/bin/env bash
# benchmarks/run_perf_ab.sh <base-ref>
#
# ONE command: benchmark <base-ref> and the current working tree on the same GPU,
# then print a CHANGELOG-ready block. Nothing to interpret, nothing to assemble.
#
#   bash benchmarks/run_perf_ab.sh v4.0.0
#
# A read-only git worktree of <base-ref> is created in a temp dir and removed on
# exit. Auto3D is pure Python, so PYTHONPATH is enough -- no reinstall, and the
# two runs cannot contaminate each other's site-packages.
#
# Note that the *measurement code* comes from the current tree in both runs
# (only PYTHONPATH differs), which is deliberate: identical instrumentation,
# different measured code. bench_optimization_perf.py aborts if it detects both
# runs importing Auto3D from the same tree.
set -euo pipefail

BASE="${1:?usage: run_perf_ab.sh <base-ref>   (e.g. v4.0.0, or a SHA)}"
REPO="$(git rev-parse --show-toplevel)"
BENCH="$REPO/benchmarks/bench_optimization_perf.py"
TMP="$(mktemp -d)"
WT="$TMP/base"

cleanup() {
    git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true
    rm -rf "$TMP"
}
trap cleanup EXIT

if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "ABORT: no CUDA device visible. This benchmark measures the removal of" >&2
    echo "       host-device synchronizations, which do not exist on CPU." >&2
    exit 1
fi

echo "== creating read-only worktree of $BASE =="
git -C "$REPO" worktree add --detach "$WT" "$BASE" >/dev/null

echo "== baseline ($BASE) =="
PYTHONPATH="$WT/src" python "$BENCH" --label before

echo "== branch ($(git -C "$REPO" rev-parse --abbrev-ref HEAD)) =="
PYTHONPATH="$REPO/src" python "$BENCH" --label after

echo
echo "== comparison (paste the block below into CHANGELOG.md) =="
PYTHONPATH="$REPO/src" python "$BENCH" --compare before after
