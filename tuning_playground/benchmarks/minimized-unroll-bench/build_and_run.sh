#!/bin/bash
# build_and_run.sh — compile, analyze registers, and benchmark
# apply_to_buffer with [ForceUnroll] vs [MaxIters(N)].
#
# Usage:
#   SLANGC=/path/to/slangc bash build_and_run.sh
#
# Optional env vars:
#   SLANGC  — path to slangc (default: slangc in PATH)
#   NVCC    — path to nvcc   (default: /usr/local/cuda/bin/nvcc)
#   ARCH    — GPU arch        (default: sm_89)
#   RUNS    — benchmark runs  (default: 5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SLANGC="${SLANGC:-slangc}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_89}"
RUNS="${RUNS:-5}"

SRC="$SCRIPT_DIR/loop_bench.slang"
BENCH="$SCRIPT_DIR/bench_launcher.cu"
OUTDIR="/tmp/loop-strategy-bench"

NVCC_FLAGS="-arch=$ARCH --use_fast_math \
  -D__CUDA_NO_HALF_OPERATORS__ \
  -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__"

mkdir -p "$OUTDIR"

echo "=== Configuration ==="
echo "  slangc : $SLANGC"
echo "  nvcc   : $NVCC"
echo "  arch   : $ARCH"
echo "  runs   : $RUNS"
echo ""

# --- Phase 1: Compile both Slang variants ---

echo "=== Phase 1: Slang compilation ==="

echo -n "  [ForceUnroll]... "
$SLANGC "$SRC" -O3 -target cuda -o "$OUTDIR/unroll.cu"
echo "done"

echo -n "  [MaxIters]...    "
$SLANGC "$SRC" -O3 -target cuda -DUSE_MAXITERS -o "$OUTDIR/maxiters.cu"
echo "done"
echo ""

# --- Phase 2: Register analysis ---

echo "=== Phase 2: Register counts (backward kernels) ==="

echo ""
echo "--- [ForceUnroll] ---"
$NVCC $NVCC_FLAGS --ptxas-options=-v -cubin "$OUTDIR/unroll.cu" \
  -o "$OUTDIR/unroll.cubin" 2>&1 | grep -E "fused_apply.*_bwd_diff$" -A2

echo ""
echo "--- [MaxIters] ---"
$NVCC $NVCC_FLAGS --ptxas-options=-v -cubin "$OUTDIR/maxiters.cu" \
  -o "$OUTDIR/maxiters.cubin" 2>&1 | grep -E "fused_apply.*_bwd_diff$" -A2
echo ""

# --- Phase 3: Build benchmark launchers ---

echo "=== Phase 3: Building benchmark launchers ==="

echo -n "  [ForceUnroll]... "
$NVCC $NVCC_FLAGS -DBENCH_CUDA_PATH="$OUTDIR/unroll.cu" \
  "$BENCH" -o "$OUTDIR/bench_unroll" 2>/dev/null
echo "done"

echo -n "  [MaxIters]...    "
$NVCC $NVCC_FLAGS -DBENCH_CUDA_PATH="$OUTDIR/maxiters.cu" \
  "$BENCH" -o "$OUTDIR/bench_maxiters" 2>/dev/null
echo "done"
echo ""

# --- Phase 4: Run benchmarks ---

echo "=== Phase 4: Runtime benchmark ($RUNS runs, 1000 iters each) ==="
echo ""

for run in $(seq 1 "$RUNS"); do
  echo "--- Run $run ---"
  echo "[ForceUnroll]:"
  "$OUTDIR/bench_unroll"
  echo "[MaxIters]:"
  "$OUTDIR/bench_maxiters"
  echo ""
done
