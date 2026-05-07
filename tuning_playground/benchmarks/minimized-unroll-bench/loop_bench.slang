// Minimal benchmark: ForceUnroll vs MaxIters for autodiff loop strategy.
//
// Demonstrates that fused kernels with large unrolled loops cause register
// pressure blowup under [ForceUnroll], while [MaxIters] uses loop checkpointing
// to keep register count low at the cost of minor recomputation.
//
// Compile variants:
//   Default:        autodiff with [ForceUnroll]
//   -DUSE_MAXITERS: autodiff with [MaxIters] instead of [ForceUnroll]
//
// Build:
//   slangc loop_bench.slang -O3 -target cuda -o out/unroll.cu
//   slangc loop_bench.slang -O3 -target cuda -DUSE_MAXITERS -o out/maxiters.cu

// ============================================================
// Utility
// ============================================================

namespace utils {

// KEY ANNOTATION #1: [PreferRecompute] on get_index.
// This is a pure function of no_diff values. Without this annotation,
// autodiff checkpoints a uint2 (8 bytes) per unrolled iteration.
[PreferRecompute]
uint2 get_index<let N : int>(no_diff uint index)
{
    const uint index_i = index / N;
    const uint index_j = index - index_i * N;
    return uint2(index_i, index_j);
}

} // namespace utils

// ============================================================
// Unary functions applied element-wise to buffer data
// ============================================================

interface IElementOp
{
    [Differentiable]
    static float apply(float x);
};

struct CopyOp : IElementOp
{
    [Differentiable]
    static float apply(float x) { return x; }
};

struct ExpOp : IElementOp
{
    [Differentiable]
    static float apply(float x) { return exp(x); }
};

[Differentiable]
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

struct SigmoidOp : IElementOp
{
    [Differentiable]
    static float apply(float x) { return sigmoid(x); }
};

// ============================================================
// apply_to_buffer: THE FUNCTION UNDER TEST
//
// KEY ANNOTATION #2: [PreferRecompute] on reload_and_apply.
// Without this, autodiff checkpoints the loaded float (4 bytes) per
// unrolled iteration. With it, the backward pass re-reads from memory
// instead — trading one extra global load for N*4 bytes less register
// pressure.
// ============================================================

[PreferRecompute]
[Differentiable]
float reload_and_apply<Op>(DiffTensorView input, uint2 index)
    where Op : IElementOp
{
    return Op.apply(input.loadOnce(index));
}

[Differentiable]
void apply_to_buffer<Op, let N : int, let ColOffset : int>(
    DiffTensorView output, DiffTensorView input,
    no_diff uint tid, no_diff uint count, no_diff uint row_offset
    ) where Op : IElementOp
{
#ifdef USE_MAXITERS
    [MaxIters(N)]
#else
    [ForceUnroll]
#endif
    for (int i = 0; i < N; ++i) {
        const uint2 index = utils::get_index<N>(i * count + tid);
        const uint2 out_index = uint2(index.x + row_offset, uint(ColOffset) + index.y);
        output.storeOnce(out_index, reload_and_apply<Op>(input, index));
    }
}

// ============================================================
// Task interface — each task processes one attribute buffer
// ============================================================

interface ITask
{
    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset);
};

// --- Simple tasks: apply a unary op to a fixed-width buffer ---

struct CopyTask<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        apply_to_buffer<CopyOp, N, 0>(output, input, tid, count, row_offset);
    }
};

struct ExpTask<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        apply_to_buffer<ExpOp, N, 0>(output, input, tid, count, row_offset);
    }
};

struct SigmoidTask<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        apply_to_buffer<SigmoidOp, N, 0>(output, input, tid, count, row_offset);
    }
};

// --- Merged copy: reads from two source buffers into one output ---
// This has a large loop (A+B iterations) and is the primary source of
// register pressure when unrolled in a fused kernel.

struct MergedCopyTask<let A : int, let B : int> : ITask
{
    DiffTensorView source_a;
    DiffTensorView source_b;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        static const int total = A + B;

#ifdef USE_MAXITERS
        [MaxIters(total)]
#else
        [ForceUnroll]
#endif
        for (int i = 0; i < total; ++i) {
            const var index = utils::get_index<total>(i * count + tid);
            float value;
            if (index.y < A) {
                value = source_a.loadOnce(index);
            } else {
                value = source_b.loadOnce(uint2(index.x, index.y - A));
            }
            const var out_index = uint2(index.x + row_offset, index.y);
            output.storeOnce(out_index, value);
        }
    }
};

// ============================================================
// Fused kernel instantiations
//
// fused_apply_small: moderate complexity (27 unrolled iterations)
//   → ForceUnroll fits in registers, MaxIters adds slight overhead
//
// fused_apply_large: high complexity (75 unrolled iterations)
//   → ForceUnroll causes register spills, MaxIters is much faster
// ============================================================

[CUDAKernel] [Differentiable] [AutoPyBindCUDA]
void fused_apply_small(
    no_diff uint row_offset, no_diff uint count,
    CopyTask<3> vectors,
    ExpTask<3> scales,
    SigmoidTask<1> opacities,
    CopyTask<20> features
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    vectors.run(tid, count, row_offset);
    scales.run(tid, count, row_offset);
    opacities.run(tid, count, row_offset);
    features.run(tid, count, row_offset);
}

[CUDAKernel] [Differentiable] [AutoPyBindCUDA]
void fused_apply_large(
    no_diff uint row_offset, no_diff uint count,
    CopyTask<3> vectors,
    ExpTask<3> scales,
    SigmoidTask<1> opacities,
    CopyTask<20> features,
    MergedCopyTask<3, 45> merged_signal
) {
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count) return;
    vectors.run(tid, count, row_offset);
    scales.run(tid, count, row_offset);
    opacities.run(tid, count, row_offset);
    features.run(tid, count, row_offset);
    merged_signal.run(tid, count, row_offset);
}
