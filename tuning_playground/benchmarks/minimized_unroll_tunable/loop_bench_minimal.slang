// SPDX-License-Identifier: Apache-2.0
//
// Minimal SlangPy/tuning adaptation of minimized-unroll-bench/loop_bench.slang.
//
// The original benchmark selects [ForceUnroll] vs [MaxIters] with
// -DUSE_MAXITERS. This version keeps the element/task structure intact and
// exposes that policy and the effective fused loop width as Slang tunables.

import slangpy;

[Tunable(1, 2, 4, 8, 20, 32)]
public extern static const int kFeatureWidth = 20;

[Tunable(1, 2, 4, 8, 16, 32, 45, 64)]
public extern static const int kMergedSignalB = 45;

static const int kSignalAWidth = 3;

namespace utils {

[PreferRecompute]
uint2 get_index<let N : int>(no_diff uint index)
{
    const uint index_i = index / N;
    const uint index_j = index - index_i * N;
    return uint2(index_i, index_j);
}

} // namespace utils

public interface IElementOp
{
    [Differentiable]
    static float apply(float x);
};

public struct CopyOp : IElementOp
{
    [Differentiable]
    static float apply(float x) { return x; }
};

public struct ExpOp : IElementOp
{
    [Differentiable]
    static float apply(float x) { return exp(x); }
};

[Differentiable]
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

public struct SigmoidOp : IElementOp
{
    [Differentiable]
    static float apply(float x) { return sigmoid(x); }
};

[PreferRecompute]
[Differentiable]
float reload_and_apply<Op>(DiffTensorView input, uint2 index)
    where Op : IElementOp
{
    return Op.apply(input.loadOnce(index));
}

public interface ILoopStrategy
{
    [Differentiable]
    static void apply_to_buffer<Op, let N : int, let ColOffset : int>(
        DiffTensorView output,
        DiffTensorView input,
        no_diff uint tid,
        no_diff uint count,
        no_diff uint row_offset)
        where Op : IElementOp;

    [Differentiable]
    static void merged_copy<let A : int, let B : int>(
        DiffTensorView source_a,
        DiffTensorView source_b,
        DiffTensorView output,
        no_diff uint tid,
        no_diff uint count,
        no_diff uint row_offset);
};

public struct ForceUnrollLoop : ILoopStrategy
{
    [Differentiable]
    public static void apply_to_buffer<Op, let N : int, let ColOffset : int>(
        DiffTensorView output,
        DiffTensorView input,
        no_diff uint tid,
        no_diff uint count,
        no_diff uint row_offset)
        where Op : IElementOp
    {
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
        {
            const uint2 index = utils::get_index<N>(i * count + tid);
            const uint2 out_index = uint2(index.x + row_offset, uint(ColOffset) + index.y);
            output.storeOnce(out_index, reload_and_apply<Op>(input, index));
        }
    }

    [Differentiable]
    public static void merged_copy<let A : int, let B : int>(
        DiffTensorView source_a,
        DiffTensorView source_b,
        DiffTensorView output,
        no_diff uint tid,
        no_diff uint count,
        no_diff uint row_offset)
    {
        static const int total = A + B;

        [ForceUnroll]
        for (int i = 0; i < total; ++i)
        {
            const var index = utils::get_index<total>(i * count + tid);
            float value;
            if (index.y < A)
            {
                value = source_a.loadOnce(index);
            }
            else
            {
                value = source_b.loadOnce(uint2(index.x, index.y - A));
            }
            const var out_index = uint2(index.x + row_offset, index.y);
            output.storeOnce(out_index, value);
        }
    }
};

public struct MaxItersLoop : ILoopStrategy
{
    [Differentiable]
    public static void apply_to_buffer<Op, let N : int, let ColOffset : int>(
        DiffTensorView output,
        DiffTensorView input,
        no_diff uint tid,
        no_diff uint count,
        no_diff uint row_offset)
        where Op : IElementOp
    {
        [MaxIters(N)]
        for (int i = 0; i < N; ++i)
        {
            const uint2 index = utils::get_index<N>(i * count + tid);
            const uint2 out_index = uint2(index.x + row_offset, uint(ColOffset) + index.y);
            output.storeOnce(out_index, reload_and_apply<Op>(input, index));
        }
    }

    [Differentiable]
    public static void merged_copy<let A : int, let B : int>(
        DiffTensorView source_a,
        DiffTensorView source_b,
        DiffTensorView output,
        no_diff uint tid,
        no_diff uint count,
        no_diff uint row_offset)
    {
        static const int total = A + B;

        [MaxIters(total)]
        for (int i = 0; i < total; ++i)
        {
            const var index = utils::get_index<total>(i * count + tid);
            float value;
            if (index.y < A)
            {
                value = source_a.loadOnce(index);
            }
            else
            {
                value = source_b.loadOnce(uint2(index.x, index.y - A));
            }
            const var out_index = uint2(index.x + row_offset, index.y);
            output.storeOnce(out_index, value);
        }
    }
};

[Tunable]
public extern struct TunableLoopStrategy : ILoopStrategy = ForceUnrollLoop;

public interface ITask
{
    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset);
};

public struct CopyTask<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        TunableLoopStrategy::apply_to_buffer<CopyOp, N, 0>(output, input, tid, count, row_offset);
    }
};

public struct ExpTask<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        TunableLoopStrategy::apply_to_buffer<ExpOp, N, 0>(output, input, tid, count, row_offset);
    }
};

public struct SigmoidTask<let N : int> : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        TunableLoopStrategy::apply_to_buffer<SigmoidOp, N, 0>(output, input, tid, count, row_offset);
    }
};

public struct MergedCopyTask<let A : int, let B : int> : ITask
{
    DiffTensorView source_a;
    DiffTensorView source_b;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        TunableLoopStrategy::merged_copy<A, B>(source_a, source_b, output, tid, count, row_offset);
    }
};

public struct FeatureTask : ITask
{
    DiffTensorView input;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        TunableLoopStrategy::apply_to_buffer<CopyOp, kFeatureWidth, 0>(output, input, tid, count, row_offset);
    }
};

public struct MergedSignalTask : ITask
{
    DiffTensorView source_a;
    DiffTensorView source_b;
    DiffTensorView output;

    [Differentiable]
    void run(no_diff uint tid, no_diff uint count, no_diff uint row_offset)
    {
        TunableLoopStrategy::merged_copy<kSignalAWidth, kMergedSignalB>(source_a, source_b, output, tid, count, row_offset);
    }
};

[CUDAKernel]
[Differentiable]
public void fused_apply_small(
    no_diff uint row_offset,
    no_diff uint count,
    CopyTask<3> vectors,
    ExpTask<3> scales,
    SigmoidTask<1> opacities,
    FeatureTask features)
{
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    vectors.run(tid, count, row_offset);
    scales.run(tid, count, row_offset);
    opacities.run(tid, count, row_offset);
    features.run(tid, count, row_offset);
}

[CUDAKernel]
[Differentiable]
public void fused_apply_large(
    no_diff uint row_offset,
    no_diff uint count,
    CopyTask<3> vectors,
    ExpTask<3> scales,
    SigmoidTask<1> opacities,
    FeatureTask features,
    MergedSignalTask merged_signal)
{
    const uint tid = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (tid >= count)
        return;
    vectors.run(tid, count, row_offset);
    scales.run(tid, count, row_offset);
    opacities.run(tid, count, row_offset);
    features.run(tid, count, row_offset);
    merged_signal.run(tid, count, row_offset);
}
