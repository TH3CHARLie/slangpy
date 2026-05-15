# Minimized Unroll Tunable Benchmark

This folder adapts the standalone CUDA minimized-unroll benchmark into the
SlangPy autotuning story while preserving the original fused task structure.

`loop_bench_minimal.slang` is the only shader used for this benchmark. It keeps
the original shape:

- `CopyTask<3>`
- `ExpTask<3>`
- `SigmoidTask<1>`
- a feature copy task
- an optional merged-signal task for the large workload
- `fused_apply_small`
- `fused_apply_large`

The shader exposes three reflected tuning axes:

- `[Tunable] extern struct TunableLoopStrategy : ILoopStrategy`
- `[Tunable(1, 2, 4, 8, 20, 32)] extern static const int kFeatureWidth`
- `[Tunable(1, 2, 4, 8, 16, 32, 45, 64)] extern static const int kMergedSignalB`

The derived unrolled-iteration counts are:

```text
small_iters = 3 + 3 + 1 + kFeatureWidth
large_iters = 3 + 3 + 1 + kFeatureWidth + 3 + kMergedSignalB
```

The original benchmark points remain in the search space:

```text
kFeatureWidth = 20
kMergedSignalB = 45
small_iters = 27
large_iters = 75
```

The full reflected shader space has `2 * 6 * 8 = 96` configs. The paper-facing
flip sweep filters that space to:

```text
kFeatureWidth: 1, 2, 4, 8, 20
kMergedSignalB: 1, 2, 4, 8, 16, 45
```

That filtered sweep executes 70 rows:

- 10 deduplicated small rows, because `kMergedSignalB` is inactive for small
- 60 large rows

The intended paper figure is the flip plot: runtime ratio vs. effective
unrolled iterations. Low-width cases favor `ForceUnroll`; increasing the
merged-signal width flips the best strategy to `MaxIters`. The resource table
reports registers/spills for representative points, especially the original
27-iteration and 75-iteration cases.
