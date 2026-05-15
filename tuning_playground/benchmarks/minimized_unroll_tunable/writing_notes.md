# Minimal Unroll Writing Notes

These notes are for paper narration. Use `README.md` for benchmark mechanics;
use this file to decide what claim the minimal-unroll example supports and
which artifact to cite.

## Workload Story

The benchmark is a minimized fused autodiff kernel built from task-style buffer
operations. The shader keeps the same small tasks as the original CUDA example:
fixed-width copy, exp, sigmoid, a feature copy, and an optional merged-signal
task. Those tasks are fused into two entry points:

- `fused_apply_small`: copy/exp/sigmoid plus feature work.
- `fused_apply_large`: the same prefix plus a merged-signal task that increases
  live state and resource pressure.

The useful paper point is that the best loop lowering strategy changes with the
workload/resource regime. Small fused loops can benefit from `[ForceUnroll]`.
The large merged-signal loop becomes register- and spill-bound when forced to
unroll, so `[MaxIters]` wins by keeping resource use under control.

Do not frame this as iteration count alone explaining the result. The 75-iter
case is also different because of the merged-signal shape and its resource
pressure.

## Reflected Tunables

The shader exposes three reflected tuning axes:

- `TunableLoopStrategy`: `ForceUnrollLoop`, `MaxItersLoop`
- `kFeatureWidth`: `1, 2, 4, 8, 20, 32`
- `kMergedSignalB`: `1, 2, 4, 8, 16, 32, 45, 64`

The derived iteration counts are:

```text
small_iters = 3 + 3 + 1 + kFeatureWidth
large_iters = 3 + 3 + 1 + kFeatureWidth + 3 + kMergedSignalB
```

The original benchmark points are in the reflected search space:

- Small: `kFeatureWidth=20`, so `small_iters=27`.
- Large: `kFeatureWidth=20` and `kMergedSignalB=45`, so `large_iters=75`.

## Generated Paper Artifacts

These files are generated from the benchmark and plotting scripts and are not
required in a source-only checkout:

- `minimized_unroll_flip.png`: the paper-facing plot showing the strategy flip.
- `minimized_unroll_flip_table.md`: timing table for each swept point.
- `minimized_unroll_flip_resource_table.md`: registers and spill evidence for
  the original 27-iter and 75-iter points.
- `results_minimized_unroll_minimal_flip.jsonl`: raw timing rows behind the
  plot and tables.

Suggested figure/table mapping:

- The figure supports the high-level claim that the preferred strategy flips as
  the workload moves into the merged-signal/resource-bound regime.
- The timing table supports the exact runtimes and ratios.
- The resource table supports the mechanism: forced unrolling in the large case
  drives register pressure and spills, while `MaxItersLoop` avoids them.

## Paper-Ready Numbers

Representative original points:

- 27-iter small workload: `ForceUnrollLoop` is `577.5 us`; `MaxItersLoop` is
  `610.8 us`. `ForceUnrollLoop` is about `5.8%` faster by runtime ratio
  (`610.8 / 577.5 = 1.058`).
- 75-iter large workload: `ForceUnrollLoop` is `4347.6 us`; `MaxItersLoop` is
  `1810.0 us`. `MaxItersLoop` is about `2.4x` faster by runtime ratio
  (`4347.6 / 1810.0 = 2.40`).
- Resource evidence for the 75-iter large workload: `ForceUnrollLoop` uses
  `255` registers/thread and spills (`3192 B` stores, `1600 B` loads), while
  `MaxItersLoop` uses `32` registers/thread and has no spills.

## Suggested Phrasing

A concise paper phrasing:

> In the minimized fused autodiff benchmark, autotuning selects different loop
> lowering strategies for different resource regimes. For the original small
> 27-iteration point, forced unrolling is modestly faster. For the original
> large 75-iteration merged-signal point, forced unrolling pushes the backward
> kernel to 255 registers/thread and introduces spills; the bounded
> `MaxItersLoop` version avoids spills and runs about 2.4x faster.

Safe claim:

- The autotuner discovers a strategy flip between unroll-friendly small fused
  loops and resource-bound merged-signal loops.

Avoid:

- Claiming that iteration count alone determines the best strategy.
- Generalizing the exact crossover point beyond this benchmark shape and GPU.
