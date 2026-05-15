# Neural Texture Writing Notes

These notes are for paper narration. Use the benchmark scripts for mechanics;
use this file to decide what the neural-texture example supports and which
artifact to cite.

## Workload Story

The benchmark is a self-contained neural-texture forward-inference kernel. For
each UV sample, the shader applies frequency encoding, evaluates a small MLP,
and returns an RGB value. The parameter buffer is initialized once per network
shape and reused across activation variants, so comparisons within a shape use
the same weights.

The useful paper point is that Slang can expose a mixed tuning space directly
from the shader:

- interface-based algorithm choice for the activation function
- integer specialization axes for network width, depth, and frequency bands

The Python driver discovers the full space by reflection and specializes
variants with module linking. It does not mirror the activation list or network
shape choices in a separate Python search-space definition.

The shader is intentionally split across two Slang files to demonstrate modular
tunables:

- `activation.slang` declares the structural activation tunable:
  `IBenchAct`, the activation implementations, and `Act`.
- `bench_net.slang` imports `activation.slang`, declares the numeric tunables
  `kWidth`, `kDepth`, and `kFreqBands`, and contains the neural-texture
  encoding, MLP evaluation, and SlangPy entrypoint.

Use this split to support the paper claim that tunable choices do not have to be
monolithic or local to one entrypoint file. A reusable activation module can own
one part of the search space, the entrypoint shader can own another part, and
reflection still presents one combined search space to the host tuner.

## Reflected Tunables

The shader exposes four reflected axes:

- `Act`: `TReLU`, `TLeakyReLU`, `TELU`, `TSmeLU`, `TSwish`, `TTanh`,
  `TSigmoid`
- `kWidth`: `16, 32, 64, 128`
- `kDepth`: `2, 3, 4`
- `kFreqBands`: `6`

Derived dimensions:

```text
input_dim = 2 + 4 * kFreqBands
param_count = input_dim * kWidth + kWidth
            + (kDepth - 1) * (kWidth * kWidth + kWidth)
            + kWidth * 3 + 3
```

For the current result set, this is `7 * 4 * 3 * 1 = 84` reflected configs.

## Generated Paper Artifacts

These files are generated from the benchmark scripts and are not required in a
source-only checkout:

- `spread_tunable.png`: the paper-facing reflected-tunable plot. It shows
  the per-shape activation throughput spread.
- `throughput_tunable.png`: the paper-facing reflected-tunable plot showing
  throughput by activation for each `(kWidth, kDepth, kFreqBands)` shape.
- `results_tunable.jsonl`: raw reflected-tunable timing rows behind
  `spread_tunable.png`.
- `opentuner_comparison.md`: workflow comparison between reflected Slang
  tunables and a black-box OpenTuner-style source-specialization loop.
- `strategy_comparison_table.md`: side-by-side comparison of exhaustive search
  and a seeded random-search policy over the same reflected result set.

Suggested figure/table mapping:

- `spread_tunable.png` supports the claim that activation choice matters within
  fixed network shapes.
- `throughput_tunable.png` supports the claim that the best activation is
  shape-dependent.
- `results_tunable.jsonl` supports exact throughput numbers.
- `opentuner_comparison.md` supports the programming-model claim: Slang keeps
  the tuning space with the shader, while the OpenTuner path mirrors choices in
  Python and generates specialized source.
- `strategy_comparison_table.md` supports the strategy-swap claim: the same
  reflected search space can be consumed by exhaustive search or a bounded
  stochastic search policy without changing shader declarations.

## Paper-Ready Numbers

Current reflected-tunable run:

- Device: `NVIDIA GeForce RTX 3080`
- Resolution: `256`
- Warmup calls: `2`
- Timed calls: `5`
- Total configs: `84`
- Throughput range across all reflected configs: `1.90` to `152.50`
  MSamples/s, a global spread of about `80.4x`.

Do not use the global spread alone as the main activation claim, because it is
dominated by network shape. The stronger activation-specific claim is the
within-shape spread:

- Maximum per-shape activation spread: about `2.09x` at
  `kWidth=128, kDepth=4, kFreqBands=6` (`TELU` at `3.96` MSamples/s versus
  `TSigmoid` at `1.90` MSamples/s).
- Other large within-shape spreads include `2.06x` at
  `kWidth=64, kDepth=4` and `2.00x` at `kWidth=32, kDepth=4`.
- The reflected sweep has four distinct activation winners across the twelve
  network shapes: `TReLU`, `TLeakyReLU`, `TELU`, and `TSmeLU`.

Representative per-shape winners:

- `kWidth=16, kDepth=2`: `TReLU`, `152.50` MSamples/s.
- `kWidth=32, kDepth=2`: `TLeakyReLU`, `139.74` MSamples/s.
- `kWidth=64, kDepth=2`: `TSmeLU`, `60.80` MSamples/s.
- `kWidth=128, kDepth=4`: `TELU`, `3.96` MSamples/s.

Strategy-swap comparison:

- Exhaustive best: `Act=TReLU,kWidth=16,kDepth=2,kFreqBands=6` at `152.50`
  MSamples/s after measuring all `84` configurations.
- Random search with seed `0` and a budget of `12` configurations finds
  `Act=TELU,kWidth=16,kDepth=2,kFreqBands=6` at `149.55` MSamples/s, or
  `98.1%` of the exhaustive best, while evaluating `14.3%` of the space.
- Treat this as a workflow demonstration, not as a claim that random search is
  the best policy. The point is that the shader-owned reflected space is
  independent of the host-side search strategy.

## Suggested Phrasing

A concise paper phrasing:

> In the neural-texture benchmark, the shader declares both interface-based and
> integer tunables: activation implementation, width, depth, and frequency-band
> count. SlangPy reflects this 84-config space directly from the shader and
> specializes each variant by module linking. On an RTX 3080, activation choice
> changes throughput by up to about 2.1x within a fixed network shape, and the
> winning activation changes across shapes.

Safe claims:

- The benchmark demonstrates reflected mixed-type tuning axes: struct/interface
  choices and integer specialization choices in one shader-owned search space.
- Activation choice has a measurable, shape-dependent performance effect.
- The best activation is not a single global choice across all network shapes.
- The reflected search space can be passed to different search strategies. The
  side-by-side strategy table demonstrates exhaustive and random policies over
  the same shader declarations.

Avoid:

- Claiming the `80.4x` global spread is caused by activation alone; width and
  depth dominate that number.
- Treating the forward-inference timing result as a convergence or image-quality
  result.
