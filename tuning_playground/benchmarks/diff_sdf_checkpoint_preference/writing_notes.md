# Diff SDF Checkpoint-Preference Writing Notes

These notes are for paper narration around the resource-bound diff SDF result.
Use `../diff_sdf/writing_notes.md` for the older all-executable schedule sweep.

## Workload Story

The benchmark trains a tiny neural SDF to match an analytic sphere target. Each
training step raymarches a fixed number of steps per pixel, evaluates a small
MLP SDF at each step, backpropagates an L2 pixel loss through the raymarch loop,
and applies an SGD update.

This variant exposes checkpointing as three link-time boolean preferences:

- checkpoint or recompute the MLP forward call
- checkpoint or recompute the SDF wrapper
- checkpoint or recompute the full raymarch loop

The useful paper point is resource-bound tuning. Some schedules are not merely
slower; they exceed backend resource limits and cannot execute on the tested
GPU/backend. The most important trigger is checkpointing `sdf_eval` inside the
raymarch loop, especially when `N_STEPS * W` reaches `1024` or higher.

## Tunables

The outer shape is selected by source-level defines:

- `N_STEPS`: `16, 32, 64`
- `W`: `16, 32, 64`

The schedule is selected by three link-time bool constants:

- `PreferCheckpointMarch`: `false`, `true`
- `PreferCheckpointSDF`: `false`, `true`
- `PreferCheckpointMLP`: `false`, `true`

This creates nine intended shapes and eight intended schedules per shape.
Schedule tags in these notes use `March / SDF / MLP` order:

- `R` means recompute (`PreferCheckpoint...=false`).
- `C` means checkpoint (`PreferCheckpoint...=true`).
- Example: `RRC` means recompute march, recompute SDF, checkpoint MLP.

The key shader site is:

```slang
[CheckpointPreference(PreferCheckpointSDF)]
float sdf_eval(no_diff float3 p, float params[kParams])
```

`sdf_eval` is called inside the fixed-step raymarch loop, so checkpointing it
can retain autodiff state that scales with both `N_STEPS` and `W`.

## Generated Paper Artifacts

These files are generated from the benchmark scripts and are not required in a
source-only checkout:

- `spread_tunable.png`: the paper-facing plot for successful timing rows in
  the checkpoint-preference sweep, summarized as per-shape schedule spread.
- `throughput_tunable.png`: the paper-facing plot showing throughput by
  checkpoint schedule for executable shapes.
- `results_diff_sdf_checkpoint_preference.jsonl`: raw `resolution=64` result
  set. This is the main paper result.
- `results_diff_sdf_checkpoint_preference_res32.jsonl`: secondary
  `resolution=32` cross-check. It shows the same six shape-level failures.

Suggested figure/table mapping:

- `spread_tunable.png` supports throughput spread among executable schedules.
- `throughput_tunable.png` supports the exact per-schedule throughput pattern
  for executable shapes.
- `results_diff_sdf_checkpoint_preference.jsonl` supports exact successful
  timing rows and shape-level failures.
- The generated result rows support the mechanism and caution: the failures are
  schedule-sensitive and appear tied to resource limits, especially `SDF=C`.

## Paper-Ready Numbers

Current `resolution=64` result set:

- Device: `NVIDIA GeForce RTX 3080`
- Resolution: `64`
- Warmup iterations: `3`
- Timed iterations: `20`
- Intended sweep: `9` shapes and `8` schedules per shape, for `72` intended
  schedule trials.
- Recorded successful timing rows: `24`
- Recorded shape-level failures: `6`

Successful shapes and timing spread:

| shape | winner | winner iters/s | slowest | slowest iters/s | spread |
|---|---|---:|---|---:|---:|
| `n=16,w=16` | `RRR` | `184.41` | `RCC` | `65.38` | `2.82x` |
| `n=16,w=32` | `RRC` | `136.35` | `CCC` | `60.58` | `2.25x` |
| `n=32,w=16` | `RRC` | `161.83` | `CCR` | `51.24` | `3.16x` |

Shape-level failures in the default result:

```text
n=16,w=64
n=32,w=32
n=32,w=64
n=64,w=16
n=64,w=32
n=64,w=64
```

The first three boundary failures include all shapes with
`N_STEPS * W = 1024`:

```text
16 * 64 = 1024
32 * 32 = 1024
64 * 16 = 1024
```

The Vulkan failure reported in the JSONL rows is:

```text
RuntimeError: m_rhi_command_encoder->finish(rhi_command_buffer.writeRef()) failed with error: -13 (unknown)
```

The `resolution=32` cross-check has the same failure set:

- `24` successful timing rows
- `6` shape-level failures
- successful-row throughput range `79.08` to `198.79` iters/s

Important caveat: the current driver records failures at shape granularity. If
one schedule fails, successful schedules already completed for that shape are
not emitted. Use `resource_bound_finding.md` for the targeted schedule probes
showing that the failure is schedule-specific and strongly associated with
`SDF=C`, rather than simply the shape alone.

## Suggested Phrasing

A concise paper phrasing:

> The checkpoint-preference diff SDF benchmark exposes a resource-bound region
> of the autodiff schedule space. For small shapes, all eight checkpoint/
> recompute combinations run, with up to a 3.16x throughput spread. At larger
> `N_STEPS * W` products, schedules that checkpoint the SDF evaluation inside
> the raymarch loop can exceed backend resource limits; the default sweep
> records six shape-level failures, beginning at `N_STEPS * W = 1024`.

Safe claims:

- The benchmark demonstrates that autotuning needs to handle failed variants as
  first-class outcomes, not only slower timings.
- Checkpointing can move a differentiable program from time-bound tuning into a
  resource-bound regime.
- Targeted probes indicate that `SDF=C` is the schedule-sensitive trigger for
  the observed failures.

Avoid:

- Saying every schedule fails for the failing shapes. The current JSONL records
  shape-level failures, and targeted probes show schedule-specific behavior.
- Treating the Vulkan `-13` error as a front-end Slang error. The CUDA
  cross-check in `resource_bound_finding.md` failed with kernel-launch out of
  memory for a targeted case, supporting a backend resource-limit
  interpretation.
- Claiming the exact `N_STEPS * W = 1024` boundary is universal; it is measured
  for this shader, backend, and GPU.
