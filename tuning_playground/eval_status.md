# Slang Autotuning — Evaluation Status (2026-04-27)

## 1. What We Built

### Tuning System (`tuning/` package)

The core autotuning infrastructure lives in `tuning/` under the SlangPy repo. The original tuner/model layer is pure Python; the current branch also adds compiler/reflection plumbing so built-in `[Tunable]` and integer-choice tunables can be discovered and specialized end to end.

**Slang-side primitives: built-in `[Tunable]` on extern structs and `extern static const int`.**
A Slang module can now declare:
- interface tunables as `[Tunable] public extern struct X : IFoo = DefaultImpl;`
- integer tunables as `[Tunable(choices...)] public extern static const int N = DefaultValue;`

Both are specialized at link time. The compiler-side `[Tunable]` support is now built-in on the Slang `tuning` branch (and merged with upstream `master`), rather than being treated as boilerplate/user-attribute-only infrastructure.

**Reflection (`reflection.py`).**
Discovers tunable axes automatically from a loaded `SlangModule` using Slang's reflection API:
- `find_tunable_decls(module)` — walks the module's declaration tree, finds all `extern struct` decls carrying the built-in `tunable` modifier.
- `find_tunable_int_decls(module)` — walks variable decls and finds `[Tunable(choices...)] extern static const int` declarations, then reads the integer argument values from the reflected `Tunable` attribute.
- `find_interfaces_for_decl(module, decl)` — resolves which interface(s) a tunable conforms to via `layout.is_sub_type()`.
- `find_conforming_types(module, interface_name)` — enumerates all non-extern structs in the module that implement that interface. These are the candidate implementations.
- `discover_tuning_space(module)` — composes the above into a `{tunable_name: {interface_name: [impl_names]}}` dict.
- `link_variant(device, module, bindings, int_bindings=None)` — creates a specialized `spy.Module` by emitting both `export struct X : IFoo = Impl;` and `export static const int N = value;` into a link module. This is the module-level specialization path.

**Current contract for integer tunables.**
The SlangPy discovery/specialization path intentionally only treats `extern static const int` as integer tunables. This matches what the runtime knows how to materialize during module linking. Other `[Tunable(...)]` variable shapes are ignored by discovery for now.

**Config types (`config.py`).**
- `TunableParam` — one tuning axis: (tunable_name, interface_name, list of impl_names).
- `IntTunableParam` — one integer tuning axis: (tunable_name, list of discrete integer choices).
- `TunableBinding` — one axis bound to one impl: (tunable_name, interface_name, impl_name).
- `IntTunableBinding` — one integer axis bound to one concrete value: (tunable_name, value).
- `TuningConfig` — a frozen, hashable tuple of mixed struct/int bindings. Represents one point in the config space. Has `to_bindings_dict()` (for module-level `link_variant`), `to_link_bindings()` (for function-level `with_settings`), and `int_bindings()` (for integer specializations).
- `TuningSpace` — wraps a list of mixed struct/int params. `TuningSpace.discover(module)` uses reflection to auto-build from a loaded module. `all_configs()` enumerates the full cartesian product. `size()` returns the product of per-axis option counts.
- `TuningResult` — records one trial: (config, elapsed_ms, iteration index).

**Search model (`model.py`).**
- `TuningModel` — abstract base class defining the search strategy interface: `initialize(space)`, `propose() -> TuningConfig`, `report(config, elapsed_ms)`, `best()`, `is_complete()`, `history`.
- `ExhaustiveSearch` — the only concrete implementation so far. Iterates through `space.all_configs()` in order, reports results, tracks the best by lowest `elapsed_ms`. Complete and deterministic — useful for small spaces (up to ~hundreds of configs). No Bayesian, bandit, or random search yet.

**Tuner orchestrator (`tuner.py`).**
- `Tuner` — user-facing class that glues the search model, tuning space, and variant linking together. Usage pattern:
  1. `tuner = Tuner(model=ExhaustiveSearch())`
  2. `space = tuner.setup(device, raw_module)` — discovers the tuning space via reflection and initializes the model.
  3. Loop: `config = tuner.propose()` -> evaluate -> `tuner.report(config, elapsed_ms)` until `tuner.is_complete()`.
  4. `tuner.best()` returns the winning config; `tuner.get_variant(config)` returns a cached `spy.Module` for any config.
- The Tuner does NOT own the execution loop. The user writes the measurement code. This is deliberate — it keeps the tuner generic across workload types (forward-only, training, rendering, etc.).
- Variant cache: `get_variant(config)` builds and caches a linked `spy.Module` per config, so repeated access is free.

**Function-level specialization (`FunctionNode.with_settings`).**
- Alternative to module-level `link_variant`. Instead of creating a new `spy.Module` per config, `func.with_settings(config)` returns a `FunctionNodeLinkTypeBindings` that emits `export struct` and `export static const int` declarations into the kernel source at codegen time. Each distinct config gets its own `CallDataCache` entry keyed by a canonical signature string.
- Lighter weight: no extra module, compilation happens lazily, reuse is transparent through the native cache.
- `TuningConfig.to_link_bindings()` bridges between the tuning types and the slangpy core (which never imports from `tuning/`).
- Landed as commit `aebf1d8`.

**Benchmark A — neural_texture (forward inference):**
- Workload: neural texture lookup with a tiny MLP. Input position -> RGB.
- Tunable axes:
  - activation function (7 impls behind one `[Tunable] extern struct`)
  - network width (`[Tunable(16, 32, 64, 128)] extern static const int`)
  - network depth (`[Tunable(2, 3, 4)] extern static const int`)
  - frequency-band count (`[Tunable(6)] extern static const int`)
- Sweep: 84 total link-time configs (7 activations x 4 widths x 3 depths x 1 freq-band choice).
- Metric: MSamples/s (forward-only throughput).
- Result: the old committed dataset captured activation-only spread across Python-driven outer-loop shapes. The current checked-in `results_tunable.jsonl` is only a short smoke run for the new all-axes-via-linking path (warmup=2, timed_calls=5), not paper-quality benchmark data yet.
- Current state: benchmark driver and shader are updated to reflection-discovered mixed struct/int tunables; full rerun still needed.

**Benchmark B — diff_sdf (differentiable autodiff schedule):**
- Workload: differentiable SDF raymarching. A tiny MLP defines a neural SDF; sphere-tracing produces a pixel image; L2 loss against an analytic unit sphere; Slang autodiff backpropagates through the entire chain (`pixel_loss` -> `render_pixel` -> `march_body` -> `sdf_body` -> `mlp_body` -> params). Forward + backward + SGD constitutes one training iteration.
- Tunable axes (3 independent):
  - `SchedMarch` — checkpoint schedule for the raymarch loop
  - `SchedSDF` — checkpoint schedule for the SDF eval wrapper
  - `SchedMLP` — checkpoint schedule for the MLP forward
- Each axis has two impls: `[PreferCheckpoint]` vs `[PreferRecompute]`. 3 binary axes compose into 2^3 = 8 link-time configs per shape.
- Sweep: 9 shapes (n_steps in {16,32,64} x width in {16,32,64}) x 8 schedule configs = 72 trials. RTX 3080. 20 timed iters per trial after 3 warmup iters.
- Metric: training iters/s (fwd + bwd + SGD wall-clock).
- Committed: `diff_sdf.slang` + `bench.py` (`91c54d7`), `bench_tunable.py` + `plot_tunable.py` + `results_diff_sdf.jsonl` (`3c30ff6`).

---

## 2. Key Findings (diff_sdf)

| Shape | Min ips | Max ips | Spread | Winner (Ma/SDF/MLP) |
|-------|---------|---------|--------|----------------------|
| n=16, w=16 | 153.0 | 309.1 | 2.02x | CRR |
| n=16, w=32 | 132.0 | 142.2 | 1.08x | CCR |
| n=16, w=64 | 59.7 | 69.8 | 1.17x | RCC |
| n=32, w=16 | 149.3 | 172.7 | 1.16x | CRR |
| n=32, w=32 | 126.3 | 132.9 | 1.05x | CRR |
| n=32, w=64 | 57.0 | 82.6 | 1.45x | CCR |
| n=64, w=16 | 125.9 | 291.8 | 2.32x | RRR |
| n=64, w=32 | 110.4 | 125.3 | 1.13x | CRC |
| n=64, w=64 | 49.4 | 66.1 | 1.34x | RRC |

- **Peak per-shape spread: 2.32x** (n=64, w=16)
- **Distinct winners: 6 of 8** possible configs
- **5 of the 6 winning configs are MIXED** Checkpoint/Recompute

**Paper-relevant takeaways:**
- The optimal autodiff checkpoint schedule is **per-call-site**, not a global toggle. A blanket `[PreferCheckpoint]` or `[PreferRecompute]` is suboptimal for most shapes.
- The optimal schedule **changes with workload shape** (n_steps, width). No single config dominates — autotuning is needed.
- The three axes **compose independently** — exactly the design space `[Tunable] extern struct` is built to express. Macro systems cannot cleanly capture this per-call-site, per-axis granularity.

---

## 3. Known Risks

**(a) Single GPU.** All numbers are RTX 3080 only. For the paper we claim autotuning matters because the winner varies across hardware. We have not demonstrated that yet. Need at least one more adapter (ideally a different vendor or generation) to show winners shift.

**(b) Modest spreads on some shapes.** Five of nine shapes have spread under 1.2x. Reviewers may argue the schedule axis "barely matters" except for a few outlier shapes. Counter: the shapes where it matters most (deep loops, small MLPs) are the ones where checkpointing overhead dominates, which is a real and common workload class.

**(c) Training loop is CPU-bottlenecked.** The SGD step (pull grads to CPU, scale, push back) is the same across all configs, so it dilutes the measured spread. A pure-GPU training loop would likely show larger spreads. Current numbers are conservative.

**(d) `[PreferRecompute]`/`[PreferCheckpoint]` are compiler hints, not guarantees.** The Slang compiler may ignore them. We are trusting that they take effect. No independent verification (e.g. inspecting IR or measuring intermediate memory). Risk is low in practice per domain expertise, but a reviewer could push on it.

**(e) No baseline comparison.** We show that schedule choice matters and autotuning finds the winner. We do not compare against: (1) always-Checkpoint (the default), (2) always-Recompute, (3) a hand-tuned schedule picked by an expert. Should compute regret vs default as a concrete number (e.g. "autotuning finds a config that is 2.32x faster than the default for shape n=64,w=16").

**(f) Two benchmarks.** neural_texture (fwd inference, activation axis) and diff_sdf (autodiff schedule). A third workload domain would strengthen the generality claim.

**(g) Reflection/API debt for integer tunables.** The compiler now treats `[Tunable]` as a built-in attribute, but SlangPy currently reads integer choice lists through the reflected attribute-argument path rather than a dedicated tunable-specific reflection API. This is workable for now, but it is a bridge, not the final abstraction.

---

## 4. Unknowns

- **Do winners shift across GPUs?** This is the central autotuning thesis and is untested. We have the infrastructure to run on another adapter; just need access.

- **Does the spread widen or shrink with larger MLP widths (w=128, 256)?** Current sweep tops out at w=64 where the MLP starts dominating compute and checkpointing overhead is proportionally smaller. Extending the sweep would clarify the tradeoff curve.

- **How does search scale to larger config spaces?** With 8 configs, exhaustive search is trivial. A real workload might compose 4-5 axes with 3+ impls each, producing hundreds of configs. Do we need smarter search (Bayesian, pruning)? Not yet demonstrated.

- **How much of the benchmark code should stay shader-declared versus Python-declared?** The new neural_texture path moves width/depth/frequency into shader-declared integer tunables. This is cleaner for the autotuning story, but it changes how users control outer-loop sweeps and may not fit every workload.

- **Does function-level specialization (`with_settings`) give a meaningful per-trial speedup over module-level relink?** We built the infrastructure but have not benchmarked specialization overhead itself. This would support the "fast iteration" claim in the paper.

- **What does the memory footprint look like?** Checkpoint vs Recompute trades memory for compute. We measure throughput but not peak VRAM. Reporting both would be more complete and could explain why Recompute wins on some shapes (VRAM pressure, cache thrashing).

- **Convergence equivalence:** all 8 schedule configs are mathematically equivalent (same gradients, same training trajectory). We verified this in `bench.py` with the default schedule. We have not explicitly verified that all 8 configs produce identical loss curves. They should, but a divergence would indicate a compiler bug worth flagging.

---

## 5. Immediate Next Steps (prioritized)

1. **Rerun neural_texture properly.** Regenerate `results_tunable.jsonl` with paper-quality settings for the new all-axes-via-linking path. The currently checked-in file is a smoke run only.

2. **Cross-GPU sweep.** Run `bench_tunable.py` / `diff_sdf` on a second adapter. Compare winners. This is still the single highest-leverage result for the paper.

3. **Compute regret vs default.** For each shape, report how much faster the autotuned winner is compared to the default (all-Checkpoint) config. This is a one-liner from existing data and directly answers "why autotune instead of picking a reasonable default?"

4. **Third benchmark workload.** Candidates: a compute reduction/scan with tunable algorithm + tile size, or a classical rendering pass.

5. **Function-level specialization overhead benchmark.** Measure trials/s for module-relink vs `with_settings` vs recompile-from-scratch, now that integer tunables also flow through the same path.

6. **Paper writing:** draft the diff_sdf figure + caption, update eval section with the 2.32x / 6-of-8 numbers, and note the new built-in/int-tunable compiler support as part of the system story.
