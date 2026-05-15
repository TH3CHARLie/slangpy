"""Slang-autotuning driver for the neural-texture forward-inference kernel.

The tunable axes are split across two Slang files using link-time
specialization:

  - ``[Tunable] extern struct Act : IBenchAct = TReLU;``
  - ``[Tunable(16, 32, 64, 128)] extern static const int kWidth = 32;``

The imported ``activation.slang`` file owns the structural activation
tunable. The benchmark entrypoint ``bench_net.slang`` owns the numeric width,
depth, and frequency-band tunables. The Python driver loads ``bench_net.slang``,
discovers the combined choice space via reflection, builds the cartesian
product, and specializes each variant through module linking. No preprocessor
defines, no Python outer loop over shapes.

The metric is forward-inference throughput (pixels / second). Weights are
initialised once (Kaiming-normal) and frozen per (width, depth, freq_bands)
combination — every activation variant for a given shape runs on identical
parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

# Ensure the tuning package is importable when run directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import slangpy as spy  # noqa: E402

from tuning import ExhaustiveSearch, RandomSearch, Tuner  # noqa: E402
from tuning.config import IntTunableBinding, TuningConfig  # noqa: E402


THIS_DIR = pathlib.Path(__file__).parent
SHADER = THIS_DIR / "bench_net.slang"


# --------------------------------------------------------------------------- #
# Param tensor + weight init
# --------------------------------------------------------------------------- #


def param_count(width: int, depth: int, freq_bands: int) -> int:
    input_dim = 2 + 4 * freq_bands
    n = 0
    # layer 0: input -> width
    n += input_dim * width + width
    # middle layers: width -> width
    n += (depth - 1) * (width * width + width)
    # output: width -> 3
    n += width * 3 + 3
    return n


def init_params(width: int, depth: int, freq_bands: int, seed: int = 0) -> np.ndarray:
    """Kaiming-normal initialisation over the flat param vector."""
    rng = np.random.default_rng(seed)
    input_dim = 2 + 4 * freq_bands
    chunks: list[np.ndarray] = []

    def kaiming(fan_in: int, shape: tuple[int, ...]) -> np.ndarray:
        std = math.sqrt(2.0 / fan_in)
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    # layer 0
    chunks.append(kaiming(input_dim, (width * input_dim,)))
    chunks.append(np.zeros((width,), dtype=np.float32))
    # middle
    for _ in range(depth - 1):
        chunks.append(kaiming(width, (width * width,)))
        chunks.append(np.zeros((width,), dtype=np.float32))
    # output
    chunks.append(kaiming(width, (3 * width,)))
    chunks.append(np.zeros((3,), dtype=np.float32))
    return np.concatenate(chunks)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def create_uv_grid(device: spy.Device, resolution: int) -> spy.NDBuffer:
    from slangpy.types import NDBuffer
    span = np.linspace(0, 1, resolution, dtype=np.float32)
    uvs_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=2)
    uvs = NDBuffer(device, "float2", shape=(resolution, resolution))
    uvs.copy_from_numpy(uvs_np)
    return uvs


def upload_params(device: spy.Device, params_np: np.ndarray) -> spy.Buffer:
    return device.create_buffer(
        element_count=params_np.size,
        struct_size=4,
        usage=spy.BufferUsage.shader_resource,
        data=params_np,
    )


def get_int_binding(cfg: TuningConfig, name: str) -> int:
    """Extract an integer tunable value from a config by name."""
    for b in cfg.bindings:
        if isinstance(b, IntTunableBinding) and b.tunable_name == name:
            return b.value
    raise KeyError(f"No integer binding for {name!r} in config")


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class TrialResult:
    config: str
    kWidth: int
    kDepth: int
    kFreqBands: int
    activation: str
    msamples_per_s: float
    elapsed_seconds: float
    calls: int


def make_tuning_model(strategy: str, budget: int | None, seed: int):
    if strategy == "exhaustive":
        return ExhaustiveSearch()
    if strategy == "random":
        return RandomSearch(budget=budget, seed=seed)
    raise ValueError(f"unknown strategy: {strategy}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--warmup-calls", type=int, default=5)
    ap.add_argument("--timed-calls", type=int, default=50)
    ap.add_argument("--output", type=pathlib.Path,
                    default=THIS_DIR / "results_tunable.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N configs")
    ap.add_argument("--strategy", choices=["exhaustive", "random"], default="exhaustive")
    ap.add_argument("--budget", type=int, default=0,
                    help="maximum configs for --strategy random; 0 means no budget")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for --strategy random")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    device = spy.create_device(
        spy.DeviceType.vulkan,
        False,
        include_paths=[THIS_DIR],
    )
    adapter = device.info.adapter_name
    coopvec = spy.Feature.cooperative_vector in device.features
    print(f"[device] {adapter}  coopvec={coopvec}")

    # Load the module once — all tunables are extern, so defaults are used.
    raw_module = device.load_module(str(SHADER))

    # Discover the full tuning space from the shader.
    model = make_tuning_model(
        args.strategy,
        args.budget if args.strategy == "random" and args.budget else None,
        args.seed,
    )
    tuner = Tuner(model=model)
    space = tuner.setup(device, raw_module)
    print(f"[discover] {space}")

    if args.dry_run:
        trial_idx = 0
        while not tuner.is_complete():
            if args.limit and trial_idx >= args.limit:
                break
            cfg = tuner.propose()
            trial_idx += 1
            print(f"  {cfg}")
        print(f"[dry-run] strategy={args.strategy} configs={trial_idx}")
        return 0

    module = spy.Module(raw_module)
    func = module.evalPixel
    uv_grid = create_uv_grid(device, args.resolution)
    output_tmp = args.output.with_name(args.output.name + ".tmp")

    # Cache params per (width, depth, freq_bands) shape to avoid re-init.
    params_cache: dict[tuple[int, int, int], spy.Buffer] = {}
    total = space.size()
    if args.strategy == "random" and args.budget:
        total = min(total, args.budget)
    if args.limit:
        total = min(total, args.limit)

    with output_tmp.open("w") as f:
        f.write(
            json.dumps(
                {
                    "_meta": {
                        "adapter": adapter,
                        "resolution": args.resolution,
                        "warmup_calls": args.warmup_calls,
                        "timed_calls": args.timed_calls,
                        "total_configs": space.size(),
                        "strategy": args.strategy,
                        "budget": (
                            args.budget
                            if args.strategy == "random" and args.budget
                            else None
                        ),
                        "seed": args.seed if args.strategy == "random" else None,
                        "planned_evaluations": total,
                        "driver": "bench_tunable.py",
                        "tuning_space": str(space),
                        "note": "forward-inference throughput; fixed Kaiming weights; "
                                "all axes via link-time specialization",
                    }
                }
            )
            + "\n"
        )
        f.flush()

        results: list[TrialResult] = []
        trial_idx = 0

        while not tuner.is_complete():
            if args.limit and trial_idx >= args.limit:
                break

            cfg: TuningConfig = tuner.propose()
            trial_idx += 1

            # Extract integer tunables for this config.
            w = get_int_binding(cfg, "kWidth")
            d = get_int_binding(cfg, "kDepth")
            fb = get_int_binding(cfg, "kFreqBands")

            # Get or create params for this shape.
            shape_key = (w, d, fb)
            if shape_key not in params_cache:
                params_np = init_params(w, d, fb)
                params_cache[shape_key] = upload_params(device, params_np)
            params = params_cache[shape_key]

            # Link the variant (struct + int tunables resolved via module linking).
            variant = func.with_settings(cfg)

            # Warm up.
            for _ in range(args.warmup_calls):
                variant(uv_grid, params)
            device.wait_for_idle()

            # Time.
            t0 = time.perf_counter()
            for _ in range(args.timed_calls):
                variant(uv_grid, params)
            device.wait_for_idle()
            elapsed = time.perf_counter() - t0

            total_samples = args.timed_calls * (args.resolution * args.resolution)
            msamples_per_s = (total_samples * 1e-6) / elapsed if elapsed > 0 else 0.0

            tuner.report(cfg, (elapsed / args.timed_calls) * 1000)

            # Extract activation name from struct bindings.
            struct_bindings = cfg.struct_bindings()
            act_name = struct_bindings[0].impl_name if struct_bindings else "N/A"

            result = TrialResult(
                config=str(cfg),
                kWidth=w,
                kDepth=d,
                kFreqBands=fb,
                activation=act_name,
                msamples_per_s=msamples_per_s,
                elapsed_seconds=elapsed,
                calls=args.timed_calls,
            )
            results.append(result)

            f.write(json.dumps({
                "config": result.config,
                "kWidth": result.kWidth,
                "kDepth": result.kDepth,
                "kFreqBands": result.kFreqBands,
                "activation": result.activation,
                "msamples_per_s": result.msamples_per_s,
                "elapsed_seconds": result.elapsed_seconds,
                "calls": result.calls,
            }) + "\n")
            f.flush()

            print(
                f"  [{trial_idx}/{total}] w={w},d={d},fb={fb} act={act_name}  "
                f"{msamples_per_s:.1f} MS/s"
            )

    output_tmp.replace(args.output)

    best = tuner.best()
    best_result = min(tuner.history, key=lambda r: r.elapsed_ms)
    print(f"\n[done] {len(results)} trials written to {args.output}")
    print(f"[best] {best}  ({best_result.elapsed_ms:.2f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
