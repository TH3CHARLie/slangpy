"""Slang-autotuning driver for the neural-texture forward-inference kernel.

Structurally parallel to ``bench.py`` but with one key difference: the
activation axis is driven through the Slang autotuning infrastructure
(``[Tunable] extern struct`` + ``TuningSpace.discover`` + ``Tuner.propose``
+ ``FunctionNode.with_settings``) instead of a Python cartesian product.

Width, depth, and frequency-band count remain Python outer-loop axes —
they're compile-time numeric parameters baked into the Slang module via
preprocessor defines, not interface impls, so they don't belong in the
``[Tunable]`` axis. This mirrors how a real user would deploy the system:
structural axes through Slang interfaces, numeric sizing axes Python-driven.

The metric is forward-inference throughput (pixels / second). We deliberately
do NOT retrain per-variant: weights are initialised once (Kaiming-normal) and
frozen, so every activation variant runs on identical parameters and the only
timing difference comes from the activation kernel itself. Loss is not
reported — the existing ``bench.py`` already captures the quality spread via
training; this driver is purely about demonstrating the tuning system.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

# Ensure the tuning package is importable when run directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import slangpy as spy  # noqa: E402

from tuning import ExhaustiveSearch, Tuner  # noqa: E402
from tuning.config import TuningConfig  # noqa: E402


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
# Helpers (target / uv grid) — lifted from bench.py so this driver is
# self-contained and doesn't depend on bench.py imports.
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


# --------------------------------------------------------------------------- #
# Config / result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Shape:
    hidden_width: int
    hidden_depth: int
    freq_bands: int

    def label(self) -> str:
        return f"w={self.hidden_width},d={self.hidden_depth},fb={self.freq_bands}"


@dataclass
class TrialResult:
    shape: dict
    activation: str
    msamples_per_s: float
    elapsed_seconds: float
    calls: int


# --------------------------------------------------------------------------- #
# Per-shape tuning run
# --------------------------------------------------------------------------- #


def run_shape(
    device: spy.Device,
    shape: Shape,
    uv_grid,
    *,
    warmup_calls: int,
    timed_calls: int,
    resolution: int,
) -> list[TrialResult]:
    """Tune activation for a single (width, depth, freq) shape.

    Returns one TrialResult per activation impl discovered in the module.
    """
    header = (
        f"#define W {shape.hidden_width}\n"
        f"#define D {shape.hidden_depth}\n"
        f"#define FB {shape.freq_bands}\n"
    )
    source = header + SHADER.read_text()
    raw_module = device.load_module_from_source(
        f"bench_net_w{shape.hidden_width}_d{shape.hidden_depth}_fb{shape.freq_bands}",
        source,
    )

    tuner = Tuner(model=ExhaustiveSearch())
    space = tuner.setup(device, raw_module)
    assert space.size() > 0, f"No tunable configs discovered for shape {shape}"

    module = spy.Module(raw_module)
    func = module.evalPixel

    # Weights are initialised once and shared across all activation variants.
    params_np = init_params(shape.hidden_width, shape.hidden_depth, shape.freq_bands)
    params = upload_params(device, params_np)

    results: list[TrialResult] = []

    while not tuner.is_complete():
        cfg: TuningConfig = tuner.propose()
        variant = func.with_settings(cfg)

        # Warm up — first call compiles the kernel & populates CallDataCache.
        for _ in range(warmup_calls):
            variant(uv_grid, params)
        device.wait_for_idle()

        # Time N dispatches.
        t0 = time.perf_counter()
        for _ in range(timed_calls):
            variant(uv_grid, params)
        device.wait_for_idle()
        elapsed = time.perf_counter() - t0

        total_samples = timed_calls * (resolution * resolution)
        msamples = total_samples * 1e-6
        msamples_per_s = msamples / elapsed if elapsed > 0 else 0.0

        tuner.report(cfg, (elapsed / timed_calls) * 1000)

        # The tunable config has a single binding; extract its impl name.
        act_name = cfg.bindings[0].impl_name
        results.append(TrialResult(
            shape=asdict(shape),
            activation=act_name,
            msamples_per_s=msamples_per_s,
            elapsed_seconds=elapsed,
            calls=timed_calls,
        ))

    return results


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128])
    ap.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--freq-bands", nargs="+", type=int, default=[6])
    ap.add_argument("--warmup-calls", type=int, default=5)
    ap.add_argument("--timed-calls", type=int, default=50)
    ap.add_argument("--output", type=pathlib.Path,
                    default=THIS_DIR / "results_tunable.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N shapes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shapes = [
        Shape(w, d, fb)
        for w, d, fb in itertools.product(args.widths, args.depths, args.freq_bands)
    ]
    if args.limit:
        shapes = shapes[: args.limit]

    print(f"[sweep] {len(shapes)} shapes × activations (tunable)")
    if args.dry_run:
        for s in shapes:
            print(f"  {s.label()}")
        return 0

    device = spy.create_device(
        spy.DeviceType.vulkan,
        False,
        include_paths=[THIS_DIR],
    )
    adapter = device.info.adapter_name
    coopvec = spy.Feature.cooperative_vector in device.features
    print(f"[device] {adapter}  coopvec={coopvec}")

    uv_grid = create_uv_grid(device, args.resolution)

    total_trials = 0
    with args.output.open("w") as f:
        f.write(
            json.dumps(
                {
                    "_meta": {
                        "adapter": adapter,
                        "resolution": args.resolution,
                        "warmup_calls": args.warmup_calls,
                        "timed_calls": args.timed_calls,
                        "num_shapes": len(shapes),
                        "driver": "bench_tunable.py",
                        "tunable_axis": "activation",
                        "note": "forward-inference throughput; fixed Kaiming weights; "
                                "no training, no loss",
                    }
                }
            )
            + "\n"
        )
        f.flush()

        for i, shape in enumerate(shapes):
            try:
                trials = run_shape(
                    device, shape, uv_grid,
                    warmup_calls=args.warmup_calls,
                    timed_calls=args.timed_calls,
                    resolution=args.resolution,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(shapes)}] {shape.label()}  FAILED: {type(e).__name__}: {e}")
                f.write(json.dumps({"shape": asdict(shape),
                                    "error": f"{type(e).__name__}: {e}"}) + "\n")
                f.flush()
                continue

            for t in trials:
                f.write(json.dumps(asdict(t)) + "\n")
            f.flush()
            total_trials += len(trials)

            thrs = sorted([t.msamples_per_s for t in trials])
            spread = thrs[-1] / thrs[0] if thrs[0] > 0 else float("inf")
            fastest = max(trials, key=lambda t: t.msamples_per_s)
            print(
                f"  [{i+1}/{len(shapes)}] {shape.label()}  "
                f"n_act={len(trials)}  thr {thrs[0]:.1f}–{thrs[-1]:.1f} MS/s  "
                f"spread {spread:.2f}×  fastest={fastest.activation}"
            )

    print(f"\n[done] {total_trials} trials written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
