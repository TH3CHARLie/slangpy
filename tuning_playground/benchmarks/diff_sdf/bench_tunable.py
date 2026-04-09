"""Slang-autotuning driver for the differentiable SDF raymarching workload.

Outer Python loop:   (N_STEPS, W) shapes — compile-time numeric axes that
                     cannot be expressed as [Tunable] extern structs.
Inner Tuner:         the 3 [Tunable] extern structs in diff_sdf.slang
                     (SchedMarch × SchedSDF × SchedMLP), 2^3 = 8 configs.

Per config we measure training throughput — specifically, time to complete
a fixed number of forward+backward+SGD iterations on a small image. This
captures the real cost of the differentiable chain under each checkpointing
schedule. Convergence is already verified in bench.py; here we care only
about per-trial wall-clock, since all 8 schedules are mathematically
equivalent (they only differ in whether intermediate values are cached or
recomputed on the backward pass).

Output: results_diff_sdf.jsonl. Plot driver: plot_tunable.py (TBD).
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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import slangpy as spy  # noqa: E402

from tuning import ExhaustiveSearch, Tuner  # noqa: E402
from tuning.config import TuningConfig  # noqa: E402

THIS_DIR = pathlib.Path(__file__).parent
SHADER = THIS_DIR / "diff_sdf.slang"


# --------------------------------------------------------------------------- #
# Target + init (lifted from bench.py)
# --------------------------------------------------------------------------- #


def make_target(resolution: int) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    U, V = np.meshgrid(xs, ys, indexing="xy")
    ro = np.array([0.0, 0.0, -3.0], dtype=np.float32)
    rd = np.stack([U, V, np.full_like(U, 2.0)], axis=-1)
    rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
    t = np.zeros_like(U)
    for _ in range(64):
        p = ro + rd * t[..., None]
        d = np.linalg.norm(p, axis=-1) - 1.0
        t += d
    return (0.5 - 0.5 * np.tanh(t - 3.0)).astype(np.float32)


def param_count(width: int) -> int:
    return 3 * width + width + width + 1


def init_params(width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = param_count(width)
    p = np.zeros(n, dtype=np.float32)
    p[: 3 * width] = (rng.standard_normal(3 * width) * 0.3).astype(np.float32)
    return p


# --------------------------------------------------------------------------- #
# Config / result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Shape:
    n_steps: int
    width: int

    def label(self) -> str:
        return f"n={self.n_steps},w={self.width}"


@dataclass
class TrialResult:
    shape: dict
    sched: dict  # {tunable_name: impl_name}
    iters_per_s: float
    elapsed_seconds: float
    iters: int


# --------------------------------------------------------------------------- #
# Per-shape tuning run
# --------------------------------------------------------------------------- #


def run_shape(
    device: spy.Device,
    shape: Shape,
    *,
    resolution: int,
    warmup_iters: int,
    timed_iters: int,
    lr: float,
) -> list[TrialResult]:
    header = f"#define N_STEPS {shape.n_steps}\n#define W {shape.width}\n"
    source = header + SHADER.read_text()
    raw_module = device.load_module_from_source(
        f"diff_sdf_n{shape.n_steps}_w{shape.width}", source
    )

    tuner = Tuner(model=ExhaustiveSearch())
    space = tuner.setup(device, raw_module)
    assert space.size() > 0, f"no tunables discovered for {shape}"

    module = spy.Module(raw_module)
    train = module.train_pixel

    # Shared inputs across all configs for this shape.
    target_np = make_target(resolution)
    target = spy.Tensor.from_numpy(device, target_np)
    span = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    uv_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=-1)
    uv = spy.Tensor.from_numpy(device, uv_np)

    n_params = param_count(shape.width)
    n_pixels = resolution * resolution

    results: list[TrialResult] = []

    while not tuner.is_complete():
        cfg: TuningConfig = tuner.propose()
        variant = train.with_settings(cfg)

        # Fresh params per trial so every config sees the same workload.
        params_np = init_params(shape.width)
        grads_np = np.zeros_like(params_np)
        params = spy.Tensor.from_numpy(device, params_np)
        grads = spy.Tensor.from_numpy(device, grads_np)

        def step() -> None:
            # Zero grads
            grads.copy_from_numpy(np.zeros(n_params, dtype=np.float32))
            # Forward + backward
            loss_out = variant(params, grads, uv, target)
            # SGD update CPU-side (same overhead across all configs → cancels
            # out in comparisons). Pull grad, scale, apply.
            g = grads.to_numpy()
            p = params.to_numpy()
            p -= lr * (g / n_pixels)
            params.copy_from_numpy(p)
            # Read the loss out once to ensure the dispatch actually fires.
            _ = loss_out.to_numpy()

        # Warmup (compile + populate caches)
        for _ in range(warmup_iters):
            step()
        device.wait_for_idle()

        # Timed
        t0 = time.perf_counter()
        for _ in range(timed_iters):
            step()
        device.wait_for_idle()
        elapsed = time.perf_counter() - t0

        iters_per_s = timed_iters / elapsed if elapsed > 0 else 0.0
        tuner.report(cfg, (elapsed / timed_iters) * 1000.0)

        sched = {b.tunable_name: b.impl_name for b in cfg.bindings}
        results.append(TrialResult(
            shape=asdict(shape),
            sched=sched,
            iters_per_s=iters_per_s,
            elapsed_seconds=elapsed,
            iters=timed_iters,
        ))

    return results


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--n-steps", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--warmup-iters", type=int, default=3)
    ap.add_argument("--timed-iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--output", type=pathlib.Path,
                    default=THIS_DIR / "results_diff_sdf.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shapes = [Shape(n, w) for n, w in itertools.product(args.n_steps, args.widths)]
    if args.limit:
        shapes = shapes[: args.limit]

    print(f"[sweep] {len(shapes)} shapes × schedule configs (tunable)")
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
    print(f"[device] {adapter}")

    total_trials = 0
    with args.output.open("w") as f:
        f.write(json.dumps({
            "_meta": {
                "adapter": adapter,
                "resolution": args.resolution,
                "warmup_iters": args.warmup_iters,
                "timed_iters": args.timed_iters,
                "lr": args.lr,
                "num_shapes": len(shapes),
                "driver": "bench_tunable.py",
                "workload": "diff_sdf raymarching",
                "tunable_axes": ["SchedMarch", "SchedSDF", "SchedMLP"],
                "note": "forward+backward+sgd throughput; checkpoint schedule axis",
            }
        }) + "\n")
        f.flush()

        for i, shape in enumerate(shapes):
            try:
                trials = run_shape(
                    device, shape,
                    resolution=args.resolution,
                    warmup_iters=args.warmup_iters,
                    timed_iters=args.timed_iters,
                    lr=args.lr,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(shapes)}] {shape.label()}  FAILED: "
                      f"{type(e).__name__}: {e}")
                f.write(json.dumps({"shape": asdict(shape),
                                    "error": f"{type(e).__name__}: {e}"}) + "\n")
                f.flush()
                continue

            for t in trials:
                f.write(json.dumps(asdict(t)) + "\n")
            f.flush()
            total_trials += len(trials)

            ips = sorted([t.iters_per_s for t in trials])
            spread = ips[-1] / ips[0] if ips[0] > 0 else float("inf")
            fastest = max(trials, key=lambda t: t.iters_per_s)
            fast_label = ",".join(f"{k[5:]}={v[:4]}" for k, v in fastest.sched.items())
            print(f"  [{i+1}/{len(shapes)}] {shape.label()}  "
                  f"n_cfg={len(trials)}  ips {ips[0]:.1f}–{ips[-1]:.1f}  "
                  f"spread {spread:.2f}×  fastest={fast_label}")

    print(f"\n[done] {total_trials} trials written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
