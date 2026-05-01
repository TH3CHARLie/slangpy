"""Differentiable SDF raymarching baseline training harness.

Precomputes a target image by raymarching an analytic unit sphere, then trains
a tiny neural SDF to reproduce the image via L2 pixel loss. Gradients flow
through the raymarch loop via Slang autodiff. The checkpointing schedule is
controlled by three extern bool constants, which default to Checkpoint.

This script validates that the workload compiles and converges end-to-end. The
bool schedule sweep lives in `bench_tunable.py`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import slangpy as spy  # noqa: E402

THIS_DIR = pathlib.Path(__file__).parent
SHADER = THIS_DIR / "diff_sdf.slang"


# --------------------------------------------------------------------------- #
# Target image
# --------------------------------------------------------------------------- #


def make_target(resolution: int) -> np.ndarray:
    """Render a unit sphere with the same camera and soft-hit used in shader."""
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

    hit = 0.5 - 0.5 * np.tanh(t - 3.0)
    return hit.astype(np.float32)


def param_count(width: int) -> int:
    return 3 * width + width + width + 1


def init_params(width: int, seed: int = 0) -> np.ndarray:
    """Init so MLP output is zero everywhere at iteration 0."""
    rng = np.random.default_rng(seed)
    n = param_count(width)
    p = np.zeros(n, dtype=np.float32)
    p[: 3 * width] = (rng.standard_normal(3 * width) * 0.3).astype(np.float32)
    return p


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="smaller defaults for a quick compile + 1-iter sanity check",
    )
    args = ap.parse_args()

    if args.smoke:
        args.resolution = 16
        args.iters = 2
        args.log_every = 1

    device = spy.create_device(
        spy.DeviceType.vulkan,
        False,
        include_paths=[THIS_DIR],
    )
    print(f"[device] {device.info.adapter_name}")

    header = f"#define N_STEPS {args.n_steps}\n#define W {args.width}\n"
    source = header + SHADER.read_text()
    raw_module = device.load_module_from_source(
        f"diff_sdf_checkpoint_preference_n{args.n_steps}_w{args.width}",
        source,
    )
    module = spy.Module(raw_module)
    print(
        f"[compile] N_STEPS={args.n_steps}  W={args.width}  "
        f"params={param_count(args.width)}"
    )

    target_np = make_target(args.resolution)
    target = spy.Tensor.from_numpy(device, target_np)
    print(f"[target]  {target_np.shape}  range=[{target_np.min():.3f}, {target_np.max():.3f}]")

    span = np.linspace(-1.0, 1.0, args.resolution, dtype=np.float32)
    uv_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=-1)
    uv = spy.Tensor.from_numpy(device, uv_np)

    params_np = init_params(args.width)
    params = spy.Tensor.from_numpy(device, params_np)
    grads_np = np.zeros_like(params_np)
    grads = spy.Tensor.from_numpy(device, grads_np)

    n_pixels = args.resolution * args.resolution

    print(f"[train]   iters={args.iters}  lr={args.lr}  pixels={n_pixels}")
    t_start = time.perf_counter()

    for it in range(args.iters):
        grads.copy_from_numpy(np.zeros_like(params_np))

        loss_out = module.train_pixel(params, grads, uv, target)

        g = grads.to_numpy()
        p = params.to_numpy()
        p -= args.lr * (g / n_pixels)
        params.copy_from_numpy(p)
        params_np = p

        if (it + 1) % args.log_every == 0 or it == 0:
            loss_np = loss_out.to_numpy()
            mean_loss = float(loss_np.mean())
            print(
                f"  iter {it + 1:4d}  loss={mean_loss:.5f}  "
                f"grad_norm={np.linalg.norm(g):.3f}"
            )

    elapsed = time.perf_counter() - t_start
    print(f"[done]    {elapsed:.2f}s  ({args.iters / elapsed:.1f} iter/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
