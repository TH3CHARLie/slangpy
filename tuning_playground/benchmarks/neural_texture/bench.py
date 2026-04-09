"""Headless neural-texture training sweep.

Measures (throughput, reconstruction loss) across a grid of architecture configs
on a procedurally generated RGB target. Reuses the slangpy-samples neuralnetwork
library for model assembly; this script owns only the sweep, timing, and metric
collection.

The purpose is to establish the performance/quality spread across a realistic
neural-shader search space — motivating input for the Syun paper (§5).
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

# Make the slangpy-samples neuralnetwork library importable.
SAMPLES_NN = pathlib.Path(
    "/home/xuanday/dev/autotune/slangpy/samples/experiments/neuralnetwork"
)
sys.path.insert(0, str(SAMPLES_NN))

import slangpy as spy  # noqa: E402
from slangpy import Module  # noqa: E402
from slangpy.types import NDBuffer  # noqa: E402

import neuralnetworks as nn  # noqa: E402


THIS_DIR = pathlib.Path(__file__).parent
SHADER = THIS_DIR / "NeuralTexture.slang"


# --------------------------------------------------------------------------- #
# Target texture
# --------------------------------------------------------------------------- #


def make_target(resolution: int, seed: int = 0) -> np.ndarray:
    """Procedural RGB target with mixed low- and high-frequency content.

    Reproducible without external image files. Combines a radial gradient,
    a checker pattern, and per-channel sinusoids to exercise encoding bands.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, resolution, dtype=np.float32),
        np.linspace(0.0, 1.0, resolution, dtype=np.float32),
        indexing="ij",
    )
    cx, cy = 0.5, 0.5
    radial = np.exp(-8.0 * ((xs - cx) ** 2 + (ys - cy) ** 2))
    checker = ((np.floor(xs * 6) + np.floor(ys * 6)) % 2).astype(np.float32)
    waves = np.stack(
        [
            0.5 + 0.5 * np.sin(12.0 * xs + 3.0 * ys),
            0.5 + 0.5 * np.sin(7.0 * ys - 5.0 * xs),
            0.5 + 0.5 * np.sin(9.0 * (xs + ys)),
        ],
        axis=-1,
    )
    base = np.stack([radial, checker, (radial * checker)], axis=-1)
    noise = rng.standard_normal((resolution, resolution, 3), dtype=np.float32) * 0.02
    rgb = np.clip(0.5 * base + 0.5 * waves + noise, 0.0, 1.0).astype(np.float32)
    return rgb


def upload_target(device: spy.Device, rgb: np.ndarray) -> spy.Texture:
    """Upload an (H,W,3) float32 array as an RGBA32F texture."""
    h, w, _ = rgb.shape
    rgba = np.concatenate([rgb, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
    tex = device.create_texture(
        width=w,
        height=h,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource,
        data=rgba,
    )
    return tex


def create_uv_grid(device: spy.Device, resolution: int) -> NDBuffer:
    span = np.linspace(0, 1, resolution, dtype=np.float32)
    uvs_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=2)
    uvs = NDBuffer(device, "float2", shape=(resolution, resolution))
    uvs.copy_from_numpy(uvs_np)
    return uvs


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


ACTIVATIONS: dict[str, type] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "SmeLU": nn.SmeLU,
    "Swish": nn.Swish,
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
}


@dataclass(frozen=True)
class Config:
    activation: str
    hidden_width: int
    hidden_depth: int  # number of hidden LinearLayer+Activation blocks
    freq_bands: int
    precision: str  # "half" | "float"

    def label(self) -> str:
        return (
            f"act={self.activation},w={self.hidden_width},d={self.hidden_depth},"
            f"fb={self.freq_bands},p={self.precision}"
        )


@dataclass
class Result:
    config: dict
    msamples_per_s: float
    final_loss: float
    train_seconds: float
    eval_seconds: float
    epochs: int
    batches_per_epoch: int
    batch_shape: tuple[int, int]


# --------------------------------------------------------------------------- #
# Model assembly
# --------------------------------------------------------------------------- #


def build_model(cfg: Config, coopvec_available: bool) -> tuple[nn.ModelChain, nn.Real, float]:
    """Build a ModelChain matching the given config.

    Returns (model, mlp_precision_enum, grad_scale).
    Falls back to float+array if the requested precision is half but the device
    does not support cooperative_vector.
    """
    if cfg.precision == "half" and coopvec_available:
        mlp_input = nn.ArrayKind.coopvec
        mlp_precision = nn.Real.half
    else:
        mlp_input = nn.ArrayKind.array
        mlp_precision = nn.Real.float

    act_cls = ACTIVATIONS[cfg.activation]

    layers: list[nn.IModel] = [
        nn.Convert.to_array(),
        nn.FrequencyEncoding(cfg.freq_bands),
        nn.Convert.to_precision(mlp_precision),
        nn.Convert.to_array_kind(mlp_input),
    ]
    for _ in range(cfg.hidden_depth):
        layers.append(nn.LinearLayer(nn.Auto, cfg.hidden_width))
        layers.append(act_cls())
    layers.append(nn.LinearLayer(nn.Auto, 3))
    layers.append(nn.Convert.to_vector())
    layers.append(nn.Convert.to_precision(nn.Real.float))
    layers.append(nn.Exp())

    model = nn.ModelChain(*layers)

    grad_scale = 128.0 if mlp_precision == nn.Real.half else 1.0
    return model, mlp_precision, grad_scale


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #


def run_one(
    device: spy.Device,
    cfg: Config,
    target_tex: spy.Texture,
    uv_grid: NDBuffer,
    target_rgb_np: np.ndarray,
    *,
    epochs: int,
    batches_per_epoch: int,
    batch_shape: tuple[int, int],
    warmup_epochs: int,
    coopvec_available: bool,
) -> Result:
    """Train one config to completion. Return measured (throughput, loss)."""

    model, mlp_precision, grad_scale = build_model(cfg, coopvec_available)

    optim = nn.AdamOptimizer(learning_rate=0.005)
    if mlp_precision == nn.Real.half:
        optim = nn.FullPrecisionOptimizer(optim, gradient_scale=grad_scale)

    module = Module.load_from_file(device, str(SHADER))
    model.initialize(module, module.float2)
    optim.initialize(module, model.parameters())

    loss_scale = grad_scale / math.prod(batch_shape)

    sampler = device.create_sampler(min_lod=0, max_lod=0)
    seeds = (
        np.random.default_rng(42)
        .integers(0, 2**31 - 1, size=batch_shape, dtype=np.uint32)
        .astype(np.uint32)
    )
    rng = module.RNG(seeds)

    def run_epochs(n: int) -> None:
        for _ in range(n):
            cmd = device.create_command_encoder()
            for _ in range(batches_per_epoch):
                module.trainTexture.append_to(
                    cmd, model, rng, target_tex, sampler, loss_scale
                )
                optim.step(cmd)
            submit_id = device.submit_command_buffer(cmd.finish())
            device.wait_for_submit(submit_id)

    # Warmup (kernel compile, first dispatches)
    run_epochs(warmup_epochs)

    # Timed training
    t0 = time.perf_counter()
    run_epochs(epochs)
    train_seconds = time.perf_counter() - t0

    msamples = (epochs * batches_per_epoch * math.prod(batch_shape)) * 1e-6
    msamples_per_s = msamples / train_seconds if train_seconds > 0 else 0.0

    # Evaluate final reconstruction loss (full UV grid, L2 vs target)
    eval_t0 = time.perf_counter()
    output = module.evalModel(model, uv_grid)
    # output is a NDBuffer of float4
    pred_np = output.to_numpy()
    eval_seconds = time.perf_counter() - eval_t0

    pred_rgb = pred_np[..., :3].astype(np.float32)
    final_loss = float(np.mean((pred_rgb - target_rgb_np) ** 2))

    return Result(
        config=asdict(cfg),
        msamples_per_s=msamples_per_s,
        final_loss=final_loss,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        batch_shape=batch_shape,
    )


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


def sweep_configs(args: argparse.Namespace) -> list[Config]:
    acts = args.activations or list(ACTIVATIONS.keys())
    widths = args.widths or [16, 32, 64, 128]
    depths = args.depths or [2, 3, 4]
    freq_bands = args.freq_bands or [6]
    precisions = args.precisions or ["float"]
    return [
        Config(a, w, d, fb, p)
        for a, w, d, fb, p in itertools.product(
            acts, widths, depths, freq_bands, precisions
        )
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batches-per-epoch", type=int, default=10)
    ap.add_argument("--batch-h", type=int, default=256)
    ap.add_argument("--batch-w", type=int, default=256)
    ap.add_argument("--activations", nargs="+", choices=list(ACTIVATIONS.keys()))
    ap.add_argument("--widths", nargs="+", type=int)
    ap.add_argument("--depths", nargs="+", type=int)
    ap.add_argument("--freq-bands", nargs="+", type=int)
    ap.add_argument("--precisions", nargs="+", choices=["half", "float"])
    ap.add_argument("--output", type=pathlib.Path, default=THIS_DIR / "results.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N configs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = sweep_configs(args)
    if args.limit:
        configs = configs[: args.limit]

    print(f"[sweep] {len(configs)} configs")
    if args.dry_run:
        for c in configs:
            print(f"  {c.label()}")
        return 0

    # Local include path holds our migrated LinearLayer.slang (new DiffTensor API).
    # Must come before nn.slang_include_paths() so it shadows the stale in-tree copy.
    local_nn_slang = THIS_DIR / "neuralnetworks" / "slang"
    device = spy.create_device(
        spy.DeviceType.vulkan,
        False,
        include_paths=[local_nn_slang] + nn.slang_include_paths(),
    )
    coopvec = spy.Feature.cooperative_vector in device.features
    adapter = device.info.adapter_name
    print(f"[device] {adapter}  coopvec={coopvec}")

    target_np = make_target(args.resolution)
    target_tex = upload_target(device, target_np)
    uv_grid = create_uv_grid(device, args.resolution)

    with args.output.open("w") as f:
        f.write(
            json.dumps(
                {
                    "_meta": {
                        "adapter": adapter,
                        "coopvec": coopvec,
                        "resolution": args.resolution,
                        "epochs": args.epochs,
                        "warmup_epochs": args.warmup_epochs,
                        "batches_per_epoch": args.batches_per_epoch,
                        "batch_shape": [args.batch_h, args.batch_w],
                        "num_configs": len(configs),
                    }
                }
            )
            + "\n"
        )
        f.flush()

        for i, cfg in enumerate(configs):
            try:
                r = run_one(
                    device,
                    cfg,
                    target_tex,
                    uv_grid,
                    target_np,
                    epochs=args.epochs,
                    batches_per_epoch=args.batches_per_epoch,
                    batch_shape=(args.batch_h, args.batch_w),
                    warmup_epochs=args.warmup_epochs,
                    coopvec_available=coopvec,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(configs)}] {cfg.label()}  FAILED: {type(e).__name__}: {e}")
                f.write(
                    json.dumps({"config": asdict(cfg), "error": f"{type(e).__name__}: {e}"})
                    + "\n"
                )
                f.flush()
                continue

            print(
                f"  [{i+1}/{len(configs)}] {cfg.label()}  "
                f"throughput={r.msamples_per_s:7.2f} MS/s  loss={r.final_loss:.5f}  "
                f"train={r.train_seconds:.2f}s"
            )
            f.write(json.dumps(asdict(r)) + "\n")
            f.flush()

    print(f"\n[done] results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
