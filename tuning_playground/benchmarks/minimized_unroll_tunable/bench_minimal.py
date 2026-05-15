"""SlangPy runner for loop_bench_minimal.slang.

The shader keeps the original CUDA benchmark's task-struct entry point shape,
so this launcher builds those structs as Python dictionaries and dispatches the
backward kernels through SlangPy.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import slangpy as spy  # noqa: E402
from slangpy import diff_pair  # noqa: E402
from slangpy.types import Tensor  # noqa: E402

from tuning import ExhaustiveSearch, Tuner  # noqa: E402
from tuning.config import IntTunableBinding, TuningConfig, TunableBinding  # noqa: E402

THIS_DIR = pathlib.Path(__file__).parent
SHADER = THIS_DIR / "loop_bench_minimal.slang"
DEFAULT_OUTPUT = THIS_DIR / "results_minimized_unroll_minimal.jsonl"
SIGNAL_A_WIDTH = 3
DEFAULT_SIGNAL_B_WIDTH = 45


@dataclass
class Tasks:
    vectors: dict
    scales: dict
    opacities: dict
    features: dict
    merged_signal: dict


@dataclass
class Buffers:
    vectors_in: object
    vectors_out: object
    scales_in: object
    scales_out: object
    opacities_in: object
    opacities_out: object
    features_in: object
    features_out: object
    signal_a: object
    signal_b: object
    signal_out: object
    input_grad_refs: list[object]


@dataclass(frozen=True)
class RunCase:
    workload: str
    cfg: TuningConfig
    strategy: str
    feature_width: int
    merged_signal_b: int
    effective_iters: int


def get_strategy(cfg: TuningConfig) -> str:
    for b in cfg.bindings:
        if isinstance(b, TunableBinding) and b.tunable_name == "TunableLoopStrategy":
            return b.impl_name
    raise KeyError(f"TunableLoopStrategy not present in config {cfg}")


def get_int_binding(cfg: TuningConfig, name: str) -> int:
    for b in cfg.bindings:
        if isinstance(b, IntTunableBinding) and b.tunable_name == name:
            return b.value
    raise KeyError(f"{name} not present in config {cfg}")


def small_iters(feature_width: int) -> int:
    return 3 + 3 + 1 + feature_width


def large_iters(feature_width: int, merged_signal_b: int) -> int:
    return small_iters(feature_width) + SIGNAL_A_WIDTH + merged_signal_b


def make_tensor(device: spy.Device, samples: int, width: int, seed: int) -> Tensor:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.25, size=(samples, width)).astype(np.float32)
    return Tensor.from_numpy(device, data).with_grads()


def make_output(device: spy.Device, samples: int, width: int) -> Tensor:
    return Tensor.zeros(device, shape=(samples, width), dtype=float).with_grads()


def make_tensor_buffers(
    device: spy.Device,
    samples: int,
    feature_width: int,
    merged_signal_b: int,
) -> Buffers:
    vectors_in = make_tensor(device, samples, 3, 1)
    scales_in = make_tensor(device, samples, 3, 2)
    opacities_in = make_tensor(device, samples, 1, 3)
    features_in = make_tensor(device, samples, feature_width, 4)
    signal_a = make_tensor(device, samples, SIGNAL_A_WIDTH, 5)
    signal_b = make_tensor(device, samples, merged_signal_b, 6)
    return Buffers(
        vectors_in=vectors_in,
        vectors_out=make_output(device, samples, 3),
        scales_in=scales_in,
        scales_out=make_output(device, samples, 3),
        opacities_in=opacities_in,
        opacities_out=make_output(device, samples, 1),
        features_in=features_in,
        features_out=make_output(device, samples, feature_width),
        signal_a=signal_a,
        signal_b=signal_b,
        signal_out=make_output(device, samples, SIGNAL_A_WIDTH + merged_signal_b),
        input_grad_refs=[
            vectors_in.grad_out,
            scales_in.grad_out,
            opacities_in.grad_out,
            features_in.grad_out,
            signal_a.grad_out,
            signal_b.grad_out,
        ],
    )


def make_torch_pair(samples: int, width: int, seed: int):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    primal = torch.randn(
        (samples, width),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) * 0.25
    grad = torch.zeros_like(primal)
    return diff_pair(primal, grad), grad


def make_torch_zero_pair(samples: int, width: int):
    import torch

    primal = torch.zeros((samples, width), device="cuda", dtype=torch.float32)
    grad = torch.zeros_like(primal)
    return diff_pair(primal, grad), grad


def make_torch_buffers(
    samples: int,
    feature_width: int,
    merged_signal_b: int,
) -> Buffers:
    vectors_in, vectors_grad = make_torch_pair(samples, 3, 1)
    vectors_out, _ = make_torch_zero_pair(samples, 3)
    scales_in, scales_grad = make_torch_pair(samples, 3, 2)
    scales_out, _ = make_torch_zero_pair(samples, 3)
    opacities_in, opacities_grad = make_torch_pair(samples, 1, 3)
    opacities_out, _ = make_torch_zero_pair(samples, 1)
    features_in, features_grad = make_torch_pair(samples, feature_width, 4)
    features_out, _ = make_torch_zero_pair(samples, feature_width)
    signal_a, signal_a_grad = make_torch_pair(samples, SIGNAL_A_WIDTH, 5)
    signal_b, signal_b_grad = make_torch_pair(samples, merged_signal_b, 6)
    signal_out, _ = make_torch_zero_pair(samples, SIGNAL_A_WIDTH + merged_signal_b)
    return Buffers(
        vectors_in=vectors_in,
        vectors_out=vectors_out,
        scales_in=scales_in,
        scales_out=scales_out,
        opacities_in=opacities_in,
        opacities_out=opacities_out,
        features_in=features_in,
        features_out=features_out,
        signal_a=signal_a,
        signal_b=signal_b,
        signal_out=signal_out,
        input_grad_refs=[
            vectors_grad,
            scales_grad,
            opacities_grad,
            features_grad,
            signal_a_grad,
            signal_b_grad,
        ],
    )


def clear_input_grads(bufs: Buffers) -> None:
    for grad in bufs.input_grad_refs:
        assert grad is not None
        grad.clear() if hasattr(grad, "clear") else grad.zero_()


def make_buffers(
    device: spy.Device,
    storage: str,
    samples: int,
    feature_width: int,
    merged_signal_b: int,
) -> Buffers:
    if storage == "torch":
        return make_torch_buffers(samples, feature_width, merged_signal_b)
    return make_tensor_buffers(device, samples, feature_width, merged_signal_b)


def make_tasks(bufs: Buffers, feature_width: int, merged_signal_b: int) -> Tasks:
    return Tasks(
        vectors={
            "_type": "CopyTask<3>",
            "input": bufs.vectors_in,
            "output": bufs.vectors_out,
        },
        scales={
            "_type": "ExpTask<3>",
            "input": bufs.scales_in,
            "output": bufs.scales_out,
        },
        opacities={
            "_type": "SigmoidTask<1>",
            "input": bufs.opacities_in,
            "output": bufs.opacities_out,
        },
        features={
            "_type": "FeatureTask",
            "input": bufs.features_in,
            "output": bufs.features_out,
        },
        merged_signal={
            "_type": "MergedSignalTask",
            "source_a": bufs.signal_a,
            "source_b": bufs.signal_b,
            "output": bufs.signal_out,
        },
    )


def make_run_cases(configs: list[TuningConfig]) -> list[RunCase]:
    cases: list[RunCase] = []
    small_representatives: dict[tuple[str, int], TuningConfig] = {}

    for cfg in configs:
        strategy = get_strategy(cfg)
        feature_width = get_int_binding(cfg, "kFeatureWidth")
        merged_signal_b = get_int_binding(cfg, "kMergedSignalB")
        cases.append(
            RunCase(
                workload="large",
                cfg=cfg,
                strategy=strategy,
                feature_width=feature_width,
                merged_signal_b=merged_signal_b,
                effective_iters=large_iters(feature_width, merged_signal_b),
            )
        )

        small_key = (strategy, feature_width)
        current = small_representatives.get(small_key)
        if current is None or merged_signal_b == DEFAULT_SIGNAL_B_WIDTH:
            small_representatives[small_key] = cfg

    for cfg in small_representatives.values():
        feature_width = get_int_binding(cfg, "kFeatureWidth")
        cases.append(
            RunCase(
                workload="small",
                cfg=cfg,
                strategy=get_strategy(cfg),
                feature_width=feature_width,
                merged_signal_b=get_int_binding(cfg, "kMergedSignalB"),
                effective_iters=small_iters(feature_width),
            )
        )

    return sorted(
        cases,
        key=lambda c: (
            0 if c.workload == "small" else 1,
            c.effective_iters,
            c.strategy,
            c.feature_width,
            c.merged_signal_b,
        ),
    )


def filter_configs(
    configs: list[TuningConfig],
    feature_widths: set[int] | None,
    merged_signal_bs: set[int] | None,
) -> list[TuningConfig]:
    if feature_widths is None and merged_signal_bs is None:
        return configs

    filtered = []
    for cfg in configs:
        feature_width = get_int_binding(cfg, "kFeatureWidth")
        merged_signal_b = get_int_binding(cfg, "kMergedSignalB")
        if feature_widths is not None and feature_width not in feature_widths:
            continue
        if merged_signal_bs is not None and merged_signal_b not in merged_signal_bs:
            continue
        filtered.append(cfg)
    return filtered


def run_backward(
    backward_function,
    workload: str,
    samples: int,
    tasks: Tasks,
    command_encoder=None,
) -> None:
    common = (
        0,
        samples,
        tasks.vectors,
        tasks.scales,
        tasks.opacities,
        tasks.features,
    )
    kwargs = {"_thread_count": samples}
    if command_encoder is not None:
        kwargs["_append_to"] = command_encoder
    if workload == "small":
        backward_function(*common, **kwargs)
    else:
        backward_function(*common, tasks.merged_signal, **kwargs)


def time_cpu_wall(device, function, workload: str, samples: int, tasks: Tasks, calls: int) -> dict:
    t0 = time.perf_counter()
    for _ in range(calls):
        run_backward(function, workload, samples, tasks)
    device.wait_for_idle()
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": elapsed,
        "us_per_call": (elapsed / calls) * 1e6,
    }


def time_gpu_timestamp(device, function, workload: str, samples: int, tasks: Tasks, calls: int) -> dict:
    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=calls * 2)
    for i in range(calls):
        command_encoder = device.create_command_encoder()
        command_encoder.write_timestamp(query_pool, i * 2)
        run_backward(function, workload, samples, tasks, command_encoder=command_encoder)
        command_encoder.write_timestamp(query_pool, i * 2 + 1)
        device.submit_command_buffer(command_encoder.finish())
    device.wait_for_idle()

    queries = np.array(query_pool.get_results(0, calls * 2), dtype=np.float64)
    frequency = float(device.info.timestamp_frequency)
    deltas_us = (queries[1::2] - queries[0::2]) / frequency * 1e6
    return {
        "elapsed_seconds": float(np.sum(deltas_us) / 1e6),
        "us_per_call": float(np.mean(deltas_us)),
        "gpu_us_min": float(np.min(deltas_us)),
        "gpu_us_max": float(np.max(deltas_us)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["vulkan", "cuda"], default="cuda")
    ap.add_argument("--samples", type=int, default=500_000)
    ap.add_argument("--warmup-calls", type=int, default=50)
    ap.add_argument("--timed-calls", type=int, default=1000)
    ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--storage", choices=["tensor", "torch"], default="torch")
    ap.add_argument("--timing", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--thread-group-size", type=int, default=32)
    ap.add_argument("--feature-widths", nargs="+", type=int)
    ap.add_argument("--merged-signal-bs", nargs="+", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    device_type = {
        "cuda": spy.DeviceType.cuda,
        "vulkan": spy.DeviceType.vulkan,
    }[args.device]
    device = spy.create_device(device_type, False, include_paths=[THIS_DIR])
    print(f"[device] {device.info.adapter_name} ({args.device})")

    raw_module = device.load_module(str(SHADER))
    tuner = Tuner(model=ExhaustiveSearch())
    space = tuner.setup(device, raw_module)
    print(f"[discover]\n{space}")

    configs = space.all_configs()
    configs = filter_configs(
        configs,
        set(args.feature_widths) if args.feature_widths else None,
        set(args.merged_signal_bs) if args.merged_signal_bs else None,
    )
    if args.limit:
        configs = configs[: args.limit]
    run_cases = make_run_cases(configs)

    if args.dry_run:
        for case in run_cases:
            print(
                f"  workload={case.workload} iters={case.effective_iters} "
                f"feature={case.feature_width} mergedB={case.merged_signal_b} "
                f"{case.cfg}"
            )
        print(f"[dry-run] shader_configs={len(configs)} run_cases={len(run_cases)}")
        return 0

    module = spy.Module(raw_module)
    funcs = {
        "small": module.fused_apply_small,
        "large": module.fused_apply_large,
    }
    if args.storage == "torch" and args.device != "cuda":
        raise ValueError("--storage torch requires --device cuda")
    buffer_cache: dict[tuple[int, int], tuple[Buffers, Tasks]] = {}

    with args.output.open("w") as f:
        f.write(
            json.dumps(
                {
                    "_meta": {
                        "adapter": device.info.adapter_name,
                        "device": args.device,
                        "samples": args.samples,
                        "warmup_calls": args.warmup_calls,
                        "timed_calls": args.timed_calls,
                        "storage": args.storage,
                        "timing": args.timing,
                        "thread_group_size": args.thread_group_size,
                        "shader_configs": len(configs),
                        "run_cases": len(run_cases),
                        "driver": "bench_minimal.py",
                        "shader": SHADER.name,
                        "note": "Task-struct SlangPy execution of minimized-unroll-bench with tunable loop strategy and fused-loop widths",
                    }
                }
            )
            + "\n"
        )
        f.flush()

        for i, case in enumerate(run_cases):
            cache_key = (case.feature_width, case.merged_signal_b)
            if cache_key not in buffer_cache:
                bufs = make_buffers(
                    device,
                    args.storage,
                    args.samples,
                    case.feature_width,
                    case.merged_signal_b,
                )
                buffer_cache[cache_key] = (
                    bufs,
                    make_tasks(bufs, case.feature_width, case.merged_signal_b),
                )
            bufs, tasks = buffer_cache[cache_key]

            function = funcs[case.workload].with_settings(case.cfg).bwds
            if args.thread_group_size != 32:
                function = function.thread_group_size(spy.uint3(args.thread_group_size, 1, 1))
            clear_input_grads(bufs)

            for _ in range(args.warmup_calls):
                run_backward(function, case.workload, args.samples, tasks)
            device.wait_for_idle()

            clear_input_grads(bufs)
            timing = (
                time_gpu_timestamp
                if args.timing == "gpu"
                else time_cpu_wall
            )(device, function, case.workload, args.samples, tasks, args.timed_calls)

            result = {
                "workload": case.workload,
                "strategy": case.strategy,
                "config": str(case.cfg),
                "feature_width": case.feature_width,
                "merged_signal_b": case.merged_signal_b,
                "effective_iters": case.effective_iters,
                "samples": args.samples,
                "calls": args.timed_calls,
                "timing": args.timing,
                **timing,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()
            tuner.report(case.cfg, result["us_per_call"] / 1000.0)
            print(
                f"  [{i + 1}/{len(run_cases)}] {case.workload} "
                f"iters={case.effective_iters} feature={case.feature_width} "
                f"mergedB={case.merged_signal_b} {case.strategy}: "
                f"{result['us_per_call']:.1f} us ({args.timing})"
            )

    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
