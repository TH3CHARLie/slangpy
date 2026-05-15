"""OpenTuner workflow showcase for the neural-texture forward benchmark.

This file intentionally avoids the tuning APIs introduced by our work. It
models what an OpenTuner user would write against ordinary SlangPy: manually
mirror the search space in Python, generate a concrete shader source for each
configuration, load that module, and measure it as a black box. The generated
shader imports activation.slang, matching the modular shader layout used by the
reflected-tunable benchmark, but OpenTuner still needs the activation list
redeclared below.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

# Ensure local slangpy is importable when run directly.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import opentuner  # noqa: E402
import slangpy as spy  # noqa: E402
from opentuner import (  # noqa: E402
    ConfigurationManipulator,
    EnumParameter,
    MeasurementInterface,
    Result,
)


THIS_DIR = pathlib.Path(__file__).parent
TEMPLATE = THIS_DIR / "opentuner_net_template.slang"


ACTIVATIONS = [
    "TReLU",
    "TLeakyReLU",
    "TELU",
    "TSmeLU",
    "TSwish",
    "TTanh",
    "TSigmoid",
]
WIDTHS = [16, 32, 64, 128]
DEPTHS = [2, 3, 4]
FREQ_BANDS = [6]
BAD_WIDTH_CHOICE = "not_an_int"


def init_params(width: int, depth: int, freq_bands: int, seed: int = 0) -> np.ndarray:
    """Kaiming-normal initialisation over the flat param vector."""
    rng = np.random.default_rng(seed)
    input_dim = 2 + 4 * freq_bands
    chunks: list[np.ndarray] = []

    def kaiming(fan_in: int, shape: tuple[int, ...]) -> np.ndarray:
        std = math.sqrt(2.0 / fan_in)
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    chunks.append(kaiming(input_dim, (width * input_dim,)))
    chunks.append(np.zeros((width,), dtype=np.float32))
    for _ in range(depth - 1):
        chunks.append(kaiming(width, (width * width,)))
        chunks.append(np.zeros((width,), dtype=np.float32))
    chunks.append(kaiming(width, (3 * width,)))
    chunks.append(np.zeros((3,), dtype=np.float32))
    return np.concatenate(chunks)


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


def specialize_source(
    template: str,
    act: str,
    width: int,
    depth: int,
    freq_bands: int,
) -> str:
    """Generate a concrete shader source for one OpenTuner configuration."""
    return (
        template.replace("__ACTIVATION__", act)
        .replace("__K_WIDTH__", str(width))
        .replace("__K_DEPTH__", str(depth))
        .replace("__K_FREQ_BANDS__", str(freq_bands))
    )


class NeuralTextureOpenTuner(MeasurementInterface):
    """OpenTuner adapter that measures generated Slang modules as black boxes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._runtime_ready = False
        self._params_cache: dict[tuple[int, int, int], spy.Buffer] = {}
        self._func_cache = {}
        self._trial_idx = 0

    def manipulator(self) -> ConfigurationManipulator:
        """Declare the search space in Python, as OpenTuner expects."""
        small_space = getattr(self.args, "small_space", False)
        activations = ["TReLU", "TELU"] if small_space else ACTIVATIONS
        widths = [16] if small_space else WIDTHS
        depths = [2] if small_space else DEPTHS
        freq_bands = [6] if small_space else FREQ_BANDS
        if getattr(self.args, "inject_bad_width_choice", False):
            widths = [BAD_WIDTH_CHOICE] if small_space else [BAD_WIDTH_CHOICE, *widths]

        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter("Act", activations))
        manipulator.add_parameter(EnumParameter("kWidth", widths))
        manipulator.add_parameter(EnumParameter("kDepth", depths))
        manipulator.add_parameter(EnumParameter("kFreqBands", freq_bands))
        return manipulator

    def ensure_runtime(self) -> None:
        if self._runtime_ready:
            return

        self.device = spy.create_device(
            spy.DeviceType.vulkan,
            False,
            include_paths=[THIS_DIR],
        )
        self.adapter = self.device.info.adapter_name
        self.template_source = TEMPLATE.read_text()
        self.uv_grid = create_uv_grid(self.device, self.args.resolution)
        width_count = 1 if self.args.small_space else len(WIDTHS)
        if self.args.inject_bad_width_choice and not self.args.small_space:
            width_count += 1
        active_configs = (
            len(["TReLU", "TELU"] if self.args.small_space else ACTIVATIONS)
            * width_count
            * len([2] if self.args.small_space else DEPTHS)
            * len([6] if self.args.small_space else FREQ_BANDS)
        )

        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        with self.args.output.open("w") as f:
            f.write(
                json.dumps(
                    {
                        "_meta": {
                            "adapter": self.adapter,
                            "resolution": self.args.resolution,
                            "warmup_calls": self.args.warmup_calls,
                            "timed_calls": self.args.timed_calls,
                            "driver": "opentuner_showcase.py",
                            "tuner": "OpenTuner",
                            "small_space": self.args.small_space,
                            "inject_bad_width_choice": (
                                self.args.inject_bad_width_choice
                            ),
                            "active_configs": active_configs,
                            "total_declared_configs": (
                                len(ACTIVATIONS)
                                * len(WIDTHS)
                                * len(DEPTHS)
                                * len(FREQ_BANDS)
                            ),
                            "note": "OpenTuner manually declares the tuning space "
                            "in Python and source-specializes one Slang module per "
                            "config. The generated shader imports activation.slang.",
                        }
                    }
                )
                + "\n"
            )
        self._runtime_ready = True

    def get_func(self, act: str, width: int, depth: int, freq_bands: int):
        key = (act, width, depth, freq_bands)
        if key not in self._func_cache:
            source = specialize_source(
                self.template_source,
                act,
                width,
                depth,
                freq_bands,
            )
            module_name = f"opentuner_net_{act}_{width}_{depth}_{freq_bands}"
            virtual_path = THIS_DIR / f"{module_name}.slang"
            raw_module = self.device.load_module_from_source(
                module_name,
                source,
                path=str(virtual_path),
            )
            self._func_cache[key] = spy.Module(raw_module).evalPixel
        return self._func_cache[key]

    def run(self, desired_result, input, limit) -> Result:
        self.ensure_runtime()

        cfg = desired_result.configuration.data
        w = int(cfg["kWidth"])
        d = int(cfg["kDepth"])
        fb = int(cfg["kFreqBands"])
        act = cfg["Act"]

        shape_key = (w, d, fb)
        if shape_key not in self._params_cache:
            params_np = init_params(w, d, fb)
            self._params_cache[shape_key] = upload_params(self.device, params_np)
        params = self._params_cache[shape_key]

        func = self.get_func(act, w, d, fb)

        for _ in range(self.args.warmup_calls):
            func(self.uv_grid, params)
        self.device.wait_for_idle()

        t0 = time.perf_counter()
        for _ in range(self.args.timed_calls):
            func(self.uv_grid, params)
        self.device.wait_for_idle()
        elapsed = time.perf_counter() - t0

        elapsed_ms = (elapsed / self.args.timed_calls) * 1000
        total_samples = self.args.timed_calls * (
            self.args.resolution * self.args.resolution
        )
        msamples_per_s = (total_samples * 1e-6) / elapsed if elapsed > 0 else 0.0

        self._trial_idx += 1
        config_label = f"Act={act},kWidth={w},kDepth={d},kFreqBands={fb}"
        with self.args.output.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "trial": self._trial_idx,
                        "config": config_label,
                        "kWidth": w,
                        "kDepth": d,
                        "kFreqBands": fb,
                        "activation": act,
                        "msamples_per_s": msamples_per_s,
                        "elapsed_seconds": elapsed,
                        "calls": self.args.timed_calls,
                    }
                )
                + "\n"
            )

        result = Result(time=elapsed_ms)
        if hasattr(result, "set_attribute"):
            result.set_attribute("msamples_per_s", msamples_per_s)
        print(
            f"  opentuner trial {self._trial_idx}: "
            f"w={w},d={d},fb={fb} act={act}  {msamples_per_s:.1f} MS/s",
            flush=True,
        )
        return result

    def save_final_config(self, configuration) -> None:
        print(f"[opentuner] best config: {configuration.data}")


def main() -> None:
    argparser = opentuner.default_argparser()
    argparser.add_argument("--resolution", type=int, default=64)
    argparser.add_argument("--warmup-calls", type=int, default=1)
    argparser.add_argument("--timed-calls", type=int, default=2)
    argparser.add_argument(
        "--small-space",
        action="store_true",
        help="restrict to two cheap variants for a quick showcase smoke run",
    )
    argparser.add_argument(
        "--inject-bad-width-choice",
        action="store_true",
        help=(
            "add a non-integer kWidth option to demonstrate that OpenTuner's "
            "Python-side enum search space is not shader type checked. With "
            "--small-space, the invalid width is the only width so the failure "
            "is deterministic."
        ),
    )
    argparser.add_argument(
        "--output",
        type=pathlib.Path,
        default=THIS_DIR / "results_opentuner_showcase.jsonl",
    )
    NeuralTextureOpenTuner.main(argparser.parse_args())


if __name__ == "__main__":
    main()
