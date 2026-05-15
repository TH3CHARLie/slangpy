# Neural Texture: Slang Tunables vs. OpenTuner Workflow

Goal: show the same neural-texture tuning task expressed in two workflows.
This is a coding-style comparison, not a full OpenTuner performance evaluation.

## Slang / SlangPy Workflow

The tuning space lives with the shader. SlangPy discovers it through reflection.

```slang
// activation.slang
[Tunable]
public extern struct Act : IBenchAct = TReLU;

// bench_net.slang
import activation;

[Tunable(16, 32, 64, 128)]
public extern static const int kWidth = 32;

[Tunable(2, 3, 4)]
public extern static const int kDepth = 2;

[Tunable(6)]
public extern static const int kFreqBands = 6;

```

```python
raw_module = device.load_module(str(SHADER))

tuner = Tuner(model=ExhaustiveSearch())
space = tuner.setup(device, raw_module)

module = spy.Module(raw_module)
func = module.evalPixel

while not tuner.is_complete():
    cfg = tuner.propose()
    variant = func.with_settings(cfg)

    elapsed_ms = measure(variant, uv_grid, params)
    tuner.report(cfg, elapsed_ms)
```

The Python driver does not duplicate the tunable choices. Adding a shader-side
choice, such as another activation implementation or width, changes the reflected
space automatically.

## OpenTuner Workflow

OpenTuner treats the shader as a black-box program to measure. Without the new
Slang tuning APIs, the tuning space is redeclared in Python and each config is
turned into a concrete shader source. The generated OpenTuner shader can still
import the shared `activation.slang` implementation module, but the host code
must manually mirror which activation names are legal choices.

```python
class NeuralTextureOpenTuner(MeasurementInterface):
    def manipulator(self):
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter("Act", [
            "TReLU", "TLeakyReLU", "TELU", "TSmeLU",
            "TSwish", "TTanh", "TSigmoid",
        ]))
        manipulator.add_parameter(EnumParameter("kWidth", [16, 32, 64, 128]))
        manipulator.add_parameter(EnumParameter("kDepth", [2, 3, 4]))
        manipulator.add_parameter(EnumParameter("kFreqBands", [6]))
        return manipulator
```

```python
    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data

        source = specialize_source(
            template_source,
            act=cfg["Act"],
            width=cfg["kWidth"],
            depth=cfg["kDepth"],
            freq_bands=cfg["kFreqBands"],
        )
        raw_module = device.load_module_from_source(module_name(cfg), source)
        func = spy.Module(raw_module).evalPixel

        elapsed_ms = measure(func, uv_grid, params)
        return Result(time=elapsed_ms)
```

This path is useful as a familiar black-box autotuning baseline. The cost is
that the search space and specialization mechanism are maintained in Python:
the driver must enumerate choices across both shader files, generate source,
compile/load each variant, and cache the resulting functions itself.

Because OpenTuner sees these choices as Python enum values, it does not know
that `kWidth`, `kDepth`, and `kFreqBands` are shader integer tunables. If the
Python manipulator includes a bad value such as `"not_an_int"` for `kWidth`,
OpenTuner accepts the search space and the failure only appears later when the
benchmark tries to cast the chosen value before generating Slang source.

## Suggested Paper Caption

Both examples specialize the same `evalPixel` Slang kernel. In the Slang/SlangPy
workflow, tunable choices are declared across `activation.slang` and
`bench_net.slang` and discovered through reflection. In the OpenTuner workflow,
the benchmark author manually mirrors the same cross-file choices in Python and
source-specializes a separate Slang module for each proposed configuration.

## Smoke Command

Use a tiny OpenTuner run only to validate the adapter:

```bash
python tuning_playground/benchmarks/neural_texture/opentuner_showcase.py \
  --small-space \
  --technique=PureRandom \
  --test-limit=2 \
  --resolution=64 \
  --warmup-calls=1 \
  --timed-calls=2 \
  --database sqlite:////tmp/neural_texture_opentuner_showcase.db \
  --output /tmp/neural_texture_opentuner_showcase.jsonl
```

OpenTuner may still request a few duplicate/probe evaluations around the small
space. That is fine for this showcase; the goal is to validate the adapter and
capture the code-shape contrast, not to collect final tuning statistics.

To demonstrate the missing type check in the OpenTuner path:

```bash
python tuning_playground/benchmarks/neural_texture/opentuner_showcase.py \
  --small-space \
  --inject-bad-width-choice \
  --technique=PureRandom \
  --test-limit=1 \
  --resolution=8 \
  --warmup-calls=0 \
  --timed-calls=1 \
  --database sqlite:////tmp/neural_texture_opentuner_bad_width.db \
  --output /tmp/neural_texture_opentuner_bad_width.jsonl
```

Expected result: OpenTuner starts normally, then the benchmark fails with the
raw Python conversion error when it evaluates `int(cfg["kWidth"])`, e.g.
`ValueError: invalid literal for int() with base 10: 'not_an_int'`.
