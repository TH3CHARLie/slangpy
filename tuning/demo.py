"""Demo: autotuning ml_pipeline.slang with the Tuner API."""

import pathlib
import sys
import time

# Ensure the tuning package is importable when run directly
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import slangpy as spy

from tuning import ExhaustiveSearch, Tuner

slang_path = pathlib.Path(__file__).parent.parent / "tuning_playground" / "ml_pipeline.slang"
slang_source = slang_path.read_text()

slangpy_slang_path = pathlib.Path(__file__).parent.parent / "slangpy" / "slang"
device = spy.Device(
    compiler_options=spy.SlangCompilerOptions(
        {"include_paths": [slang_path.parent, slangpy_slang_path]}
    ),
)
raw_module = device.load_module_from_source(slang_path.stem, slang_source)

# Set up the tuner
tuner = Tuner(model=ExhaustiveSearch())
space = tuner.setup(device, raw_module)
print(space)
print()

# Test inputs
test_values = (-1.5, 0.5, 2.0, -0.3)

# Tuning loop — user owns the loop
print(f"--- Tuning transform_reduce({test_values}) ---")
while not tuner.is_complete():
    config = tuner.propose()
    variant = tuner.get_variant(config)

    # Warm up (first call compiles the kernel)
    variant.transform_reduce(*test_values)

    # Measure
    t0 = time.perf_counter()
    result = variant.transform_reduce(*test_values)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    tuner.report(config, elapsed_ms)

    print(f"  {config}: result={result:.4f}  time={elapsed_ms:.3f} ms")

# Results
print(f"\n--- Results ({len(tuner.history)} trials) ---")
for r in sorted(tuner.history, key=lambda r: r.elapsed_ms):
    print(f"  {r.config}: {r.elapsed_ms:.3f} ms")

best = tuner.best()
print(f"\nBest: {best}")
