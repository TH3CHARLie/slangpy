import slangpy as spy
import numpy as np
import pathlib
from slangpy.core.function import AutotuneSetting

# Create a device with the local folder for slangpy includes
device = spy.create_device(
    include_paths=[
        pathlib.Path(__file__).parent.absolute(),
    ]
)

# Load module
module = spy.Module.load_from_file(device, "tmp.slang")

a = 5.1
b = 10
tuner = spy.Tuner()
tuner.init_model()

func = module.add
assert(isinstance(func, spy.Function))
N_ITERS = 50
tunable_params = tuner.get_tunable_parameters(func)

for i in range(N_ITERS):
    settings = tuner.propose_settings(func, tunable_params)
    if i % 2 == 0:
        settings = {"TunableFoo": AutotuneSetting("TunableFoo", "IFoo", "FooSlow")}
    else:
        settings = {"TunableFoo": AutotuneSetting("TunableFoo", "IFoo", "FooFast")}
    func_modified = func.with_settings(settings)
    timer = spy.Timer()
    result = func_modified(a=a, b=b)
    print(result)
    elapsed = timer.elapsed_ms()
    print(f"Iteration {i} took {elapsed} ms with settings {settings}")
    tuner.update_model(func, settings, elapsed)

best_settings = tuner.predict_optimal_settings(func, tunable_params)
func_best = func.with_settings(best_settings)
# func_best(a=a, b=b, _result="numpy")
