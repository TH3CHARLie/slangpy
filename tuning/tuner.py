"""User-facing autotuning orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import slangpy as spy

from .model import ExhaustiveSearch, TuningModel
from .reflection import link_variant
from .config import TuningConfig, TuningResult, TuningSpace

if TYPE_CHECKING:
    from slangpy import Device, SlangModule


class Tuner:
    """User-facing autotuning orchestrator.

    Bridges the search model, the tuning space, and the variant linking
    machinery. Does not own the execution loop — the user does.

    Example::

        tuner = Tuner(model=ExhaustiveSearch())
        space = tuner.setup(device, raw_module)

        while not tuner.is_complete():
            config = tuner.propose()
            variant = tuner.get_variant(config)
            # warm up
            variant.my_func(args)
            # measure
            t0 = time.perf_counter()
            variant.my_func(args)
            elapsed = (time.perf_counter() - t0) * 1000
            tuner.report(config, elapsed)

        best_module = tuner.get_variant(tuner.best())
    """

    def __init__(self, model: TuningModel | None = None) -> None:
        """Create a tuner with a given search model.

        :param model: Search strategy. Defaults to ExhaustiveSearch.
        """
        self._model = model or ExhaustiveSearch()
        self._space: TuningSpace | None = None
        self._device: Device | None = None
        self._module: SlangModule | None = None
        self._variant_cache: dict[TuningConfig, spy.Module] = {}

    def setup(self, device: Device, module: SlangModule) -> TuningSpace:
        """Discover the tuning space and initialize the model.

        Must be called before propose/report.

        :param device: The GPU device.
        :param module: A loaded Slang module with [Tunable] extern structs.
        :return: The discovered tuning space.
        """
        self._device = device
        self._module = module
        self._space = TuningSpace.discover(module)
        self._model.initialize(self._space)
        return self._space

    @property
    def space(self) -> TuningSpace:
        """The tuning space. Only available after setup()."""
        if self._space is None:
            raise RuntimeError("Call setup() before accessing the tuning space")
        return self._space

    def propose(self) -> TuningConfig:
        """Ask the model for the next config to try.

        :return: The next config to evaluate.
        :raises StopIteration: When the model has no more configs to propose.
        """
        return self._model.propose()

    def report(self, config: TuningConfig, elapsed_ms: float) -> None:
        """Report timing for a config.

        :param config: The config that was evaluated.
        :param elapsed_ms: Wall-clock time in milliseconds.
        """
        self._model.report(config, elapsed_ms)

    def best(self) -> TuningConfig:
        """Best config found so far (lowest elapsed_ms).

        :raises ValueError: If no results have been reported yet.
        """
        return self._model.best()

    def is_complete(self) -> bool:
        """Whether the model wants to stop proposing."""
        return self._model.is_complete()

    @property
    def history(self) -> list[TuningResult]:
        """All results collected so far, in order."""
        return self._model.history

    def get_variant(self, config: TuningConfig) -> spy.Module:
        """Get (or create and cache) a linked Module for a config.

        :param config: The tuning config to link.
        :return: A SlangPy Module with the specified implementations linked in.
        """
        if self._device is None or self._module is None:
            raise RuntimeError("Call setup() before get_variant()")
        if config not in self._variant_cache:
            int_bindings = {b.tunable_name: b.value for b in config.int_bindings()}
            self._variant_cache[config] = link_variant(
                self._device, self._module, config.to_bindings_dict(),
                int_bindings=int_bindings or None,
            )
        return self._variant_cache[config]
