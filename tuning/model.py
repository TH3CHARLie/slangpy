"""Tuning model interface and implementations.

A TuningModel is a search strategy that proposes configs to evaluate,
receives timing feedback, and tracks the best result found.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import TuningConfig, TuningResult, TuningSpace


class TuningModel(ABC):
    """Abstract search strategy for tuning.

    Subclass this to implement different search strategies
    (Bayesian optimization, bandits, random search, etc.).
    """

    @abstractmethod
    def initialize(self, space: TuningSpace) -> None:
        """Called once with the tuning space before any proposals.

        :param space: The tuning space to search over.
        """

    @abstractmethod
    def propose(self) -> TuningConfig:
        """Propose the next config to evaluate.

        :return: The next config to try.
        :raises StopIteration: When the model has no more configs to propose.
        """

    @abstractmethod
    def report(self, config: TuningConfig, elapsed_ms: float) -> None:
        """Report the result of evaluating a config.

        :param config: The config that was evaluated.
        :param elapsed_ms: Wall-clock time in milliseconds.
        """

    @abstractmethod
    def best(self) -> TuningConfig:
        """Return the best config found so far (lowest elapsed_ms).

        :raises ValueError: If no results have been reported yet.
        """

    @abstractmethod
    def is_complete(self) -> bool:
        """Whether the model has explored enough and won't propose more."""

    @property
    @abstractmethod
    def history(self) -> list[TuningResult]:
        """All results collected so far, in order."""


class ExhaustiveSearch(TuningModel):
    """Try every config exactly once. Simple and complete."""

    def __init__(self) -> None:
        self._all_configs: list[TuningConfig] = []
        self._next_index: int = 0
        self._results: list[TuningResult] = []

    def initialize(self, space: TuningSpace) -> None:
        self._all_configs = space.all_configs()
        self._next_index = 0
        self._results = []

    def propose(self) -> TuningConfig:
        if self._next_index >= len(self._all_configs):
            raise StopIteration("All configs have been evaluated")
        config = self._all_configs[self._next_index]
        self._next_index += 1
        return config

    def report(self, config: TuningConfig, elapsed_ms: float) -> None:
        self._results.append(
            TuningResult(config=config, elapsed_ms=elapsed_ms, iteration=len(self._results))
        )

    def best(self) -> TuningConfig:
        if not self._results:
            raise ValueError("No results reported yet")
        return min(self._results, key=lambda r: r.elapsed_ms).config

    def is_complete(self) -> bool:
        return self._next_index >= len(self._all_configs)

    @property
    def history(self) -> list[TuningResult]:
        return list(self._results)
