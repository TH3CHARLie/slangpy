"""Core data types for the tuning system."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slangpy import SlangModule


@dataclass(frozen=True)
class TunableBinding:
    """One tunable bound to one implementation."""

    tunable_name: str
    interface_name: str
    impl_name: str

    def __str__(self) -> str:
        return f"{self.tunable_name}={self.impl_name}"


@dataclass
class TunableParam:
    """One tunable axis: an extern struct, the interface it conforms to, and available impls."""

    tunable_name: str
    interface_name: str
    impl_names: list[str]

    def __str__(self) -> str:
        return f"{self.tunable_name}: {self.interface_name} -> [{', '.join(self.impl_names)}]"


@dataclass(frozen=True)
class TuningConfig:
    """A complete set of tunable bindings. One point in the tuning space.

    Immutable and hashable so it can be used as a dict key.
    """

    bindings: tuple[TunableBinding, ...]

    def to_bindings_dict(self) -> dict[str, tuple[str, str]]:
        """Convert to the format link_variant() expects."""
        return {
            b.tunable_name: (b.interface_name, b.impl_name) for b in self.bindings
        }

    def __str__(self) -> str:
        return ",".join(str(b) for b in self.bindings)


@dataclass
class TuningResult:
    """Records one trial: a config and its measured performance."""

    config: TuningConfig
    elapsed_ms: float
    iteration: int


class TuningSpace:
    """The full tuning space for a module — all tunables and their options."""

    def __init__(self, params: list[TunableParam]) -> None:
        self.params = params

    @staticmethod
    def discover(module: SlangModule) -> TuningSpace:
        """Build from a loaded module using Slang reflection.

        :param module: A loaded Slang module.
        :return: A TuningSpace describing all [Tunable] extern structs and their options.
        """
        from .reflection import discover_tuning_space

        raw = discover_tuning_space(module)
        params: list[TunableParam] = []
        for tunable_name, interfaces in raw.items():
            for interface_name, impl_names in interfaces.items():
                params.append(TunableParam(tunable_name, interface_name, impl_names))
                break  # one interface per tunable
        return TuningSpace(params)

    def all_configs(self) -> list[TuningConfig]:
        """Enumerate every combinatorial config (cartesian product)."""
        if not self.params:
            return []
        impl_lists = [
            [
                TunableBinding(p.tunable_name, p.interface_name, impl)
                for impl in p.impl_names
            ]
            for p in self.params
        ]
        return [
            TuningConfig(bindings=tuple(combo))
            for combo in itertools.product(*impl_lists)
        ]

    def size(self) -> int:
        """Total number of configs in the space."""
        if not self.params:
            return 0
        n = 1
        for p in self.params:
            n *= len(p.impl_names)
        return n

    def __str__(self) -> str:
        lines = [f"TuningSpace ({self.size()} configs):"]
        for p in self.params:
            lines.append(f"  {p}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TuningSpace(params={self.params!r})"
