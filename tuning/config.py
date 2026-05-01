"""Core data types for the tuning system."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from slangpy import SlangModule


@dataclass(frozen=True)
class TunableBinding:
    """One struct tunable bound to one implementation."""

    tunable_name: str
    interface_name: str
    impl_name: str

    def __str__(self) -> str:
        return f"{self.tunable_name}={self.impl_name}"


@dataclass(frozen=True)
class IntTunableBinding:
    """One integer tunable bound to a specific value."""

    tunable_name: str
    value: int

    def __str__(self) -> str:
        return f"{self.tunable_name}={self.value}"


# A binding is either a struct or int tunable choice.
AnyBinding = Union[TunableBinding, IntTunableBinding]


@dataclass
class TunableParam:
    """One tunable axis: an extern struct, the interface it conforms to, and available impls."""

    tunable_name: str
    interface_name: str
    impl_names: list[str]

    def __str__(self) -> str:
        return f"{self.tunable_name}: {self.interface_name} -> [{', '.join(self.impl_names)}]"


@dataclass
class IntTunableParam:
    """One integer tunable axis: an extern static const int with discrete choices."""

    tunable_name: str
    choices: list[int]

    def __post_init__(self) -> None:
        self.choices = list(dict.fromkeys(self.choices))

    def __str__(self) -> str:
        return f"{self.tunable_name}: int -> [{', '.join(str(c) for c in self.choices)}]"


# A param is either a struct or int tunable axis.
AnyParam = Union[TunableParam, IntTunableParam]


@dataclass(frozen=True)
class TuningConfig:
    """A complete set of tunable bindings. One point in the tuning space.

    Immutable and hashable so it can be used as a dict key.
    """

    bindings: tuple[AnyBinding, ...]

    def struct_bindings(self) -> list[TunableBinding]:
        """Return only the struct tunable bindings."""
        return [b for b in self.bindings if isinstance(b, TunableBinding)]

    def int_bindings(self) -> list[IntTunableBinding]:
        """Return only the integer tunable bindings."""
        return [b for b in self.bindings if isinstance(b, IntTunableBinding)]

    def to_bindings_dict(self) -> dict[str, tuple[str, str]]:
        """Convert struct bindings to the format link_variant() expects."""
        return {
            b.tunable_name: (b.interface_name, b.impl_name)
            for b in self.bindings
            if isinstance(b, TunableBinding)
        }

    def to_link_bindings(self) -> list[tuple[str, str, str]]:
        """Convert struct bindings to the list of triples that FunctionNode.with_settings() expects."""
        return [
            (b.tunable_name, b.interface_name, b.impl_name)
            for b in self.bindings
            if isinstance(b, TunableBinding)
        ]

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

    def __init__(self, params: list[AnyParam]) -> None:
        self.params = params

    @staticmethod
    def discover(module: SlangModule) -> TuningSpace:
        """Build from a loaded module using Slang reflection.

        Discovers both [Tunable] extern struct (interface) tunables and
        [Tunable(choices...)] extern static const int tunables.

        :param module: A loaded Slang module.
        :return: A TuningSpace describing all tunable axes and their options.
        """
        from .reflection import discover_tuning_space, find_tunable_int_decls

        params: list[AnyParam] = []

        # Struct tunables (interface-based).
        raw = discover_tuning_space(module)
        for tunable_name, interfaces in raw.items():
            for interface_name, impl_names in interfaces.items():
                params.append(TunableParam(tunable_name, interface_name, impl_names))
                break  # one interface per tunable

        # Integer tunables.
        for info in find_tunable_int_decls(module):
            params.append(IntTunableParam(info.variable_name, info.choices))

        return TuningSpace(params)

    def all_configs(self) -> list[TuningConfig]:
        """Enumerate every combinatorial config (cartesian product)."""
        if not self.params:
            return []
        binding_lists: list[list[AnyBinding]] = []
        for p in self.params:
            if isinstance(p, TunableParam):
                binding_lists.append([
                    TunableBinding(p.tunable_name, p.interface_name, impl)
                    for impl in p.impl_names
                ])
            else:
                binding_lists.append([
                    IntTunableBinding(p.tunable_name, val)
                    for val in p.choices
                ])
        return [
            TuningConfig(bindings=tuple(combo))
            for combo in itertools.product(*binding_lists)
        ]

    def size(self) -> int:
        """Total number of configs in the space."""
        if not self.params:
            return 0
        n = 1
        for p in self.params:
            if isinstance(p, TunableParam):
                n *= len(p.impl_names)
            else:
                n *= len(p.choices)
        return n

    def __str__(self) -> str:
        lines = [f"TuningSpace ({self.size()} configs):"]
        for p in self.params:
            lines.append(f"  {p}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TuningSpace(params={self.params!r})"
