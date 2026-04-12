"""Thin reflection utilities for discovering tunable declarations in Slang modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import slangpy as spy

if TYPE_CHECKING:
    from slangpy import DeclReflection, SlangModule


def find_tunable_decls(module: SlangModule) -> list[DeclReflection]:
    """Return all [Tunable] extern struct declarations in a module.

    :param module: A loaded Slang module.
    :return: List of DeclReflection nodes for [Tunable] extern struct declarations.
    """
    root = module.module_decl
    results = []
    for decl in root.children_of_kind(spy.DeclReflection.Kind.struct):
        if decl.has_modifier(spy.ModifierID.extern):
            if decl.has_modifier(spy.ModifierID.tunable):
                results.append(decl)
    return results


def find_interfaces_for_decl(module: SlangModule, decl: DeclReflection) -> list[spy.TypeReflection]:
    """Return all interfaces that a declaration conforms to.

    :param module: A loaded Slang module.
    :param decl: A DeclReflection node to check.
    :return: List of TypeReflection nodes for interfaces this decl conforms to.
    """
    layout = module.layout
    decl_type = decl.as_type()
    if decl_type is None:
        return []
    root = module.module_decl
    results = []
    for i in range(len(root)):
        child = root[i]
        itype = child.as_type()
        if itype and itype.kind == spy.TypeReflection.Kind.interface:
            if layout.is_sub_type(decl_type, itype):
                results.append(itype)
    return results


def find_conforming_types(module: SlangModule, interface_name: str) -> list[DeclReflection]:
    """Return all non-extern struct decls in module that conform to the named interface.

    :param module: A loaded Slang module.
    :param interface_name: Name of the Slang interface to check conformance against.
    :return: List of DeclReflection nodes for conforming struct declarations.
    """
    layout = module.layout
    interface_type = layout.find_type_by_name(interface_name)
    if interface_type is None:
        return []
    root = module.module_decl
    results = []
    for decl in root.children_of_kind(spy.DeclReflection.Kind.struct):
        if decl.has_modifier(spy.ModifierID.extern):
            continue  # skip the placeholder itself
        type_refl = decl.as_type()
        if type_refl and layout.is_sub_type(type_refl, interface_type):
            results.append(decl)
    return results


def discover_tuning_space(module: SlangModule) -> dict[str, dict[str, list[str]]]:
    """Analyze a module and return its full tuning space.

    For each [Tunable] extern struct, discovers the interface(s) it conforms to
    and all concrete implementations available for that interface.

    :param module: A loaded Slang module.
    :return: Dict mapping tunable name -> {interface name -> [impl names]}.
    """
    result: dict[str, dict[str, list[str]]] = {}
    for decl in find_tunable_decls(module):
        interfaces: dict[str, list[str]] = {}
        for itype in find_interfaces_for_decl(module, decl):
            conformers = find_conforming_types(module, itype.name)
            interfaces[itype.name] = [c.name for c in conformers]
        result[decl.name] = interfaces
    return result


def link_variant(
    device: spy.Device,
    module: spy.SlangModule,
    bindings: dict[str, tuple[str, str]],
) -> spy.Module:
    """Create a Module variant by binding one or more tunables to implementations.

    Returns a SlangPy Module with the export linkage applied, so functions
    can be called directly via the functional API (e.g. ``variant.add(2.0, 3)``).

    :param device: The GPU device.
    :param module: The loaded Slang module containing the tunables and implementations.
    :param bindings: Mapping of tunable_name -> (interface_name, impl_name).
    :return: A SlangPy Module with the specified implementations linked in.
    """
    module_name = module.name
    lines = [f'import "{module_name}";']
    name_parts = [module_name]
    for tunable_name, (interface_name, impl_name) in bindings.items():
        lines.append(f"export struct {tunable_name} : {interface_name} = {impl_name};")
        name_parts.append(f"{tunable_name}_{impl_name}")
    additional_source = "\n".join(lines)
    export_module = device.load_module_from_source(
        "_".join(name_parts), additional_source
    )
    return spy.Module(module, link=[export_module])


def generate_all_variants(
    device: spy.Device,
    module: spy.SlangModule,
) -> dict[str, spy.Module]:
    """Generate Module variants for all combinatorial tunable implementations.

    Discovers the tuning space and creates a SlangPy Module for each
    combination of implementations across all tunables.

    :param device: The GPU device.
    :param module: The loaded Slang module.
    :return: Dict mapping variant key (e.g. "TunableActivation=ReLU,TunableReduce=SumReduce")
             to Module.
    """
    space = discover_tuning_space(module)

    # Build list of (tunable_name, interface_name, [impl_names]) for each tunable.
    # Each tunable has one interface (first one found).
    axes: list[tuple[str, str, list[str]]] = []
    for tunable_name, interfaces in space.items():
        for interface_name, impl_names in interfaces.items():
            axes.append((tunable_name, interface_name, impl_names))
            break  # use first interface per tunable

    # Generate combinatorial product of all axes
    import itertools

    if not axes:
        return {}

    impl_lists = [axis[2] for axis in axes]
    variants: dict[str, spy.Module] = {}
    for combo in itertools.product(*impl_lists):
        bindings: dict[str, tuple[str, str]] = {}
        key_parts: list[str] = []
        for (tunable_name, interface_name, _), impl_name in zip(axes, combo):
            bindings[tunable_name] = (interface_name, impl_name)
            key_parts.append(f"{tunable_name}={impl_name}")
        key = ",".join(key_parts)
        variants[key] = link_variant(device, module, bindings)
    return variants


if __name__ == "__main__":
    import pathlib
    import sys

    if len(sys.argv) > 1:
        slang_path = pathlib.Path(sys.argv[1])
    else:
        slang_path = pathlib.Path(__file__).parent / "ml_pipeline.slang"

    slang_source = slang_path.read_text()

    slangpy_slang_path = pathlib.Path(__file__).parent.parent / "slangpy" / "slang"
    device = spy.Device(
        compiler_options=spy.SlangCompilerOptions(
            {"include_paths": [slang_path.parent, slangpy_slang_path]}
        ),
    )
    module = device.load_module_from_source(slang_path.stem, slang_source)

    # Discover tuning space
    space = discover_tuning_space(module)

    if not space:
        print("No [Tunable] declarations found.")
    else:
        for tunable, interfaces in space.items():
            print(f"[Tunable] extern struct {tunable}")
            for iface, impls in interfaces.items():
                print(f"  implements: {iface}")
                for impl in impls:
                    print(f"    - {impl}")

    # Generate all combinatorial variants and test them
    print("\n--- Generating variants ---")
    variants = generate_all_variants(device, module)
    print(f"  {len(variants)} variants generated")

    # Test inputs: mix of positive and negative values
    test_values = (-1.5, 0.5, 2.0, -0.3)

    import time

    print(f"\n--- transform_reduce({test_values}) ---")
    for key, variant in variants.items():
        # Warm up (first call compiles the kernel)
        variant.transform_reduce(*test_values)
        t0 = time.perf_counter()
        result = variant.transform_reduce(*test_values)
        dt = time.perf_counter() - t0
        print(f"  {key}: {result}  ({dt*1e6:.1f} µs)")

    print(f"\n--- activate_element per value ---")
    # Show activation behavior for a single variant per activation type
    seen_activations: set[str] = set()
    for key, variant in variants.items():
        act_part = key.split(",")[0]  # e.g. "TunableActivation=ReLU"
        if act_part in seen_activations:
            continue
        seen_activations.add(act_part)
        # Warm up
        variant.activate_element(test_values[0])
        t0 = time.perf_counter()
        results = [variant.activate_element(v) for v in test_values]
        dt = time.perf_counter() - t0
        print(f"  {act_part}: {test_values} -> {results}  ({dt*1e6:.1f} µs)")
