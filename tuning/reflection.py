"""Reflection utilities for discovering tunable declarations in Slang modules."""

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
