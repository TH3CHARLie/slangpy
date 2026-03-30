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
            type_refl = decl.as_type()
            if type_refl and type_refl.find_user_attribute_by_name("Tunable"):
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


if __name__ == "__main__":
    import pathlib
    import sys

    if len(sys.argv) > 1:
        slang_path = pathlib.Path(sys.argv[1])
    else:
        slang_path = pathlib.Path(__file__).parent / "example.slang"

    slang_source = slang_path.read_text()

    device = spy.Device()
    module = device.load_module_from_source(slang_path.stem, slang_source)

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
