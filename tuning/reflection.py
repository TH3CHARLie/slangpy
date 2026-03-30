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


if __name__ == "__main__":
    import pathlib

    slang_source = (pathlib.Path(__file__).parent / "example.slang").read_text()

    device = spy.Device()
    module = device.load_module_from_source("example", slang_source)

    tunables = find_tunable_decls(module)
    print(f"Tunable decls ({len(tunables)}):")
    for d in tunables:
        print(f"  {d.name}")
    assert len(tunables) == 1 and tunables[0].name == "TunableFoo", f"unexpected tunables: {tunables}"

    conformers = find_conforming_types(module, "IFoo")
    names = {d.name for d in conformers}
    print(f"Conforming types: {names}")
    assert names == {"FooSlow", "FooFast"}, f"unexpected conformers: {names}"

    print("All assertions passed.")
