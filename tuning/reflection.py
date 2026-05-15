"""Reflection utilities for discovering tunable declarations in Slang modules."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

import slangpy as spy

if TYPE_CHECKING:
    from slangpy import DeclReflection, SlangModule


_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*|\"[^\"]+\")\s*;", re.MULTILINE)


def _strip_line_comments(source: str) -> str:
    """Remove // comments for simple import discovery."""
    return "\n".join(line.split("//", 1)[0] for line in source.splitlines())


def _module_visit_key(module: SlangModule) -> str:
    path = getattr(module, "path", None)
    if path:
        return str(path)
    return module.name


def _source_import_names(module: SlangModule) -> list[str]:
    """Return direct import names written in a module source file."""
    path = getattr(module, "path", None)
    if not path:
        return []
    try:
        source = path.read_text()
    except OSError:
        return []
    names = []
    for match in _IMPORT_RE.finditer(_strip_line_comments(source)):
        raw = match.group(1)
        name = raw[1:-1] if raw.startswith('"') else raw
        if name == "slangpy":
            continue
        names.append(name)
    return names


def iter_module_and_imports(module: SlangModule) -> list[SlangModule]:
    """Return a module and Slang modules directly or transitively imported by it.

    Slang's module reflection exposes declarations from the loaded module, but
    not all imported module declarations. Loading imported modules through the
    same session lets tuning discovery see tunables declared in reusable modules.
    """
    results: list[SlangModule] = []
    visited: set[str] = set()

    def visit(current: SlangModule) -> None:
        key = _module_visit_key(current)
        if key in visited:
            return
        visited.add(key)
        results.append(current)

        session = getattr(current, "session", None)
        if session is None:
            return
        for import_name in _source_import_names(current):
            try:
                imported = session.load_module(import_name)
            except Exception:
                continue
            visit(imported)

    visit(module)
    return results


def _find_tunable_decls_in_module(module: SlangModule) -> list[DeclReflection]:
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


def find_tunable_decls(module: SlangModule) -> list[DeclReflection]:
    """Return all [Tunable] extern struct declarations in a module and imports."""
    results = []
    for source_module in iter_module_and_imports(module):
        results.extend(_find_tunable_decls_in_module(source_module))
    return results


@dataclass
class IntTunableInfo:
    """Describes a [Tunable(choices...)] integer variable discovered via reflection."""

    variable_name: str
    choices: list[int]


def _find_tunable_int_decls_in_module(module: SlangModule) -> list[IntTunableInfo]:
    """Return all [Tunable(choices...)] integer variable declarations in a module.

    Finds ``[Tunable(choices...)] extern static const int`` variable declarations,
    then reads the integer argument values from the attribute to determine the
    choice space.

    :param module: A loaded Slang module.
    :return: List of IntTunableInfo with variable name and integer choices.
    """
    root = module.module_decl
    results = []
    for decl in root.children_of_kind(spy.DeclReflection.Kind.variable):
        if not decl.has_modifier(spy.ModifierID.tunable):
            continue
        var = decl.as_variable()
        if var is None:
            continue
        if not (
            decl.has_modifier(spy.ModifierID.extern)
            and decl.has_modifier(spy.ModifierID.static)
            and decl.has_modifier(spy.ModifierID.const)
        ):
            continue
        if (
            var.type.kind != spy.TypeReflection.Kind.scalar
            or var.type.scalar_type != spy.TypeReflection.ScalarType.int32
        ):
            continue
        # Find the Tunable attribute and read its int arguments.
        attr = None
        for i in range(var.user_attribute_count):
            a = var.get_user_attribute_by_index(i)
            if a.name == "Tunable":
                attr = a
                break
        if attr is None or attr.argument_count == 0:
            continue
        choices = list(
            dict.fromkeys(attr.argument_value_int(i) for i in range(attr.argument_count))
        )
        results.append(IntTunableInfo(variable_name=var.name, choices=choices))
    return results


def find_tunable_int_decls(module: SlangModule) -> list[IntTunableInfo]:
    """Return all [Tunable(choices...)] integer variable declarations in a module and imports."""
    results = []
    for source_module in iter_module_and_imports(module):
        results.extend(_find_tunable_int_decls_in_module(source_module))
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
    for source_module in iter_module_and_imports(module):
        for decl in _find_tunable_decls_in_module(source_module):
            interfaces: dict[str, list[str]] = {}
            for itype in find_interfaces_for_decl(source_module, decl):
                conformers = find_conforming_types(source_module, itype.name)
                interfaces[itype.name] = [c.name for c in conformers]
            result[decl.name] = interfaces
    return result


def link_variant(
    device: spy.Device,
    module: spy.SlangModule,
    bindings: dict[str, tuple[str, str]],
    int_bindings: dict[str, int] | None = None,
) -> spy.Module:
    """Create a Module variant by binding tunables to concrete values.

    Generates a Slang source module with ``export`` declarations for both
    struct tunables (``export struct X : I = Impl;``) and integer tunables
    (``export static const int X = N;``), then links it against the base module.

    :param device: The GPU device.
    :param module: The loaded Slang module containing the tunables.
    :param bindings: Mapping of tunable_name -> (interface_name, impl_name) for struct tunables.
    :param int_bindings: Mapping of tunable_name -> int value for integer tunables.
    :return: A SlangPy Module with the specified bindings linked in.
    """
    module_name = module.name
    lines = [f'import "{module_name}";']
    for import_name in _source_import_names(module):
        lines.append(f'import "{import_name}";')
    name_parts = [module_name]
    for tunable_name, (interface_name, impl_name) in bindings.items():
        lines.append(f"export struct {tunable_name} : {interface_name} = {impl_name};")
        name_parts.append(f"{tunable_name}_{impl_name}")
    if int_bindings:
        for tunable_name, value in int_bindings.items():
            lines.append(f"export static const int {tunable_name} = {value};")
            name_parts.append(f"{tunable_name}_{value}")
    additional_source = "\n".join(lines)
    export_module = module.session.load_module_from_source(
        "_".join(name_parts), additional_source
    )
    return spy.Module(module, link=[export_module])
