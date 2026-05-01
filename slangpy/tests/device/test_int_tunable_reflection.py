# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for [Tunable(choices...)] integer tunable reflection.

Verifies that:
1. Attribute.argument_value_int() reads integer attribute arguments
2. VariableReflection exposes user attributes
3. The tuning reflection helper discovers integer tunables from a module
"""

import pathlib
import sys

import pytest

import slangpy as spy
from slangpy import Module
from slangpy.testing import helpers
from slangpy.testing.helpers import test_id  # type: ignore (pytest fixture)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tuning.reflection import IntTunableInfo, find_tunable_int_decls  # noqa: E402
from tuning.reflection import link_variant  # noqa: E402
from tuning.config import (  # noqa: E402
    IntTunableBinding,
    IntTunableParam,
    TunableBinding,
    TunableParam,
    TuningConfig,
    TuningSpace,
)


def create_session(device: spy.Device) -> spy.SlangSession:
    return device.create_slang_session(
        compiler_options={
            "include_paths": device.slang_session.desc.compiler_options.include_paths,
            "debug_info": spy.SlangDebugInfoLevel.standard,
        }
    )


TUNABLE_INT_SOURCE = """
[Tunable(1, 2, 4, 8)]
public extern static const int TunableWidth = 2;

[Tunable(16, 32, 16)]
public extern static const int TunableDepth = 16;

[shader("compute")]
[numthreads(1,1,1)]
void main() {}
"""


TUNABLE_DISCOVERY_SOURCE = """
public interface IOp
{
    int apply(int x);
}

public struct AddOne : IOp
{
    public int apply(int x) { return x + 1; }
}

public struct DoubleIt : IOp
{
    public int apply(int x) { return x * 2; }
}

[Tunable]
public extern struct TunableOp : IOp = AddOne;

[Tunable(2, 4)]
public extern static const int TunableScale = 2;

[Tunable(8, 16)]
public static const int NonExternScale = 8;

[Tunable(1, 2)]
public extern static const uint WrongTypeScale = 1;

[shader("compute")]
[numthreads(1,1,1)]
void main() {}
"""


TUNABLE_RUNTIME_SOURCE = """
import slangpy;

public interface IOp
{
    int apply(int x);
}

public struct AddOne : IOp
{
    public int apply(int x) { return x + 1; }
}

public struct DoubleIt : IOp
{
    public int apply(int x) { return x * 2; }
}

[Tunable]
public extern struct TunableOp : IOp = AddOne;

[Tunable(2, 4)]
public extern static const int TunableScale = 2;

public int eval(int x)
{
    return (TunableOp()).apply(x) * TunableScale;
}
"""


def test_tuning_space_enumerates_mixed_struct_and_int_axes():
    """TuningSpace should include integer axes in size() and all_configs()."""
    space = TuningSpace(
        [
            TunableParam("Op", "IOp", ["AddOne", "DoubleIt"]),
            IntTunableParam("Scale", [2, 4, 4, 8]),
        ]
    )

    configs = space.all_configs()

    assert space.size() == 6
    assert [str(cfg) for cfg in configs] == [
        "Op=AddOne,Scale=2",
        "Op=AddOne,Scale=4",
        "Op=AddOne,Scale=8",
        "Op=DoubleIt,Scale=2",
        "Op=DoubleIt,Scale=4",
        "Op=DoubleIt,Scale=8",
    ]
    assert configs[-1].to_link_bindings() == [("Op", "IOp", "DoubleIt")]
    assert configs[-1].int_bindings() == [IntTunableBinding("Scale", 8)]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_attribute_value_int(test_id: str, device_type: spy.DeviceType):
    """Attribute.argument_value_int() returns the correct integer values."""
    device = helpers.get_device(type=device_type)
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"tunable_int_attr_{test_id}",
        source=TUNABLE_INT_SOURCE,
    )

    # Find TunableWidth variable via decl reflection.
    root = module.module_decl
    found = False
    for decl in root.children_of_kind(spy.DeclReflection.Kind.variable):
        if decl.name == "TunableWidth":
            var = decl.as_variable()
            assert var is not None
            assert var.user_attribute_count >= 1
            attr = var.get_user_attribute_by_index(0)
            assert attr.name == "Tunable"
            assert attr.argument_count == 4
            assert attr.argument_value_int(0) == 1
            assert attr.argument_value_int(1) == 2
            assert attr.argument_value_int(2) == 4
            assert attr.argument_value_int(3) == 8
            found = True
            break
    assert found, "TunableWidth variable decl not found"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_find_tunable_int_decls(test_id: str, device_type: spy.DeviceType):
    """find_tunable_int_decls() discovers all integer tunables and their choices."""
    device = helpers.get_device(type=device_type)
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"tunable_int_discover_{test_id}",
        source=TUNABLE_INT_SOURCE,
    )

    int_tunables = find_tunable_int_decls(module)
    by_name = {t.variable_name: t for t in int_tunables}

    assert "TunableWidth" in by_name
    assert by_name["TunableWidth"].choices == [1, 2, 4, 8]

    assert "TunableDepth" in by_name
    assert by_name["TunableDepth"].choices == [16, 32]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_find_tunable_int_decls_ignores_unsupported_shapes(
    test_id: str, device_type: spy.DeviceType
):
    """Only extern static const int tunables should be discovered."""
    device = helpers.get_device(type=device_type)
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"tunable_int_supported_shapes_{test_id}",
        source=TUNABLE_DISCOVERY_SOURCE,
    )

    int_tunables = find_tunable_int_decls(module)

    assert int_tunables == [IntTunableInfo(variable_name="TunableScale", choices=[2, 4])]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tuning_space_discover_mixed_module(test_id: str, device_type: spy.DeviceType):
    """TuningSpace.discover() should include both struct and integer tunables."""
    device = helpers.get_device(type=device_type)
    session = create_session(device)
    module = session.load_module_from_source(
        module_name=f"tuning_space_discover_mixed_{test_id}",
        source=TUNABLE_DISCOVERY_SOURCE,
    )

    space = TuningSpace.discover(module)

    assert space.size() == 4
    assert len(space.params) == 2
    assert isinstance(space.params[0], TunableParam)
    assert space.params[0].tunable_name == "TunableOp"
    assert space.params[0].impl_names == ["AddOne", "DoubleIt"]
    assert isinstance(space.params[1], IntTunableParam)
    assert space.params[1].tunable_name == "TunableScale"
    assert space.params[1].choices == [2, 4]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_link_variant_applies_integer_tunable(test_id: str, device_type: spy.DeviceType):
    """link_variant() should specialize integer tunables, not just discover them."""
    device = helpers.get_device(type=device_type)
    raw_module = device.load_module_from_source(
        module_name=f"tunable_int_runtime_{test_id}",
        source=TUNABLE_RUNTIME_SOURCE,
    )

    default_module = Module(raw_module)
    assert default_module.eval(3) == 8

    variant = link_variant(
        device,
        raw_module,
        bindings={},
        int_bindings={"TunableScale": 4},
    )
    assert variant.eval(3) == 16


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_with_settings_applies_integer_tunable(test_id: str, device_type: spy.DeviceType):
    """FunctionNode.with_settings() should specialize integer tunables end to end."""
    device = helpers.get_device(type=device_type)
    raw_module = device.load_module_from_source(
        module_name=f"tunable_int_with_settings_{test_id}",
        source=TUNABLE_RUNTIME_SOURCE,
    )

    module = Module(raw_module)
    func = module.eval

    default_result = func(3)
    specialized = func.with_settings(
        TuningConfig(bindings=(IntTunableBinding("TunableScale", 4),))
    )

    assert default_result == 8
    assert specialized(3) == 16


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_with_settings_applies_mixed_struct_and_int_tunables(
    test_id: str, device_type: spy.DeviceType
):
    """Mixed configs should specialize both interface and integer tunables together."""
    device = helpers.get_device(type=device_type)
    raw_module = device.load_module_from_source(
        module_name=f"tunable_mixed_with_settings_{test_id}",
        source=TUNABLE_RUNTIME_SOURCE,
    )

    module = Module(raw_module)
    func = module.eval

    cfg = TuningConfig(
        bindings=(
            TunableBinding("TunableOp", "IOp", "DoubleIt"),
            IntTunableBinding("TunableScale", 4),
        )
    )

    assert func.with_settings(cfg)(3) == 24
