# Autotuning Notes

## Table of Contents

- [Reflection API](#reflection-api)
- [Roadmap](#roadmap)

---

## Reflection API

### Two Reflection Types

SlangPy exposes two parallel reflection hierarchies for the same Slang declarations:

| | `DeclReflection` | `TypeReflection` |
|---|---|---|
| Represents | A syntactic declaration (what was written) | A semantic type (what it means in the type system) |
| Accessed via | `module.module_decl` and its children | `decl.as_type()` or `layout.find_type_by_name()` |
| Has | `name`, `kind`, `has_modifier()`, `children` | `name`, `kind`, `find_user_attribute_by_name()` |
| Used with | Walking the declaration tree | `layout.is_sub_type()`, attribute queries |

`as_type()` bridges the two: it converts a `DeclReflection` into the corresponding `TypeReflection`.

### Kind Enums

#### `DeclReflection.Kind`

| Value | Meaning |
|---|---|
| `struct` | Struct/class declaration |
| `func` | Function declaration |
| `variable` | Variable declaration |
| `generic` | Generic/template declaration |
| `module` | Module root |
| `unsupported` | Catch-all for anything not categorized (including interfaces) |

#### `TypeReflection.Kind`

| Value | Meaning |
|---|---|
| `struct` | Concrete struct type |
| `interface` | Interface type |
| `scalar`, `vector`, `matrix`, `array` | Numeric/aggregate types |
| `resource`, `texture_buffer`, `constant_buffer`, `shader_storage_buffer`, `sampler_state` | GPU resource types |
| `parameter_block`, `pointer`, `feedback`, `output_stream` | Other Slang type categories |
| `specialized` | A generic specialized with concrete type args |
| `generic_type_parameter` | A `<T>` placeholder |
| `none` | No type / unknown |

### The Interface Gap

`DeclReflection.Kind` has no `interface` variant. An `interface IFoo` declaration shows up as `Kind.unsupported` at the decl level. However, `as_type().kind` correctly returns `TypeReflection.Kind.interface`.

This means you **cannot** do:
```python
root.children_of_kind(DeclReflection.Kind.interface)  # doesn't exist
```

Instead, to find interfaces you must iterate all children and check on the type side:
```python
for i in range(len(root)):
    child = root[i]
    itype = child.as_type()
    if itype and itype.kind == TypeReflection.Kind.interface:
        # found an interface
```

### Practical Consequence

Any operation that needs to distinguish structs from interfaces must go through `TypeReflection`. The decl tree alone treats interfaces as `unsupported`, which is unusable for filtering.

This is **not** a SlangPy binding gap — it's an upstream Slang limitation. The Slang reflection API (`SlangDeclKind` in `slang.h`) has no `SLANG_DECL_KIND_INTERFACE` entry. The compiler's `spReflectionDecl_getKind()` function doesn't handle `InterfaceDecl`, so it falls through to `SLANG_DECL_KIND_UNSUPPORTED_FOR_REFLECTION`. The `InterfaceDecl` AST node exists internally, but the public reflection API doesn't surface it.

Slang also defines `SLANG_DECL_KIND_NAMESPACE` and `SLANG_DECL_KIND_ENUM` which SlangPy doesn't wrap yet, but neither of those help with interface discovery.

---

## Roadmap

### Completed

- **Reflection utilities** (`tuning_playground/reflection.py`, promoted to `tuning/reflection.py`)
  - `find_tunable_decls` / `find_interfaces_for_decl` / `find_conforming_types` / `discover_tuning_space`
- **C++ binding** — `DeclReflection.has_modifier()` bound to Python
- **Reference Slang files** — `example.slang` (TunableFoo) and `ml_pipeline.slang` (TunableActivation × TunableReduce)
- **Link-time specialization** — `link_variant()` builds an export module with `export struct X : IFoo = Impl;` and returns a `spy.Module` with `link=[export]`, enabling functional-API calls on the variant
- **Multi-tunable combinatorial enumeration** — `TuningSpace.all_configs()` via `itertools.product`
- **Tuner + Model API** (`tuning/` package, separate from playground)
  - `config.py` — `TunableBinding`, `TunableParam`, `TuningConfig`, `TuningResult`, `TuningSpace`
  - `model.py` — `TuningModel` ABC + `ExhaustiveSearch`
  - `tuner.py` — user-facing orchestrator (`setup`/`propose`/`report`/`best`/`is_complete`/`get_variant`) with variant cache
  - `demo.py` — working end-to-end demo, user owns the loop (PyTorch-style)
- **Wall-clock timing** — per-variant timing in the demo loop

### Next Steps

#### 1. Function-level tuning
`func.with_settings()` on a `Function` so specialization can be scoped to a single function rather than the whole module. Requires changes to `slangpy/core/function.py`.

#### 2. GPU-side timing
Replace wall-clock timing with GPU timestamp queries (e.g. `spy.Timer`) for accurate kernel measurement, including warmup handling and multi-iteration averaging.

#### 3. Additional search strategies
Validate Model extensibility by adding `RandomSearch`, then Bayesian optimization / bandits. The `TuningModel` ABC is already designed for this.

#### 4. Caching / persistence
Store tuning results keyed by (module source hash, device info) across runs. Serve the best-known variant on subsequent loads.

#### 5. Apply to a real sample
Hook the Tuner into a slangpy-samples example (e.g. sdf-match) with activation + optimizer tunables, using time-to-target-loss as the metric.
