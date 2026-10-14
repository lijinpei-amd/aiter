# Coding style

These rules apply to the generator code: `gluon.constexpr_function`s, aggregate
constructors and methods, and host Python, the code that decides what the kernel
emits. The generated code, the `@gluon.jit` functions, takes whatever shape reads
best for the kernel. `_frozen.py` is a snapshot; it keeps its own names.

## Fields, not accessors

Read a stored value of an aggregate (`Schedule`, `KernelLayout`, the configs) as a
public field. A method earns its place only when it computes something: it indexes
by an argument, converts a component, or derives a value.

```python
#: Pipeline depth: the largest live span of a present component, in steps.
depth: gl.constexpr                  # set in __init__: self.depth = gl.constexpr(depth)

depth: gl.constexpr = sched.depth    # @gluon.jit code reads the field as is
depth = _v(sched.depth)              # constexpr functions and host code unwrap it
```

- Document a public field with a `#:` comment above its annotation.
- Keep a field private (`_name`) when only methods read it.
- A field is a `gl.constexpr`, so host code calls `unwrap` (`_v`) before arithmetic,
  `is None`, hashing or formatting. A test that runs a jit body as `.fn` and keys on
  the values it computes passes the aggregate through `_HostView`
  (`test_moe_gemm_gluon_scale_layout.py`).

## One method for both operands

A method that applies to operand A and B takes the operand as an argument:
`foobar(idx)`, written once. When it also differs between payload and scale, it
takes `is_scale` as well: `foobar(idx, is_scale)`.

```python
func_cfg.has_scale(idx)                   # one method for A and B
layout.fragment_shape_slot(idx, is_scale)  # one method for A/B and payload/scale
```

Methods that already take a component `(operand, is_scale)`, such as
`component_via_lds(component)`, satisfy the same rule.
