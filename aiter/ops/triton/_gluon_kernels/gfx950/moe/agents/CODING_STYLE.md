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
depth = sched.depth.value            # constexpr functions and host code take .value
```

- Document a public field with a `#:` comment above its annotation.
- Keep a field private (`_name`) when only methods read it.
- A field is a `gl.constexpr`, so constexpr functions and host code read it with
  `.value` before arithmetic, `is None`, hashing or formatting (see
  [Unwrap by kind](#unwrap-by-kind)). A test that runs a jit body as `.fn` and keys
  on the values it computes passes the aggregate through `_HostView`
  (`test_moe_gemm_gluon_scale_layout.py`).

## Unwrap by kind

Know whether each value is a `gl.constexpr` or raw, and read it by its kind:

| Kind | Where it comes from | How to read it |
|---|---|---|
| Wrapped | an aggregate field declared `gl.constexpr`; a module global `X: gl.constexpr = gl.constexpr(...)`; `tensor.shape[i]` | `.value` |
| Raw | a `constexpr_function` parameter, including the elements of a tuple, list or NamedTuple parameter; the result of any `constexpr_function` call | as is |
| Unknown | a value nested in a dict or aggregate argument; code that may receive a duck-typed test fake | `maybe_unwrap` (`_v`) |

```python
def num_k_tiles(self, K):          # a constexpr_function method
    BK = self._BLOCK_K.value       # wrapped: a field
    return K // BK                 # raw: a parameter, and a raw result
```

- **Raw** because `ConstexprFunction.__call__` unwraps every argument, through
  lists and tuples, before the body runs, and hands the raw result back to host
  code, to another constexpr function and to an aggregate constructor. Only a
  `@gluon.jit` caller gets it re-wrapped.
- **Accessors** return a raw value: `return self._x.value[idx]`, never a bare
  `self._x`.
- **`.value` is the strict unwrap.** On a raw value it raises `AttributeError` at
  the first compile. An `IntEnum` has a `.value` too, so store enums as
  `int(enum)`: a raw enum would pass silently.
- **A leftover `gl.constexpr` fails late.** Arithmetic on it still works and
  returns another `constexpr`, but `is None` is false, `len()` fails, `isinstance`
  misses, and its hash is `hash((value, type))`, so dict and set lookups miss.
- **`maybe_unwrap` marks a real unknown.** `Schedule.__init__` keeps it on config
  and layout fields because the schedule tests pass a duck-typed fake with raw
  fields.

## One method for both operands

A method that applies to operand A and B takes the operand as an argument:
`foobar(idx)`, written once. When it also differs between payload and scale, it
takes `is_scale` as well: `foobar(idx, is_scale)`.

```python
func_cfg.has_scale(idx)                   # one method for A and B
layout.fragment_shape_slot(idx, is_scale)  # one method for A/B and payload/scale
```

Methods that already take a `Component`, such as
`component_via_lds(component)`, satisfy the same rule.
