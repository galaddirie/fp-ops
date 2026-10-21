# Migrating to FP-Ops 0.3

FP-Ops 0.3 is a direct, breaking replacement. It has no compatibility
namespace or deprecation adapter.

## Execution and composition

Every operation is unary and executes explicitly:

```python
result = await pipeline.run(value, environment=environment)
```

Use `.then(next_operation)` for sequential composition and `.map(callback)` for
a raw callback. Use `.flat_map(callback)` only when the callback deliberately
returns an `Operation[Unit, T]`; that returned operation runs with `UNIT`.
Returning an `Operation` from `.map` leaves it as ordinary data.

| Removed | Replacement |
|---|---|
| `.execute(value)` | `.run(value)` |
| `await bound_operation` | `.run(UNIT)` |
| calling an operation to bind arguments | an operation template or Python factory |
| `a & b` | `fanout_parallel(a, b)` |
| `.bind(callback)` | `.flat_map(callback)` |
| `.apply_cont(callback)` | `.map`, `.flat_map`, or recovery |
| `Operation.sequence` | `pipe_all` |
| `sequences.map(Operation)` | `traverse` |
| `sequences.filter(Operation)` | `filter_operation` |
| `sequences.zip` | `fanout_each` |

`>>` remains a thin alias for `.then`. `|` remains a thin alias for
`or_else`. Graph ports, runtime partial binding, and the old placeholder
renderer are gone.

Binary `&` was removed in 0.3 and restored in 0.4 as a fail-fast fanout. Unlike
0.2, `a & b & c` returns a flat `(a, b, c)` tuple rather than `((a, b), c)`.

Construction-only templates retain the useful one-hole syntax without making
`Operation` callable:

```python
from fp_ops import _, operation_template


@operation_template
def add(left: int, right: int) -> int:
    return left + right


add_ten = add(_, 10)  # Operation[int, int]
pipeline = double >> add_ten
```

A template call must contain exactly one direct placeholder among its first six
positional arguments. The call immediately returns a unary `Operation`;
configuration is never appended at execution time. Nested placeholders and
multiple independent holes should be rewritten as an explicit input record or
ordinary factory. This is enforced at construction even when an `Any`
annotation prevents a type checker from proving the one-hole rule. Variadic
configuration is accepted, while keyword-argument holes remain a runtime-only
feature because mypy and Pyright model fixed positional holes. Checker support
for forwarding configured variadic arguments is limited.

Use `threaded_operation_template` when the configured callable is blocking.
For environment templates, place the injected capability in the final
positional parameter; keyword-only configuration may follow it.

## Errors

Once `.run` starts, ordinary `Exception` values become
`Result[..., Exception]` errors. Cancellation and other `BaseException`
subclasses still propagate.

`operation` treats every return value—including `Result`, `None`, and another
`Operation`—as data. Use `result_operation` only for a callable that declares
one `Result` layer.

Default collection and object helpers fail fast. Choose the name that matches
the required policy:

- `fanout_all_settled`, `traverse_all_settled`, and `build_all_settled` retain
  a `Result` for every branch, item, or field.
- `traverse_lossy`, `filter_best_effort`, and `build_lossy` explicitly drop
  failures.

## Environment

Replace implicit context injection and output-sensitive context replacement
with typed capability keys:

```python
SESSION = EnvKey("session", BrowserSession)

@environment_operation_template(SESSION)
async def read(
    selector: str,
    element: Element,
    session: BrowserSession,
    *,
    timeout: float = 5.0,
) -> str:
    ...

environment = Environment().with_value(SESSION, session)
result = await read(".title", _, timeout=2.0).run(
    element,
    environment=environment,
)
```

The environment is read-only and never replaced by a successful stage value.
Changing focus belongs in normal pipeline input/output.

## Objects and collections

- `objects.get` becomes `get_path`; a present `None` no longer triggers the
  default.
- lossy `build` becomes fail-fast `build`; field failures are
  `BuildFieldError(field, cause)`.
- `objects.merge` becomes `merge_shallow` or `assign`.
- `objects.update` becomes `assign_fields`.
- pure callbacks use `map_each`, `filter_each`, and `fold`.
- nested operations use `traverse`, `filter_operation`, and `fold_operation`.
