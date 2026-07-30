# FP-Ops

FP-Ops is a typed, immutable, asynchronous pipeline library centered on one
unary abstraction:

```text
Operation[A, B] = A -> Environment -> Awaitable[Result[B, Exception]]
```

An operation receives exactly one pipeline value. Successful values move from
left to right; a read-only `Environment` supplies typed capabilities. `None`,
`Result`, environment objects, and returned `Operation` objects remain ordinary
data unless an explicitly named combinator says otherwise.

FP-Ops requires Python 3.10 or newer.

## Install

```bash
pip install fp-ops
```

## Define, compose, and run

```python
import asyncio

from fp_ops import Ok, operation


@operation
def parse(value: str) -> int:
    return int(value)


@operation
async def double(value: int) -> int:
    return value * 2


pipeline = parse.then(double)  # `parse >> double` is the syntax alias


async def main() -> None:
    assert await pipeline.run("21") == Ok(42)


asyncio.run(main())
```

`.then` accepts only an `Operation`, making the stage relationship visible to
mypy and Pyright. Use `.map(callback)` for a raw successful-value transform.
Use `.flat_map(callback)` only when the callback returns an
`Operation[Unit, T]` that should execute with `UNIT`.

There is no callable or awaitable `Operation`, argument binding, placeholder
DSL, graph interface, or `.execute` alias.

## Raw values and declared results

`operation` lifts a callable that returns a raw value. It never guesses from
the value's runtime type:

```python
from expression import Ok
from fp_ops import UNIT, operation, result_operation

raw = operation(lambda _: Ok(3))
flattened = result_operation(lambda _: Ok(3))

# await raw.run(UNIT)       == Ok(Ok(3))
# await flattened.run(UNIT) == Ok(3)
```

Synchronous functions, async functions, and synchronous functions that return
awaitables share one normalizer. Synchronous work runs inline. Use
`threaded_operation` to opt blocking work into `asyncio.to_thread`.

After `.run` starts, ordinary `Exception` values become `Error(exception)`.
`asyncio.CancelledError`, `KeyboardInterrupt`, `SystemExit`, and other
`BaseException` subclasses propagate.

## Typed environments

```python
from dataclasses import dataclass

from fp_ops import EnvKey, Environment, environment_operation


@dataclass
class Rates:
    tax: float


RATES = EnvKey("rates", Rates)


@environment_operation(RATES)
def add_tax(subtotal: float, rates: Rates) -> float:
    return subtotal * (1 + rates.tax)


environment = Environment().with_value(RATES, Rates(tax=0.13))
# result = await add_tax.run(10.0, environment=environment)
```

Keys use object identity, not their names. `with_value` returns a new
environment and validates the capability's runtime type. Missing requirements
become an `EnvironmentError` inside `.run`. The same environment is used for
every stage and can never be replaced by a successful output.

The mapping is immutable; contained capabilities may still be mutable or
concurrency-sensitive.

## Explicit policy names

Sequential and fail-fast behavior is the default:

- `pipe_all(operations)` composes left to right.
- `recover`, `recover_with`, and `default_on_error` handle failures.
- `or_else` and `fallback` try alternatives with the original input.
- `tap` propagates effect failures; `tap_best_effort` ignores them.
- `retry(..., attempts=n, backoff=...)` counts the first call as an attempt.
- `fanout_parallel` is fail-fast same-input concurrency.
- `collect_named` is fail-fast named same-input concurrency.

All-settled behavior has a different result shape:

- `fanout_all_settled` returns one `Result` per branch.
- `traverse_all_settled` returns one `Result` per item.
- `build_all_settled` returns one `Result` per field.

Lossy behavior is also explicit:

- `traverse_lossy` retains successful items.
- `filter_best_effort` treats predicate failures as rejection.
- `build_lossy` omits failed fields.

Fail-fast parallel operations cancel and await unfinished siblings. Fixed
fanout accepts an optional concurrency `limit`; concurrent collection traversal
requires one. Returned values preserve declaration/input order, not completion
order.

## Collections

Raw callbacks and nested operations use different names:

```python
from fp_ops import (
    filter_each,
    fold,
    map_each,
    operation,
    traverse,
    traverse_parallel,
)

increment = operation(lambda value: value + 1)
mapped = map_each(str)
sequential = traverse(increment)
concurrent = traverse_parallel(increment, limit=8)
selected = filter_each(lambda value: value > 0)
total = fold(lambda acc, value: acc + value, 0)
```

List order and mapping insertion order are preserved. A fail-fast item error is
reported as `TraversalItemError(index, cause)`. Empty inputs do not invoke the
callback.

## Paths, builds, and assignment

```python
from dataclasses import dataclass

from fp_ops import build, get_path


@dataclass
class User:
    name: str
    age: int


to_user = build(
    {
        "name": get_path("profile.name"),
        "age": get_path("profile.age"),
    },
    User,
)
```

`get_path` distinguishes an absent path from a present `None`. `build`
evaluates schema fields in insertion order and fails with
`BuildFieldError(field, cause)`. `merge_shallow`/`assign` merge same-input
mapping sources; `assign_fields` copies an input mapping and applies shallow
field assignments.

Mappings and sequences used as combinator configuration are snapshotted.
Captured domain objects and environment capabilities are not transitively
immutable.

## Reference and migration

- [Generated top-level interface reference](docs/api-reference.md)
- [0.3 breaking migration guide](docs/migration-0.3.md)
- [Executable unary pipeline example](examples/unary_pipeline.py)

The source, good/bad consumer fixtures, and an isolated installed wheel are
checked with both mypy and Pyright.
