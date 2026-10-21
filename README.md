# FP-Ops: Composable Async Pipelines for Python

[![PyPI version](https://img.shields.io/badge/pypi-v0.4.0-blue.svg)](https://pypi.org/project/fp-ops/)
[![Python versions](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/fp-ops/)
[![codecov](https://codecov.io/gh/galaddirie/fp-ops/graph/badge.svg?token=8MHGFYBD8V)](https://codecov.io/gh/galaddirie/fp-ops)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Type checked: mypy + Pyright](https://img.shields.io/badge/type%20checked-mypy%20%2B%20Pyright-blue)](https://github.com/galaddirie/fp-ops/actions)

FP-Ops turns regular Python functions into small, type-safe operations that are
easy to compose, run, and test.

Use it when a task has several steps—fetching data, validating it, transforming
it, handling failures—and you want the whole pipeline to remain readable.

## Why FP-Ops?

- **Readable composition:** chain steps with `>>`, try alternatives with `|`,
  and run branches in parallel with `&`.
- **Sync and async together:** compose either kind of function in one pipeline.
- **Explicit errors:** every run returns `Ok(value)` or `Error(exception)`.
- **Strong typing:** pipeline inputs and outputs are checked by mypy and Pyright.
- **Safe concurrency:** choose fail-fast, all-settled, or lossy behavior by name.
- **Immutable building blocks:** configuring or composing an operation never
  mutates the original.

## Installation

```bash
pip install fp-ops
```

FP-Ops supports Python 3.10 and newer.

## Quick start

```python
import asyncio

from fp_ops import Ok, operation


@operation
def parse_number(value: str) -> int:
    return int(value)


@operation
async def double(value: int) -> int:
    return value * 2


pipeline = parse_number >> double


async def main() -> None:
    assert await pipeline.run("21") == Ok(42)

    invalid = await pipeline.run("not a number")
    assert invalid.is_error()
    assert isinstance(invalid.error, ValueError)


asyncio.run(main())
```

An `Operation[A, B]` accepts one value of type `A` and produces a
`Result[B, Exception]`. Ordinary exceptions become `Error` values, so a
failed step stops the pipeline without hiding the reason.

## Compose and transform

Use `>>` (or `.then()`) when the next step is another operation. Use
`.map()` for a small value transformation.

```python
from fp_ops import operation


@operation
def username(user: dict[str, str]) -> str:
    return user["name"]


display_name = username.map(str.strip).map(str.title)
```

Pipelines are immutable and reusable:

```python
raw_name = username
clean_name = username.map(str.strip)
display_name = clean_name.map(str.title)
```

Creating `clean_name` or `display_name` does not change `username`.

## Configure reusable operations

Operation templates let you configure a multi-argument function while leaving
one `_` slot for the pipeline value:

```python
from fp_ops import _, operation_template


@operation_template
def format_money(symbol: str, amount: float, *, precision: int = 2) -> str:
    return f"{symbol}{amount:.{precision}f}"


usd = format_money("$", _, precision=2)
# await usd.run(12.5) == Ok("$12.50")
```

The template immediately creates a normal unary operation. Configuration is
validated and captured when the operation is built—not later when it runs.

## Handle failures

Choose the behavior that matches your application:

```python
from fp_ops import Ok, default_on_error, operation, retry


@operation
def parse_number(value: str) -> int:
    return int(value)


safe_parse = default_on_error(parse_number, 0)
resilient_parse = retry(parse_number, attempts=3, backoff=0.1)

# await safe_parse.run("unknown") == Ok(0)
```

- `recover` turns an exception into a value.
- `recover_with` runs another operation after a failure.
- `default_on_error` supplies a fixed fallback value.
- `first | second` and `fallback(...)` try alternatives with the original
  input.
- `retry` retries an operation with a fixed or calculated backoff.

## Work with collections

Collection helpers preserve list order and mapping keys:

```python
from fp_ops import Ok, filter_each, map_each, operation


@operation
def scores(record: dict[str, list[int]]) -> list[int]:
    return record["scores"]


normalize_scores = (
    scores
    >> filter_each(lambda score: score >= 0)
    >> map_each(lambda score: score / 100)
)

# await normalize_scores.run({"scores": [80, -1, 95]})
# == Ok([0.8, 0.95])
```

Use raw callbacks with `map_each`, `filter_each`, and `fold`. Use nested
operations with `traverse`, `filter_operation`, and `fold_operation`.
`traverse_parallel` adds bounded concurrency when each item performs async
work.

## Build structured output

Create dictionaries or typed models from the same input:

```python
from dataclasses import dataclass

from fp_ops import Ok, build, get_path


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

data = {"profile": {"name": "Ada", "age": 36}}
# await to_user.run(data) == Ok(User(name="Ada", age=36))
```

`get_path` works with nested mappings, sequence indexes, and attributes.
`assign`, `assign_fields`, and `merge_shallow` cover common mapping
transformations.

## Supply shared dependencies

An `Environment` provides typed dependencies such as configuration, clients,
or sessions without mixing them into pipeline data:

```python
from dataclasses import dataclass

from fp_ops import EnvKey, Environment, Ok, environment_operation


@dataclass(frozen=True)
class Settings:
    base_url: str


SETTINGS = EnvKey("settings", Settings)


@environment_operation(SETTINGS)
def user_url(user_id: int, settings: Settings) -> str:
    return f"{settings.base_url}/users/{user_id}"


environment = Environment().with_value(
    SETTINGS,
    Settings(base_url="https://api.example.com"),
)

# await user_url.run(7, environment=environment)
# == Ok("https://api.example.com/users/7")
```

Environments are read-only and shared by every stage in a run.

## Choose an execution policy

Failure behavior is explicit in each helper's name:

| Work | Fail fast | Keep every result | Keep successes |
|---|---|---|---|
| Parallel branches | `a & b` / `fanout_parallel` | `fanout_all_settled` | — |
| Collection items | `traverse` / `traverse_parallel` | `traverse_all_settled` | `traverse_lossy` |
| Object fields | `build` | `build_all_settled` | `build_lossy` |
| Predicates | `filter_operation` | — | `filter_best_effort` |

`a & b & c` runs every branch on the same input and returns a flat
`(a, b, c)` tuple. Only an `&` chain is flattened: a branch that returns a
tuple stays one value, and `a & (b & c)` nests as `(a, (b, c))`.

Fail-fast concurrent work cancels unfinished siblings. Concurrent collection
helpers require a `limit`, and results retain input order rather than
completion order.

## Learn more

- [API reference](docs/api-reference.md)
- [0.3 migration guide](docs/migration-0.3.md)
- [Executable example](examples/unary_pipeline.py)
- [Changelog](CHANGELOG.md)

Version 0.3 is a breaking redesign. If you are upgrading from 0.2, start with
the migration guide; old `.execute()`, callable operations, and runtime argument
binding have been replaced by explicit APIs. Binary `&` returned in 0.4 as a
typed, fail-fast fanout.

## Development

```bash
poetry install
poetry run pytest
poetry run mypy -p fp_ops
poetry run pyright
```

## Contributing

Contributions are welcome. Open an issue to discuss a larger change, or submit a
pull request with tests for the behavior you are changing.

## License

FP-Ops is available under the [MIT License](LICENSE).
