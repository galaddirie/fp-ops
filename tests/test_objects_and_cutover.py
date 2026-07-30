from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import pytest
from expression import Error, Ok

import fp_ops
from fp_ops import (
    BuildFieldError,
    PathNotFoundError,
    assign,
    assign_fields,
    build,
    build_all_settled,
    build_lossy,
    get_path,
    merge_shallow,
    operation,
)


@pytest.mark.asyncio
async def test_get_path_distinguishes_missing_from_present_none() -> None:
    data = {"user": {"name": None}, "items": [{"price": 3}]}

    assert await get_path("user.name", "default").run(data) == Ok(None)
    assert await get_path("items[0].price").run(data) == Ok(3)
    assert await get_path("missing", 7).run(data) == Ok(7)
    missing = await get_path("missing").run(data)
    assert missing.is_error()
    assert isinstance(missing.error, PathNotFoundError)


@pytest.mark.asyncio
async def test_build_is_ordered_fail_fast_and_preserves_field_cause() -> None:
    trace: list[str] = []
    cause = RuntimeError("browser")

    def field(name: str, *, error: bool = False) -> Any:
        def evaluate(_data: object) -> str:
            trace.append(name)
            if error:
                raise cause
            return name

        return evaluate

    outcome = await build(
        {
            "title": field("title"),
            "price": field("price", error=True),
            "never": field("never"),
        }
    ).run({})

    assert outcome.is_error()
    assert isinstance(outcome.error, BuildFieldError)
    assert outcome.error.field == "price"
    assert outcome.error.cause is cause
    assert outcome.error.__cause__ is cause
    assert trace == ["title", "price"]


@pytest.mark.asyncio
async def test_build_all_settled_and_lossy_have_truthful_shapes() -> None:
    broken = operation(lambda _data: 1 / 0)
    schema = {"good": operation(lambda data: data["value"]), "bad": broken}

    settled = await build_all_settled(schema).run({"value": 4})
    assert settled.is_ok()
    assert settled.ok["good"] == Ok(4)
    assert settled.ok["bad"].is_error()
    assert await build_lossy(schema).run({"value": 4}) == Ok({"good": 4})


@pytest.mark.asyncio
async def test_build_instantiates_models_and_nested_schema() -> None:
    @dataclass
    class User:
        name: str
        details: dict[str, int]

    builder = build(
        {
            "name": get_path("user.name"),
            "details": {"age": get_path("user.age")},
        },
        User,
    )
    assert await builder.run({"user": {"name": "Ada", "age": 37}}) == Ok(
        User("Ada", {"age": 37})
    )


@pytest.mark.asyncio
async def test_schema_and_assignment_configuration_is_snapshotted() -> None:
    literal = {"value": 1}
    schema = {"field": literal}
    builder = build(schema)
    assigned = assign_fields({"field": literal})
    schema["later"] = 2
    literal["value"] = 99

    assert await builder.run({}) == Ok({"field": {"value": 1}})
    assert await assigned.run({"original": True}) == Ok(
        {"original": True, "field": {"value": 1}}
    )


@pytest.mark.asyncio
async def test_shallow_merge_and_assign_fail_fast() -> None:
    dynamic = operation(lambda data: {"value": data["value"]})
    expected = {"static": 1, "value": 2}

    assert await merge_shallow({"static": 1}, dynamic).run({"value": 2}) == Ok(expected)
    assert await assign({"static": 1}, dynamic).run({"value": 2}) == Ok(expected)
    invalid = await merge_shallow(lambda _data: None).run({})  # type: ignore[arg-type]
    assert invalid.is_error()
    assert isinstance(invalid.error, TypeError)


def test_hard_cutover_exports_only_the_new_interface() -> None:
    legacy_modules = [
        "fp_ops.context",
        "fp_ops.execution",
        "fp_ops.flow",
        "fp_ops.graph",
        "fp_ops.operator",
        "fp_ops.placeholder",
        "fp_ops.primitives",
        "fp_ops.sequences",
    ]
    for module in legacy_modules:
        assert find_spec(module) is None

    assert not hasattr(fp_ops.Operation, "execute")
    assert not hasattr(fp_ops.Operation, "bind")
    assert not hasattr(fp_ops.Operation, "apply_cont")
    assert not hasattr(fp_ops.Operation, "__and__")
    assert "BaseContext" not in fp_ops.__all__
    assert "Placeholder" in fp_ops.__all__

    required = {
        "CollectionOperation",
        "EMPTY_ENV",
        "EnvKey",
        "Environment",
        "Operation",
        "UNIT",
        "Unit",
        "assign",
        "assign_fields",
        "build",
        "build_all_settled",
        "build_lossy",
        "collect_named",
        "constant",
        "default_on_error",
        "environment_operation",
        "environment_operation_template",
        "environment_result_operation",
        "environment_result_operation_template",
        "fail",
        "fallback",
        "fanout_all_settled",
        "fanout_each",
        "fanout_parallel",
        "filter_best_effort",
        "filter_each",
        "filter_operation",
        "fold",
        "fold_operation",
        "get_path",
        "identity",
        "map_each",
        "merge_shallow",
        "operation",
        "operation_template",
        "or_else",
        "pipe_all",
        "recover",
        "recover_with",
        "result_operation",
        "result_operation_template",
        "retry",
        "tap",
        "tap_best_effort",
        "threaded_operation",
        "threaded_operation_template",
        "traverse",
        "traverse_all_settled",
        "traverse_lossy",
        "traverse_parallel",
    }
    assert required <= set(fp_ops.__all__)
