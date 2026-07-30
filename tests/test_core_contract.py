from __future__ import annotations

import asyncio
import inspect
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from expression import Error, Ok, Result

from fp_ops import (
    EMPTY_ENV,
    UNIT,
    Environment,
    Operation,
    constant,
    fail,
    identity,
    operation,
    result_operation,
    threaded_operation,
)


def successful(result: Result[Any, Exception]) -> Any:
    assert result.is_ok(), result
    return result.ok


@pytest.mark.asyncio
async def test_single_and_multi_stage_runs_share_unary_input_semantics() -> None:
    seen: list[object] = []

    @operation
    def first(value: object) -> object:
        seen.append(value)
        return value

    @operation
    def second(value: object) -> tuple[object, str]:
        seen.append(value)
        return value, "done"

    assert successful(await first.run(None)) is None
    assert successful(await first.then(second).run(None)) == (None, "done")
    assert seen == [None, None, None]


@pytest.mark.asyncio
async def test_identity_and_associativity_match_results_and_effect_trace() -> None:
    trace: list[str] = []

    def stage(name: str, amount: int) -> Operation[int, int]:
        @operation
        def apply(value: int) -> int:
            trace.append(name)
            return value + amount

        return apply

    first, second, third = stage("a", 1), stage("b", 2), stage("c", 3)
    direct = await first.run(1)
    direct_trace = trace[:]
    trace.clear()
    left_identity = await identity().then(first).run(1)
    left_trace = trace[:]
    trace.clear()
    right_identity = await first.then(identity()).run(1)
    right_trace = trace[:]
    trace.clear()
    left_group = await first.then(second).then(third).run(1)
    left_group_trace = trace[:]
    trace.clear()
    right_group = await first.then(second.then(third)).run(1)

    assert direct == left_identity == right_identity
    assert direct_trace == left_trace == right_trace == ["a"]
    assert left_group == right_group == Ok(7)
    assert left_group_trace == trace == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_short_circuit_preserves_original_exception() -> None:
    error = ValueError("stop")
    called = False

    @operation
    def later(value: int) -> int:
        nonlocal called
        called = True
        return value

    outcome = await fail(error).then(later).run(1)

    assert outcome.is_error()
    assert outcome.error is error
    assert called is False


@pytest.mark.asyncio
async def test_raw_result_operation_and_none_are_ordinary_values() -> None:
    inner = Ok(3)
    raw = operation(lambda _value: inner)
    preserve_none = operation(lambda _value: None).then(
        operation(lambda value: ("received", value))
    )

    raw_outcome = await raw.run(UNIT)
    assert raw_outcome.is_ok()
    assert raw_outcome.ok is inner
    assert successful(await preserve_none.run(UNIT)) == ("received", None)


@pytest.mark.asyncio
async def test_result_constructor_flattens_exactly_once() -> None:
    error = LookupError("missing")
    nested = Ok(Ok(4))

    assert await result_operation(lambda _value: Error(error)).run(UNIT) == Error(
        error
    )
    outcome = await result_operation(lambda _value: nested).run(UNIT)
    assert outcome.is_ok()
    assert outcome.ok == Ok(4)


@pytest.mark.asyncio
async def test_map_preserves_returned_operation_and_flat_map_executes_one() -> None:
    returned = constant(9)

    mapped = await operation(lambda value: value).map(
        lambda _value: returned
    ).run(1)
    flattened = await operation(lambda value: value).flat_map(
        lambda _value: returned
    ).run(1)

    assert mapped.is_ok()
    assert mapped.ok is returned
    assert flattened == Ok(9)


@pytest.mark.asyncio
async def test_sync_async_and_sync_returning_nested_awaitable_normalize() -> None:
    @operation
    def sync(value: int) -> int:
        return value + 1

    @operation
    async def async_stage(value: int) -> int:
        return value + 1

    def nested(value: int) -> object:
        async def outer() -> object:
            async def inner() -> int:
                return value + 1

            return inner()

        return outer()

    assert await sync.run(1) == Ok(2)
    assert await async_stage.run(1) == Ok(2)
    assert await operation(nested).run(1) == Ok(2)


@pytest.mark.asyncio
async def test_exception_boundary_and_cancellation_policy() -> None:
    @operation
    def raises(_value: object) -> object:
        raise RuntimeError("closed")

    @operation
    async def cancelled(_value: object) -> object:
        raise asyncio.CancelledError

    outcome = await raises.run(UNIT)
    assert outcome.is_error()
    assert isinstance(outcome.error, RuntimeError)
    with pytest.raises(asyncio.CancelledError):
        await cancelled.run(UNIT)


@pytest.mark.asyncio
async def test_threaded_operation_runs_initial_call_outside_event_loop() -> None:
    event_loop_thread = __import__("threading").get_ident()

    @threaded_operation
    def thread_id(_value: object) -> int:
        return __import__("threading").get_ident()

    outcome = await thread_id.run(UNIT)
    assert successful(outcome) != event_loop_thread


def test_operation_is_immutable_metadata_not_a_callable_or_awaitable() -> None:
    @operation
    def documented(value: int) -> int:
        """Increment a value."""

        return value + 1

    assert documented.name == "documented"
    assert documented.description == "Increment a value."
    # The stored signature belongs to the wrapped source, not Operation.run.
    assert str(documented.source_signature) == "(value: 'int') -> 'int'"
    assert "CallNode" not in repr(documented)
    assert "_node" not in repr(documented)
    assert not callable(documented)
    assert not inspect.isawaitable(documented)
    with pytest.raises(FrozenInstanceError):
        documented.name = "changed"  # type: ignore[misc]


def test_then_rejects_callable_and_run_always_requires_value() -> None:
    stage = operation(lambda value: value)
    with pytest.raises(TypeError, match="map"):
        stage.then(lambda value: value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        stage.run()  # type: ignore[call-arg]


def test_constant_fail_and_identity_are_fresh_operation_values() -> None:
    with pytest.raises(TypeError):
        Operation()  # type: ignore[call-arg]
    assert isinstance(constant(1), Operation)
    assert isinstance(fail(ValueError("x")), Operation)
    assert identity() is not identity()
    assert EMPTY_ENV is not Environment()
