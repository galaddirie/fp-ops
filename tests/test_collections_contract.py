from __future__ import annotations

import asyncio
from typing import Any

import pytest
from expression import Error, Ok, Result

from fp_ops import (
    TraversalItemError,
    fanout_each,
    filter_best_effort,
    filter_each,
    filter_operation,
    fold,
    fold_operation,
    map_each,
    operation,
    result_operation,
    traverse,
    traverse_all_settled,
    traverse_lossy,
    traverse_parallel,
)


@pytest.mark.asyncio
async def test_map_and_traverse_preserve_list_and_mapping_order() -> None:
    double = operation(lambda value: value * 2)

    assert await map_each(lambda value: value + 1).run([1, 2, 3]) == Ok(
        [2, 3, 4]
    )
    assert await traverse(double).run({"b": 2, "a": 1}) == Ok(
        {"b": 4, "a": 2}
    )


@pytest.mark.asyncio
async def test_empty_inputs_do_not_invoke_callbacks() -> None:
    called = False

    def callback(value: int) -> int:
        nonlocal called
        called = True
        return value

    assert await map_each(callback).run([]) == Ok([])
    assert await filter_each(lambda value: bool(callback(value))).run({}) == Ok({})
    assert called is False


@pytest.mark.asyncio
async def test_fail_fast_collection_error_retains_item_and_cause() -> None:
    cause = ValueError("item")

    @operation
    def broken(value: int) -> int:
        if value == 2:
            raise cause
        return value

    outcome = await traverse(broken).run([1, 2, 3])
    assert outcome.is_error()
    assert isinstance(outcome.error, TraversalItemError)
    assert outcome.error.index == 1
    assert outcome.error.cause is cause
    assert outcome.error.__cause__ is cause


@pytest.mark.asyncio
async def test_parallel_traversal_is_bounded_and_returns_input_order() -> None:
    active = 0
    maximum = 0

    @operation
    async def delayed(value: int) -> int:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep((4 - value) * 0.005)
        active -= 1
        return value * 2

    assert await traverse_parallel(delayed, limit=2).run([1, 2, 3]) == Ok(
        [2, 4, 6]
    )
    assert maximum == 2


@pytest.mark.asyncio
async def test_parallel_fail_fast_cancels_unfinished_items() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @operation
    async def work(value: int) -> int:
        if value == 1:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        await started.wait()
        raise ValueError("stop")

    outcome = await traverse_parallel(work, limit=2).run([1, 2, 3])
    assert outcome.is_error()
    assert isinstance(outcome.error, TraversalItemError)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_all_settled_and_lossy_disclose_distinct_policies() -> None:
    @result_operation
    def positive(value: int) -> Result[int, Exception]:
        return Ok(value) if value > 0 else Error(ValueError(str(value)))

    reported: list[tuple[int, str]] = []

    async def report(value: int, error: Exception) -> None:
        reported.append((value, str(error)))

    settled = await traverse_all_settled(positive, limit=2).run([1, -2, 3])
    assert settled.is_ok()
    assert settled.ok[0] == Ok(1)
    assert settled.ok[1].is_error()
    assert settled.ok[2] == Ok(3)
    assert await traverse_lossy(
        positive, limit=2, on_error=report
    ).run([1, -2, 3]) == Ok([1, 3])
    assert reported == [(-2, "-2")]


@pytest.mark.asyncio
async def test_filter_policies_are_explicit() -> None:
    @operation
    def predicate(value: int) -> bool:
        if value == 2:
            raise ValueError("predicate")
        return value % 2 == 1

    assert await filter_each(lambda value: value > 1).run([1, 2, 3]) == Ok(
        [2, 3]
    )
    failed = await filter_operation(predicate).run([1, 2, 3])
    assert failed.is_error()
    assert isinstance(failed.error, TraversalItemError)
    assert await filter_best_effort(predicate).run([1, 2, 3]) == Ok([1, 3])


@pytest.mark.asyncio
async def test_fold_callbacks_and_operations_are_distinct() -> None:
    reducer = operation(lambda pair: pair[0] + pair[1])

    assert await fold(lambda total, value: total + value, 0).run([1, 2, 3]) == Ok(
        6
    )
    assert await fold_operation(reducer, 0).run([1, 2, 3]) == Ok(6)


@pytest.mark.asyncio
async def test_fanout_each_is_positional_and_fail_fast() -> None:
    result = await fanout_each(
        operation(lambda value: value + 1),
        operation(lambda value: value * 2),
    ).run([1, 2])
    assert result == Ok([(2, 2), (3, 4)])


def test_parallel_collection_limit_is_validated_at_construction() -> None:
    stage = operation(lambda value: value)
    with pytest.raises(ValueError):
        traverse_parallel(stage, limit=0)
    with pytest.raises(ValueError):
        traverse_all_settled(stage, limit=-1)
    with pytest.raises(ValueError):
        traverse_lossy(stage, limit=0)


def test_collection_helpers_reject_invalid_configuration() -> None:
    invalid: Any = object()
    stage = operation(lambda value: value)

    with pytest.raises(TypeError):
        map_each(invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        traverse(invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        traverse_lossy(stage, limit=1, on_error=invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        filter_each(invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        filter_operation(invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fold(invalid, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fold_operation(invalid, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fanout_each(stage, invalid)  # type: ignore[arg-type]


def test_fold_configuration_tolerates_non_copyable_initial_values() -> None:
    class NonCopyable:
        def __deepcopy__(self, _memo: dict[int, object]) -> object:
            raise TypeError("not copyable")

    initial = NonCopyable()
    assert fold(lambda current, _item: current, initial).name == "fold"
    assert (
        fold_operation(operation(lambda pair: pair[0]), initial).name
        == "fold_operation"
    )


@pytest.mark.asyncio
async def test_collection_helpers_report_callback_and_input_errors() -> None:
    def broken(*_args: object) -> object:
        raise ValueError("callback")

    mapped = await map_each(broken).run([1])
    selected = await filter_each(broken).run([1])
    invalid_selection = await filter_each(lambda _value: 1).run([1])  # type: ignore[arg-type]
    invalid_operation = operation(lambda _value: 1)
    filtered = await filter_operation(invalid_operation).run([1])  # type: ignore[arg-type]
    best_effort = await filter_best_effort(invalid_operation).run([1])  # type: ignore[arg-type]
    folded = await fold(broken, 0).run([1])  # type: ignore[arg-type]
    folded_operation = await fold_operation(
        operation(broken),  # type: ignore[arg-type]
        0,
    ).run([1])
    fanned = await fanout_each(operation(broken)).run([1])  # type: ignore[arg-type]

    for outcome in (
        mapped,
        selected,
        invalid_selection,
        filtered,
        folded,
        folded_operation,
        fanned,
    ):
        assert outcome.is_error()
        assert isinstance(outcome.error, TraversalItemError)
    assert best_effort == Ok([])

    assert (
        await map_each(lambda value: value).run(1)  # type: ignore[arg-type]
    ).is_error()
    assert (
        await fold(lambda total, value: total + value, 0).run(1)  # type: ignore[arg-type]
    ).is_error()
    assert (
        await fanout_each(operation(lambda value: value)).run(1)  # type: ignore[arg-type]
    ).is_error()


@pytest.mark.asyncio
async def test_lossy_traversal_reports_on_error_callback_failures() -> None:
    failed = result_operation(lambda value: Error(ValueError(str(value))))

    def broken_report(_value: int, _error: Exception) -> None:
        raise RuntimeError("report")

    outcome = await traverse_lossy(
        failed,
        limit=1,
        on_error=broken_report,
    ).run([1])
    assert outcome.is_error()
    assert isinstance(outcome.error, TraversalItemError)
    assert isinstance(outcome.error.cause, RuntimeError)
