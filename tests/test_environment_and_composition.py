from __future__ import annotations

import asyncio
from typing import Any

import pytest
from expression import Error, Ok, Result

from fp_ops import (
    UNIT,
    EnvKey,
    Environment,
    EnvironmentError,
    Operation,
    collect_named,
    constant,
    default_on_error,
    environment_operation,
    environment_result_operation,
    fail,
    fallback,
    fanout_all_settled,
    fanout_parallel,
    operation,
    or_else,
    pipe_all,
    recover,
    recover_with,
    result_operation,
    retry,
    tap,
    tap_best_effort,
)


class Counter:
    def __init__(self, amount: int) -> None:
        self.amount = amount
        self.seen: list[int] = []


COUNTER = EnvKey("counter", Counter)


@environment_operation(COUNTER)
def add_counter(value: int, counter: Counter) -> int:
    counter.seen.append(value)
    return value + counter.amount


@pytest.mark.asyncio
async def test_environment_is_immutable_typed_and_reused_across_stages() -> None:
    counter = Counter(2)
    first = Environment()
    configured = first.with_value(COUNTER, counter)

    assert COUNTER not in first
    assert configured.require(COUNTER) is counter
    assert await add_counter.then(add_counter).run(1, environment=configured) == Ok(
        5
    )
    assert counter.seen == [1, 3]
    assert add_counter.required_environment == (COUNTER,)


@pytest.mark.asyncio
async def test_missing_environment_requirement_is_an_error_before_effects() -> None:
    called = False

    @operation
    def before(value: int) -> int:
        nonlocal called
        called = True
        return value

    outcome = await before.then(add_counter).run(1)
    assert outcome.is_error()
    assert isinstance(outcome.error, EnvironmentError)
    assert called is False


def test_environment_rejects_wrong_value_type_at_assembly() -> None:
    with pytest.raises(EnvironmentError):
        Environment().with_value(COUNTER, object())  # type: ignore[arg-type]


def test_environment_validates_keys_and_exposes_mapping_metadata() -> None:
    with pytest.raises(ValueError):
        EnvKey("", Counter)
    with pytest.raises(TypeError):
        EnvKey("counter", object())  # type: ignore[arg-type]
    with pytest.raises(EnvironmentError):
        Environment({object(): Counter(1)})  # type: ignore[dict-item]

    environment = Environment({COUNTER: Counter(1)})
    assert list(environment) == [COUNTER]
    assert len(environment) == 1
    assert repr(COUNTER) == "EnvKey('counter', Counter)"
    assert repr(environment) == "Environment(counter)"


@pytest.mark.asyncio
async def test_environment_result_constructor_flattens_declared_result() -> None:
    @environment_result_operation(COUNTER)
    def checked(value: int, counter: Counter) -> Result[int, Exception]:
        if value < 0:
            return Error(ValueError("negative"))
        return Ok(value + counter.amount)

    environment = Environment().with_value(COUNTER, Counter(3))
    assert await checked.run(2, environment=environment) == Ok(5)
    assert (await checked.run(-1, environment=environment)).is_error()


@pytest.mark.asyncio
async def test_recovery_variants_and_alternatives_use_original_input() -> None:
    error = ValueError("bad")
    failed: Operation[int, int] = result_operation(lambda _value: Error(error))

    assert await recover(failed, lambda exc: len(str(exc))).run(10) == Ok(3)
    assert await recover_with(failed, lambda _exc: constant(8)).run(10) == Ok(8)
    assert await default_on_error(failed, 7).run(10) == Ok(7)
    assert await or_else(failed, operation(lambda value: value + 1)).run(10) == Ok(
        11
    )
    assert await (failed | operation(lambda value: value + 2)).run(10) == Ok(12)
    assert await fallback(failed, failed, constant(4)).run(10) == Ok(4)


def test_invalid_fallback_and_retry_configuration_raise_synchronously() -> None:
    with pytest.raises(ValueError):
        fallback()
    with pytest.raises(ValueError):
        retry(constant(1), attempts=0)
    with pytest.raises(ValueError):
        retry(constant(1), attempts=1, backoff=-0.1)


def test_combinators_reject_invalid_configuration() -> None:
    invalid: Any = object()
    stage = constant(1)

    with pytest.raises(TypeError):
        pipe_all([invalid])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        recover(invalid, lambda _error: 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        recover_with(stage, invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        or_else(stage, invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fallback(stage, invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tap(invalid, lambda _value: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        retry(invalid, attempts=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        retry(stage, attempts=True)
    with pytest.raises(TypeError):
        retry(stage, attempts=1, backoff=invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fanout_parallel(invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fanout_parallel(stage, limit=True)
    with pytest.raises(TypeError):
        collect_named(invalid=invalid)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_recovery_retry_and_empty_fanout_edge_cases() -> None:
    handler_called = False

    def handler(_error: Exception) -> int:
        nonlocal handler_called
        handler_called = True
        return 2

    assert await recover(constant(1), handler).run(UNIT) == Ok(1)
    assert handler_called is False

    invalid_recovery = await recover_with(
        fail(ValueError("failed")),
        lambda _error: 2,  # type: ignore[arg-type]
    ).run(UNIT)
    assert invalid_recovery.is_error()
    assert isinstance(invalid_recovery.error, TypeError)

    first_error = ValueError("first")
    last_error = LookupError("last")
    exhausted = await fallback(fail(first_error), fail(last_error)).run(UNIT)
    assert exhausted == Error(last_error)

    retried = await retry(
        fail(ValueError("retry")),
        attempts=2,
        backoff=0.0001,
    ).run(UNIT)
    assert retried.is_error()
    assert str(retried.error) == "retry"

    invalid_backoff = await retry(
        fail(ValueError("retry")),
        attempts=2,
        backoff=lambda _attempt, _error: True,
    ).run(UNIT)
    assert invalid_backoff.is_error()
    assert isinstance(invalid_backoff.error, TypeError)

    negative_backoff = await retry(
        fail(ValueError("retry")),
        attempts=2,
        backoff=lambda _attempt, _error: -1,
    ).run(UNIT)
    assert negative_backoff.is_error()
    assert isinstance(negative_backoff.error, ValueError)

    assert await fanout_parallel().run(UNIT) == Ok(())


@pytest.mark.asyncio
async def test_default_value_survives_non_copyable_configuration() -> None:
    class NonCopyable:
        def __deepcopy__(self, _memo: dict[int, object]) -> object:
            raise TypeError("not copyable")

    value = NonCopyable()
    outcome = await default_on_error(fail(ValueError("failed")), value).run(UNIT)
    assert outcome.is_ok()
    assert outcome.ok is value


@pytest.mark.asyncio
async def test_retry_attempt_count_and_async_backoff() -> None:
    calls = 0
    backoffs: list[tuple[int, str]] = []

    @operation
    def flaky(value: int) -> int:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError(f"failure {calls}")
        return value

    async def backoff(attempt: int, error: Exception) -> float:
        backoffs.append((attempt, str(error)))
        return 0

    assert await retry(flaky, attempts=3, backoff=backoff).run(5) == Ok(5)
    assert calls == 3
    assert backoffs == [(1, "failure 1"), (2, "failure 2")]


@pytest.mark.asyncio
async def test_tap_propagates_and_best_effort_ignores_only_exception() -> None:
    effect_error = RuntimeError("effect")

    def broken(_value: int) -> None:
        raise effect_error

    propagated = await tap(constant(3), broken).run(UNIT)
    ignored = await tap_best_effort(constant(3), broken).run(UNIT)
    assert propagated == Error(effect_error)
    assert ignored == Ok(3)


@pytest.mark.asyncio
async def test_pipe_all_snapshots_configuration() -> None:
    operations = [operation(lambda value: value + 1)]
    pipeline = pipe_all(operations)
    operations.append(operation(lambda value: value * 100))
    assert await pipeline.run(1) == Ok(2)
    assert await pipe_all([]).run("same") == Ok("same")


@pytest.mark.asyncio
async def test_fanout_output_order_all_settled_and_named_shape() -> None:
    async def delayed(value: int, delay: float) -> int:
        await asyncio.sleep(delay)
        return value

    slow = operation(lambda value: delayed(value + 1, 0.02))
    fast = operation(lambda value: delayed(value + 2, 0))
    broken = fail(ValueError("branch"))

    assert await fanout_parallel(slow, fast).run(1) == Ok((2, 3))
    settled = await fanout_all_settled(slow, broken, fast).run(1)
    assert settled.is_ok()
    values = settled.ok
    assert values[0] == Ok(2)
    assert values[1].is_error()
    assert values[2] == Ok(3)
    assert await collect_named(first=slow, second=fast).run(1) == Ok(
        {"first": 2, "second": 3}
    )


@pytest.mark.asyncio
async def test_ampersand_fans_out_flat_only_across_an_ampersand_chain() -> None:
    increment = operation(lambda value: value + 1)
    double = operation(lambda value: value * 2)
    text = operation(lambda value: str(value))
    pair = operation(lambda value: (value, value))

    assert await (increment & double).run(3) == Ok((4, 6))
    assert await (increment & double & text).run(3) == Ok((4, 6, "3"))
    assert await (pair & text).run(3) == Ok(((3, 3), "3"))
    assert await (increment & (double & text)).run(3) == Ok((4, (6, "3")))
    assert await ((increment & double) >> operation(sum)).run(3) == Ok(10)
    assert await ((increment & double).map(sum) & text).run(3) == Ok((10, "3"))


@pytest.mark.asyncio
async def test_ampersand_fails_fast_and_merges_environment_requirements() -> None:
    key = EnvKey("offset", int)
    offset = environment_operation(key)(lambda value, amount: value + amount)
    broken = fail(ValueError("branch"))

    fanout = offset & constant("x") & offset
    assert fanout.required_environment == (key,)
    assert await fanout.run(
        1, environment=Environment().with_value(key, 10)
    ) == Ok((11, "x", 11))
    outcome = await (constant(1) & broken & constant(2)).run(UNIT)
    assert outcome.is_error()
    assert str(outcome.error) == "branch"


def test_ampersand_rejects_non_operation_operands() -> None:
    invalid: Any = object()

    with pytest.raises(TypeError):
        constant(1) & invalid
    with pytest.raises(TypeError):
        constant(1) & constant(2) & invalid


@pytest.mark.asyncio
async def test_fail_fast_fanout_cancels_and_joins_sibling() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @operation
    async def sibling(_value: object) -> None:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    @operation
    async def broken(_value: object) -> object:
        await started.wait()
        raise ValueError("fanout")

    outcome = await fanout_parallel(sibling, broken).run(UNIT)
    assert outcome.is_error()
    assert str(outcome.error) == "fanout"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_fanout_limit_bounds_concurrency() -> None:
    active = 0
    maximum = 0

    def branch() -> Operation[object, int]:
        @operation
        async def run_branch(_value: object) -> int:
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return maximum

        return run_branch

    outcome = await fanout_parallel(
        branch(), branch(), branch(), branch(), limit=2
    ).run(UNIT)
    assert outcome.is_ok()
    assert maximum == 2
