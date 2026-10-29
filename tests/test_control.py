"""Control flow: branch, passthrough, sleep, loop, bracket, provide."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from expression import Error, Ok, Result

from fp_ops import (
    Continue,
    Environment,
    EnvironmentError,
    EnvKey,
    LoopStepError,
    Operation,
    ReleaseError,
    Stop,
    UndeclaredRequirementError,
    bracket,
    branch,
    constant,
    continue_if,
    fail,
    from_environment,
    identity,
    loop,
    operation,
    passthrough,
    provide,
    result_operation,
    sleep,
    traverse_parallel,
)


@dataclass(frozen=True)
class Settings:
    base: str


SETTINGS = EnvKey("settings", Settings)
COUNT = EnvKey("count", int)


def is_even(value: int) -> bool:
    return value % 2 == 0


# branch ---------------------------------------------------------------------


async def test_branch_runs_only_the_chosen_side() -> None:
    seen: list[str] = []

    def mark(label: str) -> Operation[int, str]:
        return operation(lambda value: seen.append(label) or f"{label}:{value}")

    chosen = branch(operation(is_even), mark("even"), mark("odd"))

    assert (await chosen.run(2)).ok == "even:2"
    assert (await chosen.run(3)).ok == "odd:3"
    assert seen == ["even", "odd"]


async def test_branch_rejects_non_bool_and_merges_requirements() -> None:
    bad = branch(operation(lambda value: value), constant("a"), constant("b"))
    outcome = await bad.run(1)
    assert isinstance(outcome.error, TypeError)

    needs = branch(from_environment(SETTINGS).map(bool), constant(1), from_environment(COUNT))
    assert set(needs.required_environment) == {SETTINGS, COUNT}


# passthrough / sleep ----------------------------------------------------------


async def test_passthrough_keeps_input_and_propagates_effect_failure() -> None:
    effects: list[int] = []
    keep = passthrough(operation(effects.append))
    assert (await keep.run(5)).ok == 5
    assert effects == [5]

    boom = RuntimeError("effect")
    assert (await passthrough(fail(boom)).run(5)).error is boom


async def test_passthrough_effect_uses_environment() -> None:
    effect = passthrough(from_environment(SETTINGS))
    environment = Environment().with_value(SETTINGS, Settings("x"))
    assert (await effect.run("in", environment=environment)).ok == "in"
    assert SETTINGS in effect.required_environment


async def test_sleep_passes_value_and_validates_delay() -> None:
    assert (await sleep(0).run("v")).ok == "v"
    assert (await sleep(lambda: 0.0).run("v")).ok == "v"
    assert isinstance((await sleep(lambda: -1.0).run("v")).error, ValueError)
    with pytest.raises(ValueError, match="non-negative"):
        sleep(-1)


# loop -----------------------------------------------------------------------


def count_to(limit: int) -> Operation[int, Continue[int] | Stop[int]]:
    def step(value: int) -> Continue[int] | Stop[int]:
        return Stop(value) if value >= limit else Continue(value + 1)

    return operation(step)


async def test_loop_runs_in_constant_stack_space() -> None:
    assert (await loop(count_to(100_000)).run(0)).ok == 100_000


async def test_loop_yields_between_synchronous_steps() -> None:
    ticks: list[int] = []

    async def ticker() -> None:
        for tick in range(3):
            ticks.append(tick)
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    await loop(count_to(10)).run(0)
    await task
    assert ticks == [0, 1, 2]


async def test_loop_max_steps_returns_last_state() -> None:
    assert (await loop(count_to(1_000), max_steps=5).run(0)).ok == 5
    with pytest.raises(ValueError, match="max_steps"):
        loop(count_to(1), max_steps=0)


async def test_loop_step_failure_carries_step_and_state() -> None:
    def step(value: int) -> Continue[int]:
        if value == 3:
            raise KeyError("page 3")
        return Continue(value + 1)

    outcome = await loop(operation(step)).run(0)
    error = outcome.error
    assert isinstance(error, LoopStepError)
    assert (error.step, error.state) == (4, 3)
    assert isinstance(error.cause, KeyError)


async def test_loop_rejects_other_step_results() -> None:
    outcome = await loop(operation(lambda value: value)).run(1)
    assert isinstance(outcome.error, TypeError)


async def test_loop_law_stop_after_f_is_f() -> None:
    double: Operation[int, int] = operation(lambda value: value * 2)
    once = loop(double.map(Stop))
    assert (await once.run(21)).ok == (await double.run(21)).ok


async def test_loop_requirements_are_preflighted() -> None:
    step = from_environment(COUNT).map(Stop)
    stepped = loop(step)
    assert COUNT in stepped.required_environment
    outcome = await stepped.run(0)
    assert isinstance(outcome.error, EnvironmentError)


async def test_loop_can_be_cancelled_mid_step() -> None:
    started = asyncio.Event()

    async def slow(value: int) -> Continue[int]:
        started.set()
        await asyncio.sleep(10)
        return Continue(value)

    task = asyncio.create_task(loop(operation(slow)).run(0))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_continue_if_advances_while_condition_holds() -> None:
    below_ten: Operation[int, bool] = operation(lambda value: value < 10)
    add_three: Operation[int, int] = operation(lambda value: value + 3)
    assert (await loop(continue_if(below_ten, add_three)).run(0)).ok == 12


# bracket ----------------------------------------------------------------------


class Pool:
    def __init__(self) -> None:
        self.opened = 0
        self.closed: list[int] = []

    def acquire(self) -> Operation[object, int]:
        def open_one(_: object) -> int:
            self.opened += 1
            return self.opened

        return operation(open_one)

    def release(self, *, failing: bool = False) -> Operation[int, None]:
        def close(resource: int) -> None:
            self.closed.append(resource)
            if failing:
                raise OSError(f"close {resource}")

        return operation(close)


async def test_bracket_releases_after_success_and_failure() -> None:
    pool = Pool()
    ok = bracket(pool.acquire(), operation(lambda r: r * 10), pool.release())
    assert (await ok.run(None)).ok == 10

    boom = ValueError("use")
    failed = await bracket(pool.acquire(), fail(boom), pool.release()).run(None)
    assert failed.error is boom
    assert pool.closed == [1, 2]


async def test_bracket_acquire_failure_releases_nothing() -> None:
    pool = Pool()
    outcome = await bracket(fail(OSError("acquire")), identity(), pool.release()).run(None)
    assert isinstance(outcome.error, OSError)
    assert pool.closed == []


async def test_bracket_release_failure_is_reported_with_primary() -> None:
    pool = Pool()
    only_release = await bracket(pool.acquire(), identity(), pool.release(failing=True)).run(None)
    assert isinstance(only_release.error, ReleaseError)
    assert only_release.error.primary is None

    boom = ValueError("use")
    both = await bracket(pool.acquire(), fail(boom), pool.release(failing=True)).run(None)
    assert isinstance(both.error, ReleaseError)
    assert both.error.primary is boom


async def test_bracket_releases_when_cancelled() -> None:
    pool = Pool()
    started = asyncio.Event()

    async def use(resource: int) -> int:
        started.set()
        await asyncio.sleep(10)
        return resource

    task = asyncio.create_task(bracket(pool.acquire(), operation(use), pool.release()).run(None))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.closed == [1]


async def test_bracket_release_finishes_even_if_cancelled_during_release() -> None:
    closed: list[int] = []
    releasing = asyncio.Event()

    async def slow_close(resource: int) -> None:
        releasing.set()
        await asyncio.sleep(0.05)
        closed.append(resource)

    task = asyncio.create_task(
        bracket(constant(1), identity(), operation(slow_close)).run(None)
    )
    await releasing.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed == [1]


async def test_fail_fast_traversal_releases_every_cancelled_sibling() -> None:
    pool = Pool()

    async def use(resource: int) -> int:
        if resource == 1:
            raise RuntimeError("first fails")
        await asyncio.sleep(10)
        return resource

    each = bracket(pool.acquire(), operation(use), pool.release())
    outcome = await traverse_parallel(each, limit=4).run([0, 0, 0, 0])

    assert outcome.is_error()
    assert sorted(pool.closed) == [1, 2, 3, 4]


# provide / from_environment -------------------------------------------------------


async def test_provide_binds_a_value_for_its_body_only() -> None:
    url = provide(
        SETTINGS,
        operation(lambda path: Settings(f"https://{path}")),
        from_environment(SETTINGS).map(lambda settings: settings.base),
    )
    assert url.required_environment == ()
    assert (await url.run("example.com")).ok == "https://example.com"


async def test_provide_subtracts_only_its_key() -> None:
    body = from_environment(SETTINGS) & from_environment(COUNT)
    scoped = provide(SETTINGS, constant(Settings("x")), body)
    assert scoped.required_environment == (COUNT,)
    outcome = await scoped.run(None, environment=Environment().with_value(COUNT, 2))
    assert outcome.ok == (Settings("x"), 2)


async def test_provide_shadows_and_isolates() -> None:
    outer = Environment().with_value(COUNT, 1)
    inner = provide(COUNT, constant(2), from_environment(COUNT))
    pair = inner & from_environment(COUNT)
    assert (await pair.run(None, environment=outer)).ok == (2, 1)


async def test_provide_checks_value_type() -> None:
    wrong = provide(COUNT, constant("not an int"), from_environment(COUNT))
    assert isinstance((await wrong.run(None)).error, EnvironmentError)


async def test_provide_law() -> None:
    assert (await provide(COUNT, constant(7), from_environment(COUNT)).run(None)).ok == 7


# flat_map(requires=) ------------------------------------------------------------


async def test_flat_map_requires_is_preflighted() -> None:
    dynamic = constant(1).flat_map(lambda _: from_environment(COUNT), requires=[COUNT])
    assert COUNT in dynamic.required_environment
    assert isinstance((await dynamic.run(None)).error, EnvironmentError)
    assert (await dynamic.run(None, environment=Environment().with_value(COUNT, 3))).ok == 3


async def test_flat_map_undeclared_missing_key_is_named() -> None:
    dynamic = constant(1).flat_map(lambda _: from_environment(COUNT))
    error = (await dynamic.run(None)).error
    assert isinstance(error, UndeclaredRequirementError)
    assert error.key is COUNT


async def test_result_operation_in_loop_keeps_error_values() -> None:
    def step(value: int) -> Result[Continue[int] | Stop[int], Exception]:
        return Ok(Stop(value)) if value else Error(ValueError("zero"))

    stepped = loop(result_operation(step))
    assert (await stepped.run(4)).ok == 4
    assert isinstance((await stepped.run(0)).error, LoopStepError)
