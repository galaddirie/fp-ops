from __future__ import annotations

import inspect
import threading

import pytest
from expression import Error, Ok, Result

from fp_ops import (
    Environment,
    EnvKey,
    Operation,
    Placeholder,
    _,
    environment_operation_template,
    environment_result_operation_template,
    operation_template,
    result_operation_template,
    threaded_operation_template,
)


@pytest.mark.asyncio
async def test_placeholder_can_select_any_direct_positional_argument() -> None:
    @operation_template
    def render(
        prefix: str,
        value: int,
        suffix: bytes,
        *,
        scale: float = 1.0,
    ) -> str:
        return f"{prefix}{value * scale:g}{suffix.decode()}"

    first = render(_, 4, b"!")
    second = render("$", _, b"!", scale=2.0)
    third = render("$", 4, _, scale=2.0)

    assert await first.run("£") == Ok("£4!")
    assert await second.run(3) == Ok("$6!")
    assert await third.run(b"?") == Ok("$8?")
    assert all(isinstance(op, Operation) for op in (first, second, third))
    assert all(not callable(op) for op in (first, second, third))


@pytest.mark.asyncio
async def test_template_normalizes_async_and_result_outputs() -> None:
    @operation_template
    async def render(prefix: str, value: int) -> str:
        return f"{prefix}{value}"

    @result_operation_template
    async def checked(minimum: int, value: int) -> Result[int, Exception]:
        if value < minimum:
            return Error(ValueError("too small"))
        return Ok(value)

    assert await render("#", _).run(3) == Ok("#3")
    assert await checked(2, _).run(3) == Ok(3)
    failed = await checked(2, _).run(1)
    assert failed.is_error()
    assert isinstance(failed.error, ValueError)


@pytest.mark.asyncio
async def test_threaded_template_runs_blocking_callable_off_loop_thread() -> None:
    loop_thread = threading.get_ident()

    @threaded_operation_template
    def is_threaded(prefix: str, value: int) -> str:
        assert threading.get_ident() != loop_thread
        return f"{prefix}{value}"

    assert await is_threaded("#", _).run(3) == Ok("#3")


def test_template_validates_configuration_before_creating_operation() -> None:
    @operation_template
    def combine(left: str, right: int) -> str:
        return left * right

    with pytest.raises(TypeError, match="exactly one"):
        combine("x", 2)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="exactly one"):
        combine(_, _)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="missing"):
        combine(_)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="unexpected"):
        combine(_, 2, unknown=True)  # type: ignore[call-overload]


@pytest.mark.asyncio
async def test_template_snapshots_configuration_and_preserves_metadata() -> None:
    @operation_template
    def append_to(items: list[int], value: int) -> list[int]:
        """Append a configured value."""

        return [*items, value]

    configured = [1]
    stage = append_to(configured, _)
    configured.append(2)

    assert await stage.run(3) == Ok([1, 3])
    assert stage.name == "append_to"
    assert stage.description == "Append a configured value."
    assert str(stage.source_signature) == (
        "(items: 'list[int]', value: 'int') -> 'list[int]'"
    )
    assert inspect.signature(append_to) == inspect.signature(append_to.__wrapped__)


@pytest.mark.asyncio
async def test_keyword_placeholder_works_at_runtime() -> None:
    @operation_template
    def render(prefix: str, value: int, *, suffix: str) -> str:
        return f"{prefix}{value}{suffix}"

    stage = render("$", 4, suffix=_)  # type: ignore[call-overload]
    assert await stage.run("!") == Ok("$4!")


@pytest.mark.asyncio
async def test_template_rejects_nested_but_accepts_variadic_configuration() -> None:
    @operation_template
    def configured(options: dict[str, object], value: int) -> int:
        return value

    with pytest.raises(TypeError, match="exactly one"):
        configured({"nested": _}, 1)

    @operation_template
    def variadic(value: int, *rest: object) -> int:
        return value + len(rest)

    assert await variadic(_, "configured", True).run(3) == Ok(5)


class Counter:
    def __init__(self, amount: int) -> None:
        self.amount = amount


COUNTER = EnvKey("counter", Counter)


@pytest.mark.asyncio
async def test_environment_templates_inject_final_capability() -> None:
    @environment_operation_template(COUNTER)
    def add(
        prefix: str,
        value: int,
        counter: Counter,
        *,
        repeat: int = 1,
    ) -> str:
        return f"{prefix}{value + counter.amount}" * repeat

    @environment_result_operation_template(COUNTER)
    def checked(
        minimum: int,
        value: int,
        counter: Counter,
    ) -> Result[int, Exception]:
        adjusted = value + counter.amount
        if adjusted < minimum:
            return Error(ValueError("too small"))
        return Ok(adjusted)

    environment = Environment().with_value(COUNTER, Counter(2))
    rendered = add("#", _, repeat=2)
    validated = checked(3, _)

    assert rendered.required_environment == (COUNTER,)
    assert await rendered.run(4, environment=environment) == Ok("#6#6")
    assert await validated.run(1, environment=environment) == Ok(3)
    missing = await rendered.run(4)
    assert missing.is_error()


def test_placeholder_has_clear_public_representation() -> None:
    assert isinstance(_, Placeholder)
    assert repr(_) == "_"
