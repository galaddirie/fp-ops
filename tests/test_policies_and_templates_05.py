"""Error filters, partition, derive, build's keyword form, and & narrowing."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from expression import Error, Ok

from fp_ops import (
    Environment,
    EnvKey,
    Operation,
    Partitioned,
    _,
    build,
    constant,
    default_on_error,
    derive,
    environment_operation_template,
    exponential,
    fail,
    from_environment,
    operation,
    operation_template,
    partition,
    recover,
    retry,
    traverse_all_settled,
)


class Missing(LookupError):
    pass


def counting_failure(errors: list[Exception]) -> tuple[Operation[object, str], list[int]]:
    calls: list[int] = []

    def attempt(_: object) -> str:
        calls.append(len(calls))
        raise errors[min(len(calls) - 1, len(errors) - 1)]

    return operation(attempt), calls


# error filters --------------------------------------------------------------------


async def test_retry_on_retries_only_matching_errors() -> None:
    op, calls = counting_failure([TimeoutError("slow")])
    outcome = await retry(op, attempts=3, on=TimeoutError).run(None)
    assert isinstance(outcome.error, TimeoutError)
    assert len(calls) == 3

    op, calls = counting_failure([KeyError("fatal")])
    outcome = await retry(op, attempts=3, on=(TimeoutError, ConnectionError)).run(None)
    assert isinstance(outcome.error, KeyError)
    assert len(calls) == 1


async def test_retry_on_predicate_and_predicate_failure() -> None:
    op, calls = counting_failure([ValueError("retry me")])
    retried = retry(op, attempts=2, on=lambda error: "retry" in str(error))
    await retried.run(None)
    assert len(calls) == 2

    def broken(_error: Exception) -> bool:
        raise RuntimeError("predicate")

    op, _calls = counting_failure([ValueError("x")])
    outcome = await retry(op, attempts=2, on=broken).run(None)
    assert isinstance(outcome.error, RuntimeError)


async def test_recover_on_narrows_and_passes_others_through() -> None:
    def lookup(key: str) -> str:
        if key == "missing":
            raise Missing(key)
        raise OSError("browser crashed")

    def handle(error: Missing) -> str:
        return f"default for {error.args[0]}"

    safe = recover(operation(lookup), handle, on=Missing)
    assert (await safe.run("missing")).ok == "default for missing"
    assert isinstance((await safe.run("other")).error, OSError)

    optional = default_on_error(operation(lookup), None, on=Missing)
    assert (await optional.run("missing")).ok is None
    assert isinstance((await optional.run("other")).error, OSError)


def test_error_filter_validation() -> None:
    with pytest.raises(TypeError):
        retry(constant(1), attempts=1, on=())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        retry(constant(1), attempts=1, on="nope")  # type: ignore[arg-type]


def test_exponential_backoff() -> None:
    delay = exponential(0.5, cap=3.0)
    assert [delay(n, RuntimeError()) for n in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 3.0]
    spread = exponential(1.0, jitter=0.5)
    assert 0.5 <= spread(1, RuntimeError()) <= 1.5
    with pytest.raises(ValueError, match="factor"):
        exponential(1.0, factor=0.5)


# partition ------------------------------------------------------------------------


async def test_partition_sequences_and_mappings() -> None:
    def parse(text: str) -> int:
        return int(text)

    settled = await traverse_all_settled(operation(parse), limit=2).run(["1", "x", "3"])
    parts = partition(settled.ok)
    assert isinstance(parts, Partitioned)
    assert parts.values == {0: 1, 2: 3}
    assert list(parts.failures) == [1]
    assert parts.ok == [1, 3]
    assert len(parts.errors) == 1

    keyed = partition({"a": Ok(1), "b": Error(ValueError("b"))})
    assert keyed.values == {"a": 1}
    assert list(keyed.failures) == ["b"]

    with pytest.raises(TypeError):
        partition([1])  # type: ignore[list-item]


# derive -----------------------------------------------------------------------------


@operation_template
def join(base: str, path: str, *, sep: str = "/") -> str:
    return f"{base}{sep}{path}"


@dataclass(frozen=True)
class Page:
    url: str
    next_path: str


async def test_derived_operand_is_read_from_the_input() -> None:
    upper: Operation[str, str] = operation(str.upper)
    to_upper_suffix: Operation[str, str] = join(_, derive(upper), sep="-")
    assert (await to_upper_suffix.run("page")).ok == "page-PAGE"

    hole_second: Operation[str, str] = join(derive(upper), _)
    assert (await hole_second.run("a")).ok == "A/a"

    next_path: Operation[Page, str] = operation(lambda page: page.next_path)

    @operation_template
    def follow(page: Page, path: str) -> str:
        return f"{page.url}/{path}"

    assert (await follow(_, derive(next_path)).run(Page("a", "b"))).ok == "a/b"


async def test_derive_law_and_order_and_errors() -> None:
    plain = join(_, "x")
    derived = join(_, derive(constant("x")))
    assert (await plain.run("a")).ok == (await derived.run("a")).ok

    seen: list[str] = []

    @operation_template
    def three(first: str, second: str, third: str) -> str:
        return first + second + third

    def mark(label: str) -> Operation[str, str]:
        return operation(lambda value: seen.append(label) or label)

    combined = three(derive(mark("1")), _, derive(mark("3")))
    assert (await combined.run("2")).ok == "123"
    assert seen == ["1", "3"]

    boom = KeyError("operand")
    assert (await join(_, derive(fail(boom))).run("a")).error is boom


async def test_derived_operands_rerun_on_retry_and_carry_requirements() -> None:
    count = EnvKey("count", int)
    reads: list[int] = []

    def read(value: str) -> str:
        reads.append(1)
        if len(reads) < 2:
            raise TimeoutError
        return value

    retried = retry(join(_, derive(operation(read))), attempts=2)
    assert (await retried.run("a")).ok == "a/a"
    assert len(reads) == 2

    needs = join(_, derive(from_environment(count).map(str)))
    assert needs.required_environment == (count,)
    environment = Environment().with_value(count, 5)
    assert (await needs.run("n", environment=environment)).ok == "n/5"


async def test_environment_template_with_derived_operand() -> None:
    prefix = EnvKey("prefix", str)

    @environment_operation_template(prefix)
    def label(value: str, suffix: str, dependency: str) -> str:
        return f"{dependency}{value}{suffix}"

    op = label(_, derive(operation(str.upper)))
    assert set(op.required_environment) == {prefix}
    environment = Environment().with_value(prefix, ">")
    assert (await op.run("a", environment=environment)).ok == ">aA"


# build keyword form / & narrowing ---------------------------------------------------------


@dataclass(frozen=True)
class Record:
    name: str
    size: int
    source: str
    note: None


async def test_build_keyword_form_with_scalars() -> None:
    record = build(
        Record,
        name=operation(lambda text: text.strip()),
        size=operation(len),
        source="web",
        note=None,
    )
    assert (await record.run(" ab ")).ok == Record("ab", 4, "web", None)
    as_dict = build({"name": operation(str.strip), "source": "web"})
    assert (await as_dict.run(" x ")).ok == {"name": "x", "source": "web"}
    with pytest.raises(TypeError):
        build(Record, Record)  # type: ignore[call-overload]


async def test_constant_fans_out_with_any_input() -> None:
    pair = constant("fixed") & operation(lambda value: value + 1)
    assert (await pair.run(1)).ok == ("fixed", 2)
