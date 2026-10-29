from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Awaitable

from expression import Ok, Result
from typing_extensions import assert_type

from fp_ops import (
    UNIT,
    CollectionOperation,
    Environment,
    EnvKey,
    Fanout,
    Operation,
    Unit,
    _,
    build,
    constant,
    environment_operation,
    environment_operation_template,
    fanout_parallel,
    filter_best_effort,
    filter_each,
    filter_operation,
    operation,
    operation_template,
    pipe_all,
    result_operation,
    result_operation_template,
    threaded_operation_template,
    traverse,
)


@operation
async def parse(value: str) -> int:
    return int(value)


@operation
def double(value: int) -> int:
    return value * 2


@operation
def render(value: int) -> str:
    return str(value)


@operation
def uppercase(value: str) -> str:
    return value.upper()


@operation
def no_value(value: str) -> None:
    return None


def deferred(value: int) -> Awaitable[str]:
    async def finish() -> str:
        return str(value)

    return finish()


assert_type(parse, Operation[str, int])
assert_type(parse | no_value, Operation[str, int | None])
assert_type(operation(deferred), Operation[int, str])
assert_type(parse.then(double), Operation[str, int])
assert_type(parse.map(str), Operation[str, str])
assert_type(pipe_all((parse, double, render)), Operation[str, str])
assert_type(
    fanout_parallel(parse, uppercase),
    Operation[str, tuple[int, str]],
)
assert_type(parse & uppercase, Fanout[str, int, str])
assert_type(parse & uppercase & parse, Fanout[str, int, str, int])
assert_type(parse & (uppercase & parse), Fanout[str, int, tuple[str, int]])
flat_fanout: Operation[str, tuple[int, str, int]] = parse & uppercase & parse


class Summary:
    def __init__(self, number: int, upper: str) -> None:
        self.number = number
        self.upper = upper


assert_type(
    build({"number": parse, "upper": uppercase}, Summary),
    Operation[str, Summary],
)


def raw_result(value: int) -> Result[int, Exception]:
    return Ok(value)


def checked_result(value: int) -> Result[int, Exception]:
    return Ok(value)


assert_type(
    operation(raw_result),
    Operation[int, Result[int, Exception]],
)
assert_type(
    result_operation(checked_result),
    Operation[int, int],
)
assert_type(
    parse.flat_map(lambda value: constant(str(value))),
    Operation[str, str],
)
assert_type(constant(1), Operation[object, int])
assert_type(UNIT, Unit)


class Logger:
    def write(self, value: str) -> int:
        return len(value)


LOGGER = EnvKey("logger", Logger)


@environment_operation(LOGGER)
def write(value: str, logger: Logger) -> int:
    return logger.write(value)


assert_type(write, Operation[str, int])
environment = Environment().with_value(LOGGER, Logger())
assert_type(environment.require(LOGGER), Logger)


@operation_template
def format_value(
    prefix: str,
    value: int,
    suffix: bytes,
    *,
    scale: float = 1.0,
) -> str:
    return f"{prefix}{value * scale:g}{suffix.decode()}"


assert_type(format_value(_, 1, b"!"), Operation[str, str])
assert_type(format_value("$", _, b"!", scale=2.0), Operation[int, str])
assert_type(format_value("$", 1, _), Operation[bytes, str])


@threaded_operation_template
def blocking_format(prefix: str, value: int) -> str:
    return f"{prefix}{value}"


assert_type(blocking_format("#", _), Operation[int, str])


@result_operation_template
def checked(minimum: int, value: int) -> Result[int, Exception]:
    return Ok(value) if value >= minimum else Ok(minimum)


assert_type(checked(0, _), Operation[int, int])


@operation_template
def summarize(prefix: str, values: list[str]) -> str:
    return prefix + ", ".join(values)


@operation_template
def summarize_mapping(prefix: str, values: dict[str, str]) -> str:
    return prefix + ", ".join(values.values())


titles = traverse(uppercase)
assert_type(titles, CollectionOperation[str, str])
assert_type(
    titles >> summarize("got: ", _),
    Operation[Iterable[str], str],
)
assert_type(
    titles >> summarize_mapping("got: ", _),
    Operation[Mapping[str, str], str],
)


def positive(value: int) -> bool:
    return value > 0


@operation
def numbers(value: str) -> list[int]:
    return [len(value)]


@operation
def keyed_numbers(value: str) -> dict[str, int]:
    return {value: len(value)}


@operation
def count_list(values: list[int]) -> int:
    return len(values)


@operation
def count_mapping(values: dict[str, int]) -> int:
    return len(values)


selected = filter_each(positive)
assert_type(selected, CollectionOperation[int, int])
assert_type(
    filter_operation(operation(positive)),
    CollectionOperation[int, int],
)
assert_type(
    filter_best_effort(operation(positive)),
    CollectionOperation[int, int],
)
assert_type(numbers >> selected, Operation[str, list[int]])
assert_type(numbers.then(selected), Operation[str, list[int]])
assert_type(numbers >> selected >> count_list, Operation[str, int])
assert_type(keyed_numbers >> selected, Operation[str, dict[str, int]])
assert_type(keyed_numbers.then(selected), Operation[str, dict[str, int]])
assert_type(keyed_numbers >> selected >> count_mapping, Operation[str, int])


async def check_filter_run_shapes() -> None:
    assert_type(
        await selected.run([1, 2]),
        Result[list[int], Exception],
    )
    assert_type(
        await selected.run({"one": 1}),
        Result[dict[str, int], Exception],
    )


@environment_operation_template(LOGGER)
def write_configured(
    prefix: str,
    value: str,
    logger: Logger,
    *,
    repeat: int = 1,
) -> int:
    return logger.write((prefix + value) * repeat)


assert_type(write_configured(_, "value"), Operation[str, int])
assert_type(write_configured("prefix", _, repeat=2), Operation[str, int])


# --- 0.5: control flow, scoped environment, derive, partition, filters ---------

from fp_ops import (  # noqa: E402
    Continue,
    Partitioned,
    Stop,
    bracket,
    branch,
    continue_if,
    default_on_error,
    derive,
    exponential,
    from_environment,
    loop,
    partition,
    passthrough,
    provide,
    recover,
    retry,
    sleep,
    traverse_all_settled,
)


class Cursor:
    def __init__(self, page: int) -> None:
        self.page = page


def next_cursor(cursor: Cursor) -> Continue[Cursor] | Stop[Cursor]:
    return Continue(Cursor(cursor.page + 1)) if cursor.page < 3 else Stop(cursor)


crawl_step: Operation[Cursor, Continue[Cursor] | Stop[Cursor]] = operation(next_cursor)
assert_type(loop(crawl_step, max_steps=10), Operation[Cursor, Cursor])

def is_small(value: int) -> bool:
    return value < 10


def add_one(value: int) -> int:
    return value + 1


def forget(_resource: int) -> None:
    return None


below = operation(is_small)
increment = operation(add_one)
assert_type(loop(continue_if(below, increment)), Operation[int, int])
assert_type(branch(below, increment, constant("big")), Operation[int, int | str])
assert_type(passthrough(increment), Operation[int, int])
paused: Operation[int, int] = sleep(0.5)

LIMIT = EnvKey("limit", int)
scoped_limit = provide(LIMIT, increment, from_environment(LIMIT).map(str))
assert_type(scoped_limit, Operation[int, str])

opened: Operation[str, int] = operation(len)
closed = operation(forget)
assert_type(bracket(opened, increment, closed), Operation[str, int])


def handle_lookup(error: LookupError) -> str:
    return str(error.args)


recovered = recover(parse, handle_lookup, on=LookupError)
assert_type(recovered, Operation[str, int | str])
assert_type(default_on_error(parse, None, on=ValueError), Operation[str, int | None])
assert_type(retry(parse, attempts=3, backoff=exponential(0.5), on=(TimeoutError,)), Operation[str, int])

settled = traverse_all_settled(parse, limit=2)
partitioned = settled.map(partition)
assert_type(partitioned, Operation[Iterable[str], Partitioned[int, int]])


@operation_template
def visit(page: str, url: str, *, retries: int = 1) -> str:
    return f"{page}->{url}"


upper_url: Operation[str, str] = operation(str.upper)
assert_type(visit(_, derive(upper_url)), Operation[str, str])
assert_type(visit(derive(upper_url), _), Operation[str, str])
assert_type(visit(_, derive(constant("fixed")), retries=2), Operation[str, str])


class Pair:
    def __init__(self, left: int, right: str) -> None:
        self.left = left
        self.right = right


assert_type(build(Pair, left=parse, right="literal"), Operation[str, Pair])
assert_type(constant(1) & parse, Fanout[str, int, int])
