from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Awaitable

from expression import Ok, Result
from typing_extensions import assert_type

from fp_ops import (
    CollectionOperation,
    Environment,
    EnvKey,
    Operation,
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
assert_type(pipe_all((parse, double)), Operation[str, int])
assert_type(
    fanout_parallel(parse, uppercase),
    Operation[str, tuple[int, str]],
)


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
assert_type(result_operation(checked_result), Operation[int, int])
assert_type(
    parse.flat_map(lambda value: constant(str(value))),
    Operation[str, str],
)


class Logger:
    pass


LOGGER = EnvKey("logger", Logger)


@environment_operation(LOGGER)
def use_logger(value: str, logger: Logger) -> int:
    return len(value) if logger else 0


assert_type(use_logger, Operation[str, int])
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
def use_configured(
    prefix: str,
    value: str,
    logger: Logger,
    *,
    repeat: int = 1,
) -> int:
    return len((prefix + value) * repeat) if logger else 0


assert_type(use_configured(_, "value"), Operation[str, int])
assert_type(use_configured("prefix", _, repeat=2), Operation[str, int])
