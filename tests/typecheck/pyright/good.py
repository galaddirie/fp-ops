from __future__ import annotations

from typing import Awaitable

from expression import Ok, Result
from typing_extensions import assert_type

from fp_ops import (
    EnvKey,
    Environment,
    Operation,
    constant,
    environment_operation,
    fanout_parallel,
    operation,
    pipe_all,
    result_operation,
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


def deferred(value: int) -> Awaitable[str]:
    async def finish() -> str:
        return str(value)

    return finish()


assert_type(parse, Operation[str, int])
assert_type(operation(deferred), Operation[int, str])
assert_type(parse.then(double), Operation[str, int])
assert_type(parse.map(str), Operation[str, str])
assert_type(pipe_all((parse, double)), Operation[str, int])
assert_type(
    fanout_parallel(parse, uppercase),
    Operation[str, tuple[int, str]],
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
