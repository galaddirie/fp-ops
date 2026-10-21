from __future__ import annotations

from expression import Ok, Result

from fp_ops import (
    Environment,
    EnvKey,
    Operation,
    _,
    build,
    operation,
    operation_template,
    result_operation,
)

to_text: Operation[int, str] = operation(lambda value: str(value))
needs_bytes: Operation[bytes, float] = operation(lambda value: float(value))

# incompatible stage input
invalid_pipeline = to_text.then(needs_bytes)

# missing unary run input
missing_input = to_text.run()


def raw_result(value: int) -> Result[int, Exception]:
    return Ok(value)


def nested_result(value: int) -> Result[Result[int, Exception], Exception]:
    return Ok(Ok(value))


# raw constructors do not flatten Result
raw_nested = operation(raw_result)
nested: Operation[int, int] = raw_nested

# result constructors flatten only the one Result they declare
one_level = result_operation(nested_result)
double_nested: Operation[int, int] = one_level


class Settings:
    pass


SETTINGS = EnvKey("settings", Settings)

# typed environment keys reject incompatible values
wrong_environment = Environment().with_value(SETTINGS, "wrong")


@operation_template
def configured(prefix: str, value: int) -> str:
    return f"{prefix}{value}"


# templates require exactly one correctly positioned placeholder
no_placeholder = configured("value=", 1)
two_placeholders = configured(_, _)
wrong_configuration = configured(1, _)


class MixedBuild:
    def __init__(self, text: str, number: float) -> None:
        self.text = text
        self.number = number


# build fields must consume one shared input type
invalid_build = build(
    {"text": to_text, "number": needs_bytes},
    MixedBuild,
)
