from __future__ import annotations

from expression import Ok, Result

from fp_ops import EnvKey, Environment, Operation, operation, result_operation


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


class Capability:
    pass


CAPABILITY = EnvKey("capability", Capability)

# typed environment keys reject incompatible capability values
wrong_environment = Environment().with_value(CAPABILITY, "wrong")
