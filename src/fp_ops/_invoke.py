"""Internal callable adapters and the one async normalizer."""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast

from expression import Result

from fp_ops.environment import EnvKey, Environment


async def invoke(fn: Callable[..., Any], *args: Any) -> Any:
    """Invoke ``fn`` and recursively await until a non-awaitable value appears."""

    value = fn(*args)
    while inspect.isawaitable(value):
        value = await value
    return value


@dataclass(frozen=True, slots=True)
class Invocation:
    fn: Callable[..., Any]
    output: Literal["raw", "result"] = "raw"
    environment_key: EnvKey[Any] | None = None
    threaded: bool = False

    async def call(self, value: Any, environment: Environment) -> Any:
        args: tuple[Any, ...]
        if self.environment_key is None:
            args = (value,)
        else:
            args = (value, environment.require(self.environment_key))

        if self.threaded:
            returned = await asyncio.to_thread(self.fn, *args)
            while inspect.isawaitable(returned):
                returned = await returned
        else:
            returned = await invoke(self.fn, *args)

        if self.output == "result":
            if not isinstance(returned, Result):
                raise TypeError(
                    "result_operation callable returned "
                    f"{type(returned).__name__}, expected expression.Result"
                )
            result = cast(Result[Any, Any], returned)
            if result.is_error():
                error = result.error
                if not isinstance(error, Exception):
                    raise TypeError(
                        "result_operation error must be an Exception, "
                        f"got {type(error).__name__}"
                    )
                raise error
            return result.ok

        return returned
