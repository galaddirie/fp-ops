from __future__ import annotations
import asyncio, inspect, time
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar, Union

from expression import Result

from fp_ops.operator import Operation, operation
from fp_ops.context import BaseContext
from fp_ops.placeholder import _

S = TypeVar("S")
T = TypeVar("T")

def fail(exc: Union[str, Exception, Type[Exception]]) -> Operation[Any, Any, None]:
    """
    `fail(ValueError("boom"))` -> op that *always* returns `Result.Error`.
    `fail(ValueError)`        -> same, but instantiates on each call.
    """
    if inspect.isclass(exc) and issubclass(exc, Exception):
        def _make() -> Exception:                    # type: ignore[return-type]
            return exc()                             # type: ignore[misc]
    elif isinstance(exc, Exception):
        def _make() -> Exception:                    # type: ignore[return-type]
            return exc
    else:
        def _make() -> Exception:                    # type: ignore[return-type]
            return Exception(exc)

    @operation
    async def _always_error(*_a, **_kw):
        raise _make()

    return _always_error

def attempt(
    risky_fn: Callable[..., S],
    *,
    context: bool = False,
    context_type: Type[BaseContext] | None = None,
) -> Operation[Any, S, None]:
    """
    Like the old helper: make an Operation out of **any** sync/async callable
    and turn raised exceptions into `Result.Error`.
    """

    return operation(risky_fn, context=context, context_type=context_type)

def retry(op: Operation, *, max_retries: int = 3, delay: float = 0.1) -> Operation:
    """Just sugar for `op.retry(...)` to keep legacy code unchanged."""
    return op.retry(attempts=max_retries, delay=delay)

def tap(
    op: Operation,
    side_effect: Callable[[Any], Any],
    *,
    context: bool = False,
    context_type: Type[BaseContext] | None = None,
) -> Operation:
    """
    Attach `side_effect` to `op` and forward the original value.
    Works with sync / async side-effects.
    """
    setattr(side_effect, "requires_context", context)
    setattr(side_effect, "context_type",   context_type)
    return op.tap(side_effect)

def branch(
    condition: Union[Callable[..., bool], Operation[Any, bool, Any]],
    true_op: Operation,
    false_op: Operation,
) -> Operation[Any, Any, Any]:
    """
    Evaluate *condition* and run `true_op` or `false_op`.
    `condition` may be a plain callable or an Operation that returns bool.
    """
    cond_op: Operation[Any, bool, Any]
    if isinstance(condition, Operation):
        cond_op = condition
    else:
        cond_op = operation(condition)

    def _choose(flag: bool) -> Operation:
        return true_op if flag else false_op

    return cond_op.bind(_choose)

def loop_until(
    predicate: Callable[[T], bool],
    body: Operation[Any, T, Any],
    *,
    max_iterations: int = 10,
    delay: float = 0.1,
) -> Operation[Any, T, Any]:
    """
    Repeatedly feed the *current* value into `body` until `predicate(value)` is
    True or `max_iterations` is reached.
    """

    @operation
    async def _looper(start: T, *args, **kw) -> T:
        value: T = start
        for _ in range(max_iterations):
            if await _safe_pred(predicate, value):
                return value
            res = await body.execute(value, *args, **kw)
            if res.is_error():
                raise res.error
            value = res.default_value(value)
            await asyncio.sleep(delay)
        return value

    return _looper

def wait(
    op: Operation,
    *,
    timeout: float = 10.0,
    delay: float = 0.1,
) -> Operation:
    """
    Execute `op` repeatedly until it returns `Result.Ok` or the timeout expires.
    """

    @operation
    async def _waiter(*args, **kw):
        start = time.perf_counter()
        last_err: Exception | None = None
        while time.perf_counter() - start < timeout:
            res = await op.execute(*args, **kw)
            if res.is_ok():
                return res.default_value(None)
            last_err = res.error
            await asyncio.sleep(delay)
        raise last_err or TimeoutError(f"wait(): timed-out after {timeout}s")

    return _waiter

async def _safe_pred(pred: Callable[[T], bool], value: T) -> bool:
    """Await predicate if it's async, otherwise run it in a thread."""
    if inspect.iscoroutinefunction(pred):
        return await pred(value)
    return await asyncio.to_thread(pred, value)
