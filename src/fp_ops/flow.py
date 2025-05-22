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

def retry(op: Operation, *, max_retries: int = 3, delay: float = 0.1, backoff: float = 1.0) -> Operation:
    """Just sugar for `op.retry(...)` to keep legacy code unchanged."""
    return op.retry(attempts=max_retries, delay=delay, backoff=backoff)

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
    async def _tap_wrapper(x, *, context=None):
        try:
            maybe = (side_effect(x, context=context)            # ctx injected if declared
                     if context
                     else side_effect(x))
            if inspect.isawaitable(maybe):
                await maybe
        except Exception:
            pass          # side-effect exceptions are ignored
        return x

    return op >> Operation._from_function(
        _tap_wrapper,
        require_ctx=context,
        ctx_type=context_type,
    )

def branch(
    condition: Union[Callable[..., bool], Operation[Any, bool, Any]],
    true_op: Operation,
    false_op: Operation,
) -> Operation[Any, Any, Any]:
    """
    Evaluate *condition* and run `true_op` or `false_op`.
    `condition` may be a plain callable or an Operation that returns bool.
    """
    _UNSET = object()
    cond_op = condition if isinstance(condition, Operation) else operation(condition)

    async def _branch(value=_UNSET, *args, **kwargs):
        has_val = value is not _UNSET
        # --- evaluate condition -------------------------------------------
        if isinstance(condition, Operation):
            res = await (
                cond_op.execute(value, **kwargs) if has_val and not cond_op.is_bound
                else cond_op.execute(**kwargs)
            )
            if res.is_error():
                raise res.error
            flag = res.default_value(False)
        else:                                           # plain callable
            try:
                out = (
                    condition(value, **kwargs) if has_val
                    else condition(**kwargs)
                )        # may be async
                flag = await out if inspect.isawaitable(out) else out
            except Exception as exc:
                raise exc

        chosen = true_op if flag else false_op
        return await (
            chosen.execute(value, **kwargs) if has_val and not chosen.is_bound
            else chosen.execute(**kwargs)
        )

    # ---- context metadata (highest-order wins) ----------------------------
    ctx_type = next((op.context_type for op in (cond_op, true_op, false_op) if op.context_type), None)
    return Operation._from_function(_branch,
                                    require_ctx=ctx_type is not None,
                                    ctx_type=ctx_type)

def loop_until(
    predicate: Callable[..., bool],
    body: Operation[Any, T, Any],
    *,
    max_iterations: int = 10,
    delay: float = 0.1,
    context: bool = False,
    context_type: Type[BaseContext] | None = None,
) -> Operation[Any, T, Any]:
    """Repeat *body* until *predicate* is satisfied or the iteration limit is hit."""

    async def _eval_pred(val, *, ctx):
        if isinstance(predicate, Operation):
            res = await (predicate.execute(val, context=ctx) if not predicate.is_bound
                         else predicate.execute(context=ctx))
            if res.is_error():
                raise res.error
            return res.default_value(False)

        out = (predicate(val, context=ctx)                    # may or may not want ctx
               if getattr(predicate, "requires_context", False)
               else predicate(val))
        return await out if inspect.isawaitable(out) else out

    async def _looper(current, *extra_args, **extra_kw):
        ctx = extra_kw.get("context")
        for _ in range(max_iterations):
            if await _eval_pred(current, ctx=ctx):
                return current

            res = await body.execute(current, *extra_args, **extra_kw)
            if res.is_error():
                raise res.error
            current = res.default_value(current)
            if delay:
                await asyncio.sleep(delay)
        return current

    # Prefer an explicit context_type argument, otherwise inherit
    ctx_t = context_type or body.context_type
    return Operation._from_function(_looper,
                                    require_ctx=context or (ctx_t is not None),
                                    ctx_type=ctx_t)

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
