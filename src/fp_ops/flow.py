from __future__ import annotations
import asyncio, inspect, time
from typing import (
    Any,
    Callable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    ParamSpec,
    Awaitable,
    Concatenate,
    NoReturn,
    cast,
)

from expression import Result

from fp_ops.operator import Operation, operation, P, R, S, Q
from fp_ops.context import BaseContext
from fp_ops.placeholder import _

T = TypeVar("T")

def fail(exc: Union[str, Exception, Type[Exception]]) -> Operation[Any, None]:
    """
    `fail(ValueError("boom"))` -> op that *always* returns `Result.Error`.
    `fail(ValueError)`        -> same, but instantiates on each call.
    """
    if inspect.isclass(exc) and issubclass(exc, Exception):
        def _make() -> Exception:
            return exc()
    elif isinstance(exc, Exception):
        def _make() -> Exception:
            return exc
    else:
        def _make() -> Exception:
            return Exception(exc)

    @operation
    async def _always_error(*_a: Any, **_kw: Any) -> NoReturn:
        raise _make()

    # Statically we advertise "returns None" to callers even though it
    # actually never returns.
    return cast(Operation[Any, None], _always_error)

def attempt(
    risky_fn: Callable[P, Awaitable[S]] | Callable[P, S],
    *,
    context: bool = False,
    context_type: Type[BaseContext] | None = None,
) -> Operation[P, S]:
    """
    Wrap any synchronous or asynchronous callable as an Operation that
    automatically catches exceptions and returns them as `Result.Error`.

    This is a convenience for quickly making any function "safe" in the
    functional pipeline, so that exceptions are not raised but instead
    are captured as error results.

    Args:
        risky_fn: The function to wrap. Can be sync or async.
        context: If True, the operation will expect and pass a context argument.
        context_type: The type of context to expect (if any).

    Returns:
        Operation[P, S]: An Operation that returns Result.Ok on success,
        or Result.Error if the function raises an exception.

    Example:
        >>> @attempt
        ... async def might_fail(x: int) -> int:
        ...     if x < 0:
        ...         raise ValueError("Negative!")
        ...     return x * 2
        >>> await might_fail(2)
        Result.Ok(4)
        >>> await might_fail(-1)
        Result.Error(ValueError("Negative!"))
    """


    if context or context_type:
        return cast(Operation[P, S],
                    operation(context=context, context_type=context_type)(risky_fn))
    # fast-path no context awareness requested
    return operation(risky_fn)

def retry(op: Operation[P, R], *, max_retries: int = 3, delay: float = 0.1, backoff: float = 1.0) -> Operation[P, R]:
    """
    Retry the operation a specified number of times on failure.

    Args:
        attempts: Maximum number of attempts.
        delay: Initial delay between attempts in seconds.
        backoff: Multiplier for the delay after each retry.

    Returns:
        A new Operation that implements retry logic.
        """
    return op.retry(attempts=max_retries, delay=delay, backoff=backoff)

def tap(
    op: Operation[P, R],
    side_effect: Callable[..., Any],
) -> Operation[P, R]:
    """
    Attach `side_effect` to `op` and forward the original value.
    Works with sync / async side-effects.
    """
    return op.tap(side_effect)

def branch(
    condition: Union[Callable[[R], bool], Operation[P, bool]],
    true_op: Operation[Concatenate[R, Q], S],
    false_op: Operation[Concatenate[R, Q], S],
) -> Operation[P, S]:
    """
    Evaluate *condition* and run `true_op` or `false_op`.
    `condition` may be a plain callable or an Operation that returns bool.
    """
    _UNSET = object()
    cond_op: Operation[Any, bool] = (
        condition  # already an Operation
        if isinstance(condition, Operation)
        else operation(cast(Callable[..., bool], condition))
    )

    async def _branch(value: Any = _UNSET, *args: Any, **kwargs: Any) -> S:
        has_val = value is not _UNSET
        
        if isinstance(condition, Operation):
            cond_res: Result[bool, Exception] = await (
                cond_op.execute(value, **kwargs) if has_val and not cond_op.is_bound
                else cond_op.execute(**kwargs)
            )
            if cond_res.is_error():
                raise cond_res.error
            flag: bool = cond_res.default_value(False)
        else:
            try:
                maybe_awaitable = (
                    condition(value, **kwargs) if has_val
                    else condition(*args, **kwargs)
                )
                flag = await maybe_awaitable if inspect.isawaitable(maybe_awaitable) else bool(maybe_awaitable)
            except Exception as exc:
                raise exc

        chosen = true_op if flag else false_op

        branch_res: Result[S, Exception] = await (
            chosen.execute(value, *args, **kwargs) if has_val and not chosen.is_bound
            else chosen.execute(*args, **kwargs)
        )

        if branch_res.is_error():
            raise branch_res.error

        return branch_res.default_value(cast(Any, None))

    ctx_type = next((op.context_type for op in (cond_op, true_op, false_op) if op.context_type), None)
    return Operation._from_function(_branch,
                                    require_ctx=ctx_type is not None,
                                    ctx_type=ctx_type)

def loop_until(
    predicate: Callable[[T], bool],
    body: Operation[Concatenate[T, P], T],
    *,
    max_iterations: int = 10,
    delay: float = 0.1,
    context: bool = False,
    context_type: Type[BaseContext] | None = None,
) -> Operation[Concatenate[T, P], T]:
    """Repeat *body* until *predicate* is satisfied or the iteration limit is hit."""

    async def _eval_pred(val: T, *, ctx: BaseContext | None) -> bool:
        if isinstance(predicate, Operation):
            res = await (predicate.execute(val, context=ctx) if not predicate.is_bound
                         else predicate.execute(context=ctx))
            if res.is_error():
                raise res.error
            return res.default_value(False)

        out = (
            cast(Any, predicate)(val, context=ctx)            # want ctx
            if getattr(predicate, "requires_context", False)
            else predicate(val)
        )
        return await out if inspect.isawaitable(out) else out

    async def _looper(
        current: T, *extra_args: P.args, **extra_kw: P.kwargs
    ) -> T:
        ctx = cast(BaseContext | None, extra_kw.get("context"))
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

    ctx_t = context_type or body.context_type
    return Operation._from_function(
        cast(Callable[..., Awaitable[T]], _looper),
        require_ctx=context or (ctx_t is not None),
        ctx_type=ctx_t)

def wait(
    op: Operation[P, R],
    *,
    timeout: float = 10.0,
    delay: float = 0.1,
) -> Operation[P, R]:
    """
    Execute `op` repeatedly until it returns `Result.Ok` or the timeout expires.
    """

    async def _waiter(*args: P.args, **kw: P.kwargs) -> R:          # helper
        start = time.perf_counter()
        last_err: Exception | None = None
        while time.perf_counter() - start < timeout:
            res = await op.execute(*args, **kw)
            if res.is_ok():
                return cast(R, res.default_value(cast(Any, None)))
            last_err = res.error
            await asyncio.sleep(delay)
        raise last_err or TimeoutError(f"wait(): timed-out after {timeout}s")

    return Operation._from_function(
        cast(Callable[..., Awaitable[R]], _waiter)
    )
Predicate = Callable[[T], bool] | Callable[[T], Awaitable[bool]]

async def _safe_pred(pred: Predicate, value: T) -> bool:
    """Await predicate if it's async; otherwise run it in a thread."""
    if inspect.iscoroutinefunction(pred):
        # mypy still sees Any here, so cast once after awaiting
        return cast(bool, await pred(value))
    return cast(bool, await asyncio.to_thread(pred, value))
