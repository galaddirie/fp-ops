"""Named composition, recovery, retry, and fanout combinators."""
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import copy
import random
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    TypeVar,
    Union,
    cast,
    overload,
)

from expression import Result

from fp_ops._ast import (
    AlternativeNode,
    FanoutNode,
    RecoverNode,
    RetryNode,
    TapNode,
)
from fp_ops.core import Operation, Unit, _merge_requirements, identity
from fp_ops.environment import EnvKey

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
T = TypeVar("T")
X = TypeVar("X", bound=Exception)


@overload
def pipe_all(operations: tuple[Operation[A, B]]) -> Operation[A, B]: ...


@overload
def pipe_all(
    operations: tuple[Operation[A, B], Operation[B, C]],
) -> Operation[A, C]: ...


@overload
def pipe_all(
    operations: tuple[Operation[A, B], Operation[B, C], Operation[C, D]],
) -> Operation[A, D]: ...


@overload
def pipe_all(
    operations: tuple[
        Operation[A, B],
        Operation[B, C],
        Operation[C, D],
        Operation[D, E],
    ],
) -> Operation[A, E]: ...


@overload
def pipe_all(operations: Iterable[Operation[T, T]]) -> Operation[T, T]: ...


def pipe_all(
    operations: Iterable[Operation[Any, Any]],
) -> Operation[Any, Any]:
    """Snapshot and compose operations from left to right."""

    snapshot = tuple(operations)
    if not snapshot:
        return identity()
    if any(not isinstance(item, Operation) for item in snapshot):
        raise TypeError("pipe_all() accepts only Operation instances")
    current = snapshot[0]
    for next_operation in snapshot[1:]:
        current = current.then(next_operation)
    return current


ErrorFilter = Union[
    "type[Exception]",
    "tuple[type[Exception], ...]",
    Callable[[Exception], bool],
]
"""Which errors a policy applies to: exception types or a predicate."""


def _error_matcher(on: ErrorFilter) -> Callable[[Exception], bool] | None:
    if isinstance(on, type):
        if not issubclass(on, Exception):
            raise TypeError("on= exception types must subclass Exception")
        if on is Exception:
            return None
        return lambda error: isinstance(error, on)
    if isinstance(on, tuple):
        types = cast("tuple[type[Exception], ...]", on)
        if not types or not all(
            isinstance(item, type) and issubclass(item, Exception) for item in types
        ):
            raise TypeError("on= must be a non-empty tuple of Exception subclasses")
        return lambda error: isinstance(error, types)
    if callable(on):
        predicate = on

        def matches(error: Exception) -> bool:
            decision = predicate(error)
            if not isinstance(decision, bool):
                raise TypeError("on= predicate must return a bool")
            return decision

        return matches
    raise TypeError("on= must be an exception type, a tuple of types, or a predicate")


@overload
def recover(
    op: Operation[A, B],
    handler: Callable[[X], C | Awaitable[C]],
    *,
    on: type[X] | tuple[type[X], ...],
) -> Operation[A, B | C]: ...


@overload
def recover(
    op: Operation[A, B],
    handler: Callable[[Exception], C | Awaitable[C]],
    *,
    on: Callable[[Exception], bool] = ...,
) -> Operation[A, B | C]: ...


def recover(
    op: Operation[A, B],
    handler: Callable[[Any], Any],
    *,
    on: ErrorFilter = Exception,
) -> Operation[A, Any]:
    """Recover an error with a raw value.

    ``on`` limits recovery to matching errors (exception types or a predicate);
    any other error is returned unchanged.
    """

    if not isinstance(op, Operation) or not callable(handler):
        raise TypeError("recover() requires an Operation and a callable")
    return Operation._from_node(
        RecoverNode(op._node, handler, with_operation=False, matches=_error_matcher(on)),
        name=f"recover({op.name})",
        requirements=op._requirements,
    )


def recover_with(
    op: Operation[A, B],
    handler: Callable[
        [Exception],
        Operation[Unit, C] | Awaitable[Operation[Unit, C]],
    ],
    *,
    on: ErrorFilter = Exception,
    requires: Iterable[EnvKey[Any]] = (),
) -> Operation[A, B | C]:
    """Recover an error by running a returned nullary operation.

    ``requires`` declares environment keys the returned operation needs, so
    they are checked before the run starts.
    """

    if not isinstance(op, Operation) or not callable(handler):
        raise TypeError("recover_with() requires an Operation and a callable")
    declared = _declared(requires)
    return cast(
        Operation[A, B | C],
        Operation._from_node(
            RecoverNode(
                op._node,
                handler,
                with_operation=True,
                matches=_error_matcher(on),
                declared=declared,
            ),
            name=f"recover_with({op.name})",
            requirements=_merge_requirements(op._requirements, declared),
        ),
    )


def default_on_error(
    op: Operation[A, B],
    value: C,
    *,
    on: ErrorFilter = Exception,
) -> Operation[A, B | C]:
    """Recover matching errors with a construction-time default value."""

    try:
        snapshot = copy.deepcopy(value)
    except Exception:
        snapshot = value
    return recover(op, lambda _error: snapshot, on=cast(Callable[[Exception], bool], on))


def or_else(
    first: Operation[A, B],
    second: Operation[A, C],
) -> Operation[A, B | C]:
    """Run ``second`` on the original input only if ``first`` fails."""

    if not isinstance(first, Operation) or not isinstance(second, Operation):
        raise TypeError("or_else() requires two Operation instances")
    return cast(
        Operation[A, B | C],
        Operation._from_node(
            AlternativeNode((first._node, second._node)),
            name=f"or_else({first.name}, {second.name})",
            requirements=_merge_requirements(
                first._requirements, second._requirements
            ),
        ),
    )


def fallback(*operations: Operation[A, B]) -> Operation[A, B]:
    """Try a snapshotted sequence of operations until one succeeds."""

    snapshot = tuple(operations)
    if not snapshot:
        raise ValueError("fallback() requires at least one operation")
    if any(not isinstance(item, Operation) for item in snapshot):
        raise TypeError("fallback() accepts only Operation instances")
    requirements = _merge_requirements(
        *(item._requirements for item in snapshot)
    )
    return cast(
        Operation[A, B],
        Operation._from_node(
            AlternativeNode(tuple(item._node for item in snapshot)),
            name="fallback",
            requirements=requirements,
        ),
    )


def tap(
    op: Operation[A, B],
    effect: Callable[[B], object | Awaitable[object]],
) -> Operation[A, B]:
    """Run an effect after success and propagate an effect failure."""

    return _tap(op, effect, best_effort=False)


def tap_best_effort(
    op: Operation[A, B],
    effect: Callable[[B], object | Awaitable[object]],
) -> Operation[A, B]:
    """Run an effect after success and deliberately ignore its exceptions."""

    return _tap(op, effect, best_effort=True)


def _tap(
    op: Operation[A, B],
    effect: Callable[[B], object | Awaitable[object]],
    *,
    best_effort: bool,
) -> Operation[A, B]:
    if not isinstance(op, Operation) or not callable(effect):
        raise TypeError("tap combinators require an Operation and a callable")
    return cast(
        Operation[A, B],
        Operation._from_node(
            TapNode(op._node, effect, best_effort=best_effort),
            name=f"{'tap_best_effort' if best_effort else 'tap'}({op.name})",
            requirements=op._requirements,
        ),
    )


Backoff = float | Callable[[int, Exception], float | Awaitable[float]]


def retry(
    op: Operation[A, B],
    *,
    attempts: int,
    backoff: Backoff = 0.0,
    on: ErrorFilter = Exception,
) -> Operation[A, B]:
    """Retry failures; ``attempts`` includes the first invocation.

    ``on`` limits retries to matching errors; any other error is returned at
    once, unchanged.
    """

    if not isinstance(op, Operation):
        raise TypeError("retry() requires an Operation")
    if isinstance(attempts, bool) or attempts < 1:
        raise ValueError("attempts must be at least 1")
    if isinstance(backoff, (int, float)):
        if backoff < 0:
            raise ValueError("backoff must not be negative")
        normalized_backoff: Backoff = float(backoff)
    elif callable(backoff):
        normalized_backoff = backoff
    else:
        raise TypeError("backoff must be a non-negative number or callable")

    return cast(
        Operation[A, B],
        Operation._from_node(
            RetryNode(op._node, attempts, normalized_backoff, _error_matcher(on)),
            name=f"retry({op.name})",
            requirements=op._requirements,
        ),
    )


def exponential(
    base: float,
    *,
    factor: float = 2.0,
    cap: float = 60.0,
    jitter: float = 0.0,
) -> Callable[[int, Exception], float]:
    """A ``retry`` backoff: ``base * factor ** (attempt - 1)`` seconds, capped,
    with up to ``jitter`` (a fraction, e.g. ``0.1``) of random spread."""

    if base < 0 or factor < 1 or cap < 0 or not 0 <= jitter <= 1:
        raise ValueError("exponential() requires base >= 0, factor >= 1, cap >= 0, 0 <= jitter <= 1")

    def delay(attempt: int, _error: Exception) -> float:
        wait = min(cap, base * factor ** (attempt - 1))
        return wait * (1 + random.uniform(-jitter, jitter)) if jitter else wait  # noqa: S311

    return delay


def _declared(keys: Iterable[EnvKey[Any]]) -> tuple[EnvKey[object], ...]:
    snapshot = tuple(keys)
    if any(not isinstance(key, EnvKey) for key in snapshot):
        raise TypeError("requires= accepts only EnvKey instances")
    return _merge_requirements(cast(tuple[EnvKey[object], ...], snapshot))


def _fanout(
    operations: tuple[Operation[A, object], ...],
    *,
    settled: bool,
    limit: int | None,
) -> Operation[A, object]:
    if any(not isinstance(item, Operation) for item in operations):
        raise TypeError("fanout accepts only Operation instances")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be at least 1")
    requirements = _merge_requirements(
        *(item._requirements for item in operations)
    )
    return Operation._from_node(
        FanoutNode(
            tuple((None, item._node) for item in operations),
            mode="all_settled" if settled else "fail_fast",
            named=False,
            limit=limit,
        ),
        name="fanout_all_settled" if settled else "fanout_parallel",
        requirements=requirements,
    )


@overload
def fanout_parallel(
    op1: Operation[A, B],
    /,
    *,
    limit: int | None = None,
) -> Operation[A, tuple[B]]: ...


@overload
def fanout_parallel(
    op1: Operation[A, B],
    op2: Operation[A, C],
    /,
    *,
    limit: int | None = None,
) -> Operation[A, tuple[B, C]]: ...


@overload
def fanout_parallel(
    op1: Operation[A, B],
    op2: Operation[A, C],
    op3: Operation[A, D],
    /,
    *,
    limit: int | None = None,
) -> Operation[A, tuple[B, C, D]]: ...


@overload
def fanout_parallel(
    *operations: Operation[A, T],
    limit: int | None = None,
) -> Operation[A, tuple[T, ...]]: ...


def fanout_parallel(
    *operations: Operation[Any, Any],
    limit: int | None = None,
) -> Operation[Any, tuple[Any, ...]]:
    """Run same-input branches concurrently, failing and cancelling early."""

    return cast(
        Operation[Any, tuple[Any, ...]],
        _fanout(tuple(operations), settled=False, limit=limit),
    )


@overload
def fanout_all_settled(
    op1: Operation[A, B],
    /,
    *,
    limit: int | None = None,
) -> Operation[A, tuple[Result[B, Exception]]]: ...


@overload
def fanout_all_settled(
    op1: Operation[A, B],
    op2: Operation[A, C],
    /,
    *,
    limit: int | None = None,
) -> Operation[
    A,
    tuple[Result[B, Exception], Result[C, Exception]],
]: ...


@overload
def fanout_all_settled(
    *operations: Operation[A, T],
    limit: int | None = None,
) -> Operation[A, tuple[Result[T, Exception], ...]]: ...


def fanout_all_settled(
    *operations: Operation[Any, Any],
    limit: int | None = None,
) -> Operation[Any, tuple[Result[Any, Exception], ...]]:
    """Run every same-input branch and retain each branch ``Result``."""

    return cast(
        Operation[Any, tuple[Result[Any, Exception], ...]],
        _fanout(tuple(operations), settled=True, limit=limit),
    )


def collect_named(
    **operations: Operation[A, object],
) -> Operation[A, dict[str, object]]:
    """Run named same-input branches concurrently and fail fast."""

    snapshot = tuple(operations.items())
    if any(not isinstance(item, Operation) for _, item in snapshot):
        raise TypeError("collect_named() accepts only Operation values")
    requirements = _merge_requirements(
        *(item._requirements for _, item in snapshot)
    )
    return cast(
        Operation[A, dict[str, object]],
        Operation._from_node(
            FanoutNode(
                tuple((name, item._node) for name, item in snapshot),
                mode="fail_fast",
                named=True,
                limit=None,
            ),
            name="collect_named",
            requirements=requirements,
        ),
    )


__all__ = [
    "Backoff",
    "ErrorFilter",
    "collect_named",
    "default_on_error",
    "exponential",
    "fallback",
    "fanout_all_settled",
    "fanout_parallel",
    "or_else",
    "pipe_all",
    "recover",
    "recover_with",
    "retry",
    "tap",
    "tap_best_effort",
]
