"""Named composition, recovery, retry, and fanout combinators."""
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import copy
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    TypeVar,
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

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
T = TypeVar("T")


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


def recover(
    op: Operation[A, B],
    handler: Callable[[Exception], C | Awaitable[C]],
) -> Operation[A, B | C]:
    """Recover an error with a raw value."""

    if not isinstance(op, Operation) or not callable(handler):
        raise TypeError("recover() requires an Operation and a callable")
    return cast(
        Operation[A, B | C],
        Operation._from_node(
            RecoverNode(op._node, handler, with_operation=False),
            name=f"recover({op.name})",
            requirements=op._requirements,
        ),
    )


def recover_with(
    op: Operation[A, B],
    handler: Callable[
        [Exception],
        Operation[Unit, C] | Awaitable[Operation[Unit, C]],
    ],
) -> Operation[A, B | C]:
    """Recover an error by running a returned nullary operation."""

    if not isinstance(op, Operation) or not callable(handler):
        raise TypeError("recover_with() requires an Operation and a callable")
    return cast(
        Operation[A, B | C],
        Operation._from_node(
            RecoverNode(op._node, handler, with_operation=True),
            name=f"recover_with({op.name})",
            requirements=op._requirements,
        ),
    )


def default_on_error(op: Operation[A, B], value: C) -> Operation[A, B | C]:
    """Recover an error with a construction-time default value."""

    try:
        snapshot = copy.deepcopy(value)
    except Exception:
        snapshot = value
    return recover(op, lambda _error: snapshot)


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
) -> Operation[A, B]:
    """Retry failures; ``attempts`` includes the first invocation."""

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
            RetryNode(op._node, attempts, normalized_backoff),
            name=f"retry({op.name})",
            requirements=op._requirements,
        ),
    )


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
    "collect_named",
    "default_on_error",
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
