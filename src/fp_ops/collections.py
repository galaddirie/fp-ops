"""Collection operations with explicit callback, effect, and error policies."""
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any, Awaitable, Callable, Hashable, Literal, TypeVar, cast

from expression import Result

from fp_ops._ast import (
    FanoutEachNode,
    FanoutNode,
    FilterEachNode,
    FilterOperationNode,
    FoldNode,
    MapEachNode,
    TraverseNode,
)
from fp_ops.core import Operation, _merge_requirements

T = TypeVar("T")
U = TypeVar("U")
A = TypeVar("A")


def map_each(
    fn: Callable[[T], U | Awaitable[U]],
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[U] | dict[Hashable, U],
]:
    """Map a raw-value callback in input order and fail on its first error."""

    if not callable(fn):
        raise TypeError("map_each() requires a callable")
    return cast(
        Operation[
            Iterable[T] | Mapping[Hashable, T],
            list[U] | dict[Hashable, U],
        ],
        Operation._from_node(MapEachNode(fn), name="map_each"),
    )


def traverse(
    op: Operation[T, U],
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[U] | dict[Hashable, U],
]:
    """Run an operation sequentially in input order and fail fast."""

    return _traverse(op, mode="fail_fast", limit=None, on_error=None)


def traverse_parallel(
    op: Operation[T, U],
    *,
    limit: int,
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[U] | dict[Hashable, U],
]:
    """Run a bounded concurrent traversal and fail fast."""

    return _traverse(op, mode="fail_fast", limit=limit, on_error=None)


def traverse_all_settled(
    op: Operation[T, U],
    *,
    limit: int,
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[Result[U, Exception]] | dict[Hashable, Result[U, Exception]],
]:
    """Run every item and retain its ``Result`` in input order."""

    return _traverse(op, mode="all_settled", limit=limit, on_error=None)


def traverse_lossy(
    op: Operation[T, U],
    *,
    limit: int,
    on_error: Callable[[T, Exception], object | Awaitable[object]] | None = None,
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[U] | dict[Hashable, U],
]:
    """Run every item and retain only successes, optionally reporting errors."""

    if on_error is not None and not callable(on_error):
        raise TypeError("on_error must be callable")
    return _traverse(op, mode="lossy", limit=limit, on_error=on_error)


def _traverse(
    op: Operation[T, U],
    *,
    mode: Literal["fail_fast", "all_settled", "lossy"],
    limit: int | None,
    on_error: Callable[[T, Exception], object | Awaitable[object]] | None,
) -> Operation[Any, Any]:
    if not isinstance(op, Operation):
        raise TypeError("traversal requires an Operation")
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be at least 1")
    return cast(
        Operation[Iterable[T] | Mapping[Hashable, T], object],
        Operation._from_node(
            TraverseNode(
                op._node,
                mode=mode,
                limit=limit,
                on_error=on_error,
            ),
            name=f"traverse_{mode}" if mode != "fail_fast" else "traverse",
            requirements=op._requirements,
        ),
    )


def filter_each(
    predicate: Callable[[T], bool | Awaitable[bool]],
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[T] | dict[Hashable, T],
]:
    """Filter with a raw predicate and fail if the predicate raises."""

    if not callable(predicate):
        raise TypeError("filter_each() requires a callable")
    return cast(
        Operation[
            Iterable[T] | Mapping[Hashable, T],
            list[T] | dict[Hashable, T],
        ],
        Operation._from_node(FilterEachNode(predicate), name="filter_each"),
    )


def filter_operation(
    predicate: Operation[T, bool],
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[T] | dict[Hashable, T],
]:
    """Filter sequentially with an operation and fail fast."""

    return _filter_operation(predicate, best_effort=False)


def filter_best_effort(
    predicate: Operation[T, bool],
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[T] | dict[Hashable, T],
]:
    """Filter with an operation, treating predicate failures as rejection."""

    return _filter_operation(predicate, best_effort=True)


def _filter_operation(
    predicate: Operation[T, bool],
    *,
    best_effort: bool,
) -> Operation[
    Iterable[T] | Mapping[Hashable, T],
    list[T] | dict[Hashable, T],
]:
    if not isinstance(predicate, Operation):
        raise TypeError("operation filtering requires an Operation")
    return cast(
        Operation[
            Iterable[T] | Mapping[Hashable, T],
            list[T] | dict[Hashable, T],
        ],
        Operation._from_node(
            FilterOperationNode(predicate._node, best_effort=best_effort),
            name="filter_best_effort" if best_effort else "filter_operation",
            requirements=predicate._requirements,
        ),
    )


def fold(
    fn: Callable[[A, T], A | Awaitable[A]],
    initial: A,
) -> Operation[Iterable[T], A]:
    """Fold left with a raw-value reducer."""

    if not callable(fn):
        raise TypeError("fold() requires a callable")
    try:
        snapshot = copy.deepcopy(initial)
    except Exception:
        snapshot = initial
    return cast(
        Operation[Iterable[T], A],
        Operation._from_node(
            FoldNode(fn, snapshot, operation_reducer=False),
            name="fold",
        ),
    )


def fold_operation(
    op: Operation[tuple[A, T], A],
    initial: A,
) -> Operation[Iterable[T], A]:
    """Fold left with an operation consuming ``(accumulator, item)``."""

    if not isinstance(op, Operation):
        raise TypeError("fold_operation() requires an Operation")
    try:
        snapshot = copy.deepcopy(initial)
    except Exception:
        snapshot = initial
    return cast(
        Operation[Iterable[T], A],
        Operation._from_node(
            FoldNode(op._node, snapshot, operation_reducer=True),
            name="fold_operation",
            requirements=op._requirements,
        ),
    )


def fanout_each(
    *operations: Operation[T, object],
) -> Operation[list[T], list[tuple[object, ...]]]:
    """Apply a fail-fast same-input fanout to each item sequentially."""

    snapshot = tuple(operations)
    if any(not isinstance(item, Operation) for item in snapshot):
        raise TypeError("fanout_each() accepts only Operation instances")
    requirements = _merge_requirements(
        *(item._requirements for item in snapshot)
    )
    fanout = FanoutNode(
        tuple((None, item._node) for item in snapshot),
        mode="fail_fast",
        named=False,
        limit=None,
    )
    return cast(
        Operation[list[T], list[tuple[object, ...]]],
        Operation._from_node(
            FanoutEachNode(fanout),
            name="fanout_each",
            requirements=requirements,
        ),
    )


__all__ = [
    "fanout_each",
    "filter_best_effort",
    "filter_each",
    "filter_operation",
    "fold",
    "fold_operation",
    "map_each",
    "traverse",
    "traverse_all_settled",
    "traverse_lossy",
    "traverse_parallel",
]
