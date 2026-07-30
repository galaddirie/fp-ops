"""Public exceptions and helpers for FP-Ops' ``Result`` channel."""

from __future__ import annotations

from typing import Generic, TypeVar

from expression import Error, Ok, Result

T = TypeVar("T")


class BuildFieldError(Exception):
    """A schema field failed while being built."""

    def __init__(self, field: str, cause: Exception) -> None:
        self.field = field
        self.cause = cause
        super().__init__(f"failed to build field {field!r}: {cause}")
        self.__cause__ = cause


class TraversalItemError(Exception, Generic[T]):
    """An item failed during a fail-fast collection operation."""

    def __init__(self, index: T, cause: Exception) -> None:
        self.index = index
        self.cause = cause
        super().__init__(f"collection item {index!r} failed: {cause}")
        self.__cause__ = cause


class PathNotFoundError(Exception):
    """A requested object path does not exist and no default was supplied."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"path not found: {path!r}")


def ok_value(result: Result[T, Exception]) -> T:
    """Extract a known-success value without using a sentinel default."""

    if result.is_error():
        raise result.error
    return result.ok


__all__ = [
    "BuildFieldError",
    "Error",
    "Ok",
    "PathNotFoundError",
    "Result",
    "TraversalItemError",
    "ok_value",
]
