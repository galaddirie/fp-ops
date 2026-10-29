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


class LoopStepError(Exception, Generic[T]):
    """A ``loop`` step failed; ``state`` is that step's input, so work can resume."""

    def __init__(self, step: int, state: T, cause: Exception) -> None:
        self.step = step
        self.state = state
        self.cause = cause
        super().__init__(f"loop step {step} failed: {cause}")
        self.__cause__ = cause


class ReleaseError(Exception):
    """A ``bracket`` release failed.

    ``primary`` is the error from the ``use`` operation when it also failed, or
    ``None`` when ``use`` succeeded and only the release went wrong.
    """

    def __init__(self, cause: Exception, primary: Exception | None = None) -> None:
        self.cause = cause
        self.primary = primary
        detail = "" if primary is None else f" (after use failed: {primary})"
        super().__init__(f"resource release failed: {cause}{detail}")
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
    "LoopStepError",
    "Ok",
    "PathNotFoundError",
    "ReleaseError",
    "Result",
    "TraversalItemError",
    "ok_value",
]
