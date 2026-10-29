"""Typed, immutable environments of shared dependencies."""
# pyright: reportUnnecessaryIsInstance=false

from __future__ import annotations

from types import MappingProxyType
from typing import Generic, Iterator, Mapping, TypeVar, cast

T = TypeVar("T")


class EnvironmentError(Exception):
    """Raised when an environment is missing a key or holds the wrong type."""


class UndeclaredRequirementError(EnvironmentError):
    """An operation returned at run time needs a key that is missing and was
    never declared with ``requires=`` where it was built."""

    def __init__(self, key: EnvKey[object], operation: str) -> None:
        self.key = key
        self.operation = operation
        super().__init__(
            f"{operation} needs environment key {key.name!r}, which is missing "
            "and was not declared with requires="
        )


class EnvKey(Generic[T]):
    """An identity key carrying the runtime type of one environment value."""

    __slots__ = ("name", "value_type")

    def __init__(self, name: str, value_type: type[T]) -> None:
        if not name:
            raise ValueError("environment key name must not be empty")
        if not isinstance(value_type, type):
            raise TypeError("environment key value_type must be a type")
        self.name = name
        self.value_type = value_type

    def __repr__(self) -> str:
        return f"EnvKey({self.name!r}, {self.value_type.__name__})"


class Environment:
    """A read-only mapping from typed identity keys to values."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[EnvKey[object], object] | None = None) -> None:
        copied: dict[EnvKey[object], object] = {}
        if values is not None:
            for key, value in values.items():
                self._validate(key, value)
                copied[key] = value
        self._values = MappingProxyType(copied)

    @staticmethod
    def _validate(key: EnvKey[object], value: object) -> None:
        if not isinstance(key, EnvKey):
            raise EnvironmentError(f"invalid environment key: {key!r}")
        if not isinstance(value, key.value_type):
            raise EnvironmentError(
                f"environment key {key.name!r} requires {key.value_type.__name__}, "
                f"got {type(value).__name__}"
            )

    def require(self, key: EnvKey[T]) -> T:
        """Return the value for ``key`` or raise :class:`EnvironmentError`."""

        try:
            value = self._values[cast(EnvKey[object], key)]
        except KeyError:
            raise EnvironmentError(
                f"missing required environment key {key.name!r}"
            ) from None
        self._validate(cast(EnvKey[object], key), value)
        return cast(T, value)

    def with_value(self, key: EnvKey[T], value: T) -> Environment:
        """Return a new environment containing ``value`` under ``key``."""

        object_key = cast(EnvKey[object], key)
        self._validate(object_key, value)
        values = dict(self._values)
        values[object_key] = value
        return Environment(values)

    def __contains__(self, key: object) -> bool:
        return key in self._values

    def __iter__(self) -> Iterator[EnvKey[object]]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        names = ", ".join(key.name for key in self._values)
        return f"Environment({names})"


EMPTY_ENV = Environment()


__all__ = [
    "EMPTY_ENV",
    "EnvKey",
    "Environment",
    "EnvironmentError",
    "UndeclaredRequirementError",
]
