"""The public unary operation seam and canonical constructors."""
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    cast,
    overload,
)

from expression import Result
from typing_extensions import Never

from fp_ops._ast import (
    CallNode,
    ConstantNode,
    FailNode,
    FlatMapNode,
    IdentityNode,
    Node,
    SequenceNode,
)
from fp_ops._invoke import Invocation
from fp_ops.environment import EMPTY_ENV, EnvKey, Environment

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
T = TypeVar("T")
E = TypeVar("E")
T_co = TypeVar("T_co", covariant=True)
A_contra = TypeVar("A_contra", contravariant=True)
B_co = TypeVar("B_co", covariant=True)


class Unit(Enum):
    """The single input value for deliberately nullary sources."""

    UNIT = "UNIT"

    def __repr__(self) -> str:
        return "UNIT"


UNIT = Unit.UNIT


def _merge_requirements(
    *groups: tuple[EnvKey[object], ...],
) -> tuple[EnvKey[object], ...]:
    merged: list[EnvKey[object]] = []
    for group in groups:
        for key in group:
            if all(key is not existing for existing in merged):
                merged.append(key)
    return tuple(merged)


@dataclass(frozen=True, slots=True, init=False)
class Operation(Generic[A_contra, B_co]):
    """An immutable ``A -> Environment -> Awaitable[Result[B, Exception]]``."""

    _node: Node = field(repr=False)
    name: str
    description: str | None = None
    source_signature: inspect.Signature | None = None
    _requirements: tuple[EnvKey[object], ...] = field(default=(), repr=False)

    def __init__(self, _token: Never) -> None:
        raise TypeError(
            "Operation cannot be instantiated directly; use an FP-Ops constructor"
        )

    @classmethod
    def _create(
        cls,
        node: Node,
        *,
        name: str,
        description: str | None = None,
        source_signature: inspect.Signature | None = None,
        requirements: tuple[EnvKey[object], ...] = (),
    ) -> Operation[Any, Any]:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_node", node)
        object.__setattr__(instance, "name", name)
        object.__setattr__(instance, "description", description)
        object.__setattr__(instance, "source_signature", source_signature)
        object.__setattr__(instance, "_requirements", requirements)
        return instance

    async def run(
        self,
        value: A_contra,
        *,
        environment: Environment = EMPTY_ENV,
    ) -> Result[B_co, Exception]:
        """Execute this operation with exactly one input value."""

        from fp_ops._runner import run

        return cast(
            Result[B_co, Exception],
            await run(self._node, value, environment, self._requirements),
        )

    def then(self, next: Operation[B_co, C]) -> Operation[A_contra, C]:
        """Compose two operations left to right, preserving one environment."""

        if not isinstance(next, Operation):
            raise TypeError("then() requires an Operation; use map() for a callable")

        left = (
            self._node.stages
            if isinstance(self._node, SequenceNode)
            else (self._node,)
        )
        right = (
            next._node.stages
            if isinstance(next._node, SequenceNode)
            else (next._node,)
        )
        return cast(
            Operation[A_contra, C],
            Operation._create(
                SequenceNode(left + right),
                name=f"{self.name}.then({next.name})",
                requirements=_merge_requirements(
                    self._requirements, next._requirements
                ),
            ),
        )

    def map(
        self,
        fn: Callable[[B_co], C | Awaitable[C]],
    ) -> Operation[A_contra, C]:
        """Transform a successful value without flattening its runtime type."""

        return self.then(operation(cast(Callable[[B_co], Any], fn)))

    def flat_map(
        self,
        fn: Callable[
            [B_co],
            Operation[Unit, C] | Awaitable[Operation[Unit, C]],
        ],
    ) -> Operation[A_contra, C]:
        """Run the nullary operation returned by ``fn`` after success."""

        return cast(
            Operation[A_contra, C],
            Operation._create(
                FlatMapNode(self._node, fn),
                name=f"{self.name}.flat_map",
                requirements=self._requirements,
            ),
        )

    def __rshift__(self, next: Operation[B_co, C]) -> Operation[A_contra, C]:
        return self.then(next)

    def __or__(
        self, alternative: Operation[A_contra, B_co]
    ) -> Operation[A_contra, B_co]:
        from fp_ops.composition import or_else

        return or_else(self, alternative)

    @property
    def required_environment(self) -> tuple[EnvKey[object], ...]:
        """The statically known environment keys required by this AST."""

        return self._requirements

    @classmethod
    def _from_node(
        cls,
        node: Node,
        *,
        name: str,
        requirements: tuple[EnvKey[object], ...] = (),
        description: str | None = None,
        source_signature: inspect.Signature | None = None,
    ) -> Operation[Any, Any]:
        return cls._create(
            node,
            name=name,
            description=description,
            source_signature=source_signature,
            requirements=requirements,
        )


def _metadata(
    fn: Callable[..., object],
) -> tuple[str, str | None, inspect.Signature | None]:
    name = getattr(fn, "__name__", type(fn).__name__)
    description = inspect.getdoc(fn)
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    return name, description, signature


def _from_invocation(
    fn: Callable[..., object],
    *,
    output: str,
    key: EnvKey[object] | None = None,
    threaded: bool = False,
) -> Operation[Any, Any]:
    name, description, signature = _metadata(fn)
    requirements = () if key is None else (key,)
    invocation = Invocation(
        cast(Callable[..., Any], fn),
        output=cast(Any, output),
        environment_key=cast(EnvKey[Any] | None, key),
        threaded=threaded,
    )
    return Operation._from_node(
        CallNode(invocation),
        name=name,
        description=description,
        source_signature=signature,
        requirements=requirements,
    )


@overload
def operation(fn: Callable[[A], Awaitable[B]]) -> Operation[A, B]: ...


@overload
def operation(fn: Callable[[A], B]) -> Operation[A, B]: ...


def operation(fn: Callable[[A], object]) -> Operation[A, Any]:
    """Lift a unary raw-value callable into an operation."""

    if not callable(fn):
        raise TypeError("operation() requires a callable")
    return cast(Operation[A, Any], _from_invocation(fn, output="raw"))


@overload
def result_operation(
    fn: Callable[[A], Awaitable[Result[B, Exception]]],
) -> Operation[A, B]: ...


@overload
def result_operation(
    fn: Callable[[A], Result[B, Exception]],
) -> Operation[A, B]: ...


def result_operation(fn: Callable[[A], object]) -> Operation[A, Any]:
    """Lift a unary callable whose declared output is exactly one ``Result``."""

    if not callable(fn):
        raise TypeError("result_operation() requires a callable")
    return cast(Operation[A, Any], _from_invocation(fn, output="result"))


@overload
def threaded_operation(fn: Callable[[A], Awaitable[B]]) -> Operation[A, B]: ...


@overload
def threaded_operation(fn: Callable[[A], B]) -> Operation[A, B]: ...


def threaded_operation(fn: Callable[[A], object]) -> Operation[A, Any]:
    """Lift a callable whose initial invocation must run in a worker thread."""

    if not callable(fn):
        raise TypeError("threaded_operation() requires a callable")
    return cast(
        Operation[A, Any],
        _from_invocation(fn, output="raw", threaded=True),
    )


class EnvironmentOperationDecorator(Protocol[T_co]):
    @overload
    def __call__(
        self,
        fn: Callable[[A, T_co], Awaitable[B]],
    ) -> Operation[A, B]: ...

    @overload
    def __call__(self, fn: Callable[[A, T_co], B]) -> Operation[A, B]: ...


class EnvironmentResultOperationDecorator(Protocol[T_co]):
    @overload
    def __call__(
        self,
        fn: Callable[[A, T_co], Awaitable[Result[B, Exception]]],
    ) -> Operation[A, B]: ...

    @overload
    def __call__(
        self,
        fn: Callable[[A, T_co], Result[B, Exception]],
    ) -> Operation[A, B]: ...


class _EnvironmentDecorator(Generic[T]):
    def __init__(self, key: EnvKey[T], *, result: bool) -> None:
        self._key = key
        self._result = result

    def __call__(self, fn: Callable[..., object]) -> Operation[Any, Any]:
        if not callable(fn):
            raise TypeError("environment operation decorator requires a callable")
        return _from_invocation(
            fn,
            output="result" if self._result else "raw",
            key=cast(EnvKey[object], self._key),
        )


def environment_operation(key: EnvKey[T]) -> EnvironmentOperationDecorator[T]:
    """Bind one capability type, then decorate a raw-value callable."""

    return cast(
        EnvironmentOperationDecorator[T],
        _EnvironmentDecorator(key, result=False),
    )


def environment_result_operation(
    key: EnvKey[T],
) -> EnvironmentResultOperationDecorator[T]:
    """Bind one capability type, then decorate a ``Result`` callable."""

    return cast(
        EnvironmentResultOperationDecorator[T],
        _EnvironmentDecorator(key, result=True),
    )


def identity() -> Operation[T, T]:
    """Return the unary identity operation."""

    return cast(
        Operation[T, T],
        Operation._from_node(IdentityNode(), name="identity"),
    )


def constant(value: T) -> Operation[object, T]:
    """Return ``value`` for every input."""

    return cast(
        Operation[object, T],
        Operation._from_node(ConstantNode(value), name="constant"),
    )


def fail(error: Exception) -> Operation[object, Never]:
    """Return an operation that always fails with ``error``."""

    if not isinstance(error, Exception):
        raise TypeError("fail() requires an Exception")
    return cast(
        Operation[object, Never],
        Operation._from_node(FailNode(error), name="fail"),
    )


__all__ = [
    "EnvironmentOperationDecorator",
    "EnvironmentResultOperationDecorator",
    "Operation",
    "UNIT",
    "Unit",
    "constant",
    "environment_operation",
    "environment_result_operation",
    "fail",
    "identity",
    "operation",
    "result_operation",
    "threaded_operation",
]
