"""Construction-only placeholder templates for unary operations.

Templates adapt a configured multi-argument callable into a normal unary
``Operation``.  Placeholder resolution happens once, when the template is
called; the placeholder is never stored in the operation AST or interpreted by
the runner.
"""

# mypy: disable-error-code=overload-overlap
# pyright: reportOverlappingOverload=false

from __future__ import annotations

import copy
import functools
import inspect
from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
)

from expression import Result

from fp_ops.core import (
    Operation,
    environment_operation,
    environment_result_operation,
    operation,
    result_operation,
    threaded_operation,
)
from fp_ops.environment import EnvKey

A = TypeVar("A", contravariant=True)
B = TypeVar("B", contravariant=True)
C = TypeVar("C", contravariant=True)
D = TypeVar("D", contravariant=True)
E = TypeVar("E", contravariant=True)
F = TypeVar("F", contravariant=True)
R = TypeVar("R", covariant=True)
T = TypeVar("T")
P = ParamSpec("P")


class Placeholder:
    """The construction-time hole filled by an operation's unary input."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_"


_ = Placeholder()


def _snapshot(value: object) -> object:
    if isinstance(value, Placeholder):
        return value
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


class _OperationTemplate(Generic[R]):
    def __init__(
        self,
        fn: Callable[..., object],
        *,
        result: bool,
        environment_key: EnvKey[object] | None = None,
        threaded: bool = False,
    ) -> None:
        if not callable(fn):
            raise TypeError("operation template requires a callable")

        try:
            source_signature = inspect.signature(fn)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "operation template requires an inspectable callable"
            ) from error

        source_parameters = tuple(source_signature.parameters.values())
        capability_parameter: inspect.Parameter | None = None
        visible_parameters = source_parameters
        if environment_key is not None:
            positional_parameters = tuple(
                parameter
                for parameter in source_parameters
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            if not positional_parameters:
                raise TypeError(
                    "environment operation template requires a capability as its "
                    "final positional parameter"
                )
            capability_parameter = positional_parameters[-1]
            visible_parameters = tuple(
                parameter
                for parameter in source_parameters
                if parameter is not capability_parameter
            )

        positional_count = sum(
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in visible_parameters
        )
        if not 1 <= positional_count <= 6:
            raise TypeError(
                "operation templates require between 1 and 6 visible positional "
                "parameters"
            )

        self._fn = fn
        self._result = result
        self._environment_key = environment_key
        self._threaded = threaded
        self._source_signature = source_signature
        self._visible_signature = source_signature.replace(
            parameters=visible_parameters
        )
        self._capability_parameter = capability_parameter
        self._positional_count = positional_count
        functools.update_wrapper(self, fn)
        self.__signature__ = self._visible_signature

    def __call__(self, *args: object, **kwargs: object) -> Operation[Any, R]:
        bound = self._visible_signature.bind(*args, **kwargs)
        holes = [
            name
            for name, value in bound.arguments.items()
            if isinstance(value, Placeholder)
        ]
        if len(holes) != 1:
            raise TypeError(
                "operation template call requires exactly one direct placeholder; "
                f"received {len(holes)}"
            )

        hole_name = holes[0]
        captured = OrderedDict(
            (name, _snapshot(value)) for name, value in bound.arguments.items()
        )

        if self._environment_key is None:

            def configured(value: object) -> object:
                arguments = OrderedDict(captured)
                arguments[hole_name] = value
                resolved = inspect.BoundArguments(
                    self._source_signature,
                    arguments,
                )
                return self._fn(*resolved.args, **resolved.kwargs)

            functools.update_wrapper(configured, self._fn)
            if self._result:
                return cast(
                    Operation[Any, R],
                    result_operation(cast(Callable[[Any], Any], configured)),
                )
            if self._threaded:
                return cast(
                    Operation[Any, R],
                    threaded_operation(cast(Callable[[Any], Any], configured)),
                )
            return cast(
                Operation[Any, R],
                operation(cast(Callable[[Any], Any], configured)),
            )

        capability_parameter = self._capability_parameter
        assert capability_parameter is not None

        def configured_with_environment(
            value: object,
            capability: object,
        ) -> object:
            arguments = OrderedDict(captured)
            arguments[hole_name] = value
            arguments[capability_parameter.name] = capability
            resolved = inspect.BoundArguments(
                self._source_signature,
                arguments,
            )
            return self._fn(*resolved.args, **resolved.kwargs)

        functools.update_wrapper(configured_with_environment, self._fn)
        key = self._environment_key
        if self._result:
            return cast(
                Operation[Any, R],
                environment_result_operation(key)(
                    cast(Callable[[Any, object], Any], configured_with_environment)
                ),
            )
        return cast(
            Operation[Any, R],
            environment_operation(key)(
                cast(Callable[[Any, object], Any], configured_with_environment)
            ),
        )


class OperationTemplate1(Protocol[A, P, R]):
    def __call__(
        self,
        first: Placeholder,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...


class OperationTemplate2(Protocol[A, B, P, R]):
    @overload
    def __call__(
        self,
        first: Placeholder,
        second: B,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: Placeholder,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...


class OperationTemplate3(Protocol[A, B, C, P, R]):
    @overload
    def __call__(
        self,
        first: Placeholder,
        second: B,
        third: C,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: Placeholder,
        third: C,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: Placeholder,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[C, R]: ...


class OperationTemplate4(Protocol[A, B, C, D, P, R]):
    @overload
    def __call__(
        self,
        first: Placeholder,
        second: B,
        third: C,
        fourth: D,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: Placeholder,
        third: C,
        fourth: D,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: Placeholder,
        fourth: D,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[C, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: C,
        fourth: Placeholder,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[D, R]: ...


class OperationTemplate5(Protocol[A, B, C, D, E, P, R]):
    @overload
    def __call__(
        self,
        first: Placeholder,
        second: B,
        third: C,
        fourth: D,
        fifth: E,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: Placeholder,
        third: C,
        fourth: D,
        fifth: E,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: Placeholder,
        fourth: D,
        fifth: E,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[C, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: C,
        fourth: Placeholder,
        fifth: E,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[D, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: C,
        fourth: D,
        fifth: Placeholder,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[E, R]: ...


class OperationTemplate6(
    Protocol[A, B, C, D, E, F, P, R],
):
    @overload
    def __call__(
        self,
        first: Placeholder,
        second: B,
        third: C,
        fourth: D,
        fifth: E,
        sixth: F,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: Placeholder,
        third: C,
        fourth: D,
        fifth: E,
        sixth: F,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: Placeholder,
        fourth: D,
        fifth: E,
        sixth: F,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[C, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: C,
        fourth: Placeholder,
        fifth: E,
        sixth: F,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[D, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: C,
        fourth: D,
        fifth: Placeholder,
        sixth: F,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[E, R]: ...

    @overload
    def __call__(
        self,
        first: A,
        second: B,
        third: C,
        fourth: D,
        fifth: E,
        sixth: Placeholder,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[F, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, D, E, F, P], Awaitable[R]],
) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, D, E, F, P], R],
) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, D, E, P], Awaitable[R]],
) -> OperationTemplate5[A, B, C, D, E, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, D, E, P], R],
) -> OperationTemplate5[A, B, C, D, E, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, D, P], Awaitable[R]],
) -> OperationTemplate4[A, B, C, D, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, D, P], R],
) -> OperationTemplate4[A, B, C, D, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, P], Awaitable[R]],
) -> OperationTemplate3[A, B, C, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, C, P], R],
) -> OperationTemplate3[A, B, C, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, P], Awaitable[R]],
) -> OperationTemplate2[A, B, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, B, P], R],
) -> OperationTemplate2[A, B, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, P], Awaitable[R]],
) -> OperationTemplate1[A, P, R]: ...


@overload
def operation_template(
    fn: Callable[Concatenate[A, P], R],
) -> OperationTemplate1[A, P, R]: ...


def operation_template(fn: Callable[..., Any]) -> Any:
    """Decorate a raw-value callable as a construction-only operation template."""

    return cast(Any, _OperationTemplate[Any](fn, result=False))


@overload
def result_operation_template(
    fn: Callable[
        Concatenate[A, B, C, D, E, F, P],
        Awaitable[Result[R, Exception]],
    ],
) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, B, C, D, E, F, P], Result[R, Exception]],
) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[
        Concatenate[A, B, C, D, E, P],
        Awaitable[Result[R, Exception]],
    ],
) -> OperationTemplate5[A, B, C, D, E, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, B, C, D, E, P], Result[R, Exception]],
) -> OperationTemplate5[A, B, C, D, E, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[
        Concatenate[A, B, C, D, P],
        Awaitable[Result[R, Exception]],
    ],
) -> OperationTemplate4[A, B, C, D, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, B, C, D, P], Result[R, Exception]],
) -> OperationTemplate4[A, B, C, D, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, B, C, P], Awaitable[Result[R, Exception]]],
) -> OperationTemplate3[A, B, C, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, B, C, P], Result[R, Exception]],
) -> OperationTemplate3[A, B, C, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, B, P], Awaitable[Result[R, Exception]]],
) -> OperationTemplate2[A, B, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, B, P], Result[R, Exception]],
) -> OperationTemplate2[A, B, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, P], Awaitable[Result[R, Exception]]],
) -> OperationTemplate1[A, P, R]: ...


@overload
def result_operation_template(
    fn: Callable[Concatenate[A, P], Result[R, Exception]],
) -> OperationTemplate1[A, P, R]: ...


def result_operation_template(fn: Callable[..., Any]) -> Any:
    """Decorate a Result-returning callable as an operation template."""

    return cast(Any, _OperationTemplate[Any](fn, result=True))


if TYPE_CHECKING:
    threaded_operation_template = operation_template
else:

    def threaded_operation_template(fn: Callable[..., Any]) -> Any:
        """Decorate a blocking callable as a threaded operation template."""

        return cast(
            Any,
            _OperationTemplate[Any](fn, result=False, threaded=True),
        )


class EnvironmentOperationTemplateDecorator(Generic[T]):
    def __init__(self, key: EnvKey[T]) -> None:
        self._key = key

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, T, P], Awaitable[R]],
    ) -> OperationTemplate1[A, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, T, P], R],
    ) -> OperationTemplate1[A, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, T, P], Awaitable[R]],
    ) -> OperationTemplate2[A, B, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, T, P], R],
    ) -> OperationTemplate2[A, B, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, T, P], Awaitable[R]],
    ) -> OperationTemplate3[A, B, C, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, T, P], R],
    ) -> OperationTemplate3[A, B, C, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, D, T, P], Awaitable[R]],
    ) -> OperationTemplate4[A, B, C, D, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, D, T, P], R],
    ) -> OperationTemplate4[A, B, C, D, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, D, E, T, P], Awaitable[R]],
    ) -> OperationTemplate5[A, B, C, D, E, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, D, E, T, P], R],
    ) -> OperationTemplate5[A, B, C, D, E, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[
            Concatenate[A, B, C, D, E, F, T, P],
            Awaitable[R],
        ],
    ) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, D, E, F, T, P], R],
    ) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...

    def __call__(self, fn: Callable[..., Any]) -> Any:
        return cast(
            Any,
            _OperationTemplate[Any](
                fn,
                result=False,
                environment_key=cast(EnvKey[object], self._key),
            ),
        )


class EnvironmentResultOperationTemplateDecorator(Generic[T]):
    def __init__(self, key: EnvKey[T]) -> None:
        self._key = key

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, T, P], Awaitable[Result[R, Exception]]],
    ) -> OperationTemplate1[A, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, T, P], Result[R, Exception]],
    ) -> OperationTemplate1[A, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[
            Concatenate[A, B, T, P],
            Awaitable[Result[R, Exception]],
        ],
    ) -> OperationTemplate2[A, B, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, T, P], Result[R, Exception]],
    ) -> OperationTemplate2[A, B, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[
            Concatenate[A, B, C, T, P],
            Awaitable[Result[R, Exception]],
        ],
    ) -> OperationTemplate3[A, B, C, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, T, P], Result[R, Exception]],
    ) -> OperationTemplate3[A, B, C, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[
            Concatenate[A, B, C, D, T, P],
            Awaitable[Result[R, Exception]],
        ],
    ) -> OperationTemplate4[A, B, C, D, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, D, T, P], Result[R, Exception]],
    ) -> OperationTemplate4[A, B, C, D, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[
            Concatenate[A, B, C, D, E, T, P],
            Awaitable[Result[R, Exception]],
        ],
    ) -> OperationTemplate5[A, B, C, D, E, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[Concatenate[A, B, C, D, E, T, P], Result[R, Exception]],
    ) -> OperationTemplate5[A, B, C, D, E, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[
            Concatenate[A, B, C, D, E, F, T, P],
            Awaitable[Result[R, Exception]],
        ],
    ) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...

    @overload
    def __call__(
        self,
        fn: Callable[
            Concatenate[A, B, C, D, E, F, T, P],
            Result[R, Exception],
        ],
    ) -> OperationTemplate6[A, B, C, D, E, F, P, R]: ...

    def __call__(self, fn: Callable[..., Any]) -> Any:
        return cast(
            Any,
            _OperationTemplate[Any](
                fn,
                result=True,
                environment_key=cast(EnvKey[object], self._key),
            ),
        )


def environment_operation_template(
    key: EnvKey[T],
) -> EnvironmentOperationTemplateDecorator[T]:
    """Bind a capability and decorate a raw-value operation template."""

    return EnvironmentOperationTemplateDecorator(key)


def environment_result_operation_template(
    key: EnvKey[T],
) -> EnvironmentResultOperationTemplateDecorator[T]:
    """Bind a capability and decorate a Result-returning operation template."""

    return EnvironmentResultOperationTemplateDecorator(key)


__all__ = [
    "EnvironmentOperationTemplateDecorator",
    "EnvironmentResultOperationTemplateDecorator",
    "OperationTemplate1",
    "OperationTemplate2",
    "OperationTemplate3",
    "OperationTemplate4",
    "OperationTemplate5",
    "OperationTemplate6",
    "Placeholder",
    "_",
    "environment_operation_template",
    "environment_result_operation_template",
    "operation_template",
    "result_operation_template",
    "threaded_operation_template",
]
