"""Construction-only placeholder templates for unary operations.

Templates adapt a configured multi-argument callable into a normal unary
``Operation``.  Placeholder resolution happens once, when the template is
called; the placeholder is never stored in the operation AST or interpreted by
the runner.
"""

# mypy: disable-error-code=overload-overlap
# pyright: reportOverlappingOverload=false, reportPrivateUsage=false
# pyright: reportUnnecessaryIsInstance=false

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
    final,
    overload,
)

from expression import Result

from fp_ops._ast import ApplyNode
from fp_ops._invoke import Invocation
from fp_ops.core import (
    Operation,
    _merge_requirements,
    environment_operation,
    environment_result_operation,
    operation,
    result_operation,
    threaded_operation,
)
from fp_ops.environment import EnvKey

# Template protocol parameters are invariant: each one is the hole's input type
# in one overload and appears inside ``Derived[...]`` operands in the others.
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
E = TypeVar("E")
F = TypeVar("F")
R = TypeVar("R", covariant=True)
DA = TypeVar("DA", contravariant=True)
DT = TypeVar("DT", covariant=True)
T = TypeVar("T")
P = ParamSpec("P")


class Placeholder:
    """The construction-time hole filled by an operation's unary input."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_"


_ = Placeholder()


@final
class Derived(Generic[DA, DT]):
    """A template operand read from the pipeline input at run time.

    Built with :func:`derive`. Use it for a value that is only known while the
    pipeline runs, such as the URL a page should navigate to next.
    """

    __slots__ = ("operation",)

    def __init__(self, operation: Operation[DA, DT]) -> None:
        if not isinstance(operation, Operation):
            raise TypeError("derive() requires an Operation")
        self.operation = operation

    def __repr__(self) -> str:
        return f"derive({self.operation.name})"


def derive(op: Operation[DA, DT]) -> Derived[DA, DT]:
    """Mark a template operand to be computed from the input at run time.

    ``op`` runs on the same input as the template's ``_`` hole, just before the
    call; derived operands run in declaration order and their errors propagate
    unchanged. ``Navigate(_, derive(next_url))`` navigates a page to a URL read
    from that page. Each run (and each retry) evaluates the operand again.
    """

    return Derived(op)


def _snapshot(value: object) -> object:
    """Shallow-copy builtin mutable containers; keep every other operand as-is.

    Operands such as clients, handles, or frozen records are stored by
    identity, so templates never need to know how to copy them.
    """

    if type(value) in (list, dict, set, bytearray):
        return copy.copy(value)
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
        environment_parameter: inspect.Parameter | None = None
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
                    "environment operation template requires the environment value "
                    "as its final positional parameter"
                )
            environment_parameter = positional_parameters[-1]
            visible_parameters = tuple(
                parameter
                for parameter in source_parameters
                if parameter is not environment_parameter
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
        self._environment_parameter = environment_parameter
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
        derived: tuple[tuple[str, Derived[Any, Any]], ...] = tuple(
            (name, cast("Derived[Any, Any]", value))
            for name, value in captured.items()
            if isinstance(value, Derived)
        )
        if derived:
            return self._apply(hole_name, captured, derived)

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

        environment_parameter = self._environment_parameter
        assert environment_parameter is not None

        def configured_with_environment(
            value: object,
            dependency: object,
        ) -> object:
            arguments = OrderedDict(captured)
            arguments[hole_name] = value
            arguments[environment_parameter.name] = dependency
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


    def _apply(
        self,
        hole_name: str,
        captured: OrderedDict[str, object],
        derived: tuple[tuple[str, Derived[Any, Any]], ...],
    ) -> Operation[Any, R]:
        """Build an operation whose derived operands are read from the input."""

        environment_parameter = self._environment_parameter
        derived_names = tuple(name for name, _value in derived)

        def arguments_for(packed: object) -> OrderedDict[str, object]:
            value, operands = cast(tuple[object, tuple[object, ...]], packed)
            arguments = OrderedDict(captured)
            arguments[hole_name] = value
            for name, operand in zip(derived_names, operands):
                arguments[name] = operand
            return arguments

        def configured(packed: object) -> object:
            resolved = inspect.BoundArguments(
                self._source_signature, arguments_for(packed)
            )
            return self._fn(*resolved.args, **resolved.kwargs)

        def configured_with_environment(packed: object, dependency: object) -> object:
            assert environment_parameter is not None
            arguments = arguments_for(packed)
            arguments[environment_parameter.name] = dependency
            resolved = inspect.BoundArguments(self._source_signature, arguments)
            return self._fn(*resolved.args, **resolved.kwargs)

        key = self._environment_key
        call: Callable[..., object]
        if key is None:
            call = functools.update_wrapper(configured, self._fn)
        else:
            call = functools.update_wrapper(configured_with_environment, self._fn)
        invocation = Invocation(
            cast(Callable[..., Any], call),
            output="result" if self._result else "raw",
            environment_key=cast(Any, key),
            threaded=self._threaded,
        )
        operations = tuple(value.operation for _name, value in derived)
        name = getattr(self._fn, "__name__", type(self._fn).__name__)
        return cast(
            Operation[Any, R],
            Operation._from_node(
                ApplyNode(tuple(op._node for op in operations), invocation),
                name=name,
                description=inspect.getdoc(self._fn),
                source_signature=self._source_signature,
                requirements=_merge_requirements(
                    *(op._requirements for op in operations),
                    () if key is None else (key,),
                ),
            ),
        )


class OperationTemplate1(Protocol[DA, P, R]):
    def __call__(
        self,
        first: Placeholder,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[DA, R]: ...


class OperationTemplate2(Protocol[A, B, P, R]):

    @overload
    def __call__(
        self,
        first: Placeholder,
        second: B | Derived[A, B],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[B, A],
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
        second: B | Derived[A, B],
        third: C | Derived[A, C],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[B, A],
        second: Placeholder,
        third: C | Derived[B, C],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[C, A],
        second: B | Derived[C, B],
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
        second: B | Derived[A, B],
        third: C | Derived[A, C],
        fourth: D | Derived[A, D],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[B, A],
        second: Placeholder,
        third: C | Derived[B, C],
        fourth: D | Derived[B, D],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[C, A],
        second: B | Derived[C, B],
        third: Placeholder,
        fourth: D | Derived[C, D],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[C, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[D, A],
        second: B | Derived[D, B],
        third: C | Derived[D, C],
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
        second: B | Derived[A, B],
        third: C | Derived[A, C],
        fourth: D | Derived[A, D],
        fifth: E | Derived[A, E],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[B, A],
        second: Placeholder,
        third: C | Derived[B, C],
        fourth: D | Derived[B, D],
        fifth: E | Derived[B, E],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[C, A],
        second: B | Derived[C, B],
        third: Placeholder,
        fourth: D | Derived[C, D],
        fifth: E | Derived[C, E],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[C, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[D, A],
        second: B | Derived[D, B],
        third: C | Derived[D, C],
        fourth: Placeholder,
        fifth: E | Derived[D, E],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[D, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[E, A],
        second: B | Derived[E, B],
        third: C | Derived[E, C],
        fourth: D | Derived[E, D],
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
        second: B | Derived[A, B],
        third: C | Derived[A, C],
        fourth: D | Derived[A, D],
        fifth: E | Derived[A, E],
        sixth: F | Derived[A, F],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[A, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[B, A],
        second: Placeholder,
        third: C | Derived[B, C],
        fourth: D | Derived[B, D],
        fifth: E | Derived[B, E],
        sixth: F | Derived[B, F],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[B, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[C, A],
        second: B | Derived[C, B],
        third: Placeholder,
        fourth: D | Derived[C, D],
        fifth: E | Derived[C, E],
        sixth: F | Derived[C, F],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[C, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[D, A],
        second: B | Derived[D, B],
        third: C | Derived[D, C],
        fourth: Placeholder,
        fifth: E | Derived[D, E],
        sixth: F | Derived[D, F],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[D, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[E, A],
        second: B | Derived[E, B],
        third: C | Derived[E, C],
        fourth: D | Derived[E, D],
        fifth: Placeholder,
        sixth: F | Derived[E, F],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Operation[E, R]: ...

    @overload
    def __call__(
        self,
        first: A | Derived[F, A],
        second: B | Derived[F, B],
        third: C | Derived[F, C],
        fourth: D | Derived[F, D],
        fifth: E | Derived[F, E],
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
    """Bind one environment key and decorate a raw-value operation template."""

    return EnvironmentOperationTemplateDecorator(key)


def environment_result_operation_template(
    key: EnvKey[T],
) -> EnvironmentResultOperationTemplateDecorator[T]:
    """Bind one environment key and decorate a Result-returning operation template."""

    return EnvironmentResultOperationTemplateDecorator(key)


__all__ = [
    "Derived",
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
    "derive",
    "environment_operation_template",
    "environment_result_operation_template",
    "operation_template",
    "result_operation_template",
    "threaded_operation_template",
]
