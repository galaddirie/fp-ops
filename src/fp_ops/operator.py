from __future__ import annotations
from functools import wraps

import inspect
import uuid
import asyncio
from dataclasses import replace
from functools import wraps
from types import MappingProxyType
from typing import (
    cast,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Sequence,
    Tuple,
    Generic,
    ParamSpec,
    TypeVar,
    Union,
    Concatenate,
    overload,
)
from collections.abc import Generator

from expression import Error, Ok, Result

from fp_ops.graph import OpGraph
from fp_ops.execution import Executor, ExecutionPlan
from fp_ops.context import BaseContext
from fp_ops.primitives import (
    Placeholder,
    Template,
    OpSpec,
    Port,
    PortType,
    Edge,
    _ as PLACEHOLDER,
)

# Core generics
P = ParamSpec("P")  # parameters of *this* operation
Q = ParamSpec("Q")  # parameters of *another* operation
R = TypeVar("R")  # return value of *this* operation
S = TypeVar("S")  # return value of *another* operation
E = TypeVar("E", bound=Exception)  # error type

# extra generics used only inside _from_function
P2 = ParamSpec("P2")
R2 = TypeVar("R2")


def _ensure_async(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Wrap a sync function so that it is awaitable.

    Args:
        fn: The function to wrap.

    Returns:
        An awaitable version of the function.
    """
    if inspect.iscoroutinefunction(fn):
        return fn

    async def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return _wrapper


def _wrap_result(value: Any) -> Result[Any, Exception]:
    """Lift raw value into `Result` if it is not one already.

    Args:
        value: Any value or Result object.

    Returns:
        A Result object containing the value.
    """
    if isinstance(value, Result):
        return value
    return Ok(value)


class Operation(Generic[P, R]):
    """An Operation represents a composable unit of computation,
    which can be a single function or a pipeline of multiple steps.

    Operations can be composed using the '>>' operator to build pipelines.

    Examples:
    ```python
    # Calling an *un-bound* Operation *executes* it immediately:
    >>> await add.execute(a=1, b=2)

    # Calling a *pipeline* returns a `_BoundCall` proxy that stores runtime
    # arguments until you finally ask for `.execute()`:
    >>> await (add >> add_one)(1, 2).execute()

    # You can also call the pipeline directly with arguments:
    >>> await (add >> add_one)(1, 2)

    # You can also use the `>>` operator to compose operations:
    >>> pipeline = add >> add_one
    >>> await pipeline(1, 2)
    ```
    """

    _ctx_factory: Callable[[], BaseContext | Awaitable[BaseContext]] | None
    _ctx_type: type[BaseContext] | None

    def __init__(
        self,
        graph: OpGraph,
        *,
        head_id: str,
        tail_id: str,
        ctx_factory: Callable[[], BaseContext | Awaitable[BaseContext]] | None = None,
        ctx_type: type[BaseContext] | None = None,
    ):
        """Initialize an Operation.

        Args:
            graph: The operation graph.
            head_id: ID of the first operation in the pipeline.
            tail_id: ID of the last operation in the pipeline.
        """
        self._graph = graph
        self._head_id = head_id
        self._tail_id = tail_id
        self._plan: ExecutionPlan | None = None
        self._ctx_factory = ctx_factory
        self._ctx_type = ctx_type

    # BACK-COMPAT SHIMS
    @property
    def context_type(self) -> type[BaseContext] | None:
        """Backwards-compat attr expected by composition helpers & tests."""
        return self._ctx_type

    @property
    def is_bound(self) -> bool:
        """
        "Bound" ≈ the first node already owns its first argument.
        We treat an op as bound when its head-template contains *no* placeholders
        but does contain constants (positional or keyword).
        """
        head = self._graph._nodes[self._head_id]
        tpl = head.template
        # TODO: should placeholders be considered bound?
        return bool(tpl.args or tpl.kwargs) and not tpl.has_placeholders()

    @classmethod
    def _from_function(
        cls,
        fn: Callable[P2, Awaitable[R2]] | Callable[P2, R2],
        *,
        require_ctx: bool = False,
        ctx_type: type[BaseContext] | None = None,
    ) -> "Operation[P2, R2]":
        """Create an Operation from a function.

        Args:
            fn: The function to wrap as an Operation.

        Returns:
            A new Operation instance.
        """
        spec = OpSpec(
            id=str(uuid.uuid4()),
            func=_ensure_async(fn),
            signature=inspect.signature(fn),
            ctx_type=ctx_type,
            require_ctx=require_ctx,
            template=Template(),
        )

        g = OpGraph()
        g.add_node(spec)
        return cast(
            "Operation[P2, R2]",
            cls(
                graph=g,
                head_id=spec.id,
                tail_id=spec.id,
                ctx_type=ctx_type,  # ← preserve the context metadata
            ),
        )

    def __call__(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> "_BoundCall[P, R]" | "Operation[P, R]":
        """Call the operation with the given arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Either a new Operation (if single step) or a _BoundCall.
        """
        if self._is_single_step():
            first_step_id = self._head_id
            spec = self._graph._nodes[first_step_id]
            
            # Merge old + new arguments
            prev_tpl = spec.template
            
            # positional: simply concatenate
            merged_args = (*prev_tpl.args, *args) if args else prev_tpl.args
            
            # kwargs: runtime overrides template
            merged_kwargs = (
                {**prev_tpl.kwargs, **kwargs} if kwargs else dict(prev_tpl.kwargs)
            )
            
            bound_tpl: Template = Template(args=merged_args, kwargs=merged_kwargs)

            # new id avoids duplicate node errors when the same op is reused
            new_spec = replace(spec, id=str(uuid.uuid4()), template=bound_tpl)
            new_graph = OpGraph()
            new_graph.add_node(new_spec)

            return Operation(
                graph=new_graph,
                head_id=new_spec.id,
                tail_id=new_spec.id,
                ctx_factory=self._ctx_factory,
                ctx_type=self._ctx_type,
            )

        return _BoundCall(pipeline=self, args=args, kwargs=kwargs)

    def __await__(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> Generator[Any, None, Result[R, Exception]]:
        """Make the operation awaitable.

        Returns:
            Awaitable that resolves to the execution result.
        """
        return self.execute(*args, **kwargs).__await__()

    def __rshift__(self, other: "Operation[Concatenate[R, Q], S]") -> "Operation[P, S]":
        """Compose this operation with another operation.

        Args:
            other: Another Operation or callable to compose with.

        Returns:
            A new Operation representing the composition.

        Raises:
            TypeError: If other is not an Operation or callable.
        """
        if not isinstance(other, Operation):
            if callable(other):
                other = Operation._from_function(other)
            else:
                raise TypeError("Right operand to >> must be an Operation.")

        # make sure we can re-use the *same* op multiple times: clone RHS
        other = other._clone()

        merged = self._graph.merged_with(other._graph)
        combined: Operation[P, S] = Operation(
            graph=merged,
            head_id=self._head_id,
            tail_id=other._tail_id,
        )
        # copy whichever side already has ctx meta
        combined._ctx_factory = self._ctx_factory or other._ctx_factory
        combined._ctx_type = self._ctx_type or other._ctx_type
        return combined

    def __repr__(self) -> str:
        """Return a string representation of the operation.

        Returns:
            A string representation.
        """
        names = " >> ".join(
            spec.func.__name__ for spec in self._graph.topological_order()
        )
        return f"<Operation {names}>"

    def validate(self) -> None:
        """Validate the operation pipeline.

        Raises:
            ValueError: If the pipeline is empty.
        """
        if not self._graph.nodes:
            raise ValueError("Pipeline is empty.")
        self._compile()

    def _compile(self) -> ExecutionPlan:
        """Compile the operation graph into an execution plan.

        Returns:
            The compiled execution plan.
        """
        if self._plan is None:
            self._plan = ExecutionPlan.from_graph(self._graph)
        return self._plan

    @classmethod
    def with_context(
        cls,
        ctx_or_factory: (
            BaseContext | Callable[[], BaseContext | Awaitable[BaseContext]]
        ),
        *,
        context_type: type[BaseContext],
    ) -> "Operation[..., BaseContext]":
        """
        Produce a single-step pipeline that **creates and yields** a context
        instance.  The factory may be synchronous or asynchronous.
        """
        factory = ctx_or_factory if callable(ctx_or_factory) else lambda: ctx_or_factory

        # the step just returns whatever context the executor injects
        async def _yield(**kwargs: Any) -> BaseContext:
            return cast(BaseContext, kwargs["context"])

        op: Operation[..., BaseContext] = cls._from_function(  # type: ignore[arg-type]
            _yield, require_ctx=True, ctx_type=context_type
        )
        op._ctx_factory = factory  # remember how to build the ctx
        return op

    def _clone(self) -> "Operation[P, R]":
        """
        Duplicate this operation *and its internal graph*; every OpSpec receives
        a brand-new UUID so the clone can live side-by-side with the original.
        """
        id_map: dict[str, str] = {}
        new_graph = OpGraph()

        # ---- nodes
        for spec in self._graph.nodes:
            new_id = uuid.uuid4().hex
            id_map[spec.id] = new_id
            new_graph.add_node(replace(spec, id=new_id))

        # ---- edges
        for e in self._graph._all_edges():
            new_graph.add_edge(
                Edge(
                    source=replace(e.source, node_id=id_map[e.source.node_id]),
                    target=replace(e.target, node_id=id_map[e.target.node_id]),
                    type=e.type,
                    transform=e.transform,
                )
            )

        return Operation(
            graph=new_graph,
            head_id=id_map[self._head_id],
            tail_id=id_map[self._tail_id],
            ctx_factory=self._ctx_factory,
            ctx_type=self._ctx_type,
        )

    async def execute(self, *args: P.args, **kwargs: P.kwargs) -> Result[R, Exception]:
        """
        Executes the operation pipeline, resolving the context as needed before delegating
        to the executor.

        Context resolution follows these rules (in order of precedence):

            1. If the caller provides a ``context`` keyword argument, it is used directly.
            2. If the pipeline was initialized with a context factory (``_ctx_factory``), the factory
               is called to produce the context. The factory may be synchronous or asynchronous.
            3. If neither of the above, no context is provided (``None``).

        Args:
            *args: Positional arguments to pass to the first operation in the pipeline.
            **kwargs: Keyword arguments to pass to the first operation in the pipeline.
                If ``context`` is present, it will be used as the context.

        Returns:
            Result[Any, Exception]: The result of executing the pipeline, wrapped in a Result.

        Raises:
            Exception: Any exception raised during context creation or pipeline execution
                will be wrapped in an Error result.
        """
        if "context" in kwargs:  # explicit from caller
            ctx = kwargs["context"]
        elif self._ctx_factory is not None:  # implicit via factory
            maybe_ctx = self._ctx_factory()
            ctx = await maybe_ctx if inspect.isawaitable(maybe_ctx) else maybe_ctx
        else:
            ctx = None  # no context in play

        plan = self._compile()
        # We forward original kwargs – the executor will inject the ctx
        # keyword if it is missing but required by a node.
        return await Executor(plan).run(
            *args, _context=cast(BaseContext | None, ctx), **kwargs
        )

    def _is_single_step(self) -> bool:
        """Check if the operation is a single step.

        Returns:
            True if the underlying graph contains exactly one node.
        """
        return len(self._graph.nodes) == 1

    def transform(self, fn: Callable[[R], S]) -> "Operation[P, S]":
        """Apply a function to the successful result of this operation.

        Args:
            fn: Function to apply to the successful value.

        Returns:
            A new Operation representing the composition.
        """
        return self >> Operation._from_function(fn)

    def filter(
        self,
        pred: Callable[[R], bool],
        err_msg: str | Exception = "filter predicate failed",
    ) -> "Operation[P, R]":
        """Filter values based on a predicate.

        Args:
            pred: Predicate function that returns True to keep the value.
            err_msg: Error message or exception to use when the predicate fails.

        Returns:
            A new Operation that filters values.
        """

        # Wrap the filter function to make it compatible with _from_function
        async def _filter_wrapper(*args: Any, **kwargs: Any) -> R:
            result = await self.execute(*args, **kwargs)
            if result.is_error():
                raise result.error

            value = cast(R, result.default_value(None))
            if not pred(value):
                if isinstance(err_msg, Exception):
                    raise err_msg
                raise ValueError(err_msg)
            return value

        return Operation._from_function(_filter_wrapper)

    def tap(self, side: Callable[[R], Any]) -> "Operation[P, R]":
        """Apply a side-effect function without changing the value.

        Args:
            side: Function to apply for side effects.

        Returns:
            A new Operation that applies the side effect.
        """
        side_sig = inspect.signature(side)
        needs_ctx = "context" in side_sig.parameters

        async def _t(x: R, **kwargs: Any) -> R:
            try:
                result = side(x, **kwargs) if needs_ctx else side(x)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
            return x

        TapStep: Operation[[R], R] = Operation._from_function(
            cast(Callable[[R], Awaitable[R]], _t),
            require_ctx=(needs_ctx or self._ctx_type is not None),
            ctx_type=self._ctx_type,
        )

        return self >> TapStep

    def catch(self, handler: Callable[[Exception], S]) -> "Operation[P, Union[R, S]]":
        """Handle errors that occur during operation execution.

        Args:
            handler: Function that takes an exception and returns a value.

        Returns:
            A new Operation that handles errors.
        """

        async def _catch(*args: P.args, **kwargs: P.kwargs) -> Union[R, S]:
            res = await self.execute(*args, **kwargs)
            if res.is_error():
                return handler(res.error)
            return cast(R, res.default_value(None))

        return Operation._from_function(_catch)

    def default_value(self, value: S) -> "Operation[P, Union[R, S]]":
        """Provide a default value if this operation fails.

        Args:
            value: Default value to use if the operation fails.

        Returns:
            A new Operation that provides a default on failure.
        """

        async def _def(*args: P.args, **kwargs: P.kwargs) -> Union[R, S]:
            res = await self.execute(*args, **kwargs)
            if res.is_error():
                return value
            return cast(R, res.default_value(None))

        return Operation._from_function(_def)

    def retry(
        self, *, attempts: int = 3, delay: float = 0.0, backoff: float = 0.0
    ) -> "Operation[P, R]":
        """Retry the operation a specified number of times on failure.

        Args:
            attempts: Maximum number of attempts.
            delay: Initial delay between attempts in seconds.
            backoff: Multiplier for the delay after each retry.

        Returns:
            A new Operation that implements retry logic.
        """

        async def _retry(*args: P.args, **kwargs: P.kwargs) -> R:
            tries_left = attempts
            sleep_for = delay
            last_error: Exception | None = None

            while tries_left:
                res = await self.execute(*args, **kwargs)
                if res.is_ok():
                    return cast(R, res.default_value(None))

                last_error = res.error
                tries_left -= 1
                if tries_left and sleep_for:
                    await asyncio.sleep(sleep_for)
                    if backoff:
                        sleep_for *= backoff

            assert last_error is not None  # for mypy
            raise last_error

        return Operation._from_function(_retry)

    def __and__(self, other: "Operation[P, S]") -> "Operation[P, tuple[R, S]]":
        """Run both operations in parallel and return a tuple of results.

        Args:
            other: Another Operation to run in parallel.

        Returns:
            A new Operation that runs both in parallel.

        Raises:
            TypeError: If other is not an Operation.
        """
        if not isinstance(other, Operation):
            raise TypeError("Operand to & must be an Operation.")

        async def _parallel(*args: P.args, **kwargs: P.kwargs) -> tuple[R, S]:
            r1, r2 = await asyncio.gather(
                self.execute(*args, **kwargs),
                other.execute(*args, **kwargs),
            )
            if r1.is_error():
                raise r1.error
            if r2.is_error():
                raise r2.error

            return (cast(R, r1.default_value(None)), cast(S, r2.default_value(None)))

        parallel_op: Operation[P, tuple[R, S]] = Operation._from_function(_parallel)
        parallel_op._ctx_factory = self._ctx_factory or getattr(
            other, "_ctx_factory", None
        )
        parallel_op._ctx_type = self._ctx_type or getattr(other, "_ctx_type", None)
        return parallel_op

    def __or__(self, other: "Operation[P, R]") -> "Operation[P, R]":
        """Try this operation first, falling back to the other if this fails.

        Args:
            other: Another Operation to use as fallback.

        Returns:
            A new Operation representing the fallback logic.

        Raises:
            TypeError: If other is not an Operation.
        """
        if not isinstance(other, Operation):
            raise TypeError("Operand to | must be an Operation.")

        async def _fallback(*args: P.args, **kwargs: P.kwargs) -> R:
            first = await self.execute(*args, **kwargs)
            if first.is_ok():
                return cast(R, first.default_value(None))
            second = await other.execute(*args, **kwargs)
            if second.is_error():
                raise second.error
            return cast(R, second.default_value(None))

        fallback_op: Operation[P, R] = Operation._from_function(_fallback)
        fallback_op._ctx_factory = self._ctx_factory or getattr(
            other, "_ctx_factory", None
        )
        fallback_op._ctx_type = self._ctx_type or getattr(other, "_ctx_type", None)
        return fallback_op

    @classmethod
    def sequence(cls, ops: Sequence["Operation"]) -> "Operation[..., Any]":
        """Create a pipeline from a sequence of operations.

        Args:
            ops: A sequence of Operations to compose.

        Returns:
            A new Operation representing the composition.

        Raises:
            ValueError: If the sequence is empty.
        """
        if not ops:
            raise ValueError("empty sequence")
        pip = ops[0]
        for op in ops[1:]:
            pip = pip >> op
        return pip

    @classmethod
    def combine(cls, **kw_ops: "Operation") -> "Operation[..., Dict[str, Any]]":
        """Combine multiple operations into one that returns a dictionary of results.

        Args:
            **kw_ops: Named operations to combine.

        Returns:
            A new Operation that runs all operations and returns their results as a dict.
        """

        async def _comb(val: Any) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for name, op in kw_ops.items():
                res = await op.execute(val)
                if res.is_error():
                    raise res.error
                out[name] = res.default_value(None)
            return out

        return Operation._from_function(_comb)

    def bind(
        self, builder: Callable[[R], "Operation[Q, S]" | Callable[..., Any]]
    ) -> "Operation[P, S]":
        """Chain this operation with another operation built from its result.

        Args:
            builder: Function that takes the result of this operation and returns
                   another Operation or callable.

        Returns:
            A new Operation representing the chained computation.

        Example:
            >>> pipeline = subtract(10, 4).bind(lambda a: divide(a, 2))
            >>> (await pipeline.execute()).default_value(None)      # 3.0
        """

        # Create a wrapper function with a more specific type signature
        # that the type checker can handle
        @wraps(builder)
        async def _bind_wrapper(*args: Any, **kwargs: Any) -> S:
            if not args:
                raise TypeError("Expected at least one argument")

            x = cast(R, args[0])
            rest_args = args[1:]

            # build the next operation (or wrap it if it's just a function)
            nxt = builder(x)
            if not isinstance(nxt, Operation):
                # make **sure** the wrapped function is marked as context-aware
                nxt = Operation._from_function(
                    nxt,
                    require_ctx=self._ctx_type is not None,
                    ctx_type=self._ctx_type,
                )

            context = kwargs.get("context")
            if context is not None:
                kwargs["context"] = context

            result = await nxt.execute(*rest_args, **kwargs)
            if result.is_error():
                raise result.error
            return cast(S, result.default_value(None))

        op: Operation[Concatenate[R, Q], S] = Operation._from_function(  # type: ignore
            _bind_wrapper,
            require_ctx=self._ctx_type is not None,
            ctx_type=self._ctx_type,
        )
        return self >> op

    def apply_cont(
        self,
        cont: Callable[
            [R], Awaitable[S] | "Operation[..., S]" | Result[S, Exception] | S
        ],
    ) -> "Operation[P, S]":
        """Apply a continuation to the successful result of this operation.

        Args:
            cont: Continuation function that takes the unwrapped result and returns
                 a raw/awaitable value, Result, or Operation.

        Returns:
            A new Operation that applies the continuation.
        """

        async def _cont_wrapper(*args: Any, **kwargs: Any) -> S:
            res = await self.execute(*args, **kwargs)
            if res.is_error():
                raise res.error
            val = cast(R, res.default_value(None))

            out = cont(val)

            if isinstance(out, Operation):
                op_result = await out.execute()
                if op_result.is_error():
                    raise op_result.error
                return cast(S, op_result.default_value(None))

            if inspect.isawaitable(out):
                out = await out

            if isinstance(out, Result):
                if out.is_error():
                    raise out.error
                return cast(S, out.default_value(None))

            return cast(S, out)

        return Operation._from_function(_cont_wrapper)

    def partial(self, *args: P.args, **kwargs: P.kwargs) -> "Operation[P, R]":
        """Create a new operation with merged arguments.
        
        For single-step operations, this behaves like functools.partial,
        preserving existing constants and overlaying new ones.
        
        Args:
            *args: Positional arguments to append.
            **kwargs: Keyword arguments to merge (new values override existing ones).
            
        Returns:
            A new Operation with merged arguments.
        """
        return cast(Operation[P, R], self(*args, **kwargs))


class _BoundCall(Generic[P, R]):
    """Stores runtime arguments until `.execute()` is invoked.

    This is an internal class used by Operation to defer execution.
    """

    __slots__ = ("_pipeline", "_args", "_kwargs")

    def __init__(
        self,
        *,
        pipeline: Operation[P, R],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """Initialize a BoundCall.

        Args:
            pipeline: The Operation to be executed.
            args: Positional arguments for the operation.
            kwargs: Keyword arguments for the operation.
        """
        self._pipeline = pipeline
        self._args = args
        self._kwargs = kwargs

    async def execute(self) -> Result[R, Exception]:
        """Execute the bound operation with the stored arguments.

        Returns:
            A Result containing either the successful value or an exception.
        """
        return await self._pipeline.execute(*self._args, **self._kwargs)

    def validate(self) -> None:
        """Validate the underlying pipeline."""
        self._pipeline.validate()

    def __repr__(self) -> str:
        """Return a string representation of the bound call.

        Returns:
            A string representation.
        """
        return f"<BoundCall of {self._pipeline!r}>"

    # TODO: is this good practice or bad practice?
    # adds the misconception that the pipeline is a function instead of a class
    # users might confuse instantiation with function call ex:
    # pipeline = add >> add_one
    # pipeline(1, 2) # this is wrong, it should be pipeline(1, 2).execute() OR pipeline.execute(1, 2)
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        """Call the underlying pipeline with new arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A new BoundCall with the given arguments.
        """
        return self._pipeline(*args, **kwargs)

    def __await__(self) -> Generator[Any, None, Result[R, Exception]]:
        """Make the bound call awaitable.

        Returns:
            Awaitable that resolves to the execution result.
        """
        return self.execute().__await__()


@overload
def operation(
    _fn: Callable[P, Awaitable[R]] | Callable[P, R],
) -> Operation[P, R]: ...


@overload
def operation(
    _fn: None = None,
    *,
    context: bool = False,
    context_type: type[BaseContext] | None = None,
) -> Callable[[Callable[P, Awaitable[R]] | Callable[P, R]], Operation[P, R]]: ...


def operation(
    _fn: Callable[P, Awaitable[R]] | Callable[P, R] | None = None,
    *,
    context: bool = False,
    context_type: type[BaseContext] | None = None,
) -> (
    Operation[P, R]
    | Callable[[Callable[P, Awaitable[R]] | Callable[P, R]], Operation[P, R]]
):
    """
    Decorator to create an Operation from a function, with optional context-awareness.

    Usage:
    ```python
        @operation
        def my_func(...):
            ...

        @operation(context=True)
        def my_ctx_func(context, ...):
            ...
    ```

    Args:
        _fn: The function to wrap (used when decorator is applied without parentheses).
        context (bool): If True, marks the operation as requiring a context argument.
        context_type (type[BaseContext] | None): Optional type to enforce for the context.

    Returns:
        Operation | Callable[[Callable[..., Any]], Operation]:
            - If used as @operation, returns an Operation wrapping the function.
            - If used as @operation(...), returns a decorator that produces an Operation.

    When used without arguments (bare @operation), wraps the function as a standard Operation.
    When used with arguments (e.g., @operation(context=True)), wraps the function and marks it as context-aware,
    enforcing that a context is provided at execution time.
    """

    def _decorate(fn: Callable[P, Awaitable[R]] | Callable[P, R]) -> Operation[P, R]:
        if isinstance(fn, Operation):
            return fn
        impl = Operation._from_function(
            fn,
            require_ctx=context,
            ctx_type=context_type,
        )
        # Preserve the docstring for better IDE experience
        if fn.__doc__:
            impl.__doc__ = fn.__doc__
        return impl

    if _fn is None:
        return _decorate

    return _decorate(_fn)


def constant(value: R) -> Operation[..., R]:
    """Create an Operation that always returns the specified value.

    Args:
        value: The constant value to return.

    Returns:
        An Operation that always returns the value.
    """

    @operation
    async def _const(*_args: Any, **_kw: Any) -> Any:
        return value

    return _const


@operation
def identity(value: Any) -> Any:
    """Create an Operation that returns its input unchanged.

    Args:
        value: The input value.

    Returns:
        The same value.
    """
    return value
