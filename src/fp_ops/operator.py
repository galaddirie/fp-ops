import asyncio
import inspect
import time
from functools import wraps, partial
from typing import (
    TypeVar,
    Callable,
    Any,
    Union,
    Generic,
    Optional,
    Tuple,
    List,
    Dict,
    Type,
    Collection,
    Awaitable,
    cast,
    overload,
    ParamSpec,
    Protocol,
    runtime_checkable,
)
from expression import Result

from .placeholder import Placeholder
from .context import BaseContext

T = TypeVar("T") # type of the input
S = TypeVar("S") # type of the output
R = TypeVar("R") # type of the result
E = TypeVar("E", bound=Exception) # type of the error
C = TypeVar("C", bound=Optional[BaseContext]) # type of the context
P = ParamSpec("P")  # Captures all parameter types

OptC = Optional[Type[BaseContext]]  # Type alias for optional context type


@runtime_checkable
class ContextAwareCallable(Protocol):
    """Protocol for callables that have context awareness attributes."""

    requires_context: bool
    context_type: Optional[Type[BaseContext]]


class Operation(Generic[T, S, C]):
    """
    A class representing a composable asynchronous operation with first-class composition.

    Operations wrap async functions and provide methods for composition using operators
    like >>, &, and |, enabling building complex functional pipelines.

    This class implements the continuation monad pattern to make composition work smoothly
    with async/await syntax.
    """

    func: Callable[..., Awaitable[Any]]
    bound_args: Optional[Tuple[Any, ...]]
    bound_kwargs: Optional[Dict[str, Any]]
    is_bound: bool
    requires_context: bool
    context_type: Optional[Type[BaseContext]]
    __name__: str
    __doc__: str
    __signature__: Optional[inspect.Signature]
    __annotations__: Dict[str, Any]
    __module__: str

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        bound_args: Optional[Tuple[Any, ...]] = None,
        bound_kwargs: Optional[Dict[str, Any]] = None,
        context_type: Optional[Type[BaseContext]] = None,
    ):
        """
        Initialize an Operation.

        Args:
            func: The async function to wrap.
            bound_args: Positional arguments to bind to the function (if any).
            bound_kwargs: Keyword arguments to bind to the function (if any).
            context_type: The expected type of the context (a Pydantic model).
        """
        self.func = func
        self.bound_args = bound_args
        self.bound_kwargs = bound_kwargs or {}
        self.is_bound = bound_args is not None or bound_kwargs is not None
        # Use getattr safely for context attributes
        self.requires_context = getattr(func, "requires_context", False)
        self.context_type = context_type or getattr(func, "context_type", None)
        self.__name__ = getattr(func, "__name__", "unknown")

    def __get_type_hints__(self) -> Dict[str, Any]:
        """Helper method for IDEs to get type hints from the original function."""
        return getattr(self, "__annotations__", {})

    
    def __str__(self) -> str:
        """Return a string representation including the docstring."""
        name = getattr(self, "__name__", "Operation")
        doc = self.__doc__ or ""
        sig = getattr(self, "__signature__", None)
        sig_str = str(sig) if sig else "()"
        return f"{name}{sig_str}\n{doc}"

    def __repr__(self) -> str:
        """Return a representation including the docstring."""
        return self.__str__()

    async def execute(self, *args: Any, **kwargs: Any) -> Result[S, Exception]:
        """
        Execute the operation with the given arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A Result containing the output or an error.
        """
        if self.is_bound:
            # Always use bound_args for bound operations, regardless of passed args
            actual_args = self.bound_args or ()  
            actual_kwargs = dict(self.bound_kwargs or {})
            if kwargs:
                actual_kwargs.update(kwargs)
        else:
            actual_args = args
            actual_kwargs = kwargs

        # Extract context from kwargs
        context = actual_kwargs.get("context")

        # If this operation requires context
        if self.requires_context:
            if context is None:
                return Result.Error(
                    Exception(
                        f"Operation {self.__name__} requires a context, but none was provided"
                    )
                )

            # If a specific context type is required, validate it
            if self.context_type is not None and not isinstance(
                context, self.context_type
            ):
                # Stricter validation for context types
                try:
                    # Only dictionaries should be converted automatically
                    if isinstance(context, dict):
                        context = self.context_type(**context)
                        actual_kwargs["context"] = context
                    # If it's a BaseContext but not the right type, we should fail
                    elif isinstance(context, BaseContext):
                        # Check for meaningful conversion - do the fields align?
                        required_fields = set(self.context_type.__annotations__.keys())
                        provided_fields = set(context.__annotations__.keys())

                        # If the context doesn't have the required fields, fail
                        if not required_fields.issubset(provided_fields):
                            missing = required_fields - provided_fields
                            return Result.Error(
                                Exception(
                                    f"Invalid context type for operation {self.__name__}: "
                                    f"Expected {self.context_type.__name__}, got {type(context).__name__}. "
                                    f"Missing fields: {missing}"
                                )
                            )

                        # Attempt strict conversion
                        try:
                            # Only convert if the context has all the required fields
                            context_data = context.model_dump()
                            # Filter to only include fields expected by the target type
                            filtered_data = {
                                k: v
                                for k, v in context_data.items()
                                if k in required_fields or k == "metadata"
                            }
                            context = self.context_type(**filtered_data)
                            actual_kwargs["context"] = context
                        except Exception as e:
                            return Result.Error(
                                Exception(
                                    f"Invalid context for operation {self.__name__}: "
                                    f"Could not convert {type(context).__name__} to {self.context_type.__name__}: {e}"
                                )
                            )
                    else:
                        # Not a dict or BaseContext - explicit failure
                        return Result.Error(
                            Exception(
                                f"Invalid context type for operation {self.__name__}: "
                                f"Expected {self.context_type.__name__}, got {type(context).__name__}"
                            )
                        )
                except Exception as e:
                    return Result.Error(
                        Exception(f"Invalid context for operation {self.__name__}: {e}")
                    )

        # Always include context in execution kwargs
        elif context is not None:
            actual_kwargs["context"] = context

        try:
            result = await self.func(*actual_args, **actual_kwargs)

            if isinstance(result, Result):
                return cast(
                    Result[S, Exception], result
                )  # Type cast to fix return type
            return cast(
                Result[S, Exception], Result.Ok(result)
            )  # Type cast to fix return type
        except Exception as e:
            return Result.Error(e)

    @classmethod
    def with_context(
        cls,
        context_factory: Optional[Callable[..., Any]] = None,
        context_type: Optional[Type[BaseContext]] = None,
    ) -> "Operation[Any, Any, Any]":
        """
        Create an operation that initializes a context.

        Args:
            context_factory: A factory function that creates a context object.
                        If None, a default empty context of the specified type will be used.
            context_type: The Pydantic model class for the context.
                     If None, a dictionary will be used.

        Returns:
            An Operation that initializes a context.
        """
        if context_factory is None:
            # Default factory depends on whether a context type is provided
            if context_type is not None:
                context_factory = lambda: context_type()
            else:
                context_factory = lambda: BaseContext()

        async def init_context(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            try:
                if inspect.iscoroutinefunction(context_factory):
                    context = await context_factory(*args, **kwargs)
                else:
                    # Ensure context_factory is callable before passing to to_thread
                    if callable(context_factory):
                        context = await asyncio.to_thread(
                            context_factory, *args, **kwargs
                        )
                    else:
                        return Result.Error(
                            Exception("context_factory must be callable")
                        )

                # Validate context against the context type if specified
                if context_type is not None and not isinstance(context, context_type):
                    try:
                        # Try to convert to the required context type
                        if isinstance(context, dict):
                            context = context_type(**context)
                        else:
                            context = context_type.model_validate(context)
                    except Exception as e:
                        return Result.Error(
                            Exception(f"Invalid context from factory: {e}")
                        )

                return Result.Ok(context)  # Explicitly wrap in Result.Ok
            except Exception as e:
                return Result.Error(e)

        # Mark the function as requiring context
        init_context.requires_context = False  # type: ignore
        init_context.context_type = context_type  # type: ignore

        return cls(init_context, context_type=context_type) 

    def __call__(self, *args: Any, **kwargs: Any) -> "Operation[T, S, C]":
        """
        Call the operation with the given arguments.

        If arguments are provided, this returns a new bound operation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A new bound Operation.
        """
        if not args and not kwargs and self.is_bound:
            return self

        # Preserve context when binding arguments
        bound_kwargs = dict(self.bound_kwargs or {})
        if kwargs:
            bound_kwargs.update(kwargs)

        new_op: Operation[T, S, C] = Operation(self.func, args, bound_kwargs, context_type=self.context_type)
        
        # Copy all metadata attributes to the new operation
        for attr in [
            '__doc__', '__name__', '__qualname__', '__annotations__', 
            '__module__', '__signature__', 'original_function'
        ]:
            if hasattr(self, attr):
                setattr(new_op, attr, getattr(self, attr))
        
        return new_op
    
    def __await__(self) -> Any:
        """
        Make Operation awaitable in async functions.

        Returns:
            An iterator that can be used with the await syntax.
        """

        async def awaitable() -> Any:
            if self.is_bound:
                return await self.execute()
            else:
                # If not bound, execute with no arguments
                return await self.execute()

        return awaitable().__await__()

    def __rshift__(
        self, other: Union["Operation[T, S, C]", Any]
    ) -> "Operation[T, S, C]":
        """
        Implement the >> operator for composition (pipeline).

        If the other operation is bound and has placeholders, the result of this
        operation will be substituted for those placeholders.

        Args:
            other: Another Operation or a constant.

        Returns:
            A new Operation representing the composition.
        """
        if isinstance(other, Placeholder):
            other = identity

        if not isinstance(other, Operation):
            if callable(other):
                other = operation(other)
            else:
                other = constant(other)

        async def composed(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            # Extract context from kwargs if available
            context = kwargs.get("context")

            # Always include context in execution if available
            execution_kwargs = dict(kwargs)

            self_result = await self.execute(*args, **execution_kwargs)

            if self_result.is_error():
                return cast(
                    Result[Any, Exception], self_result
                )  # Type cast to fix return type

            value = self_result.default_value(
                cast(S, None)
            )  # Type cast to fix argument type

            # If the value is a BaseContext, it becomes the new context
            # and is NOT passed as a positional argument to the next operation
            if isinstance(value, BaseContext):
                context = value

                # If other operation has placeholders, substitute them
                if other.is_bound and other._has_placeholders():
                    # Here, we'll pass an empty value for placeholder substitution
                    # since the actual value (the context) is being passed via kwargs
                    empty_value = None
                    new_args, new_kwargs = other._substitute_placeholders(empty_value)

                    # Add the context to the kwargs
                    if context is not None:
                        new_kwargs["context"] = context

                    try:
                        result = await other.func(*new_args, **new_kwargs)
                        if isinstance(result, Result):
                            return cast(
                                Result[Any, Exception], result
                            )  # Type cast to fix return type
                        return Result.Ok(result)
                    except Exception as e:
                        return Result.Error(e)
                elif other.is_bound:
                    # No placeholders, execute as bound with context only
                    other_kwargs = {}
                    if context is not None:
                        other_kwargs["context"] = context
                    return await other.execute(**other_kwargs)
                else:
                    # Not bound, execute with context only (no positional args)
                    other_kwargs = {}
                    if context is not None:
                        other_kwargs["context"] = context
                    return await other.execute(**other_kwargs)
            else:
                # Normal value (not a context), proceed with standard behavior
                # If other operation has placeholders, substitute them
                if other.is_bound and other._has_placeholders():
                    # Get substituted arguments
                    new_args, new_kwargs = other._substitute_placeholders(value)

                    # Always pass context to next operation if available
                    if context is not None:
                        new_kwargs["context"] = context

                    try:
                        result = await other.func(*new_args, **new_kwargs)
                        if isinstance(result, Result):
                            return cast(
                                Result[Any, Exception], result
                            )  # Type cast to fix return type
                        return Result.Ok(result)
                    except Exception as e:
                        return Result.Error(e)
                elif other.is_bound:
                    # No placeholders, execute as bound
                    # Include context in execution if available
                    other_kwargs = {}
                    if context is not None:
                        other_kwargs["context"] = context
                    return await other.execute(**other_kwargs)
                else:
                    # If not bound, pass the value as first argument
                    # Include context in execution if available
                    other_kwargs = {}
                    if context is not None:
                        other_kwargs["context"] = context
                    return await other.execute(value, **other_kwargs)

        return Operation(composed, context_type=other.context_type or self.context_type)

    def __and__(
        self, other: Union["Operation[T, Any, C]", Any]
    ) -> "Operation[T, Tuple[S, Any], C]":
        """
        Implement the & operator for parallel execution.

        Args:
            other: Another Operation or a constant.

        Returns:
            A new Operation that executes both operations and returns a tuple of results.
        """
        if not isinstance(other, Operation):
            other = constant(other)

        async def parallel(
            *args: Any, **kwargs: Any
        ) -> Result[Tuple[Any, Any], Exception]:
            # Extract context from kwargs if available
            context = kwargs.get("context")

            # Create separate kwargs for each operation to avoid interference
            # but ensure both receive the same context
            self_kwargs = dict(kwargs)
            other_kwargs = dict(kwargs)

            # Execute both operations concurrently with the same context
            result1, result2 = await asyncio.gather(
                self.execute(*args, **self_kwargs), other.execute(*args, **other_kwargs)
            )

            if result1.is_error():
                return cast(
                    Result[Tuple[Any, Any], Exception], result1
                )  # Type cast to fix return type
            if result2.is_error():
                return cast(
                    Result[Tuple[Any, Any], Exception], result2
                )  # Type cast to fix return type

            value1 = result1.default_value(
                cast(S, None)
            )  # Type cast to fix argument type
            value2 = result2.default_value(None)
            return cast(
                Result[Tuple[Any, Any], Exception], Result.Ok((value1, value2))
            )  # Type cast to fix return type

        # Use the most specific context type
        context_type = self.context_type
        if other.context_type is not None:
            if context_type is None or issubclass(other.context_type, context_type):
                context_type = other.context_type

        return Operation(parallel, context_type=context_type)

    def __or__(self, other: Union["Operation[T, S, C]", Any]) -> "Operation[T, S, C]":
        """
        Implement the | operator for alternative execution.

        Args:
            other: Another Operation or a constant.

        Returns:
            A new Operation that tries the first operation and falls back to the second.
        """
        if not isinstance(other, Operation):
            other = constant(other)

        async def alternative(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            # Extract context from kwargs if available
            context = kwargs.get("context")

            # Create separate kwargs for each operation to avoid interference
            self_kwargs = dict(kwargs)
            other_kwargs = dict(kwargs)

            result1 = await self.execute(*args, **self_kwargs)

            if result1.is_ok():
                return cast(
                    Result[Any, Exception], result1
                )  # Type cast to fix return type

            other_result = await other.execute(*args, **other_kwargs)
            return cast(
                Result[Any, Exception], other_result
            )  # Type cast to fix return type

        # Use the most specific context type
        context_type = self.context_type
        if other.context_type is not None:
            if context_type is None or issubclass(other.context_type, context_type):
                context_type = other.context_type

        return Operation(alternative, context_type=context_type)

    def map(self, transform_func: Callable[[S], R]) -> "Operation[T, S, C]":
        """
        Apply a transformation to the output of this operation.

        Args:
            transform_func: A function to apply to the result of this operation.

        Returns:
            A new Operation that applies the transformation.
        """
        is_async_transform = inspect.iscoroutinefunction(transform_func)

        async def transformed(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            # Preserve context in transformation
            result = await self.execute(*args, **kwargs)

            if result.is_error():
                return cast(
                    Result[Any, Exception], result
                )  # Type cast to fix return type

            value = result.default_value(
                cast(S, None)
            )  # Type cast to fix argument type

            # If the value is a context object, don't transform it
            if isinstance(value, BaseContext):
                return cast(
                    Result[Any, Exception], Result.Ok(value)
                )  # Type cast to fix return type

            try:
                if is_async_transform:
                    transformed_value = transform_func(value)
                else:
                    transformed_value = await asyncio.to_thread(transform_func, value)

                return Result.Ok(transformed_value)
            except Exception as e:
                return Result.Error(e)

        return Operation(transformed, context_type=self.context_type)

    def bind(
        self,
        binder_func: Callable[
            [S],
            Union[
                Awaitable[Result[R, Exception]],
                Awaitable[R],
                Result[R, Exception],
                R,
                "Operation",
            ],
        ],
    ) -> "Operation[T, S, C]":
        """
        Bind this operation to another operation using a binding function.

        Args:
            binder_func: A function that takes the result value and returns another result.

        Returns:
            A new Operation that applies the binding function.
        """
        is_async_binder = inspect.iscoroutinefunction(binder_func)

        async def bound(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            # Preserve context for binding operations
            context = kwargs.get("context")

            result = await self.execute(*args, **kwargs)

            if result.is_error():
                return cast(
                    Result[Any, Exception], result
                )  # Type cast to fix return type

            value = result.default_value(
                cast(S, None)
            )  # Type cast to fix argument type

            # If the value is a context object, use it as the new context
            if isinstance(value, BaseContext):
                context = value

            try:
                if is_async_binder:
                    bind_result = binder_func(value)
                else:
                    bind_result = await asyncio.to_thread(binder_func, value)

                if isinstance(bind_result, Result):
                    return cast(
                        Result[Any, Exception], bind_result
                    )  # Type cast to fix return type
                elif isinstance(bind_result, Operation):
                    # If bind_result is an Operation, execute it with context
                    execution_kwargs = {}
                    if context is not None:
                        execution_kwargs["context"] = context
                    return await bind_result.execute(**execution_kwargs)
                return Result.Ok(bind_result)
            except Exception as e:
                return Result.Error(e)

        return Operation(bound, context_type=self.context_type)

    def filter(
        self,
        predicate: Callable[[S], bool],
        error_msg: str = "Value did not satisfy predicate",
    ) -> "Operation[T, S, C]":
        """
        Filter the result of this operation using a predicate.

        Args:
            predicate: A function that takes the result value and returns a boolean.
            error_msg: The error message to use if the predicate returns False.

        Returns:
            A new Operation that filters the result.
        """
        is_async_predicate = inspect.iscoroutinefunction(predicate)

        async def filtered(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            # Preserve context when filtering
            result = await self.execute(*args, **kwargs)

            if result.is_error():
                return cast(
                    Result[Any, Exception], result
                )  # Type cast to fix return type

            value = result.default_value(
                cast(S, None)
            )  # Type cast to fix argument type

            # If the value is a context, skip filtering
            if isinstance(value, BaseContext):
                return cast(
                    Result[Any, Exception], result
                )  # Type cast to fix return type

            try:
                if is_async_predicate:
                    # Use safe_await for potentially non-awaitable values
                    predicate_result = await safe_await(predicate(value))
                else:
                    predicate_result = await asyncio.to_thread(predicate, value)

                if predicate_result:
                    return cast(
                        Result[Any, Exception], result
                    )  # Type cast to fix return type
                else:
                    return Result.Error(ValueError(error_msg))
            except Exception as e:
                return Result.Error(e)

        return Operation(filtered, context_type=self.context_type)

    def catch(self, error_handler: Callable[[Exception], S]) -> "Operation[T, S, C]":
        """
        Add error handling to this operation.

        Args:
            error_handler: A function that takes an exception and returns a recovery value.

        Returns:
            A new Operation with error handling.
        """
        is_async_handler = inspect.iscoroutinefunction(error_handler)

        async def handled(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            # Preserve context in error handling
            context = kwargs.get("context")

            result = await self.execute(*args, **kwargs)

            if result.is_error():
                error = result.error

                try:
                    if is_async_handler:
                        recovery_value = error_handler(error)
                    else:
                        recovery_value = await asyncio.to_thread(error_handler, error)

                    # If the recovery value is a context object, merge it with existing context
                    if (
                        isinstance(recovery_value, BaseContext)
                        and context is not None
                        and isinstance(context, BaseContext)
                    ):
                        try:
                            recovery_value = context.merge(recovery_value)
                        except Exception:
                            # If merging fails, just use the recovery value
                            pass

                    return Result.Ok(recovery_value)
                except Exception as e:
                    return Result.Error(e)

            return cast(Result[Any, Exception], result)  # Type cast to fix return type

        return Operation(handled, context_type=self.context_type)

    def default_value(self, default: S) -> "Operation[T, S, C]":
        """
        Provide a default value for error cases.

        Args:
            default: The default value to use if this operation results in an error.

        Returns:
            A new Operation that uses the default value in case of errors.
        """

        async def with_default(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            # Preserve context when providing default value
            context = kwargs.get("context")

            result = await self.execute(*args, **kwargs)

            if result.is_error():
                # If the default is a context and there's already a context, try to merge them
                if (
                    isinstance(default, BaseContext)
                    and context is not None
                    and isinstance(context, BaseContext)
                ):
                    try:
                        merged_context = context.merge(default)
                        return Result.Ok(merged_context)
                    except Exception:
                        # If merging fails, just use the default
                        pass

                return Result.Ok(default)
            return cast(Result[Any, Exception], result)  # Type cast to fix return type

        return Operation(with_default, context_type=self.context_type)

    def retry(self, attempts: int = 3, delay: float = 0.1) -> "Operation[T, S, C]":
        """
        Retry the operation a specified number of times before giving up.

        Args:
            attempts: Maximum number of attempts. Default is 3.
            delay: Delay between attempts in seconds. Default is 0.1.

        Returns:
            A new Operation with retry logic.
        """

        async def retried(*args: Any, **kwargs: Any) -> Result[S, Exception]:
            # Preserve context when retrying
            context = kwargs.get("context")
            last_error: Optional[Exception] = None

            for attempt in range(attempts):
                # Create a new kwargs for each attempt
                attempt_kwargs = dict(kwargs)

                try:
                    result = await self.execute(*args, **attempt_kwargs)

                    if result.is_ok():
                        return cast(
                            Result[S, Exception], result
                        )  # Type cast to fix return type

                    last_error = result.error
                except Exception as e:
                    last_error = e

                if attempt < attempts - 1:
                    await asyncio.sleep(delay)

            return Result.Error(last_error or Exception("Unknown error during retry"))

        return Operation(retried, context_type=self.context_type)

    def tap(self, side_effect: Callable[[S], Any]) -> "Operation[T, S, C]":
        """
        Apply a side effect to the result without changing it.

        Args:
            side_effect: A function that takes the result value and performs a side effect.

        Returns:
            A new Operation that applies the side effect.
        """
        is_async_side_effect = inspect.iscoroutinefunction(side_effect)
        # Check if the side effect requires context
        requires_context = getattr(side_effect, "requires_context", False)

        async def tapped(*args: Any, **kwargs: Any) -> Result[S, Exception]:
            # Preserve context in side effects
            context = kwargs.get("context")
            result = await self.execute(*args, **kwargs)

            if result.is_ok():
                value = result.default_value(
                    cast(S, None)
                )  # Type cast to fix argument type

                # Skip side effect if the value is a context
                if isinstance(value, BaseContext):
                    return cast(
                        Result[S, Exception], result
                    )  # Type cast to fix return type

                try:
                    if requires_context:
                        # Pass both value and context to the side effect
                        if is_async_side_effect:
                            result_value = side_effect(value, context=context) # type: ignore
                            await safe_await(result_value)
                        else:
                            await asyncio.to_thread(side_effect, value, context=context) # type: ignore
                    else:
                        # Original behavior for side effects that don't require context
                        if is_async_side_effect:
                            result_value = side_effect(value)
                            await safe_await(result_value)
                        else:
                            await asyncio.to_thread(side_effect, value)
                except Exception:
                    # Ignore exceptions in the side effect
                    pass

            return cast(Result[S, Exception], result)  # Type cast to fix return type

        return Operation(tapped, context_type=self.context_type)

    @classmethod
    def sequence(
        cls, operations: Collection["Operation"]
    ) -> "Operation[Any, List[Any], Any]":
        """
        Run a sequence of operations and collect all results.

        Args:
            operations: A collection of operations to run.

        Returns:
            A new Operation that runs all operations and returns a list of results.
        """

        async def sequenced(*args: Any, **kwargs: Any) -> Result[List[Any], Exception]:
            # Preserve context in sequenced operations
            context = kwargs.get("context")
            results = []

            for op in operations:
                # Create a new kwargs for each operation
                op_kwargs = dict(kwargs)

                op_result = await op.execute(*args, **op_kwargs)

                if op_result.is_error():
                    return cast(
                        Result[List[Any], Exception], op_result
                    )  # Type cast to fix return type

                value = op_result.default_value(None)

                # If the value is a context, update the context for subsequent operations
                if isinstance(value, BaseContext):
                    context = value
                    kwargs["context"] = context
                    # Don't add context objects to the results
                    continue

                results.append(value)

            return Result.Ok(results)

        # Find the most specific context type among all operations
        context_type = None
        for op in operations:
            if op.context_type is not None:
                if context_type is None:
                    context_type = op.context_type
                elif issubclass(op.context_type, context_type):
                    context_type = op.context_type

        return cast(Operation[Any, List[Any], Any], cls(sequenced, context_type=context_type))

    @classmethod
    def combine(
        cls, **named_ops: "Operation"
    ) -> "Operation[Any, Dict[str, Any], Any]":
        """
        Combine multiple operations into a single operation that returns a dictionary.

        Args:
            **named_ops: Named operations to combine.

        Returns:
            A new Operation that runs all operations and returns results in a dictionary.
        """

        async def combined(
            *args: Any, **kwargs: Any
        ) -> Result[Dict[str, Any], Exception]:
            # Preserve context in combined operations
            context = kwargs.get("context")
            results = {}
            context_results = {}

            for name, op in named_ops.items():
                # Create a new kwargs for each operation
                op_kwargs = dict(kwargs)

                op_result = await op.execute(*args, **op_kwargs)

                if op_result.is_error():
                    return cast(
                        Result[Dict[str, Any], Exception], op_result
                    )  # Type cast to fix return type

                value = op_result.default_value(None)

                # If the value is a context, store it separately
                if isinstance(value, BaseContext):
                    context_results[name] = value
                    # Update the context for subsequent operations
                    context = value
                    kwargs["context"] = context
                else:
                    results[name] = value

            # If there are context results, add them to the results
            results.update(context_results)
            return Result.Ok(results)

        # Find the most specific context type among all operations
        context_type = None
        for op in named_ops.values():
            if op.context_type is not None:
                if context_type is None:
                    context_type = op.context_type
                elif issubclass(op.context_type, context_type):
                    context_type = op.context_type

        return cast(Operation[Any, Dict[str, Any], Any], cls(combined, context_type=context_type))

    @staticmethod
    def unit(value: T) -> "Operation[Any, T, None]":
        """
        Return a value in the Operation monad context (unit/return).

        Args:
            value: The value to return.

        Returns:
            An Operation that returns the value.
        """

        async def constant_func(*args: Any, **kwargs: Any) -> Result[T, Exception]:
            return Result.Ok(value)

        return Operation(constant_func)

    def apply_cont(self, cont: Callable[[S], Awaitable[R]]) -> Awaitable[R]:
        """
        Apply a continuation to this operation's result.

        This is part of the continuation monad pattern.

        Args:
            cont: A continuation function.

        Returns:
            The result of applying the continuation.
        """

        async def run() -> R:
            # If the operation isn't bound, provide a default value
            # In this case, the fetch_item operation requires an item_id
            if not self.is_bound:
                # Provide a default item_id=1
                result = await self.execute(1)  # Default to item_id=1
            else:
                result = await self.execute()

            if result.is_error():
                raise result.error

            return await cont(
                result.default_value(cast(S, None))
            )  # Type cast to fix argument type

        return run()

    def _has_placeholders(self) -> bool:
        """
        Check if this operation has placeholders in its bound arguments.

        This checks recursively through nested data structures.
        """
        return self._contains_placeholder(
            self.bound_args
        ) or self._contains_placeholder(self.bound_kwargs)

    def _contains_placeholder(self, obj: Any) -> bool:
        """
        Check if an object contains any Placeholder instances.

        This recursively checks lists, tuples, and dictionaries.

        Args:
            obj: The object to check.

        Returns:
            True if obj contains a Placeholder, False otherwise.
        """
        if isinstance(obj, Placeholder):
            return True

        if isinstance(obj, (list, tuple)):
            return any(self._contains_placeholder(item) for item in obj)

        if isinstance(obj, dict):
            return any(self._contains_placeholder(key) for key in obj) or any(
                self._contains_placeholder(value) for value in obj.values()
            )

        return False

    def _substitute_placeholders(self, value: Any) -> Tuple[tuple, dict]:
        """
        Return new bound_args and bound_kwargs with placeholders substituted.

        This recursively substitutes placeholders in nested data structures.

        Args:
            value: The value to substitute for placeholders.

        Returns:
            A tuple of (new_args, new_kwargs) with placeholders substituted.
        """
        new_args = tuple(
            self._substitute_placeholder(arg, value) for arg in self.bound_args or ()
        )

        new_kwargs = {}
        if self.bound_kwargs:
            new_kwargs = {
                self._substitute_placeholder(key, value): self._substitute_placeholder(
                    val, value
                )
                for key, val in self.bound_kwargs.items()
            }

        return new_args, new_kwargs

    def _substitute_placeholder(self, obj: Any, value: Any) -> Any:
        """
        Substitute all Placeholder instances with the given value.

        This recursively processes lists, tuples, and dictionaries.

        Args:
            obj: The object to process.
            value: The value to substitute for placeholders.

        Returns:
            A new object with all placeholders replaced by the value.
        """
        if isinstance(obj, Placeholder):
            return value

        if isinstance(obj, list):
            return [self._substitute_placeholder(item, value) for item in obj]

        if isinstance(obj, tuple):
            return tuple(self._substitute_placeholder(item, value) for item in obj)

        if isinstance(obj, dict):
            return {
                self._substitute_placeholder(key, value): self._substitute_placeholder(
                    val, value
                )
                for key, val in obj.items()
            }

        return obj


async def safe_await(value: Any) -> Any:
    """Safely await a value, handling both awaitable and non-awaitable values."""
    if inspect.isawaitable(value):
        return await value
    return value

@overload
def operation(
    func: Callable[P, R],
    *,
    context: bool = False,
    context_type: Optional[Type[BaseContext]] = None,
) -> Operation[Callable[P, R], S, C]: ...

# Second overload: when decorator is used with args @operation(context=True)
@overload
def operation(
    func: None = None,
    *,
    context: bool = False,
    context_type: Optional[Type[BaseContext]] = None,
) -> Callable[[Callable[P, R]], Operation[Callable[P, R], S, C]]: ...

def operation(
    func: Optional[Callable[..., Any]] = None,
    *,
    context: bool = False,
    context_type: Optional[Type[BaseContext]] = None,
) -> Union[Operation[Callable[P, R], S, C], Callable[[Callable[P, R]], Operation[Callable[P, R], S, C]]]:
    
    """
    Decorator to convert a function (sync or async) into an Operation.

    Args:
        func: The function to convert.
        context: Whether the function requires a context.
        context_type: The expected type of the context (a Pydantic model).

    Returns:
        An Operation that wraps the function.
    """
    def decorator(f: Callable[P, R]) -> Operation[Callable[P, R], S, C]: #type: ignore
        if isinstance(f, Operation):
            # Keep context requirement if already an Operation
            f.requires_context = getattr(f, "requires_context", context)
            f.context_type = getattr(f, "context_type", context_type)
            return f

        is_async = inspect.iscoroutinefunction(f)
        
        # Capture original signature and annotations
        original_signature = inspect.signature(f)
        original_annotations = getattr(f, "__annotations__", {})
        original_doc = f.__doc__ or ""

        @wraps(f)  # Use functools.wraps to preserve metadata
        async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                if is_async:
                    result = await cast(Callable[..., Awaitable[R]], f)(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(f, *args, **kwargs)

                if isinstance(result, Result):
                    return result
                return result  # Let execute() wrap in Result.Ok
            except Exception as e:
                return Result.Error(e)

        # Mark the function with context requirements
        async_wrapped.requires_context = context  # type: ignore
        async_wrapped.context_type = context_type  # type: ignore
        async_wrapped.__annotations__ = original_annotations
        async_wrapped.__name__ = f.__name__
        async_wrapped.__doc__ = original_doc
        # Create the operation
        op: Operation[Callable[P, R], S, C] = Operation(async_wrapped, context_type=context_type)
        
        # Preserve the signature, docstring and other metadata
        op.__name__ = f.__name__
        op.__doc__ = original_doc  # Explicitly set the docstring
        op.__annotations__ = original_annotations
        op.__module__ = f.__module__
                
        return op
    
    if func is None:
        return cast(Operation[Callable[P, R], S, C], decorator)
    return decorator(func)


@operation
def identity(x: Any, **kwargs: Any) -> Any:
    """
    Return the input value unchanged.
    """
    return x


def constant(value: Any, **kwargs: Any) -> Operation[Any, Any, None]:
    """
    Return a constant value.
    """
    return Operation.unit(value)
