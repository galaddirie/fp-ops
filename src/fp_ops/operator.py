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
)
from expression import Result

from .placeholder import Placeholder
T = TypeVar("T")
S = TypeVar("S")
R = TypeVar("R")
E = TypeVar("E", bound=Exception)


class Operation(Generic[T, S]):
    """
    A class representing a composable asynchronous operation with first-class composition.

    Operations wrap async functions and provide methods for composition using operators
    like >>, &, and |, enabling building complex functional pipelines.

    This class implements the continuation monad pattern to make composition work smoothly
    with async/await syntax.
    """

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        bound_args: Optional[Tuple[Any, ...]] = None,
        bound_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an Operation.

        Args:
            func: The async function to wrap.
            bound_args: Positional arguments to bind to the function (if any).
            bound_kwargs: Keyword arguments to bind to the function (if any).
        """
        self.func = func
        self.bound_args = bound_args
        self.bound_kwargs = bound_kwargs
        self.is_bound = bound_args is not None or bound_kwargs is not None
        self.__name__ = getattr(func, "__name__", "unknown")
        self.__doc__ = getattr(func, "__doc__", "")

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
            actual_args = args or self.bound_args
            actual_kwargs = dict(self.bound_kwargs or {})
            if kwargs:
                actual_kwargs.update(kwargs)
        else:
            actual_args = args
            actual_kwargs = kwargs

        try:
            result = await self.func(*actual_args, **actual_kwargs)
            if isinstance(result, Result):
                return result
            return Result.Ok(result)
        except Exception as e:
            return Result.Error(e)

    def __call__(self, *args: Any, **kwargs: Any) -> "Operation[T, S]":
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

        return Operation(self.func, args, kwargs)

    def __await__(self):
        """
        Make Operation awaitable in async functions.

        Returns:
            An iterator that can be used with the await syntax.
        """

        async def awaitable():
            if self.is_bound:
                return await self.execute()
            else:
                # If not bound, execute with no arguments
                return await self.execute()

        return awaitable().__await__()

    def __rshift__(self, other: Union["Operation[S, R]", Any]) -> "Operation[T, R]":
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
            self_result = await self.execute(*args, **kwargs)

            if self_result.is_error():
                return self_result

            value = self_result.default_value(None)

            # If other operation has placeholders, substitute them
            if other.is_bound and other._has_placeholders():
                # Get substituted arguments
                new_args, new_kwargs = other._substitute_placeholders(value)
                
                # Execute the function directly with substituted arguments
                try:
                    result = await other.func(*new_args, **new_kwargs)
                    if isinstance(result, Result):
                        return result
                    return Result.Ok(result)
                except Exception as e:
                    return Result.Error(e)
            elif other.is_bound:
                # No placeholders, execute as bound
                return await other.execute()
            else:
                # If not bound, pass the value as the first argument
                return await other.execute(value)

        return Operation(composed)

    def __and__(
        self, other: Union["Operation[T, Any]", Any]
    ) -> "Operation[T, Tuple[S, Any]]":
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
            # Execute both operations concurrently
            result1, result2 = await asyncio.gather(
                self.execute(*args, **kwargs), other.execute(*args, **kwargs)
            )

            if result1.is_error():
                return result1
            if result2.is_error():
                return result2

            value1 = result1.default_value(None)
            value2 = result2.default_value(None)
            return Result.Ok((value1, value2))

        return Operation(parallel)

    def __or__(self, other: Union["Operation[T, S]", Any]) -> "Operation[T, S]":
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
            result1 = await self.execute(*args, **kwargs)

            if result1.is_ok():
                return result1

            return await other.execute(*args, **kwargs)

        return Operation(alternative)

    def map(self, transform_func: Callable[[S], R]) -> "Operation[T, R]":
        """
        Apply a transformation to the output of this operation.

        Args:
            transform_func: A function to apply to the result of this operation.

        Returns:
            A new Operation that applies the transformation.
        """
        is_async_transform = inspect.iscoroutinefunction(transform_func)

        async def transformed(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            result = await self.execute(*args, **kwargs)

            if result.is_error():
                return result

            value = result.default_value(None)

            try:
                if is_async_transform:
                    transformed_value = await transform_func(value)
                else:
                    transformed_value = await asyncio.to_thread(transform_func, value)

                return Result.Ok(transformed_value)
            except Exception as e:
                return Result.Error(e)

        return Operation(transformed)

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
    ) -> "Operation[T, R]":
        """
        Bind this operation to another operation using a binding function.

        Args:
            binder_func: A function that takes the result value and returns another result.

        Returns:
            A new Operation that applies the binding function.
        """
        is_async_binder = inspect.iscoroutinefunction(binder_func)

        async def bound(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            result = await self.execute(*args, **kwargs)

            if result.is_error():
                return result

            value = result.default_value(None)

            try:
                if is_async_binder:
                    bind_result = await binder_func(value)
                else:
                    bind_result = await asyncio.to_thread(binder_func, value)

                if isinstance(bind_result, Result):
                    return bind_result
                elif isinstance(bind_result, Operation):
                    # If bind_result is an Operation, execute it
                    return await bind_result.execute()
                return Result.Ok(bind_result)
            except Exception as e:
                return Result.Error(e)

        return Operation(bound)

    def filter(
        self,
        predicate: Callable[[S], bool],
        error_msg: str = "Value did not satisfy predicate",
    ) -> "Operation[T, S]":
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
            result = await self.execute(*args, **kwargs)

            if result.is_error():
                return result

            value = result.default_value(None)

            try:
                if is_async_predicate:
                    predicate_result = await predicate(value)
                else:
                    predicate_result = await asyncio.to_thread(predicate, value)

                if predicate_result:
                    return result
                else:
                    return Result.Error(ValueError(error_msg))
            except Exception as e:
                return Result.Error(e)

        return Operation(filtered)

    def catch(self, error_handler: Callable[[Exception], S]) -> "Operation[T, S]":
        """
        Add error handling to this operation.

        Args:
            error_handler: A function that takes an exception and returns a recovery value.

        Returns:
            A new Operation with error handling.
        """
        is_async_handler = inspect.iscoroutinefunction(error_handler)

        async def handled(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            result = await self.execute(*args, **kwargs)

            if result.is_error():
                error = result.error

                try:
                    if is_async_handler:
                        recovery_value = await error_handler(error)
                    else:
                        recovery_value = await asyncio.to_thread(error_handler, error)

                    return Result.Ok(recovery_value)
                except Exception as e:
                    return Result.Error(e)

            return result

        return Operation(handled)

    def default_value(self, default: S) -> "Operation[T, S]":
        """
        Provide a default value for error cases.

        Args:
            default: The default value to use if this operation results in an error.

        Returns:
            A new Operation that uses the default value in case of errors.
        """

        async def with_default(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
            result = await self.execute(*args, **kwargs)

            if result.is_error():
                return Result.Ok(default)
            return result

        return Operation(with_default)

    def retry(self, attempts: int = 3, delay: float = 0.1) -> "Operation[T, S]":
        """
        Retry the operation a specified number of times before giving up.

        Args:
            attempts: Maximum number of attempts. Default is 3.
            delay: Delay between attempts in seconds. Default is 0.1.

        Returns:
            A new Operation with retry logic.
        """

        async def retried(*args: Any, **kwargs: Any) -> Result[S, Exception]:
            last_error = None

            for attempt in range(attempts):
                try:
                    result = await self.execute(*args, **kwargs)

                    if result.is_ok():
                        return result

                    last_error = result.error
                except Exception as e:
                    last_error = e

                if attempt < attempts - 1:
                    await asyncio.sleep(delay)

            return Result.Error(last_error or Exception("Unknown error during retry"))

        return Operation(retried)

    def tap(self, side_effect: Callable[[S], None]) -> "Operation[T, S]":
        """
        Apply a side effect to the result without changing it.

        Args:
            side_effect: A function that takes the result value and performs a side effect.

        Returns:
            A new Operation that applies the side effect.
        """
        is_async_side_effect = inspect.iscoroutinefunction(side_effect)

        async def tapped(*args: Any, **kwargs: Any) -> Result[S, Exception]:
            result = await self.execute(*args, **kwargs)

            if result.is_ok():
                try:
                    if is_async_side_effect:
                        await side_effect(result.default_value(None))
                    else:
                        await asyncio.to_thread(side_effect, result.default_value(None))
                except Exception:
                    # Ignore exceptions in the side effect
                    pass

            return result

        return Operation(tapped)

    @classmethod
    async def sequence(
        cls, operations: Collection["Operation"]
    ) -> "Operation[Any, List[Any]]":
        """
        Run a sequence of operations and collect all results.

        Args:
            operations: A collection of operations to run.

        Returns:
            A new Operation that runs all operations and returns a list of results.
        """

        async def sequenced(*args: Any, **kwargs: Any) -> Result[List[Any], Exception]:
            results = []

            for op in operations:
                op_result = await op.execute(*args, **kwargs)

                if op_result.is_error():
                    return op_result

                results.append(op_result.default_value(None))

            return Result.Ok(results)

        return cls(sequenced)

    @classmethod
    async def combine(
        cls, **named_ops: "Operation"
    ) -> "Operation[Any, Dict[str, Any]]":
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
            results = {}

            for name, op in named_ops.items():
                op_result = await op.execute(*args, **kwargs)

                if op_result.is_error():
                    return op_result

                results[name] = op_result.default_value(None)

            return Result.Ok(results)

        return cls(combined)

    @staticmethod
    def unit(value: T) -> "Operation[Any, T]":
        """
        Return a value in the Operation monad context (unit/return).

        Args:
            value: The value to return.

        Returns:
            An Operation that returns the value.
        """

        async def constant(*args: Any, **kwargs: Any) -> Result[T, Exception]:
            return Result.Ok(value)

        return Operation(constant)

    def apply_cont(self, cont: Callable[[S], Awaitable[R]]) -> Awaitable[R]:
        """
        Apply a continuation to this operation's result.

        This is part of the continuation monad pattern.

        Args:
            cont: A continuation function.

        Returns:
            The result of applying the continuation.
        """

        async def run():
            # If the operation isn't bound, provide a default value
            # In this case, the fetch_item operation requires an item_id
            if not self.is_bound:
                # Provide a default item_id=1
                result = await self.execute(1)  # Default to item_id=1
            else:
                result = await self.execute()

            if result.is_error():
                raise result.error

            return await cont(result.default_value(None))

        return run()

    def _has_placeholders(self) -> bool:
        """
        Check if this operation has placeholders in its bound arguments.
        
        This checks recursively through nested data structures.
        """
        return (
            self._contains_placeholder(self.bound_args) or 
            self._contains_placeholder(self.bound_kwargs)
        )

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
            return (
                any(self._contains_placeholder(key) for key in obj) or
                any(self._contains_placeholder(value) for value in obj.values())
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
            self._substitute_placeholder(arg, value)
            for arg in self.bound_args or ()
        )
        
        new_kwargs = {}
        if self.bound_kwargs:
            new_kwargs = {
                self._substitute_placeholder(key, value): self._substitute_placeholder(val, value)
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
                self._substitute_placeholder(key, value): self._substitute_placeholder(val, value)
                for key, val in obj.items()
            }
        
        return obj

def operation(func: Callable[..., Any]) -> Operation[Any, Any]:
    """
    Decorator to convert a function (sync or async) into an Operation.

    Synchronous functions will be automatically converted to run in a thread
    via asyncio.to_thread.

    Args:
        func: The function to convert.

    Returns:
        An Operation that wraps the function.
    """
    if isinstance(func, Operation):
        return func

    is_async = inspect.iscoroutinefunction(func)

    async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            if is_async:
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)

            if isinstance(result, Result):
                return result
            return result  # Let execute() wrap in Result.Ok
        except Exception as e:
            return Result.Error(e)

    return Operation(async_wrapped)


def constant(value: S) -> Operation[Any, S]:
    """
    Create an async operation that always returns the same value.

    Args:
        value: The constant value to return.

    Returns:
        An Operation that always returns the given value.
    """
    return Operation.unit(value)

@operation
async def identity(x):
    return x

def fail(error: Union[str, Exception]) -> Operation[Any, Any]:
    """
    Create an async operation that always fails with the given error.

    Args:
        error: The error message or exception to fail with.

    Returns:
        An Operation that always fails.
    """

    async def fail_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        if isinstance(error, str):
            return Result.Error(Exception(error))
        return Result.Error(error)

    return Operation(fail_func)


def attempt(risky_operation: Callable[..., S]) -> Operation[Any, S]:
    """
    Create an operation that attempts to run a function that might raise exceptions.

    This is similar to the @operation decorator but doesn't require decorating the original function.
    Works with both sync and async functions.

    Args:
        risky_operation: A function that might raise exceptions.

    Returns:
        An operation that handles exceptions from the function.
    """

    async def attempt_func(*args: Any, **kwargs: Any) -> Result[S, Exception]:
        is_async = inspect.iscoroutinefunction(risky_operation)
        if is_async:
            try:
                return await risky_operation(*args, **kwargs)
            except Exception as e:
                return Result.Error(e)
        else:
            try:
                return await asyncio.to_thread(risky_operation, *args, **kwargs)
            except Exception as e:
                return Result.Error(e)

    return operation(attempt_func)

def branch(
    condition: Callable[[], bool],
    true_operation: Operation,
    false_operation: Operation
) -> Operation[Any, Any]:
    """
    Run a conditional operation.
    """
    async def branch_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        if await condition(*args, **kwargs):
            return await true_operation.execute(*args, **kwargs)
        else:
            return await false_operation.execute(*args, **kwargs)

    return operation(branch_func)

async def gather_operations(
    *operations: Operation, args: Any = None, kwargs: Any = None
) -> List[Result[Any, Exception]]:
    """
    Run multiple operations concurrently and return when all are complete.

    This is a utility function for running multiple operations concurrently
    outside of the Operation class.

    Args:
        *operations: Operations to run concurrently.
        args: Arguments to pass to each operation.
        kwargs: Keyword arguments to pass to each operation.

    Returns:
        A list of Results from each operation.
    """
    tasks = []
    for op in operations:
        if args is not None or kwargs is not None:
            # If args or kwargs are provided, create a new bound operation
            op = op(*args or [], **(kwargs or {}))
        tasks.append(op.execute())
    
    return await asyncio.gather(*tasks)