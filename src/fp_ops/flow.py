from typing import Any, Callable, Optional, Type, TypeVar, Union
import inspect
import asyncio
from fp_ops.operator import Operation
from fp_ops.context import BaseContext
from expression import Result

S = TypeVar("S")

def branch(
    condition: Callable[..., bool],
    true_operation: Operation,
    false_operation: Operation,
) -> Operation[Any, Any, Any]:
    """
    Run a conditional operation.
    
    Args:
        condition: A function that determines which branch to take.
        true_operation: The operation to run if condition returns True.
        false_operation: The operation to run if condition returns False.
        
    Returns:
        An operation that conditionally executes one of two operations.
    """
    # Determine if the condition function requires context
    condition_requires_context = getattr(condition, "requires_context", False)
    
    async def branch_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        # Preserve context when branching
        context = kwargs.get("context")
        
        # Evaluate the condition
        try:
            # Make a copy of kwargs to avoid modifying the original
            condition_kwargs = dict(kwargs)
            
            if inspect.iscoroutinefunction(condition):
                condition_result = await condition(*args, **condition_kwargs)
            else:
                condition_result = condition(*args, **condition_kwargs)
                
            # Choose the appropriate branch based on the condition result
            if condition_result:
                return await true_operation.execute(*args, **kwargs)
            else:
                return await false_operation.execute(*args, **kwargs)
                
        except Exception as e:
            return Result.Error(e)

    # Determine the most specific context type for the branch operation
    context_type = None
    if true_operation.context_type is not None:
        context_type = true_operation.context_type
    if false_operation.context_type is not None:
        if context_type is None:
            context_type = false_operation.context_type
        elif issubclass(false_operation.context_type, context_type):
            context_type = false_operation.context_type

    return Operation(branch_func, context_type=context_type)


def attempt(
    risky_operation: Callable[..., S],
    context: bool = False,
    context_type: Optional[Type[BaseContext]] = None,
) -> Operation[Any, S, Any]:
    """
    Create an operation that attempts to run a function that might raise exceptions.

    This is similar to the @operation decorator but doesn't require decorating the original function.
    Works with both sync and async functions.

    Args:
        risky_operation: A function that might raise exceptions.
        context: Whether the function requires a context.
        context_type: The expected type of the context.

    Returns:
        An operation that handles exceptions from the function.
    """

    async def attempt_func(*args: Any, **kwargs: Any) -> Result[S, Exception]:
        # Validate context if required
        if context and context_type is not None:
            ctx = kwargs.get("context")
            if ctx is not None and not isinstance(ctx, context_type):
                try:
                    # Try to convert to the required context type
                    if isinstance(ctx, dict):
                        ctx = context_type(**ctx)
                    elif isinstance(ctx, BaseContext):
                        ctx = context_type(**ctx.model_dump())
                    else:
                        ctx = context_type.model_validate(ctx)

                    kwargs["context"] = ctx
                except Exception as e:
                    return Result.Error(Exception(f"Invalid context: {e}"))

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

    # Mark the wrapped function with context requirements
    attempt_func.requires_context = context
    attempt_func.context_type = context_type

    return Operation(attempt_func, context_type=context_type)


def fail(
    error: Union[str, Exception], context_type: Optional[Type[BaseContext]] = None
) -> Operation[Any, Any, Any]:
    """
    Create an async operation that always fails with the given error.

    Args:
        error: The error message or exception to fail with.
        context_type: Optional type for the context this operation will use.

    Returns:
        An Operation that always fails.
    """

    async def fail_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        # Even though this operation always fails, we should specify the context type
        # for proper type checking in the operation chain
        if isinstance(error, str):
            return Result.Error(Exception(error))
        return Result.Error(error)

    return Operation(fail_func, context_type=context_type)

