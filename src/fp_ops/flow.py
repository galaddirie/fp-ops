from typing import Any, Callable, Optional, Type, TypeVar, Union, cast, overload
import inspect
import time
import asyncio
from fp_ops.operator import Operation
from fp_ops.context import BaseContext
from expression import Result

S = TypeVar("S")

def branch(
    condition: Union[Callable[..., bool], Operation],
    true_operation: Operation,
    false_operation: Operation,
) -> Operation[Any, Any, Any]:
    """
    Run a conditional operation.
    
    Args:
        condition: A function or Operation that determines which branch to take.
        true_operation: The operation to run if condition returns True.
        false_operation: The operation to run if condition returns False.
        
    Returns:
        An operation that conditionally executes one of two operations.
    """
    is_operation = isinstance(condition, Operation)
    is_decorated = hasattr(condition, "requires_context") and callable(condition)
    
    async def branch_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        try:
            condition_kwargs = dict(kwargs)
            condition_value: bool = False
            
            if is_operation:
                condition_op = cast(Operation, condition)
                condition_result = await condition_op.execute(*args, **condition_kwargs)
                if condition_result.is_error():
                    return condition_result
                condition_value = condition_result.default_value(False)
            elif is_decorated:
                op = condition(*args, **condition_kwargs)
                if isinstance(op, Operation):
                    result = await op.execute()
                    if result.is_error():
                        return result
                    condition_value = result.default_value(False)
                else:
                    condition_value = bool(op)
            elif inspect.iscoroutinefunction(condition):
                condition_func = cast(Callable[..., bool], condition)
                condition_value = await condition_func(*args, **condition_kwargs)  # type: ignore
            else:
                condition_func = cast(Callable[..., bool], condition)
                condition_value = condition_func(*args, **condition_kwargs)
            
            if condition_value:
                return await true_operation.execute(*args, **kwargs)
            else:
                return await false_operation.execute(*args, **kwargs)
                
        except Exception as e:
            return Result.Error(e)

    context_type = None
    
    condition_context_type = getattr(condition, "context_type", None)
    if condition_context_type is not None:
        context_type = condition_context_type
    
    if true_operation.context_type is not None:
        if context_type is None:
            context_type = true_operation.context_type
        elif issubclass(true_operation.context_type, context_type):
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
        if context and context_type is not None:
            ctx = kwargs.get("context")
            if ctx is not None and not isinstance(ctx, context_type):
                try:
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
        
        try:
            if is_async:
                result = await risky_operation(*args, **kwargs)  # type: ignore
                if isinstance(result, Result):
                    return cast(Result[S, Exception], result)
                return Result.Ok(result)
            else:
                result = await asyncio.to_thread(lambda: risky_operation(*args, **kwargs))
                if isinstance(result, Result):
                    return cast(Result[S, Exception], result)
                return Result.Ok(result)
        except Exception as e:
            return Result.Error(e)

    setattr(attempt_func, "requires_context", context)
    setattr(attempt_func, "context_type", context_type)

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
        if isinstance(error, str):
            return Result.Error(Exception(error))
        return Result.Error(error)

    return Operation(fail_func, context_type=context_type)


def retry(
    operation: Operation,
    max_retries: int = 3,
    delay: float = 0.1,
) -> Operation:
    """
    Create an operation that retries another operation a specified number of times.
    
    Args:
        operation: The operation to retry.
        max_retries: Maximum number of attempts. Default is 3.
        delay: Delay between attempts in seconds. Default is 0.1.
        
    Returns:
        An operation that retries the original operation.
    """
    return operation.retry(max_retries, delay)


def tap(
    operation: Operation,
    side_effect: Callable[..., Any],
    context: bool = False,
    context_type: Optional[Type[BaseContext]] = None,
) -> Operation:
    """
    Create an operation that performs a side effect without changing the value.
    
    Args:
        operation: The original operation.
        side_effect: A function that takes the result value and performs a side effect.
        context: Whether the side effect function requires a context.
        context_type: The expected type of the context.
        
    Returns:
        An operation that applies the side effect without changing the value.
    """
    if callable(side_effect) and not hasattr(side_effect, "requires_context"):
        setattr(side_effect, "requires_context", context)
        setattr(side_effect, "context_type", context_type)
        
    return operation.tap(side_effect)


def loop_until(
    condition: Callable[..., bool],
    body: Operation,
    max_iterations: int = 10,
    delay: float = 0.1,
    context: bool = False,
    context_type: Optional[Type[BaseContext]] = None,
) -> Operation:
    """
    Create an operation that loops until a condition is met.
    
    Args:
        condition: A function that determines when to stop looping.
        body: The operation to execute in each iteration.
        max_iterations: Maximum number of iterations to prevent infinite loops.
        delay: Delay between iterations in seconds.
        context: Whether the condition function requires a context.
        context_type: The expected type of the context.
        
    Returns:
        An operation that loops until the condition is met.
    """
    condition_requires_context = getattr(condition, "requires_context", context)
    
    async def loop_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        ctx = kwargs.get("context")
        iteration_value = args[0] if args else None
        
        try:
            for i in range(max_iterations):
                condition_kwargs = dict(kwargs)
                condition_args = (iteration_value,) + args[1:] if args and len(args) > 1 else (iteration_value,)
                if inspect.iscoroutinefunction(condition):
                    should_exit = await condition(*condition_args, **condition_kwargs)
                else:
                    should_exit = condition(*condition_args, **condition_kwargs)

                if should_exit:
                    return Result.Ok(iteration_value)
                
                body_kwargs = dict(kwargs)
                body_args = condition_args
                result = await body.execute(*body_args, **body_kwargs)
                
                if result.is_error():
                    return result
                
                iteration_value = result.default_value(None)
                
                if isinstance(iteration_value, BaseContext):
                    ctx = iteration_value
                    kwargs["context"] = ctx
                
                if i < max_iterations - 1:
                    await asyncio.sleep(delay)
            
            return Result.Ok(iteration_value)
                
        except Exception as e:
            return Result.Error(e)
    
    loop_context_type = context_type
    if body.context_type is not None:
        if loop_context_type is None:
            loop_context_type = body.context_type
        elif issubclass(body.context_type, loop_context_type):
            loop_context_type = body.context_type
    
    return Operation(loop_func, context_type=loop_context_type)



def wait(
    operation: Operation,
    timeout: float = 10.0,
    delay: float = 0.1,
) -> Operation:
    """
    Create an operation that waits for another operation to complete with a timeout.
    
    Args:
        operation: The operation to wait for.
        timeout: Maximum time to wait in seconds.
        delay: Delay between checks in seconds.
        
    Returns:
        An operation that waits for the original operation to complete.
    """
    async def wait_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        start_time = time.time()
        last_error = None
        
        while time.time() - start_time < timeout:
            try:
                result = await operation.execute(*args, **kwargs)
                
                if result.is_ok():
                    return result
                
                last_error = result.error
            except Exception as e:
                last_error = e
            
            await asyncio.sleep(delay)
        
        if last_error:
            return Result.Error(last_error)
        else:
            return Result.Error(TimeoutError(f"Operation timed out after {timeout} seconds"))
    
    return Operation(wait_func, context_type=operation.context_type)