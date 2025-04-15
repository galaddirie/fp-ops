from typing import Any, Callable, Optional, Type, TypeVar, Union, cast, overload, List
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
    # Check if condition is an Operation or was decorated with @operation
    is_operation = isinstance(condition, Operation)
    is_decorated = hasattr(condition, "requires_context") and callable(condition)
    
    async def branch_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        try:
            # Make a copy of kwargs to avoid modifying the original
            condition_kwargs = dict(kwargs)
            
            # Evaluate the condition - the key part is correctly executing Operations
            condition_value: bool = False
            
            if is_operation:
                # If condition is an Operation, execute it
                condition_op = cast(Operation, condition)
                condition_result = await condition_op.execute(*args, **condition_kwargs)
                if condition_result.is_error():
                    return condition_result
                condition_value = condition_result.default_value(False)
            elif is_decorated:
                # If condition was decorated with @operation, it returns an Operation when called
                # We need to execute that Operation to get the boolean result
                op = condition(*args, **condition_kwargs)
                if isinstance(op, Operation):
                    result = await op.execute()
                    if result.is_error():
                        return result
                    condition_value = result.default_value(False)
                else:
                    # If it unexpectedly doesn't return an Operation, use the value directly
                    condition_value = bool(op)
            elif inspect.iscoroutinefunction(condition):
                # For async functions, await them
                condition_func = cast(Callable[..., bool], condition)
                condition_value = await condition_func(*args, **condition_kwargs)  # type: ignore
            else:
                # For regular functions, call directly
                condition_func = cast(Callable[..., bool], condition)
                condition_value = condition_func(*args, **condition_kwargs)
            
            # Choose the appropriate branch based on the condition result
            if condition_value:
                return await true_operation.execute(*args, **kwargs)
            else:
                return await false_operation.execute(*args, **kwargs)
                
        except Exception as e:
            return Result.Error(e)

    # Determine the most specific context type for the branch operation
    context_type = None
    
    # Check condition context type
    condition_context_type = getattr(condition, "context_type", None)
    if condition_context_type is not None:
        context_type = condition_context_type
    
    # Check true_operation context type
    if true_operation.context_type is not None:
        if context_type is None:
            context_type = true_operation.context_type
        elif issubclass(true_operation.context_type, context_type):
            context_type = true_operation.context_type
    
    # Check false_operation context type
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
        
        try:
            if is_async:
                # For async functions, await the coroutine to get the actual result
                result = await risky_operation(*args, **kwargs)  # type: ignore
                
                # If the function already returns a Result, use it directly
                if isinstance(result, Result):
                    return cast(Result[S, Exception], result)
                # Otherwise, wrap the value in Result.Ok
                return Result.Ok(result)
            else:
                # For sync functions, run in a thread and handle the result
                result = await asyncio.to_thread(lambda: risky_operation(*args, **kwargs))
                
                # If the function already returns a Result, use it directly
                if isinstance(result, Result):
                    return cast(Result[S, Exception], result)
                # Otherwise, wrap the value in Result.Ok
                return Result.Ok(result)
        except Exception as e:
            return Result.Error(e)

    # Mark the wrapped function with context requirements
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
        # Even though this operation always fails, we should specify the context type
        # for proper type checking in the operation chain
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
    # Mark the side_effect function with context requirements if needed
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
    # Determine if the condition function requires context
    condition_requires_context = getattr(condition, "requires_context", context)
    
    async def loop_func(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        # Preserve context when looping
        ctx = kwargs.get("context")
        iteration_value = args[0] if args else None
        
        try:
            for i in range(max_iterations):
                # Pass the latest value to the condition
                condition_kwargs = dict(kwargs)
                
                # Always use the current iteration_value, regardless of context requirement
                condition_args = (iteration_value,) + args[1:] if args and len(args) > 1 else (iteration_value,)
                
                # Evaluate the condition
                if inspect.iscoroutinefunction(condition):
                    should_exit = await condition(*condition_args, **condition_kwargs)
                else:
                    should_exit = condition(*condition_args, **condition_kwargs)

                if should_exit:
                    # When condition is met, return the current value (before executing body again)
                    return Result.Ok(iteration_value)
                
                # Execute the body operation
                body_kwargs = dict(kwargs)
                body_args = condition_args  # Use the same args we passed to condition
                
                result = await body.execute(*body_args, **body_kwargs)
                
                if result.is_error():
                    return result
                
                iteration_value = result.default_value(None)
                
                # If the result is a context, update the context for the next iteration
                if isinstance(iteration_value, BaseContext):
                    ctx = iteration_value
                    kwargs["context"] = ctx
                
                # Add delay between iterations
                if i < max_iterations - 1:
                    await asyncio.sleep(delay)
            
            # If we reached the max iterations, return the last value
            return Result.Ok(iteration_value)
                
        except Exception as e:
            return Result.Error(e)
    
    # Determine the context type for the loop operation
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
        
        # If we reached the timeout, return the last error or a timeout error
        if last_error:
            return Result.Error(last_error)
        else:
            return Result.Error(TimeoutError(f"Operation timed out after {timeout} seconds"))
    
    return Operation(wait_func, context_type=operation.context_type)

def map_operations(
    operation: Operation,
    parallel: bool = False,
    context_type: Optional[Type[BaseContext]] = None,
) -> Operation[List[Any], List[Any], Any]:
    """
    Create an operation that applies another operation to each item in an array.
    
    This can handle:
    - Simple operations: map_operations(double)
    - Operations with placeholders: map_operations(multiply(_, 2))
    - Partially applied operations: map_operations(multiply(2))
    - Composed operations: map_operations(double >> add_five)
    - Composed operations with placeholders: map_operations(multiply(_, 2) >> add(_, 5))
    - Composed partially applied operations: map_operations(multiply(2) >> add(5))
    
    Args:
        operation: The operation to apply to each item. Can contain placeholders for partial application.
        parallel: Whether to execute operations in parallel (True) or sequentially (False).
        context_type: Optional type for the context this operation will use.
        
    Returns:
        An operation that applies the given operation to each item in an array input.
    """
    async def mapped(*args: Any, **kwargs: Any) -> Result[List[Any], Exception]:
        if not args or not isinstance(args[0], (list, tuple)):
            return Result.Error(Exception("First argument must be a list or tuple"))
        
        values = args[0]
        context = kwargs.get("context")
        
        try:
            if parallel:
                # Create tasks for parallel execution
                tasks = []
                for value in values:
                    # Create a task for each value
                    tasks.append(_process_single_value(operation, value, context))
                
                # Execute all tasks in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results, handling any exceptions
                processed_results = []
                for result in results:
                    if isinstance(result, Exception):
                        return Result.Error(result)
                    processed_results.append(result)
                
                return Result.Ok(processed_results)
            else:
                # Sequential execution
                results = []
                for value in values:
                    # Process each value sequentially
                    try:
                        result = await _process_single_value(operation, value, context)
                        results.append(result)
                    except Exception as e:
                        return Result.Error(e)
                
                return Result.Ok(results)
                
        except Exception as e:
            return Result.Error(e)
    
    # Determine context type for the operation
    op_context_type = context_type
    if operation.context_type is not None:
        if op_context_type is None:
            op_context_type = operation.context_type
        elif issubclass(operation.context_type, op_context_type):
            op_context_type = operation.context_type
    
    return Operation(mapped, context_type=op_context_type)

async def _process_single_value(operation: Operation, value: Any, context: Optional[Any] = None) -> Any:
    """
    Process a single value through an operation, handling all the special cases.
    
    This function handles the core logic of executing an operation with a value,
    including placeholder substitution and composition handling.
    
    Args:
        operation: The operation to execute
        value: The value to process
        context: Optional context to pass to the operation
        
    Returns:
        The result of processing the value through the operation
    """
    # Detect if this is a composed operation by checking for the attribute
    is_composed = hasattr(operation.func, 'is_composed_func') and operation.func.is_composed_func
    
    # Prepare execution kwargs
    execution_kwargs = {}
    if context is not None:
        execution_kwargs["context"] = context
    
    # Case 1: Composed operation with placeholders
    if is_composed and operation._has_placeholders():
        # For composed operations with placeholders, we need special handling
        # Create a clone of the operation with the value substituted for placeholders
        new_args, new_kwargs = operation._substitute_placeholders(value)
        # Ensure context is included
        if context is not None:
            new_kwargs["context"] = context
        
        # Execute the composed operation with the substituted placeholders
        result = await operation.func(*new_args, **new_kwargs)
        if isinstance(result, Result):
            if result.is_error():
                raise result.error
            return result.default_value(None)
        return result
    
    # Case 2: Composed operation without placeholders, but possibly with partial application
    elif is_composed:
        try:
            # First try to execute the composed operation directly
            result = await operation.execute(**execution_kwargs)
            if isinstance(result, Result):
                if result.is_error():
                    raise result.error
                return result.default_value(None)
            return result
        except TypeError as e:
            # If we get a TypeError about missing arguments, the operation might be
            # partially applied. Let's try to add the value as the next argument.
            if "missing" in str(e) and "argument" in str(e):
                # Create a new operation with the value added as an argument
                args = operation.bound_args or ()
                args = args + (value,)
                op_instance = Operation(
                    operation.func, 
                    args, 
                    operation.bound_kwargs, 
                    context_type=operation.context_type
                )
                result = await op_instance.execute(**execution_kwargs)
                if isinstance(result, Result):
                    if result.is_error():
                        raise result.error
                    return result.default_value(None)
                return result
            # If it's not a missing argument error, re-raise
            raise
    
    # Case 3: Simple operation with placeholders
    elif operation._has_placeholders():
        # For operations with placeholders, substitute the value
        new_args, new_kwargs = operation._substitute_placeholders(value)
        # Ensure context is included
        if context is not None:
            new_kwargs["context"] = context
        
        # Create a new operation with substituted values
        op_instance = Operation(
            operation.func, 
            new_args, 
            new_kwargs, 
            context_type=operation.context_type
        )
        result = await op_instance.execute()
        if isinstance(result, Result):
            if result.is_error():
                raise result.error
            return result.default_value(None)
        return result
    
    # Case 4: Partially applied operation (no placeholders)
    elif operation.is_bound:
        try:
            # First try to execute the bound operation as-is
            result = await operation.execute(**execution_kwargs)
            if isinstance(result, Result):
                if result.is_error():
                    raise result.error
                return result.default_value(None)
            return result
        except TypeError as e:
            # If we get a TypeError about missing arguments, add our value
            if "missing" in str(e) and "argument" in str(e):
                # Create a new operation with the value added as an argument
                args = operation.bound_args or ()
                args = args + (value,)
                op_instance = Operation(
                    operation.func, 
                    args, 
                    operation.bound_kwargs, 
                    context_type=operation.context_type
                )
                result = await op_instance.execute(**execution_kwargs)
                if isinstance(result, Result):
                    if result.is_error():
                        raise result.error
                    return result.default_value(None)
                return result
            # If it's not a missing argument error, re-raise
            raise
    
    # Case 5: Simple unbound operation
    else:
        # For unbound operations, just pass the value as the first argument
        result = await operation.execute(value, **execution_kwargs)
        if isinstance(result, Result):
            if result.is_error():
                raise result.error
            return result.default_value(None)
        return result