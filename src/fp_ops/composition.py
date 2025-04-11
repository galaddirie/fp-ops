import asyncio
from typing import Any, List, Union, Callable, Tuple, Dict
from fp_ops.operator import Operation, identity
from fp_ops.context import BaseContext
from expression import Result

# BUG: currently we do not bind the result of the previous operation to the next one, 
# that might be desired but we need to also have the option to bind the result of the previous operation to the next one
# maybe a bind=True argument?
def sequence(*operations: Operation) -> Operation:
    """
    Combines multiple operations into a single operation that executes them in order.
    Unlike 'compose', this function collects and returns ALL results as a Block.

    Args:
        *operations: Operations to execute in sequence.

    Returns:
        An Operation that executes the input operations in sequence.

    Example:
    ```python
    result = await sequence(op1, op2, op3)(*args, **kwargs)
    # result is a Block containing the results of op1, op2, and op3
    ```
    """
    async def sequenced_op(*args: Any, **kwargs: Any) -> Result[List[Any], Exception]:
        results = []
        context = kwargs.get("context")

        for op in operations:
            # Create a new kwargs for each operation
            op_kwargs = dict(kwargs)

            op_result = await op.execute(*args, **op_kwargs)

            if op_result.is_error():
                return op_result

            value = op_result.default_value(None)

            # If the value is a context, update the context for subsequent operations
            if isinstance(value, BaseContext):
                context = value
                kwargs["context"] = context
            else:
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

    return Operation(sequenced_op, context_type=context_type)


def pipe(*steps: Union[Operation, Callable[[Any], Operation]]) -> Operation:
    """
    Create a pipeline of operations where each step can be either an Operation or
    a function that takes the previous result and returns an Operation.

    This is the most flexible composition function:
    - For simple cases, use compose() or the >> operator
    - For complex cases where you need to inspect values or decide which action to run next,
      use pipe() with lambda functions
    """
    async def piped(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        if not steps:
            return Result.Ok(None)

        # Handle first step
        first_step = steps[0]
        if not isinstance(first_step, Operation):
            if callable(first_step):
                try:
                    first_step = first_step(*args)
                except Exception as e:
                    return Result.Error(e)
                
                if not isinstance(first_step, Operation):
                    return Result.Error(TypeError(f"Step function must return an Operation, got {type(first_step)}"))
            else:
                return Result.Error(TypeError(f"Step must be an Operation or callable, got {type(first_step)}"))

        # Execute the first operation
        result = await first_step.execute(*args, **kwargs)
        if result.is_error() or len(steps) == 1:
            return result

        value = result.default_value(None)
        context = kwargs.get("context")
        last_context_value = None

        # If the value is a context, update the context but don't pass it as positional arg
        if isinstance(value, BaseContext):
            context = value
            kwargs["context"] = context
            last_context_value = value
            value = None  # Don't pass context as positional arg to next op

        # Process remaining steps
        for step in steps[1:]:
            if isinstance(step, Operation):
                next_op = step
            elif callable(step):
                try:
                    next_op = step(value)
                    if not isinstance(next_op, Operation):
                        return Result.Error(TypeError(f"Step function must return an Operation, got {type(next_op)}"))
                except Exception as e:
                    return Result.Error(e)
            else:
                return Result.Error(TypeError(f"Step must be an Operation or callable, got {type(step)}"))

            # Execute the next operation
            if next_op.is_bound:
                result = await next_op.execute(**kwargs)
            else:
                result = await next_op.execute(value, **kwargs)
            
            if result.is_error():
                return result

            value = result.default_value(None)
            
            # Update context if value is a context
            if isinstance(value, BaseContext):
                context = value
                kwargs["context"] = context
                last_context_value = value
                value = None  # Don't pass context as positional arg to next op

        # If the last operation returned a context, return that as the result
        if last_context_value is not None and isinstance(value, BaseContext):
            return Result.Ok(value)
        # If any operation in the chain returned a context (not just the last one)
        elif last_context_value is not None:
            return Result.Ok(last_context_value)
        else:
            return Result.Ok(value)

    # Determine context type - use the most specific among all operations
    context_type = None
    for step in steps:
        if isinstance(step, Operation) and step.context_type is not None:
            if context_type is None:
                context_type = step.context_type
            elif issubclass(step.context_type, context_type):
                context_type = step.context_type

    return Operation(piped, context_type=context_type)
def compose(*operations: Operation) -> Operation:
    """
    Compose a list of operations into a single operation.
    """
    if not operations:
        return identity
    
    if len(operations) == 1:
        return operations[0]
    
    # Use the >> operator to compose operations from right to left
    result = operations[-1]
    for op in reversed(operations[:-1]):
        result = op >> result
    
    return result


def parallel(*operations: Operation) -> Operation:
    """
    Run multiple operations concurrently and return when all are complete.
    """
    async def parallel_op(*args: Any, **kwargs: Any) -> Result[Tuple[Any, ...], Exception]:
        if not operations:
            return Result.Ok(())
        
        # Extract context from kwargs if available
        context = kwargs.get("context")
        
        # Create tasks for each operation
        tasks = []
        for op in operations:
            # Create separate kwargs for each operation
            op_kwargs = dict(kwargs)
            tasks.append(op.execute(*args, **op_kwargs))
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Check if any operation resulted in an error
        for result in results:
            if result.is_error():
                return result
        
        # Collect values from all results
        values = tuple(result.default_value(None) for result in results)
        return Result.Ok(values)
    
    # Use the most specific context type among all operations
    context_type = None
    for op in operations:
        if op.context_type is not None:
            if context_type is None:
                context_type = op.context_type
            elif issubclass(op.context_type, context_type):
                context_type = op.context_type
                
    return Operation(parallel_op, context_type=context_type)


def fallback(*operations: Operation) -> Operation:
    """
    Try each operation in order until one succeeds.
    """
    async def fallback_op(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        if not operations:
            return Result.Error(ValueError("No operations provided to fallback"))
        
        last_error = None
        
        for op in operations:
            # Create separate kwargs for each operation
            op_kwargs = dict(kwargs)
            result = await op.execute(*args, **op_kwargs)
            
            if result.is_ok():
                return result
            
            last_error = result.error
        
        # If all operations failed, return the last error
        return Result.Error(last_error or Exception("All operations failed"))
    
    # Use the most specific context type among all operations
    context_type = None
    for op in operations:
        if op.context_type is not None:
            if context_type is None:
                context_type = op.context_type
            elif issubclass(op.context_type, context_type):
                context_type = op.context_type
                
    return Operation(fallback_op, context_type=context_type)



def map(operation: Operation, func: Callable[[Any], Any]) -> Operation:
    """
    Map a function to an operation.
    """
    return operation.map(func)


def filter(operation: Operation, func: Callable[[Any], bool]) -> Operation:
    """
    Filter a list of operations.
    """
    return operation.filter(func)


def reduce(operation: Operation, func: Callable[[Any, Any], Any]) -> Operation:
    """
    Reduce a list of operations.
    """
    async def reduced(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            return result
            
        value = result.default_value(None)
        
        if not isinstance(value, (list, tuple)):
            return Result.Error(TypeError(f"Expected a list or tuple, got {type(value)}"))
        
        if not value:
            return Result.Ok(None)
        
        try:
            from functools import reduce as functools_reduce
            result_value = functools_reduce(func, value)
            return Result.Ok(result_value)
        except Exception as e:
            return Result.Error(e)
    
    return Operation(reduced, context_type=operation.context_type)

#TODO:  should zip be a tuple or a list?
def zip(*operations: Operation) -> Operation:
    """
    Zip a list of operations.
    """
    # Similar to parallel but with a different output structure
    async def zip_op(*args: Any, **kwargs: Any) -> Result[Tuple[Any, ...], Exception]:
        if not operations:
            return Result.Ok(())
        
        # Run all operations in parallel
        results = await parallel(*operations).execute(*args, **kwargs)
        
        if results.is_error():
            return results
            
        # Zip the results together
        values = results.default_value(())
        return Result.Ok(values)
    
    # Use the most specific context type among all operations
    context_type = None
    for op in operations:
        if op.context_type is not None:
            if context_type is None:
                context_type = op.context_type
            elif issubclass(op.context_type, context_type):
                context_type = op.context_type
                
    return Operation(zip_op, context_type=context_type)


def flat_map(operation: Operation, func: Callable[[Any], List[Any]]) -> Operation:
    """
    Flat map a function to an operation.
    """
    async def flat_mapped(*args: Any, **kwargs: Any) -> Result[List[Any], Exception]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            return result
            
        value = result.default_value(None)
        
        try:
            mapped_values = func(value)
            # Flatten the list of lists
            flattened = [item for sublist in mapped_values for item in sublist]
            return Result.Ok(flattened)
        except Exception as e:
            return Result.Error(e)
    
    return Operation(flat_mapped, context_type=operation.context_type)


def group_by(operation: Operation, func: Callable[[Any], Any]) -> Operation:
    """
    Group a list of operations by a function.
    """
    async def grouped(*args: Any, **kwargs: Any) -> Result[Dict[Any, List[Any]], Exception]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            return result
            
        value = result.default_value(None)
        
        if not isinstance(value, (list, tuple)):
            return Result.Error(TypeError(f"Expected a list or tuple, got {type(value)}"))
        
        try:
            # Group items by the key function
            groups: Dict[Any, List[Any]] = {}
            for item in value:
                key = func(item)
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
            
            return Result.Ok(groups)
        except Exception as e:
            return Result.Error(e)
    
    return Operation(grouped, context_type=operation.context_type)


def partition(operation: Operation, func: Callable[[Any], bool]) -> Operation:
    """
    Partition a list of operations.
    """
    async def partitioned(*args: Any, **kwargs: Any) -> Result[Tuple[List[Any], List[Any]], Exception]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            return result
            
        value = result.default_value(None)
        
        if not isinstance(value, (list, tuple)):
            return Result.Error(TypeError(f"Expected a list or tuple, got {type(value)}"))
        
        try:
            # Partition items based on the predicate
            truthy = []
            falsy = []
            
            for item in value:
                if func(item):
                    truthy.append(item)
                else:
                    falsy.append(item)
            
            return Result.Ok((truthy, falsy))
        except Exception as e:
            return Result.Error(e)
    
    return Operation(partitioned, context_type=operation.context_type)


def first(operation: Operation) -> Operation:
    """
    Return the first operation.
    """
    async def first_op(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            return result
            
        value = result.default_value(None)
        
        if not isinstance(value, (list, tuple)):
            return Result.Error(TypeError(f"Expected a list or tuple, got {type(value)}"))
        
        if not value:
            return Result.Error(IndexError("Sequence is empty"))
        
        return Result.Ok(value[0])
    
    return Operation(first_op, context_type=operation.context_type)


def last(operation: Operation) -> Operation:
    """
    Return the last operation.
    """
    async def last_op(*args: Any, **kwargs: Any) -> Result[Any, Exception]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            return result
            
        value = result.default_value(None)
        
        if not isinstance(value, (list, tuple)):
            return Result.Error(TypeError(f"Expected a list or tuple, got {type(value)}"))
        
        if not value:
            return Result.Error(IndexError("Sequence is empty"))
        
        return Result.Ok(value[-1])
    
    return Operation(last_op, context_type=operation.context_type)


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

    # Ensure context is passed to all operations
    execution_kwargs = kwargs or {}
    context = execution_kwargs.get("context")

    for op in operations:
        # Create a separate kwargs dictionary for each operation
        # to prevent potential interference between operations
        op_kwargs = dict(execution_kwargs)

        if args is not None or kwargs is not None:
            # If args or kwargs are provided, create a new bound operation
            op = op(*args or [], **op_kwargs)

        # Validate context if the operation has a specific context type
        if (
            context is not None
            and hasattr(op, "context_type")
            and op.context_type is not None
        ):
            try:
                if not isinstance(context, op.context_type):
                    # Try to convert context to the required type
                    if isinstance(context, dict):
                        op_kwargs["context"] = op.context_type(**context)
                    elif isinstance(context, BaseContext):
                        op_kwargs["context"] = op.context_type(**context.model_dump())
                    else:
                        op_kwargs["context"] = op.context_type.model_validate(context)
            except Exception:
                # If conversion fails, use the original context
                pass

        tasks.append(op.execute(**op_kwargs))

    return await asyncio.gather(*tasks)
