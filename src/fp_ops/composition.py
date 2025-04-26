import asyncio
from typing import Any, List, Union, Callable, Tuple, Dict
from fp_ops.operator import Operation, identity
from fp_ops.context import BaseContext
from expression import Result

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
            op_kwargs = dict(kwargs)
            op_result = await op.execute(*args, **op_kwargs)

            if op_result.is_error():
                return op_result

            value = op_result.default_value(None)

            if isinstance(value, BaseContext):
                context = value
                kwargs["context"] = context
            else:
                results.append(value)

        return Result.Ok(results)

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

        result = await first_step.execute(*args, **kwargs)
        if result.is_error() or len(steps) == 1:
            return result

        value = result.default_value(None)
        context = kwargs.get("context")
        last_context_value = None

        if isinstance(value, BaseContext):
            context = value
            kwargs["context"] = context
            last_context_value = value
            value = None

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

            if next_op.is_bound:
                result = await next_op.execute(**kwargs)
            else:
                result = await next_op.execute(value, **kwargs)
            
            if result.is_error():
                return result

            value = result.default_value(None)
            
            if isinstance(value, BaseContext):
                context = value
                kwargs["context"] = context
                last_context_value = value
                value = None

        if last_context_value is not None and isinstance(value, BaseContext):
            return Result.Ok(value)
        elif last_context_value is not None:
            return Result.Ok(last_context_value)
        else:
            return Result.Ok(value)

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
        
        context = kwargs.get("context")
        
        tasks = []
        for op in operations:
            op_kwargs = dict(kwargs)
            tasks.append(op.execute(*args, **op_kwargs))
            
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result.is_error():
                return result
        
        values = tuple(result.default_value(None) for result in results)
        return Result.Ok(values)
    
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
            op_kwargs = dict(kwargs)
            result = await op.execute(*args, **op_kwargs)
            
            if result.is_ok():
                return result
            
            last_error = result.error
        
        return Result.Error(last_error or Exception("All operations failed"))
    
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

def zip(*operations: Operation) -> Operation:
    """
    Zip a list of operations.
    """
    async def zip_op(*args: Any, **kwargs: Any) -> Result[Tuple[Any, ...], Exception]:
        if not operations:
            return Result.Ok(())
        
        results = await parallel(*operations).execute(*args, **kwargs)
        
        if results.is_error():
            return results
            
        values = results.default_value(())
        return Result.Ok(values)
    
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

    execution_kwargs = kwargs or {}
    context = execution_kwargs.get("context")

    for op in operations:
        op_kwargs = dict(execution_kwargs)

        if args is not None or kwargs is not None:
            op = op(*args or [], **op_kwargs)

        if (
            context is not None
            and hasattr(op, "context_type")
            and op.context_type is not None
        ):
            try:
                if not isinstance(context, op.context_type):
                    if isinstance(context, dict):
                        op_kwargs["context"] = op.context_type(**context)
                    elif isinstance(context, BaseContext):
                        op_kwargs["context"] = op.context_type(**context.model_dump())
                    else:
                        op_kwargs["context"] = op.context_type.model_validate(context)
            except Exception:
                pass

        tasks.append(op.execute(**op_kwargs))

    return await asyncio.gather(*tasks)
