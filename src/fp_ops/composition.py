import asyncio
from typing import Any, List, Union, Callable, Tuple, Dict, TypeVar, Concatenate, cast, Iterable, Awaitable, ParamSpec

from fp_ops.operator import Operation, identity, _ensure_async
from fp_ops.context import BaseContext
from expression import Result

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def sequence(*operations: Operation) -> Operation[P, List[Any]]:
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
    async def sequenced_op(*args: Any, **kwargs: Any) -> List[Any]:
        results = []
        context = kwargs.get("context")

        for op in operations:
            op_kwargs = dict(kwargs)
            op_result = await op.execute(*args, **op_kwargs)

            if op_result.is_error():
                raise op_result.error

            value = op_result.default_value(cast(Any, None))

            if isinstance(value, BaseContext):
                context = value
                kwargs["context"] = context
            else:
                results.append(value)

        return results

    context_type = None
    for op in operations:
        if op.context_type is not None:
            if context_type is None:
                context_type = op.context_type
            elif issubclass(op.context_type, context_type):
                context_type = op.context_type

    return Operation._from_function(sequenced_op, ctx_type=context_type, require_ctx=context_type is not None)


def pipe(*steps: Union[Operation, Callable[[Any], Operation]]) -> Operation[P, Any]:
    """
    Create a pipeline of operations where each step can be either an Operation or
    a function that takes the previous result and returns an Operation.

    This is the most flexible composition function:
    - For simple cases, use compose() or the >> operator
    - For complex cases where you need to inspect values or decide which action to run next,
      use pipe() with lambda functions
    """
    async def piped(*args: Any, **kwargs: Any) -> Any:
        if not steps:
            return None

        first_step = steps[0]
        if not isinstance(first_step, Operation):
            if callable(first_step):
                try:
                    first_step = first_step(*args)
                except Exception as e:
                    raise e
                
                if not isinstance(first_step, Operation):
                    raise TypeError(f"Step function must return an Operation, got {type(first_step)}")
            else:
                raise TypeError(f"Step must be an Operation or callable, got {type(first_step)}")

        result = await first_step.execute(*args, **kwargs)
        if result.is_error():
            raise result.error
            
        if len(steps) == 1:
            return result.default_value(cast(Any, None))

        value: Any = result.default_value(cast(Any, None))
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
                        raise TypeError(f"Step function must return an Operation, got {type(next_op)}")
                except Exception as e:
                    raise e
            else:
                raise TypeError(f"Step must be an Operation or callable, got {type(step)}")

            if next_op.is_bound:
                result = await next_op.execute(**kwargs)
            else:
                result = await next_op.execute(value, **kwargs)
            
            if result.is_error():
                raise result.error

            value = result.default_value(cast(Any, None))
            
            if isinstance(value, BaseContext):
                context = value
                kwargs["context"] = context
                last_context_value = value
                value = None

        if last_context_value is not None and isinstance(value, BaseContext):
            return value
        elif last_context_value is not None:
            return last_context_value
        else:
            return value

    context_type = None
    for step in steps:
        if isinstance(step, Operation) and step.context_type is not None:
            if context_type is None:
                context_type = step.context_type
            elif issubclass(step.context_type, context_type):
                context_type = step.context_type

    return Operation._from_function(piped, ctx_type=context_type, require_ctx=context_type is not None)


def compose(*operations: Operation) -> Operation[P, R]:
    """
    Compose a list of operations into a single operation.
    """
    if not operations:
        # identity is still an Operation; the cast quiets mypy
        return cast(Operation[P, R], identity)
    
    if len(operations) == 1:
        return operations[0]
    
    result = operations[-1]
    for op in reversed(operations[:-1]):
        result = op >> result
    
    return result


def parallel(*operations: Operation) -> Operation[P, Tuple[Any, ...]]:
    """
    Run multiple operations concurrently and return when all are complete.
    """
    async def parallel_op(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
        if not operations:
            return ()
        
        context = kwargs.get("context")
        
        tasks = []
        for op in operations:
            op_kwargs = dict(kwargs)
            tasks.append(op.execute(*args, **op_kwargs))
            
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result.is_error():
                raise result.error
        
        values = tuple(result.default_value(cast(Any, None)) for result in results)
        return values
    
    context_type = None
    for op in operations:
        if op.context_type is not None:
            if context_type is None:
                context_type = op.context_type
            elif issubclass(op.context_type, context_type):
                context_type = op.context_type
                
    return Operation._from_function(parallel_op, ctx_type=context_type, require_ctx=context_type is not None)


def fallback(*operations: Operation[P, T]) -> Operation[P, T]:
    """
    Try each operation in order until one succeeds.
    """
    async def fallback_op(*args: Any, **kwargs: Any) -> T:
        if not operations:
            raise ValueError("No operations provided to fallback")
        
        last_error = None
        
        for op in operations:
            op_kwargs = dict(kwargs)
            result = await op.execute(*args, **op_kwargs)
            
            if result.is_ok():
                return result.default_value(cast(Any, None))
            
            last_error = result.error
        
        raise last_error or Exception("All operations failed")
    
    context_type = None
    for op in operations:
        if op.context_type is not None:
            if context_type is None:
                context_type = op.context_type
            elif issubclass(op.context_type, context_type):
                context_type = op.context_type
                
    return Operation._from_function(fallback_op, ctx_type=context_type, require_ctx=context_type is not None)

