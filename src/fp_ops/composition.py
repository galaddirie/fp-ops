import asyncio
from typing import (
    Any,
    List,
    Union,
    Callable,
    Tuple,
    Dict,
    TypeVar,
    Concatenate,
    cast,
    Iterable,
    Awaitable,
    ParamSpec,
    Optional,
    Type,
)
from functools import reduce

from fp_ops.operator import Operation, operation, identity
from fp_ops.context import BaseContext
from expression import Result

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")



def pipe(*ops: Union[Operation, Callable]) -> Operation[..., Any]:
    """
    Left-to-right composition of operations and functions.
    
    Args:
        *ops: Operations or callables to compose from left to right.
        
    Returns:
        A new Operation representing the composition.
        
    Examples:
        pipe(add_one, multiply_by_two, str)  # (x + 1) * 2 -> str
        pipe(operation1, operation2, operation3)
    """
    if not ops:
        return identity
    
    # Convert all callables to Operations
    operations = []
    for op in ops:
        if isinstance(op, Operation):
            operations.append(op)
        else:
            operations.append(operation(op))
    
    # Chain them using the >> operator (left-to-right)
    result = operations[0]
    for op in operations[1:]:
        result = result >> op
    
    return result


def compose(*ops: Union[Operation, Callable]) -> Operation[..., Any]:
    """
    Right-to-left composition of operations and functions.
    
    Args:
        *ops: Operations or callables to compose from right to left.
        
    Returns:
        A new Operation representing the composition.
        
    Examples:
        compose(str, multiply_by_two, add_one)  # str((x + 1) * 2)
        compose(operation3, operation2, operation1)
    """
    return pipe(*reversed(ops))


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

    return Operation._from_function(
        parallel_op, ctx_type=context_type, require_ctx=context_type is not None
    )


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

    return Operation._from_function(
        fallback_op, ctx_type=context_type, require_ctx=context_type is not None
    )
