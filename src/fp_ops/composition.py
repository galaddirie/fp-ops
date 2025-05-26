import asyncio
from typing import Any, List, Union, Callable, Tuple, Dict, TypeVar, Concatenate, cast, Iterable, Awaitable

from fp_ops.operator import Operation, identity, _ensure_async, P, R, S, Q
from fp_ops.context import BaseContext
from expression import Result

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

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


def transform(operation: Operation[P, T], func: Callable[[T], U]) -> Operation[P, U]:
    """
    Map a function to an operation.
    """
    return operation.transform(func)


def map(
    item_op: Operation[[T], U],
    *,
    max_concurrency: int | None = None,
) -> Operation[[Iterable[T]], List[U]]:
    """
    Lift a single-item `Operation` into one that runs on every element of an
    iterable (such as a sequence, generator, or async-generator).

    Args:
        item_op: The `Operation` to apply to each element.
        max_concurrency: The maximum number of operations to run concurrently.
                         If None or 0, concurrency is unbounded (uses asyncio.gather).

    Returns:
        Operation[[Iterable[T]], List[U]]: An operation that takes an
        `Iterable[T]` (with optional `context=...` kwarg) and returns a
        `List[U]`.

    Notes:
        - Shares the same context instance with every mapped run.
        - Propagates the first encountered error (maintaining fp-ops semantics).
    """
    async def _runner(
        items: Iterable[T],
        *,
        context: BaseContext | None = None,
    ) -> List[U]:
        async def _exec(v: T) -> U:
            exec_kwargs: dict[str, BaseContext] = (
                {"context": context} if context is not None and item_op.context_type else {}
            )

            res: Result[U, Exception] = await (
                item_op.execute(v, **exec_kwargs)
            )
            if res.is_error():
                raise res.error
            return res.default_value(cast(U, None))

        coro_iter = (_exec(v) for v in items)

        if max_concurrency is None:
            return await asyncio.gather(*coro_iter)

        sem = asyncio.Semaphore(max_concurrency)
        async def _bounded(coro: Awaitable[U]) -> U:
            async with sem:
                return await coro
        return await asyncio.gather(*(_bounded(c) for c in coro_iter))  # noqa

    return Operation._from_function(
        _runner,
        require_ctx=item_op.context_type is not None,
        ctx_type=item_op.context_type,
    )


def filter(operation: Operation[P, T], func: Callable[[T], bool]) -> Operation[P, T]:
    """
    Filter a list of operations.
    """
    return operation.filter(func)


def reduce(operation: Operation[P, List[T]], func: Callable[[T, T], T]) -> Operation[P, T]:
    """
    Reduce a list of operations.
    """
    async def reduced(*args: Any, **kwargs: Any) -> T:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            raise result.error
            
        value = result.default_value(cast(Any, None))
        
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a list or tuple, got {type(value)}")
        
        if not value:
            return cast(T, None)
        
        try:
            from functools import reduce as functools_reduce
            result_value = functools_reduce(func, value)
            return result_value
        except Exception as e:
            raise e
    
    return Operation._from_function(reduced, ctx_type=operation.context_type, require_ctx=operation.context_type is not None)


def zip(*operations: Operation) -> Operation[P, Tuple[Any, ...]]:
    """
    Zip a list of operations.
    """
    async def zip_op(*args: Any, **kwargs: Any) -> Tuple[Any, ...]:
        if not operations:
            return ()
        
        results = await parallel(*operations).execute(*args, **kwargs)
        
        if results.is_error():
            raise results.error
            
        values = results.default_value(cast(Any, None))
        return values
    
    context_type = None
    for op in operations:
        if op.context_type is not None:
            if context_type is None:
                context_type = op.context_type
            elif issubclass(op.context_type, context_type):
                context_type = op.context_type
                
    return Operation._from_function(zip_op, ctx_type=context_type, require_ctx=context_type is not None)


def flat_map(
    operation: Operation[P, T],
    func: Callable[[T], List[U] | List[List[U]]],
) -> Operation[P, List[U]]:
    """
    Apply *func* to the **single** value returned by *operation*,
    then flatten the resulting list one level.

    Equivalent to lodash's `flatMap` or Kotlin's `flatMap`.
    """

    async def flat_mapped(*args: Any, **kwargs: Any) -> List[U]:
        res = await operation.execute(*args, **kwargs)
        if res.is_error():
            raise res.error

        value = res.default_value(cast(Any, None))

        try:
            mapped = func(cast(T, value))
        except Exception as exc:          # propagate mapper failure cleanly
            raise exc

        if not isinstance(mapped, list):
            raise TypeError(
                f"Mapper must return a list, got {type(mapped)}"
            )

        # flatten *one* level
        out: List[U] = []
        for item in mapped:
            if isinstance(item, list):
                out.extend(item)
            else:                         # already an element
                out.append(cast(U, item))
        return out

    return Operation._from_function(
        flat_mapped,
        ctx_type=operation.context_type,
        require_ctx=operation.context_type is not None,
    )


def group_by(operation: Operation[P, List[T]], func: Callable[[T], U]) -> Operation[P, Dict[U, List[T]]]:
    """
    Group a list of operations by a function.
    """
    async def grouped(*args: Any, **kwargs: Any) -> Dict[U, List[T]]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            raise result.error
            
        value = result.default_value(cast(Any, None))
        
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a list or tuple, got {type(value)}")
        
        try:
            groups: Dict[U, List[T]] = {}
            for item in value:
                key = func(item)
                if key not in groups:
                    groups[key] = []
                groups[key].append(item)
            
            return groups
        except Exception as e:
            raise e
    
    return Operation._from_function(grouped, ctx_type=operation.context_type, require_ctx=operation.context_type is not None)


def partition(operation: Operation[P, List[T]], func: Callable[[T], bool]) -> Operation[P, Tuple[List[T], List[T]]]:
    """
    Partition a list of operations.
    """
    async def partitioned(*args: Any, **kwargs: Any) -> Tuple[List[T], List[T]]:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            raise result.error
            
        value = result.default_value(cast(Any, None))
        
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a list or tuple, got {type(value)}")
        
        try:
            truthy = []
            falsy = []
            
            for item in value:
                if func(item):
                    truthy.append(item)
                else:
                    falsy.append(item)
            
            return (truthy, falsy)
        except Exception as e:
            raise e
    
    return Operation._from_function(partitioned, ctx_type=operation.context_type, require_ctx=operation.context_type is not None)


def first(operation: Operation[P, List[T]]) -> Operation[P, T]:
    """
    Return the first operation.
    """
    async def first_op(*args: Any, **kwargs: Any) -> T:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            raise result.error
            
        value = result.default_value(cast(Any, None))
        
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a list or tuple, got {type(value)}")
        
        if not value:
            raise IndexError("Sequence is empty")
        
        return value[0]
    
    return Operation._from_function(first_op, ctx_type=operation.context_type, require_ctx=operation.context_type is not None)


def last(operation: Operation[P, List[T]]) -> Operation[P, T]:
    """
    Return the last operation.
    """
    async def last_op(*args: Any, **kwargs: Any) -> T:
        result = await operation.execute(*args, **kwargs)
        
        if result.is_error():
            raise result.error
            
        value = result.default_value(cast(Any, None))
        
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a list or tuple, got {type(value)}")
        
        if not value:
            raise IndexError("Sequence is empty")
        
        return value[-1]
    
    return Operation._from_function(last_op, ctx_type=operation.context_type, require_ctx=operation.context_type is not None)


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
            maybe = op(*args or [], **op_kwargs)
            # Both Operation and _BoundCall expose .execute()
            tasks.append(maybe.execute(**op_kwargs))  # type: ignore[attr-defined]
            continue

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
