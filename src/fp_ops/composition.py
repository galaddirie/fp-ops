import asyncio
from typing import Any, List
from fp_ops.operator import Operation
from fp_ops.context import BaseContext
from expression import Result

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
