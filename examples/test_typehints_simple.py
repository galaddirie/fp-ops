from typing import List
from pydantic import BaseModel, Field
from fp_ops.operator import operation
from fp_ops.context import BaseContext

# Simple context types
class MathContext(BaseContext):
    """Context for math operations."""
    precision: int = 2
    max_value: float = 1000.0
    min_value: float = -1000.0

class StringContext(BaseContext):
    """Context for string operations."""
    max_length: int = 100

# Simple math operations with context
@operation(context=True, context_type=MathContext) # type: ignore
async def add(a: float, b: float, *, context: MathContext) -> float:
    """Add two numbers with context-aware precision."""
    result = a + b
    if result > context.max_value:
        raise ValueError(f"Result {result} exceeds max_value {context.max_value}")
    if result < context.min_value:
        raise ValueError(f"Result {result} below min_value {context.min_value}")
    return round(result, context.precision)

@operation(context=True, context_type=MathContext)
async def subtract(a: float, b: float, *, context: MathContext) -> float:
    """Subtract two numbers with context-aware precision."""
    result = a - b
    if result > context.max_value:
        raise ValueError(f"Result {result} exceeds max_value {context.max_value}")
    if result < context.min_value:
        raise ValueError(f"Result {result} below min_value {context.min_value}")
    return round(result, context.precision)

# Simple string operations with context
@operation(context=True, context_type=StringContext)
async def uppercase(text: str, *, context: StringContext) -> str:
    """Convert text to uppercase with context-aware validation."""
    if len(text) > context.max_length:
        raise ValueError(f"Text length {len(text)} exceeds max_length {context.max_length}")
    return text.upper()

@operation(context=True, context_type=StringContext)
async def lowercase(text: str, *, context: StringContext) -> str:
    """Convert text to lowercase with context-aware validation."""
    if len(text) > context.max_length:
        raise ValueError(f"Text length {len(text)} exceeds max_length {context.max_length}")
    return text.lower()

# Simple list operations with context
@operation(context=True, context_type=MathContext)
async def sum_list(numbers: List[float], *, context: MathContext) -> float:
    """Sum a list of numbers with context-aware validation."""
    result = sum(numbers)
    if result > context.max_value:
        raise ValueError(f"Sum {result} exceeds max_value {context.max_value}")
    if result < context.min_value:
        raise ValueError(f"Sum {result} below min_value {context.min_value}")
    return round(result, context.precision) 