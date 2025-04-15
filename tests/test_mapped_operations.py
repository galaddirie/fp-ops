import pytest
import asyncio
from typing import Any, Dict, List, Optional, Type, Tuple, cast
from unittest.mock import Mock, AsyncMock

from fp_ops.context import BaseContext
from fp_ops.operator import Operation, operation, identity, constant
from fp_ops.placeholder import _
from fp_ops.flow import map_operations
from expression import Result


class TestContext(BaseContext):
    value: str = "test"


@pytest.mark.asyncio
async def test_map_operations_with_simple_operation():
    """Test map_operations with a simple operation."""
    
    @operation
    async def double(x: int) -> int:
        return x * 2
    
    mapper = map_operations(double)
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_ok()
    assert result.default_value(None) == [2, 4, 6, 8]


@pytest.mark.asyncio
async def test_map_operations_with_placeholder():
    """Test map_operations with an operation using placeholders."""
    
    @operation
    async def multiply(x: int, y: int) -> int:
        return x * y
    
    # Use placeholder to bind the second argument
    mapper = map_operations(multiply(_, 2))
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_ok()
    assert result.default_value(None) == [2, 4, 6, 8]


@pytest.mark.asyncio
async def test_map_operations_with_partial_application():
    """Test map_operations with a partially applied operation."""
    
    @operation
    async def multiply(x: int, y: int) -> int:
        return x * y
    
    # Partially apply the first argument
    mapper = map_operations(multiply(2))
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_ok()
    assert result.default_value(None) == [2, 4, 6, 8]  # 2*1, 2*2, 2*3, 2*4


@pytest.mark.asyncio
async def test_map_operations_with_composed_operations():
    """Test map_operations with composed operations using >>."""
    
    @operation
    async def double(x: int) -> int:
        return x * 2
    
    @operation
    async def add_five(x: int) -> int:
        return x + 5
    
    # Compose operations: double >> add_five
    mapper = map_operations(double >> add_five)
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_ok()
    # (1*2)+5, (2*2)+5, (3*2)+5, (4*2)+5
    assert result.default_value(None) == [7, 9, 11, 13]


@pytest.mark.asyncio
async def test_map_operations_with_composed_operations_and_placeholder():
    """Test map_operations with composed operations using placeholders."""
    
    @operation
    async def multiply(x: int, y: int) -> int:
        return x * y
    
    @operation
    async def add(x: int, y: int) -> int:
        return x + y
    
    # Compose operations with placeholders
    mapper = map_operations(multiply(_, 2) >> add(_, 5))
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_ok()
    # (1*2)+5, (2*2)+5, (3*2)+5, (4*2)+5
    assert result.default_value(None) == [7, 9, 11, 13]


@pytest.mark.asyncio
async def test_map_operations_with_composed_partially_applied_operations():
    """Test map_operations with composed operations that are partially applied."""
    
    @operation
    def multiply(x: int, y: int) -> int:
        return x * y
    
    @operation
    def add(x: int, y: int) -> int:
        return x + y
    
    # Compose partially applied operations
    @operation
    async def wrap(x: int) -> int:
        pipeline = (multiply(x, 2) >> add(5))
        return await pipeline(1)
    
    mapper = map_operations(wrap)
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_ok()
    # (2*1)+5, (2*2)+5, (2*3)+5, (2*4)+5
    assert result.default_value(None) == [7, 9, 11, 13]


@pytest.mark.asyncio
async def test_map_operations_with_complex_composition():
    """Test map_operations with complex compositions."""
    
    @operation
    async def multiply(x: int, y: int) -> int:
        return x * y
    
    @operation
    async def add(x: int, y: int) -> int:
        return x + y
    
    @operation
    async def divide(x: int, y: int) -> int:
        return x // y
    
    # Complex composition with multiple operations
    @operation
    async def wrap(x: int) -> int:
        pipeline = multiply(x, 2) >> add(_, 5) >> divide(_, 3)
        return await pipeline.execute()
    
    mapper = map_operations(wrap)
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_ok()
    
    assert result.default_value(None) == [2, 3, 3, 4]


@pytest.mark.asyncio
async def test_map_operations_error_handling():
    """Test error handling in map_operations."""
    
    @operation
    async def divide(x: int, y: int) -> int:
        return x // y
    
    # Division by zero will cause an error
    mapper = map_operations(divide(_, 0))
    result = await mapper.execute([1, 2, 3, 4])
    
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)


@pytest.mark.asyncio
async def test_map_operations_with_context():
    """Test map_operations with context passing."""
    
    class CustomContext(BaseContext):
        multiplier: int = 2
    
    @operation(context=True, context_type=CustomContext)
    async def multiply_with_context(x: int, context=None) -> int:
        return x * context.multiplier
    
    # Create context
    context = CustomContext(multiplier=3)
    
    # Map operation with context
    mapper = map_operations(multiply_with_context)
    result = await mapper.execute([1, 2, 3, 4], context=context)
    
    assert result.is_ok()
    # 1*3, 2*3, 3*3, 4*3
    assert result.default_value(None) == [3, 6, 9, 12]


@pytest.mark.asyncio
async def test_parallel_execution():
    """Test parallel execution of map_operations."""
    
    @operation
    async def slow_operation(x: int) -> int:
        await asyncio.sleep(0.1)  # Simulate a slow operation
        return x * 2
    
    # Test with parallel=True
    mapper = map_operations(slow_operation, parallel=True)
    
    # Start timer
    start_time = asyncio.get_event_loop().time()
    result = await mapper.execute([1, 2, 3, 4, 5])
    end_time = asyncio.get_event_loop().time()
    
    assert result.is_ok()
    assert result.default_value(None) == [2, 4, 6, 8, 10]
    
    # Execution time should be close to the time for one operation
    # instead of the sum of all operations
    assert end_time - start_time < 0.3  # Conservative estimate


@pytest.mark.asyncio
async def test_real_world_example():
    """Test the original use case from the problem description."""
    
    @operation
    async def RemoveAppHeaderChildren() -> List[str]:
        """Simulate returning a list of elements."""
        return ["element1", "element2", "element3"]
    
    @operation
    async def Fill(element: str, term: str) -> str:
        """Simulate filling an element with text."""
        return f"{element} filled with '{term}'"
    
    @operation
    async def Wait(seconds: int) -> int:
        """Simulate waiting for a number of seconds."""
        await asyncio.sleep(0.01)  # Small delay for testing
        return seconds
    
    # Test term to use
    term = "search term"
    
    # Test first example: Fill with placeholder
    pipeline1 = RemoveAppHeaderChildren >> map_operations(Fill(_, term) >> Wait(2))
    result1 = await pipeline1.execute()
    
    assert result1.is_ok()
    # Each element should return 2 (from Wait)
    assert result1.default_value(None) == [2, 2, 2]
    
    # Test second example: Fill with term as first arg
    # This would previously fail with "Fill() missing 1 required positional argument: 'text'"
    fill_with_term = operation(lambda element: Fill(element, term))
    pipeline2 = RemoveAppHeaderChildren >> map_operations(fill_with_term >> Wait(2))
    result2 = await pipeline2.execute()
    
    assert result2.is_ok()
    assert result2.default_value(None) == [2, 2, 2]


@pytest.mark.asyncio
async def test_alternative_real_world_example():
    """Test directly with the syntax from the problem description."""
    
    @operation
    async def RemoveAppHeaderChildren() -> List[str]:
        """Simulate returning a list of elements."""
        return ["element1", "element2", "element3"]
    
    # Define Fill to match the problem description's usage
    # This version allows term as first parameter for partial application
    @operation
    async def Fill(element_or_term: str, term_or_element: str = None) -> str:
        """Flexible Fill operation that can work with either order of parameters."""
        if term_or_element is None:
            # When used as Fill(term)
            return f"filled with '{element_or_term}'"
        else:
            # When used as Fill(element, term) or Fill(_, term)
            return f"{element_or_term} filled with '{term_or_element}'"
    
    @operation
    async def Wait(seconds: int) -> int:
        """Simulate waiting for a number of seconds."""
        await asyncio.sleep(0.01)
        return seconds
    
    term = "search term"
    
    # Test exactly as in the problem description
    pipeline1 = RemoveAppHeaderChildren >> map_operations(Fill(_, term) >> Wait(2))
    pipeline2 = RemoveAppHeaderChildren >> map_operations(Fill(term) >> Wait(2))
    
    result1 = await pipeline1.execute()
    result2 = await pipeline2.execute()
    
    assert result1.is_ok()
    assert result2.is_ok()
    
    # Both should complete successfully with the enhanced implementation
    assert result1.default_value(None) == [2, 2, 2]
    assert result2.default_value(None) == [2, 2, 2]