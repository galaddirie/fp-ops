import asyncio
import pytest
import time
import random
from typing import Dict, List, Any, Optional, Callable, TypeVar, Awaitable
from expression import Result

from fp_ops.operator import (
     constant, 
     identity,
     Operation
)
from fp_ops.flow import branch, attempt, fail
from fp_ops.decorators import operation
T = TypeVar("T")
S = TypeVar("S")


@pytest.fixture
def event_loop():
    """Create an event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def fetch_item():
    @operation
    async def _fetch_item(item_id: int) -> Dict[str, Any]:
        """Fetch an item by ID."""
        await asyncio.sleep(0.1)
        return {"id": item_id, "name": f"Item {item_id}", "category_id": item_id % 3 + 1}
    
    return _fetch_item

@pytest.fixture
def fetch_category():
    @operation
    async def _fetch_category(category_id: int) -> Dict[str, Any]:
        """Fetch a category by ID."""
        await asyncio.sleep(0.1)
        return {"id": category_id, "name": f"Category {category_id}"}
    
    return _fetch_category

@pytest.fixture
def fetch_related_items():
    @operation
    async def _fetch_related_items(category_id: int) -> List[Dict[str, Any]]:
        """Fetch items related to a category."""
        await asyncio.sleep(0.1)
        return [
            {"id": 100 + category_id, "name": f"Related Item {100 + category_id}"},
            {"id": 200 + category_id, "name": f"Related Item {200 + category_id}"}
        ]
    
    return _fetch_related_items

@pytest.fixture
def load_data():
    @operation
    async def _load_data() -> List[Dict[str, Any]]:
        """Load some sample data."""
        await asyncio.sleep(0.1)
        return [
            {"id": 1, "values": [10, 20, 30], "active": True},
            {"id": 2, "values": [5, 15, 25], "active": False},
            {"id": 3, "values": [100, 200], "active": True},
            {"id": 4, "values": [], "active": True}
        ]
    
    return _load_data

@pytest.fixture
def random_service():
    """Create a service with controllable failure rate for testing."""
    failure_rate = 0.0  # Start with no failures
    
    @operation
    async def _random_service() -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        if random.random() < failure_rate:
            raise ConnectionError("Random service failure")
        return {"status": "success"}
    
    # Add a method to control failure rate for testing
    def set_failure_rate(rate: float):
        nonlocal failure_rate
        failure_rate = max(0.0, min(1.0, rate))  # Ensure rate is between 0 and 1
    
    _random_service.set_failure_rate = set_failure_rate
    
    return _random_service

# -----------------------------------------------
# CONTINUATION MONAD PATTERN TESTS
# -----------------------------------------------

class TestContinuationMonad:
    """Test the continuation monad pattern functionality."""
    
    @pytest.mark.asyncio
    async def test_bind_chaining(self, fetch_item, fetch_category, fetch_related_items):
        """Test chaining operations with bind."""
        # Create a chain using bind
        get_item_details = fetch_item.bind(
            lambda item: fetch_category(item["category_id"]).bind(
                lambda category: fetch_related_items(category["id"]).map(
                    lambda related: {
                        "item": item,
                        "category": category,
                        "related_items": related
                    }
                )
            )
        )
        
        # Execute
        result = await get_item_details(1)
        
        assert result.is_ok()
        data = result.default_value(None)
        
        # Verify structure and content
        assert "item" in data
        assert "category" in data
        assert "related_items" in data
        assert data["item"]["id"] == 1
        assert data["category"]["id"] == data["item"]["category_id"]
        assert len(data["related_items"]) == 2
    
    # @pytest.mark.asyncio
    # async def test_binding_with_symbols(self, fetch_item, fetch_category, fetch_related_items):
    #     """Test binding with symbols."""
       
    #     get_item_details = (
    #         fetch_item 
    #         >> (lambda item: fetch_category(item["category_id"])) 
    #         >> (lambda category: fetch_related_items(category["id"])) 
    #     )
    #     result = await get_item_details(1)
    #     assert result.is_ok()
    #     data = result.default_value(None)
    #     assert "item" in data
    #     assert "category" in data
    #     assert "related_items" in data
            
            
            
    @pytest.mark.asyncio
    async def test_operator_composition(self, fetch_item, fetch_category, fetch_related_items):
        """Test achieving the same result with operator composition."""
        # Extract category_id
        @operation
        async def extract_category_id(item: Dict[str, Any]) -> int:
            return item["category_id"]
        
        # Combine results
        @operation
        async def combine_results(data_tuple: tuple) -> Dict[str, Any]:
            item, category_and_related = data_tuple
            category, related_items = category_and_related
            return {
                "item": item,
                "category": category,
                "related_items": related_items
            }
        
        # Create a pipeline with >> and & operators
        composed_pipeline = (
            fetch_item >> (
                operation(lambda item: item) & (
                    extract_category_id >> (
                        fetch_category & fetch_related_items
                    )
                )
            ) >> combine_results
        )
        
        # Execute
        result = await composed_pipeline(1)
        
        assert result.is_ok()
        data = result.default_value(None)
        
        # Verify structure and content
        assert "item" in data
        assert "category" in data
        assert "related_items" in data
        assert data["item"]["id"] == 1
        assert data["category"]["id"] == data["item"]["category_id"]
        assert len(data["related_items"]) == 2
    
    @pytest.mark.asyncio
    async def test_apply_cont(self, fetch_item, fetch_category):
        """Test the apply_cont method."""
        async def continuation(item):
            # Process the item in some way
            category_result = await fetch_category.execute(item["category_id"])
            if category_result.is_error():
                raise category_result.error
                
            category = category_result.default_value(None)
            return {"item": item, "category": category}
        
        # Use apply_cont to apply the continuation
        result = await fetch_item.apply_cont(continuation)
        
        # Verify results
        assert "item" in result
        assert "category" in result
        assert result["item"]["id"] == 1  # Using default item_id=1
        assert result["category"]["id"] == result["item"]["category_id"]

# -----------------------------------------------
# COMPLEX TRANSFORMATION TESTS
# -----------------------------------------------

class TestComplexTransformations:
    """Test complex transformation chains."""
    
    @pytest.mark.asyncio
    async def test_data_processing_pipeline(self, load_data):
        """Test a complex data processing pipeline."""
        # Define processing operations
        @operation
        async def filter_active(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [item for item in items if item.get("active", False)]
        
        @operation
        async def extract_values(items: List[Dict[str, Any]]) -> List[List[int]]:
            return [item.get("values", []) for item in items]
        
        @operation
        async def filter_non_empty(value_lists: List[List[int]]) -> List[List[int]]:
            return [values for values in value_lists if values]
        
        @operation
        async def flatten_lists(value_lists: List[List[int]]) -> List[int]:
            return [value for sublist in value_lists for value in sublist]
        
        @operation
        async def calculate_stats(values: List[int]) -> Dict[str, float]:
            if not values:
                return {"count": 0, "sum": 0, "average": 0, "min": 0, "max": 0}
                
            return {
                "count": len(values),
                "sum": sum(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        # Build pipeline
        stats_pipeline = (
            load_data >>
            filter_active >>
            extract_values >>
            filter_non_empty >>
            flatten_lists >>
            calculate_stats
        )
        
        # Execute
        result = await stats_pipeline()
        
        assert result.is_ok()
        stats = result.default_value(None)
        
        # Verify results
        assert stats["count"] == 5  # 10, 20, 30, 100, 200
        assert stats["sum"] == 360  # 10 + 20 + 30 + 100 + 200
        assert stats["average"] == 72.0  # 360 / 5
        assert stats["min"] == 10
        assert stats["max"] == 200
    
    @pytest.mark.asyncio
    async def test_nested_map_transformations(self, load_data):
        """Test alternative approach using nested map transformations."""
        # Define the last operation to calculate stats
        @operation
        async def calculate_stats(values: List[int]) -> Dict[str, float]:
            if not values:
                return {"count": 0, "sum": 0, "average": 0, "min": 0, "max": 0}
                
            return {
                "count": len(values),
                "sum": sum(values),
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        # Build pipeline with nested maps
        alternative_pipeline = load_data.map(
            # Filter active items
            lambda items: [item for item in items if item.get("active", False)]
        ).map(
            # Extract value lists
            lambda items: [item.get("values", []) for item in items]
        ).map(
            # Flatten values (including filtering empty lists)
            lambda value_lists: [value for sublist in value_lists if sublist for value in sublist]
        ).bind(calculate_stats)
        
        # Execute
        result = await alternative_pipeline()
        
        assert result.is_ok()
        stats = result.default_value(None)
        
        # Verify results (should be the same as previous test)
        assert stats["count"] == 5  # 10, 20, 30, 100, 200
        assert stats["sum"] == 360  # 10 + 20 + 30 + 100 + 200
        assert stats["average"] == 72.0  # 360 / 5
        assert stats["min"] == 10
        assert stats["max"] == 200

# -----------------------------------------------
# EDGE CASES TESTS
# -----------------------------------------------

class TestEdgeCases:
    """Test edge cases and error propagation."""
    
    @pytest.mark.asyncio
    async def test_operation_returns_none(self):
        """Test operation that returns None."""
        @operation
        async def returns_none() -> None:
            await asyncio.sleep(0.1)
            return None
        
        result = await returns_none()
        
        assert result.is_ok()
        assert result.default_value("default") is None
    
    @pytest.mark.asyncio
    async def test_operation_returns_result_directly(self):
        """Test operation that returns a Result directly."""
        @operation
        async def returns_result() -> Result[str, Exception]:
            await asyncio.sleep(0.1)
            return Result.Ok("Direct result")
        
        result = await returns_result()
        
        assert result.is_ok()
        assert result.default_value(None) == "Direct result"
    
    @pytest.mark.asyncio
    async def test_operation_returns_error_result(self):
        """Test operation that returns an error Result directly."""
        @operation
        async def returns_error() -> Result[str, Exception]:
            await asyncio.sleep(0.1)
            return Result.Error(ValueError("Direct error"))
        
        result = await returns_error()
        
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "Direct error"
    
    @pytest.mark.asyncio
    async def test_operation_raises_error(self):
        """Test operation that raises an exception."""
        @operation
        async def raises_error() -> Any:
            await asyncio.sleep(0.1)
            raise RuntimeError("Deliberate error")
        
        result = await raises_error()
        
        assert result.is_error()
        assert isinstance(result.error, RuntimeError)
        assert str(result.error) == "Deliberate error"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_type_error(self):
        """Test pipeline that causes a type error."""
        @operation
        async def returns_int() -> int:
            await asyncio.sleep(0.1)
            return 42
        
        # Create a pipeline that expects a string but gets an int
        pipeline = returns_int >> (lambda s: s.upper())
        
        result = await pipeline()
        
        assert result.is_error()
        assert isinstance(result.error, AttributeError)  # int has no upper() method
    
    @pytest.mark.asyncio
    async def test_error_recovery_in_pipeline(self):
        """Test error recovery in a pipeline."""
        @operation
        async def raises_error() -> Any:
            await asyncio.sleep(0.1)
            raise ValueError("Deliberate error")
        
        # Create a pipeline with error recovery
        pipeline = (
            raises_error.catch(lambda e: f"Recovered from {type(e).__name__}") >>
            (lambda s: f"Processed: {s}")
        )
        
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == "Processed: Recovered from ValueError"

# -----------------------------------------------
# CUSTOM COMBINATORS TESTS
# -----------------------------------------------

class TestCustomCombinators:
    """Test custom combinators and higher-order operations."""
    
    @pytest.mark.asyncio
    async def test_retry_if_error(self, random_service):
        """Test a custom retry_if_error combinator."""
        # Define a higher-order retry combinator
        def retry_if_error(op: Callable, max_attempts: int = 3, delay: float = 0.1) -> Operation:
            """Create a retry operation."""
            operation_obj = operation(op)
            return operation_obj.retry(attempts=max_attempts, delay=delay)
        
        # Set the service to fail on first attempt
        random_service.set_failure_rate(0.0)  # Start with no failures
        
        # Create retry combinator with random_service
        reliable_op = retry_if_error(random_service, max_attempts=3, delay=0.05)
        
        # Set failure rate to 100% for first call, then 0%
        call_count = 0
        original_execute = random_service.execute
        
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Result.Error(ConnectionError("First call fails"))
            return await original_execute(*args, **kwargs)
        
        # Replace execute method with mock
        random_service.execute = mock_execute
        
        # Execute
        result = await reliable_op()
        
        assert result.is_ok()
        assert call_count == 2  # Should succeed on second attempt
    
    @pytest.mark.asyncio
    async def test_when_combinator(self):
        """Test a custom when combinator for conditional execution."""
        # Define a when combinator for conditional branching
        def when(
            condition: Callable[[T], bool],
            then_branch: Operation,
            else_branch: Operation
        ) -> Operation:
            """Create conditional branching operation."""
            @operation
            async def conditional(input_value: T) -> Any:
                try:
                    if condition(input_value):
                        return await then_branch.execute(input_value)
                    else:
                        return await else_branch.execute(input_value)
                except Exception as e:
                    return Result.Error(e)
                    
            return conditional
        
        # Define a condition and branches
        def is_even(n: int) -> bool:
            return n % 2 == 0
            
        then_op = operation(lambda n: f"{n} is even")
        else_op = operation(lambda n: f"{n} is odd")
        
        # Create conditional operation
        conditional_op = when(is_even, then_op, else_op)
        
        # Test with even number
        even_result = await conditional_op(4)
        assert even_result.is_ok()
        assert even_result.default_value(None) == "4 is even"
        
        # Test with odd number
        odd_result = await conditional_op(7)
        assert odd_result.is_ok()
        assert odd_result.default_value(None) == "7 is odd"
    
    @pytest.mark.asyncio
    async def test_for_each_combinator(self):
        """Test a custom for_each combinator for list processing."""
        # Define a for_each combinator
        def for_each(item_operation: Operation) -> Operation:
            """Create an operation that applies an operation to each item in a list."""
            @operation
            async def process_items(items: List[Any]) -> List[Any]:
                results = []
                
                for item in items:
                    item_result = await item_operation.execute(item)
                    
                    if item_result.is_error():
                        return item_result
                        
                    results.append(item_result.default_value(None))
                    
                return results
                
            return process_items
        
        # Define an operation to process a single item
        @operation
        async def process_number(n: int) -> int:
            await asyncio.sleep(0.05)
            return n * 2
        
        # Create for_each operation
        process_all = for_each(process_number)
        
        # Test with a list of numbers
        numbers = [1, 2, 3, 4, 5]
        result = await process_all(numbers)
        
        assert result.is_ok()
        processed = result.default_value(None)
        assert processed == [2, 4, 6, 8, 10]

# -----------------------------------------------
# LAZY EVALUATION AND RECURSIVE TESTS
# -----------------------------------------------

class TestLazyAndRecursive:
    """Test lazy evaluation and recursive operations."""
    
    @pytest.mark.asyncio
    async def test_lazy_operation(self):
        """Test lazy evaluation of operations."""
        # Global counter to track recursion depth
        # We use a list to allow modification inside nested functions
        counter = [0] 
        max_depth = 5
        
        # Define a simple operation producer that can track its depth
        def create_leveled_operation(level: int) -> Operation:
            @operation
            async def level_op(x: Any) -> str:
                return f"Level {level}: {x}"
            return level_op
        
        # Define a lazy operation that manages recursion
        def create_recursive_op(current_depth: int) -> Operation:
            # Return a terminal operation if we've reached max depth
            if current_depth >= max_depth:
                return operation(lambda x: f"Max depth {current_depth} reached for {x}")
            
            # Create the current level operation
            curr_level_op = create_leveled_operation(current_depth)
            
            # Create a function that will lazily produce the next level
            # only when executed
            @operation
            async def next_level_connector(x: Any) -> Any:
                # Increment counter to track actual executions
                counter[0] += 1
                
                # Get result from current level
                curr_result = await curr_level_op(x)
                
                # Only proceed to next level if needed
                if isinstance(curr_result, Result):
                    curr_value = curr_result.default_value(None)
                else:
                    curr_value = curr_result
                    
                # Create and execute next level operation
                next_op = create_recursive_op(current_depth + 1)
                next_result = await next_op(f"From {curr_value}")
                
                return next_result
                
            return next_level_connector
        
        # Create our recursive operation starting at depth 0
        lazy_recursive_op = create_recursive_op(0)
        
        # Execute with different inputs
        for i in range(3):
            # Reset counter for each test case
            counter[0] = 0
            
            # Execute the operation
            result = await lazy_recursive_op(f"Input {i}")
            
            # Verify results
            assert result.is_ok()
            value = result.default_value(None)
            
            # Verify the operation worked correctly
            assert "Level 0" in value
            assert f"Input {i}" in value
            
            # Verify we didn't exceed max depth
            assert counter[0] <= max_depth
    
    @pytest.mark.asyncio
    async def test_recursive_factorial(self):
        """Test recursive factorial calculation."""
        # Define a recursive factorial operation
        @operation
        async def factorial(n: int) -> int:
            if n <= 1:
                return 1
            else:
                result = await factorial(n - 1)
                if result.is_error():
                    return result
                return n * result.default_value(1)
        
        # Test with various inputs
        expected_results = {
            0: 1,
            1: 1,
            2: 2,
            3: 6,
            4: 24,
            5: 120
        }
        
        for n, expected in expected_results.items():
            result = await factorial(n)
            assert result.is_ok()
            assert result.default_value(0) == expected
    
    @pytest.mark.asyncio
    async def test_recursive_fibonacci(self):
        """Test recursive Fibonacci calculation with parallel execution."""
        # Define a recursive Fibonacci operation using parallel execution
        @operation
        async def fibonacci(n: int) -> int:
            if n <= 1:
                return n
                
            # Calculate fib(n-1) and fib(n-2) in parallel
            results = await (fibonacci(n - 1) & fibonacci(n - 2))()
            
            if results.is_error():
                return results
                
            a, b = results.default_value((0, 0))
            return a + b
        
        # Test with various inputs
        expected_results = {
            0: 0,
            1: 1,
            2: 1,
            3: 2,
            4: 3,
            5: 5,
            6: 8
        }
        
        for n, expected in expected_results.items():
            result = await fibonacci(n)
            assert result.is_ok()
            assert result.default_value(0) == expected

# Run the tests with: pytest -xvs test_advanced_operations.py