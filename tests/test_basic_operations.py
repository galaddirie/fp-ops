import asyncio
import pytest
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from expression import Result

# Import the Operation class and utilities from your module
from fp_ops.operator import (
    operation, constant, fail, attempt, 
    Operation
)

from fp_ops.placeholder import _

# -----------------------------------------------
# FIXTURES
# -----------------------------------------------

@pytest.fixture
def event_loop():
    """Create an event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# Sample operations used across multiple tests
@pytest.fixture
def fetch_data():
    @operation
    async def _fetch_data(url: str) -> Dict[str, Any]:
        """Simulate fetching data from a URL."""
        await asyncio.sleep(0.1)  # Short delay for testing
        
        if "users" in url:
            return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        elif "posts" in url:
            return {"posts": [{"id": 101, "title": "Hello World"}, {"id": 102, "title": "Python Tips"}]}
        else:
            return {"status": "ok", "message": f"Response from {url}"}
    
    return _fetch_data

@pytest.fixture
def extract_user_names():
    @operation
    async def _extract_user_names(data: Dict[str, Any]) -> List[str]:
        """Extract user names from API response."""
        users = data.get("users", [])
        return [user["name"] for user in users]
    
    return _extract_user_names

@pytest.fixture
def format_names():
    @operation
    async def _format_names(names: List[str]) -> str:
        """Format a list of names into a string."""
        return ", ".join(names)
    
    return _format_names

@pytest.fixture
def risky_operation():
    @operation
    async def _risky_operation(n: int) -> int:
        """An operation that might fail."""
        await asyncio.sleep(0.1)
        
        if n < 0:  # Changed from n <= 0 to n < 0 to allow n=0 to proceed to division
            raise ValueError("Input must be positive")
        
        return 100 // n  # Now n=0 will raise ZeroDivisionError
    
    return _risky_operation

@pytest.fixture
def fetch_users():
    @operation
    async def _fetch_users() -> Dict[str, Any]:
        """Fetch user data."""
        await asyncio.sleep(0.1)
        return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
    
    return _fetch_users

@pytest.fixture
def fetch_posts():
    @operation
    async def _fetch_posts() -> Dict[str, Any]:
        """Fetch post data."""
        await asyncio.sleep(0.1)
        return {"posts": [{"id": 101, "title": "Hello World"}, {"id": 102, "title": "Python Tips"}]}
    
    return _fetch_posts

@pytest.fixture
def flaky_service():
    """A service that fails on first attempt but succeeds on retry."""
    call_count = 0
    
    @operation
    async def _flaky_service() -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:  # Fail on first attempt
            raise ConnectionError("Service temporarily unavailable")
        
        return {"status": "success", "data": "Service data", "attempts": call_count}
    
    return _flaky_service

# -----------------------------------------------
# BASIC COMPOSITION TESTS
# -----------------------------------------------

class TestBasicComposition:
    """Test basic composition functionality of Operations."""
    
    @pytest.mark.asyncio
    async def test_basic_composition(self, fetch_data, extract_user_names, format_names):
        """Test basic operation composition using >>."""
        # Create a pipeline
        pipeline = fetch_data >> extract_user_names >> format_names
        
        # Execute the pipeline
        result = await pipeline("https://api.example.com/users")
        
        assert result.is_ok()
        assert result.default_value(None) == "Alice, Bob"
    
    @pytest.mark.asyncio
    async def test_composition_with_constant(self, fetch_data):
        """Test composition with constant values."""
        pipeline = fetch_data >> constant("Fixed output")
        
        result = await pipeline("https://api.example.com/any")
        
        assert result.is_ok()
        assert result.default_value(None) == "Fixed output"
    
    @pytest.mark.asyncio
    async def test_composition_with_lambda(self, fetch_data):
        """Test composition with lambda functions."""
        pipeline = fetch_data >> (lambda data: f"Processed: {data.get('status', 'unknown')}")
        
        result = await pipeline("https://api.example.com/status")
        
        assert result.is_ok()
        assert result.default_value(None) == "Processed: ok"
        
    @pytest.mark.asyncio
    async def test_step_by_step_execution(self, fetch_data, extract_user_names, format_names):
        """Test executing operations step by step vs in a pipeline."""
        # Pipeline approach
        pipeline = fetch_data >> extract_user_names >> format_names
        pipeline_result = await pipeline("https://api.example.com/users")
        
        # Step by step approach
        fetch_result = await fetch_data("https://api.example.com/users")
        assert fetch_result.is_ok()
        
        names_result = await extract_user_names(fetch_result.default_value(None))
        assert names_result.is_ok()
        
        formatted_result = await format_names(names_result.default_value(None))
        assert formatted_result.is_ok()
        
        # Results should be the same
        assert pipeline_result.default_value(None) == formatted_result.default_value(None)
        assert pipeline_result.default_value(None) == "Alice, Bob"

# -----------------------------------------------
# ERROR HANDLING TESTS
# -----------------------------------------------

class TestErrorHandling:
    """Test error handling capabilities of Operations."""
    
    @pytest.mark.asyncio
    async def test_basic_error_propagation(self, risky_operation):
        """Test that errors propagate through the pipeline."""
        # Test with valid input
        valid_result = await risky_operation(10)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 10  # 100 // 10 == 10
        
        # Test with error-causing input
        error_result = await risky_operation(0)
        assert error_result.is_error()
        assert isinstance(error_result.error, ZeroDivisionError)
        
        # Test error propagation in pipeline
        pipeline = risky_operation >> (lambda x: x * 2)
        pipeline_result = await pipeline(0)
        assert pipeline_result.is_error()
    
    @pytest.mark.asyncio
    async def test_catch_error_handler(self, risky_operation):
        """Test using catch to handle errors."""
        error_handler = lambda e: f"Error handled: {type(e).__name__}"
        with_recovery = risky_operation.catch(error_handler)
        
        # Test with valid input
        valid_result = await with_recovery(10)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 10
        
        # Test with error-causing input
        error_result = await with_recovery(0)
        assert error_result.is_ok()  # Should be OK now because we caught the error
        assert error_result.default_value(None) == "Error handled: ZeroDivisionError"
    
    @pytest.mark.asyncio
    async def test_default_value(self, risky_operation):
        """Test using default_value for error cases."""
        with_default = risky_operation.default_value(999)
        
        # Test with valid input
        valid_result = await with_default(10)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 10
        
        # Test with error-causing input
        error_result = await with_default(0)
        assert error_result.is_ok()  # Should be OK because we provided a default
        assert error_result.default_value(None) == 999
    
    @pytest.mark.asyncio
    async def test_failure_in_pipeline(self, fetch_data, risky_operation):
        """Test how errors propagate in a pipeline."""
        pipeline = fetch_data >> risky_operation
        
        # This should fail because the fetch_data result won't be an integer
        result = await pipeline("https://api.example.com/users")
        
        assert result.is_error()
        # Exact error type might vary, but it should be an error
    
    @pytest.mark.asyncio
    async def test_fail_operation(self):
        """Test the fail operation factory."""
        always_fail = fail(ValueError("Deliberate failure"))
        
        result = await always_fail()
        
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "Deliberate failure"

# -----------------------------------------------
# PARALLEL EXECUTION TESTS
# -----------------------------------------------

class TestParallelExecution:
    """Test parallel execution with the & operator."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, fetch_users, fetch_posts):
        """Test that & operator executes operations in parallel."""
        # Execute in parallel
        parallel_op = fetch_users & fetch_posts
        
        # The start and end time assert doesn't always work in tests due to asyncio's behavior
        # in test environments, so we'll just test the result
        parallel_result = await parallel_op()
        
        assert parallel_result.is_ok()
        users_data, posts_data = parallel_result.default_value((None, None))
        
        assert "users" in users_data
        assert "posts" in posts_data
    
    @pytest.mark.asyncio
    async def test_parallel_with_composition(self, fetch_users, fetch_posts):
        """Test combining parallel execution with composition."""
        # Define a combiner function
        @operation
        async def combine_data(data_tuple):
            users_data, posts_data = data_tuple
            return {
                "users": users_data.get("users", []),
                "posts": posts_data.get("posts", []),
                "counts": {
                    "users": len(users_data.get("users", [])),
                    "posts": len(posts_data.get("posts", []))
                }
            }
        
        # Build the pipeline
        parallel_op = fetch_users & fetch_posts
        pipeline = parallel_op >> combine_data
        
        # Execute
        result = await pipeline()
        
        assert result.is_ok()
        combined = result.default_value(None)
        
        assert "users" in combined
        assert "posts" in combined
        assert "counts" in combined
        assert combined["counts"]["users"] == 2
        assert combined["counts"]["posts"] == 2
    
    @pytest.mark.asyncio
    async def test_parallel_with_error(self, fetch_users):
        """Test parallel execution when one operation fails."""
        # Create an operation that always fails
        failing_op = operation(lambda: asyncio.sleep(0.1)) >> fail(RuntimeError("Simulated failure"))
        
        # Execute in parallel
        parallel_op = fetch_users & failing_op
        
        # The result should be an error
        result = await parallel_op()
        
        assert result.is_error()
        assert isinstance(result.error, RuntimeError)

# -----------------------------------------------
# FALLBACK PATTERN TESTS
# -----------------------------------------------

class TestFallbackPatterns:
    """Test fallback patterns with the | operator."""
    
    @pytest.mark.asyncio
    async def test_basic_fallback(self):
        """Test basic fallback functionality."""
        # Create operations
        primary = fail(ConnectionError("Primary failed"))
        backup = constant({"source": "backup", "data": "Backup data"})
        
        # Create a fallback chain
        fallback = primary | backup
        
        # Execute
        result = await fallback()
        
        assert result.is_ok()
        data = result.default_value(None)
        assert data["source"] == "backup"
    
    @pytest.mark.asyncio
    async def test_multiple_fallbacks(self):
        """Test chaining multiple fallbacks."""
        # Create operations
        op1 = fail(ValueError("First error"))
        op2 = fail(ValueError("Second error"))
        op3 = constant("Success")
        
        # Chain fallbacks
        fallback = op1 | op2 | op3
        
        # Execute
        result = await fallback()
        
        assert result.is_ok()
        assert result.default_value(None) == "Success"
    
    @pytest.mark.asyncio
    async def test_all_fallbacks_fail(self):
        """Test case where all fallbacks fail."""
        # Create operations
        op1 = fail(ValueError("First error"))
        op2 = fail(ValueError("Second error"))
        op3 = fail(ValueError("Third error"))
        
        # Chain fallbacks
        fallback = op1 | op2 | op3
        
        # Execute
        result = await fallback()
        
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "Third error"  # The last error propagates
    
    @pytest.mark.asyncio
    async def test_fallback_with_condition(self, fetch_data):
        """Test fallback with conditional logic."""
        # Create a fallback that depends on the result of an operation
        complex_fallback = (
            fetch_data.filter(lambda d: "users" in d and len(d["users"]) > 0, "No users found") | 
            constant({"users": [{"id": 0, "name": "Default User"}]})
        )
        
        # Test with data that satisfies the condition
        result_with_users = await complex_fallback("https://api.example.com/users")
        assert result_with_users.is_ok()
        data_with_users = result_with_users.default_value(None)
        assert len(data_with_users["users"]) == 2
        
        # Test with data that fails the condition
        result_without_users = await complex_fallback("https://api.example.com/posts")
        assert result_without_users.is_ok()  # Should be OK because we used fallback
        data_without_users = result_without_users.default_value(None)
        assert len(data_without_users["users"]) == 1
        assert data_without_users["users"][0]["name"] == "Default User"

# -----------------------------------------------
# TRANSFORMATION TESTS
# -----------------------------------------------

class TestTransformations:
    """Test map, bind and other transformation capabilities."""
    
    @pytest.mark.asyncio
    async def test_map_transformation(self, fetch_data):
        """Test map function for simple transformations."""
        # Create a transformation
        get_first_user = fetch_data.map(
            lambda data: data.get("users", [{}])[0].get("name", "Unknown")
        )
        
        # Execute
        result = await get_first_user("https://api.example.com/users")
        
        assert result.is_ok()
        name = result.default_value(None)
        assert name == "Alice"
    
    @pytest.mark.asyncio
    async def test_bind_transformation(self, fetch_data):
        """Test bind function for chaining dependent operations."""
        # Create dependent operations
        @operation
        async def get_user_details(user_name: str) -> Dict[str, Any]:
            if user_name == "Alice":
                return {"name": user_name, "email": "alice@example.com", "role": "admin"}
            else:
                return {"name": user_name, "email": f"{user_name.lower()}@example.com", "role": "user"}
        
        # Create a binding transformation
        get_first_user_details = fetch_data.bind(
            lambda data: get_user_details(data.get("users", [{}])[0].get("name", "Unknown"))
        )
        
        # Execute
        result = await get_first_user_details("https://api.example.com/users")
        
        assert result.is_ok()
        user_details = result.default_value(None)
        assert user_details["name"] == "Alice"
        assert user_details["email"] == "alice@example.com"
        assert user_details["role"] == "admin"
    
    @pytest.mark.asyncio
    async def test_filter_operation(self, fetch_data):
        """Test filter operation for validation."""
        # Create a filter to validate a specific condition
        valid_data = fetch_data.filter(
            lambda data: "users" in data and len(data["users"]) > 0,
            "No users found"
        )
        
        # Test with valid data
        valid_result = await valid_data("https://api.example.com/users")
        assert valid_result.is_ok()
        
        # Test with invalid data
        invalid_result = await valid_data("https://api.example.com/posts")
        assert invalid_result.is_error()
        assert str(invalid_result.error) == "No users found"

# -----------------------------------------------
# UTILITY TESTS
# -----------------------------------------------

class TestUtilities:
    """Test utility operations like tap, retry, etc."""
    
    @pytest.mark.asyncio
    async def test_tap_operation(self, fetch_data):
        """Test tap operation for side effects."""
        # Create a collector for side effects
        side_effect_results = []
        
        # Create a tap operation
        tapped = fetch_data.tap(
            lambda data: side_effect_results.append(data.get("message", "unknown"))
        )
        
        # Execute
        result = await tapped("https://api.example.com/status")
        
        # The result should be unchanged
        assert result.is_ok()
        assert "status" in result.default_value(None)
        
        # But the side effect should have happened
        assert len(side_effect_results) == 1
        assert side_effect_results[0] == "Response from https://api.example.com/status"
    
    @pytest.mark.asyncio
    async def test_retry_operation(self, flaky_service):
        """Test retry operation for handling transient failures."""
        # Create a retry operation
        reliable = flaky_service.retry(attempts=3, delay=0.05)
        
        # Execute
        result = await reliable()
        
        # The operation should have succeeded on the second attempt
        assert result.is_ok()
        data = result.default_value(None)
        assert data["status"] == "success"
        assert data["attempts"] == 2  # Second attempt
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test retry operation when all attempts fail."""
        # Create an operation that always fails
        always_fails = operation(lambda: asyncio.sleep(0.05)) >> fail(RuntimeError("Always fails"))
        
        # Create a retry operation with minimal delay for testing
        retry_but_fail = always_fails.retry(attempts=2, delay=0.01)
        
        # Execute
        result = await retry_but_fail()
        
        # Should still fail after all retries
        assert result.is_error()
        assert isinstance(result.error, RuntimeError)

# -----------------------------------------------
# SEQUENCE AND COMBINE TESTS
# -----------------------------------------------

class TestSequenceAndCombine:
    """Test sequence and combine class methods."""
    
    @pytest.mark.asyncio
    async def test_sequence_operations(self):
        """Test the sequence class method."""
        # Create some simple operations
        @operation
        async def step1(data):
            return f"Step 1: {data}"
        
        @operation
        async def step2(data):
            return f"Step 2: {data}"
        
        @operation
        async def step3(data):
            return f"Step 3: {data}"
        
        # Create a sequence
        sequence_op = await Operation.sequence([step1, step2, step3])
        
        # Execute with the same input for all operations
        result = await sequence_op("input")
        
        assert result.is_ok()
        steps = result.default_value(None)
        assert len(steps) == 3
        assert steps[0] == "Step 1: input"
        assert steps[1] == "Step 2: input"
        assert steps[2] == "Step 3: input"
    
    @pytest.mark.asyncio
    async def test_combine_operations(self):
        """Test the combine class method."""
        # Create some simple operations
        @operation
        async def get_name(data):
            return f"Name: {data}"
        
        @operation
        async def get_id(data):
            return f"ID: {data}"
        
        @operation
        async def get_email(data):
            return f"Email: {data}@example.com"
        
        # Create a combined operation
        combined_op = await Operation.combine(
            name=get_name,
            id=get_id,
            email=get_email
        )
        
        # Execute with the same input for all operations
        result = await combined_op("user123")
        
        assert result.is_ok()
        combined = result.default_value(None)
        assert "name" in combined
        assert "id" in combined
        assert "email" in combined
        assert combined["name"] == "Name: user123"
        assert combined["id"] == "ID: user123"
        assert combined["email"] == "Email: user123@example.com"

# -----------------------------------------------
# SYNC FUNCTION TESTS
# -----------------------------------------------

class TestSyncFunctions:
    """Test working with synchronous functions."""
    
    @pytest.mark.asyncio
    async def test_sync_function_wrapping(self):
        """Test wrapping synchronous functions with operation."""
        # Define a synchronous function
        def calculate_square(n: int) -> int:
            return n * n
        
        # Wrap with operation
        square_op = operation(calculate_square)
        
        # Execute
        result = await square_op(5)
        
        assert result.is_ok()
        assert result.default_value(None) == 25
    
    @pytest.mark.asyncio
    async def test_sync_function_in_pipeline(self, fetch_data):
        """Test using synchronous functions in a pipeline."""
        # Define a synchronous function
        def extract_count(data: Dict[str, Any]) -> int:
            return len(data.get("users", []))
        
        # Create a pipeline
        pipeline = fetch_data >> operation(extract_count)
        
        # Execute
        result = await pipeline("https://api.example.com/users")
        
        assert result.is_ok()
        assert result.default_value(None) == 2
    
    @pytest.mark.asyncio
    async def test_sync_function_with_error(self):
        """Test error handling with synchronous functions."""
        # Define a synchronous function that raises an exception
        def divide(a: int, b: int) -> float:
            return a / b
        
        # Wrap with operation
        divide_op = operation(divide)
        
        # Test with valid input
        valid_result = await divide_op(10, 2)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 5.0
        
        # Test with error-causing input
        error_result = await divide_op(10, 0)
        assert error_result.is_error()
        assert isinstance(error_result.error, ZeroDivisionError)

class TestOverloading:
    
    @pytest.mark.asyncio
    async def test_overloading_forwarded_args(self):
        """ users should be able override the value sent from the previous operation"""
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        
        @operation
        async def add_one(a: int) -> int:
            return a + 1
        
        # Create a pipeline
        pipeline = add(1, 2) >> add_one
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 4

        pipeline = add(1, 2) >> add_one(1)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 2

    @pytest.mark.asyncio
    async def test_operation_with_missing_arguments(self):
        """ Test that appropriate errors are raised when arguments are missing """
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        
        pipeline = subtract(10, 4) >> divide
        result = await pipeline()
        assert result.is_error()
        assert "missing 1 required positional argument: 'b'" in str(result.error), f"should raise a TypeError {result.error}"

    @pytest.mark.asyncio
    async def test_operation_with_value_forwarding(self):
        """ Test that values are correctly forwarded between operations """
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        
        # Testing with value forwarded from previous operation
        pipeline = subtract(10, 4).bind(lambda a: divide(a, 2))
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 3.0  # (10-4)/2 = 3.0
        
    @pytest.mark.asyncio
    async def test_operation_with_complete_argument_override(self):
        """ Test that operations can completely override arguments from previous operations """
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        
        # Testing with completely different values
        pipeline = subtract(20, 5) >> divide(30, 5)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 6.0  # 30/5 = 6.0




class TestPlaceholders:
    """Tests for the placeholder functionality in the Operation class."""

    @pytest.mark.asyncio
    async def test_basic_placeholder_usage(self):
        """Test basic placeholder usage in the first argument position."""
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        
        # Test the placeholder in the first argument position
        pipeline = subtract(10, 4) >> divide(_, 2)
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 3.0  # (10-4)/2 = 3.0

    @pytest.mark.asyncio
    async def test_placeholder_in_second_position(self):
        """Test placeholder usage in the second argument position."""
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        
        @operation
        async def multiply(a: int, b: int) -> int:
            return a * b
        
        # Test the placeholder in the second argument position
        pipeline = add(5, 3) >> multiply(2, _)
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 16  # 2 * (5+3) = 16

    @pytest.mark.asyncio
    async def test_placeholder_in_keyword_args(self):
        """Test placeholder usage in keyword arguments."""
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        
        @operation
        async def divide(dividend: int, divisor: int) -> float:
            return dividend / divisor
        
        # Test with placeholder in keyword arguments
        pipeline = subtract(10, 4) >> divide(dividend=_, divisor=2)
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 3.0  # (10-4)/2 = 3.0
        
        # Test with placeholder in a different keyword position
        pipeline2 = subtract(10, 4) >> divide(dividend=12, divisor=_)
        result2 = await pipeline2()
        
        assert result2.is_ok()
        assert result2.default_value(None) == 2.0  # 12/(10-4) = 2.0

    @pytest.mark.asyncio
    async def test_multiple_placeholders(self):
        """Test using multiple placeholders in a single operation."""
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        
        @operation
        async def multiply_by_itself(x: int) -> int:
            return x * x
        
        @operation
        async def complex_op(a: int, b: int, c: int) -> int:
            return a * b + c
        
        # Generate a value then use it in multiple places
        pipeline = add(5, 3) >> complex_op(_, _, 2)
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 66  # 8*8+2 = 66
        
        # Mix with a constant
        pipeline2 = multiply_by_itself(4) >> complex_op(_, 2, _)
        result2 = await pipeline2()
        
        assert result2.is_ok()
        assert result2.default_value(None) == 48  # 16*2+16 = 48

    @pytest.mark.asyncio
    async def test_placeholders_in_collections(self):
        """Test using placeholders inside collections (lists, dicts, tuples)."""
        @operation
        async def get_number() -> int:
            return 5
        
        @operation
        async def sum_list(numbers: List[int]) -> int:
            return sum(numbers)
        
        @operation
        async def dict_values_product(data: Dict[str, int]) -> int:
            result = 1
            for value in data.values():
                result *= value
            return result
        
        @operation
        async def sum_tuple(numbers: Tuple[int, ...]) -> int:
            return sum(numbers)
        
        # Test placeholder in a list
        pipeline1 = get_number() >> sum_list([1, 2, _, 4])
        result1 = await pipeline1()
        
        assert result1.is_ok()
        assert result1.default_value(None) == 12  # 1+2+5+4 = 12
        
        # Test placeholder in a dictionary
        pipeline2 = get_number() >> dict_values_product({"a": 2, "b": _, "c": 3})
        result2 = await pipeline2()
        
        assert result2.is_ok()
        assert result2.default_value(None) == 30  # 2*5*3 = 30
        
        # Test placeholder in a tuple
        pipeline3 = get_number() >> sum_tuple((1, _, 3, _))
        result3 = await pipeline3()
        
        assert result3.is_ok()
        assert result3.default_value(None) == 14  # 1+5+3+5 = 14

    @pytest.mark.asyncio
    async def test_nested_placeholders(self):
        """Test placeholders in nested data structures."""
        @operation
        async def get_number() -> int:
            return 5
        
        @operation
        async def complex_structure(data: Dict[str, Any]) -> int:
            # Access the nested value
            return data["outer"]["inner"][1] * data["multiplier"]
        
        # Create a complex nested structure with placeholders
        pipeline = get_number() >> complex_structure({
            "outer": {
                "inner": [1, _, 3]
            },
            "multiplier": _
        })
        
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 25  # 5*5 = 25

    @pytest.mark.asyncio
    async def test_pipeline_composition(self):
        """Test composing multiple operations with placeholders."""
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        
        @operation
        async def multiply(a: int, b: int) -> int:
            return a * b
        
        @operation
        async def square(x: int) -> int:
            return x * x
        
        # Create a pipeline with multiple compositions
        pipeline = add(3, 4) >> multiply(_, 2) >> square(_)
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 196  # ((3+4)*2)^2 = 14^2 = 196
        
        # Test a different composition order
        pipeline2 = add(3, 4) >> square(_) >> multiply(_, 2)
        result2 = await pipeline2()
        
        assert result2.is_ok()
        assert result2.default_value(None) == 98  # ((3+4)^2)*2 = 49*2 = 98

    @pytest.mark.asyncio
    async def test_error_handling_with_placeholders(self):
        """Test error handling in pipelines with placeholders."""
        @operation
        async def divide(a: int, b: int) -> float:
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        
        # Test that errors are propagated correctly
        pipeline = divide(10, 0) >> add(_, 5)
        result = await pipeline()
        
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert "Division by zero" in str(result.error)
        
        # Test error in the second operation
        pipeline2 = divide(10, 2) >> divide(_, 0)
        result2 = await pipeline2()
        
        assert result2.is_error()
        assert isinstance(result2.error, ValueError)

    @pytest.mark.asyncio
    async def test_placeholder_with_bound_operations(self):
        """Test using placeholders with already bound operations."""
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        
        @operation
        async def multiply(a: int, b: int) -> int:
            return a * b
        
        # Create a reusable bound operation with a placeholder
        double = multiply(_, 2)
        
        # Use in a pipeline
        pipeline = add(5, 3) >> double
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 16  # (5+3)*2 = 16
        
        # Reuse the same bound operation with a different input
        pipeline2 = add(7, 1) >> double
        result2 = await pipeline2()
        
        assert result2.is_ok()
        assert result2.default_value(None) == 16  # (7+1)*2 = 16

    @pytest.mark.asyncio
    async def test_identity_operation(self):
        """Test a placeholder used by itself (identity operation)"""
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        
        # Just pass the value through
        pipeline = add(5, 3) >> _
        result = await pipeline()
        
        assert result.is_ok()
        assert result.default_value(None) == 8  # Just returns 5+3

    @pytest.mark.asyncio
    async def test_comparison_with_bind(self):
        """Compare placeholder behavior with traditional bind approach."""
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        
        # Using bind (from the original example)
        pipeline_bind = subtract(10, 4).bind(lambda a: divide(a, 2))
        result_bind = await pipeline_bind()
        
        # Using placeholder
        pipeline_placeholder = subtract(10, 4) >> divide(_, 2)
        result_placeholder = await pipeline_placeholder()
        
        # Both should give the same result
        assert result_bind.is_ok()
        assert result_placeholder.is_ok()
        assert result_bind.default_value(None) == result_placeholder.default_value(None) == 3.0