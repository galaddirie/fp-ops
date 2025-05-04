import asyncio
import pytest
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from expression import Result

from fp_ops.operator import (
    constant,
    identity,
    Operation
)
from fp_ops.flow import branch, attempt, fail
from fp_ops.decorators import operation
from fp_ops.placeholder import _

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def fetch_data():
    @operation
    async def _fetch_data(url: str) -> Dict[str, Any]:
        await asyncio.sleep(0.1)
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
        users = data.get("users", [])
        return [user["name"] for user in users]
    return _extract_user_names

@pytest.fixture
def format_names():
    @operation
    async def _format_names(names: List[str]) -> str:
        return ", ".join(names)
    return _format_names

@pytest.fixture
def risky_operation():
    @operation
    async def _risky_operation(n: int) -> int:
        await asyncio.sleep(0.1)
        if n < 0:
            raise ValueError("Input must be positive")
        return 100 // n
    return _risky_operation

@pytest.fixture
def fetch_users():
    @operation
    async def _fetch_users() -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
    return _fetch_users

@pytest.fixture
def fetch_posts():
    @operation
    async def _fetch_posts() -> Dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"posts": [{"id": 101, "title": "Hello World"}, {"id": 102, "title": "Python Tips"}]}
    return _fetch_posts

@pytest.fixture
def flaky_service():
    call_count = 0
    @operation
    async def _flaky_service() -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Service temporarily unavailable")
        return {"status": "success", "data": "Service data", "attempts": call_count}
    return _flaky_service

class TestBasicComposition:
    @pytest.mark.asyncio
    async def test_basic_composition(self, fetch_data, extract_user_names, format_names):
        pipeline = fetch_data >> extract_user_names >> format_names
        result = await pipeline("https://api.example.com/users")
        assert result.is_ok()
        assert result.default_value(None) == "Alice, Bob"

    @pytest.mark.asyncio
    async def test_composition_with_constant(self, fetch_data):
        pipeline = fetch_data >> constant("Fixed output")
        result = await pipeline("https://api.example.com/any")
        assert result.is_ok()
        assert result.default_value(None) == "Fixed output"

    @pytest.mark.asyncio
    async def test_composition_with_lambda(self, fetch_data):
        pipeline = fetch_data >> (lambda data: f"Processed: {data.get('status', 'unknown')}")
        result = await pipeline("https://api.example.com/status")
        assert result.is_ok()
        assert result.default_value(None) == "Processed: ok"

    @pytest.mark.asyncio
    async def test_step_by_step_execution(self, fetch_data, extract_user_names, format_names):
        pipeline = fetch_data >> extract_user_names >> format_names
        pipeline_result = await pipeline("https://api.example.com/users")
        fetch_result = await fetch_data("https://api.example.com/users")
        assert fetch_result.is_ok()
        names_result = await extract_user_names(fetch_result.default_value(None))
        assert names_result.is_ok()
        formatted_result = await format_names(names_result.default_value(None))
        assert formatted_result.is_ok()
        assert pipeline_result.default_value(None) == formatted_result.default_value(None)
        assert pipeline_result.default_value(None) == "Alice, Bob"

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_basic_error_propagation(self, risky_operation):
        valid_result = await risky_operation(10)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 10
        error_result = await risky_operation(0)
        assert error_result.is_error()
        assert isinstance(error_result.error, ZeroDivisionError)
        pipeline = risky_operation >> (lambda x: x * 2)
        pipeline_result = await pipeline(0)
        assert pipeline_result.is_error()

    @pytest.mark.asyncio
    async def test_catch_error_handler(self, risky_operation):
        error_handler = lambda e: f"Error handled: {type(e).__name__}"
        with_recovery = risky_operation.catch(error_handler)
        valid_result = await with_recovery(10)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 10
        error_result = await with_recovery(0)
        assert error_result.is_ok()
        assert error_result.default_value(None) == "Error handled: ZeroDivisionError"

    @pytest.mark.asyncio
    async def test_default_value(self, risky_operation):
        with_default = risky_operation.default_value(999)
        valid_result = await with_default(10)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 10
        error_result = await with_default(0)
        assert error_result.is_ok()
        assert error_result.default_value(None) == 999

    @pytest.mark.asyncio
    async def test_failure_in_pipeline(self, fetch_data, risky_operation):
        pipeline = fetch_data >> risky_operation
        result = await pipeline("https://api.example.com/users")
        assert result.is_error()

    @pytest.mark.asyncio
    async def test_fail_operation(self):
        always_fail = fail(ValueError("Deliberate failure"))
        result = await always_fail()
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "Deliberate failure"

class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_parallel_execution(self, fetch_users, fetch_posts):
        parallel_op = fetch_users & fetch_posts
        parallel_result = await parallel_op()
        assert parallel_result.is_ok()
        users_data, posts_data = parallel_result.default_value((None, None))
        assert "users" in users_data
        assert "posts" in posts_data

    @pytest.mark.asyncio
    async def test_parallel_with_composition(self, fetch_users, fetch_posts):
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
        parallel_op = fetch_users & fetch_posts
        pipeline = parallel_op >> combine_data
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
        failing_op = operation(lambda: asyncio.sleep(0.1)) >> fail(RuntimeError("Simulated failure"))
        parallel_op = fetch_users & failing_op
        result = await parallel_op()
        assert result.is_error()
        assert isinstance(result.error, RuntimeError)

class TestFallbackPatterns:
    @pytest.mark.asyncio
    async def test_basic_fallback(self):
        primary = fail(ConnectionError("Primary failed"))
        backup = constant({"source": "backup", "data": "Backup data"})
        fallback = primary | backup
        result = await fallback()
        assert result.is_ok()
        data = result.default_value(None)
        assert data["source"] == "backup"

    @pytest.mark.asyncio
    async def test_multiple_fallbacks(self):
        op1 = fail(ValueError("First error"))
        op2 = fail(ValueError("Second error"))
        op3 = constant("Success")
        fallback = op1 | op2 | op3
        result = await fallback()
        assert result.is_ok()
        assert result.default_value(None) == "Success"

    @pytest.mark.asyncio
    async def test_all_fallbacks_fail(self):
        op1 = fail(ValueError("First error"))
        op2 = fail(ValueError("Second error"))
        op3 = fail(ValueError("Third error"))
        fallback = op1 | op2 | op3
        result = await fallback()
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert str(result.error) == "Third error"

    @pytest.mark.asyncio
    async def test_fallback_with_condition(self, fetch_data):
        complex_fallback = (
            fetch_data.filter(lambda d: "users" in d and len(d["users"]) > 0, "No users found") |
            constant({"users": [{"id": 0, "name": "Default User"}]})
        )
        result_with_users = await complex_fallback("https://api.example.com/users")
        assert result_with_users.is_ok()
        data_with_users = result_with_users.default_value(None)
        assert len(data_with_users["users"]) == 2
        result_without_users = await complex_fallback("https://api.example.com/posts")
        assert result_without_users.is_ok()
        data_without_users = result_without_users.default_value(None)
        assert len(data_without_users["users"]) == 1
        assert data_without_users["users"][0]["name"] == "Default User"

class TestTransformations:
    @pytest.mark.asyncio
    async def test_map_transformation(self, fetch_data):
        get_first_user = fetch_data.map(
            lambda data: data.get("users", [{}])[0].get("name", "Unknown")
        )
        result = await get_first_user("https://api.example.com/users")
        assert result.is_ok()
        name = result.default_value(None)
        assert name == "Alice"

    @pytest.mark.asyncio
    async def test_bind_transformation(self, fetch_data):
        @operation
        async def get_user_details(user_name: str) -> Dict[str, Any]:
            if user_name == "Alice":
                return {"name": user_name, "email": "alice@example.com", "role": "admin"}
            else:
                return {"name": user_name, "email": f"{user_name.lower()}@example.com", "role": "user"}
        get_first_user_details = fetch_data.bind(
            lambda data: get_user_details(data.get("users", [{}])[0].get("name", "Unknown"))
        )
        result = await get_first_user_details("https://api.example.com/users")
        assert result.is_ok()
        user_details = result.default_value(None)
        assert user_details["name"] == "Alice"
        assert user_details["email"] == "alice@example.com"
        assert user_details["role"] == "admin"

    @pytest.mark.asyncio
    async def test_filter_operation(self, fetch_data):
        valid_data = fetch_data.filter(
            lambda data: "users" in data and len(data["users"]) > 0,
            "No users found"
        )
        valid_result = await valid_data("https://api.example.com/users")
        assert valid_result.is_ok()
        invalid_result = await valid_data("https://api.example.com/posts")
        assert invalid_result.is_error()
        assert str(invalid_result.error) == "No users found"

class TestUtilities:
    @pytest.mark.asyncio
    async def test_tap_operation(self, fetch_data):
        side_effect_results = []
        tapped = fetch_data.tap(
            lambda data: side_effect_results.append(data.get("message", "unknown"))
        )
        result = await tapped("https://api.example.com/status")
        assert result.is_ok()
        assert "status" in result.default_value(None)
        assert len(side_effect_results) == 1
        assert side_effect_results[0] == "Response from https://api.example.com/status"

    @pytest.mark.asyncio
    async def test_retry_operation(self, flaky_service):
        reliable = flaky_service.retry(attempts=3, delay=0.05)
        result = await reliable()
        assert result.is_ok()
        data = result.default_value(None)
        assert data["status"] == "success"
        assert data["attempts"] == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        always_fails = operation(lambda: asyncio.sleep(0.05)) >> fail(RuntimeError("Always fails"))
        retry_but_fail = always_fails.retry(attempts=2, delay=0.01)
        result = await retry_but_fail()
        assert result.is_error()
        assert isinstance(result.error, RuntimeError)

class TestSequenceAndCombine:
    @pytest.mark.asyncio
    async def test_sequence_operations(self):
        @operation
        async def step1(data):
            return f"Step 1: {data}"
        @operation
        async def step2(data):
            return f"Step 2: {data}"
        @operation
        async def step3(data):
            return f"Step 3: {data}"
        sequence_op = Operation.sequence([step1, step2, step3])
        result = await sequence_op("input").execute()
        assert result.is_ok()
        steps = result.default_value(None)
        expected = "Step 3: Step 2: Step 1: input"
        assert steps == expected, f"got {steps} expected {expected}"

    @pytest.mark.asyncio
    async def test_combine_operations(self):
        @operation
        async def get_name(data):
            return f"Name: {data}"
        @operation
        async def get_id(data):
            return f"ID: {data}"
        @operation
        async def get_email(data):
            return f"Email: {data}@example.com"
        combined_op = Operation.combine(
            name=get_name,
            id=get_id,
            email=get_email
        )
        result = await combined_op("user123")
        assert result.is_ok()
        combined = result.default_value(None)
        assert "name" in combined
        assert "id" in combined
        assert "email" in combined
        assert combined["name"] == "Name: user123"
        assert combined["id"] == "ID: user123"
        assert combined["email"] == "Email: user123@example.com"

class TestSyncFunctions:
    @pytest.mark.asyncio
    async def test_sync_function_wrapping(self):
        def calculate_square(n: int) -> int:
            return n * n
        square_op = operation(calculate_square)
        result = await square_op(5)
        assert result.is_ok()
        assert result.default_value(None) == 25

    @pytest.mark.asyncio
    async def test_sync_function_in_pipeline(self, fetch_data):
        def extract_count(data: Dict[str, Any]) -> int:
            return len(data.get("users", []))
        pipeline = fetch_data >> operation(extract_count)
        result = await pipeline("https://api.example.com/users")
        assert result.is_ok()
        assert result.default_value(None) == 2

    @pytest.mark.asyncio
    async def test_sync_function_with_error(self):
        def divide(a: int, b: int) -> float:
            return a / b
        divide_op = operation(divide)
        valid_result = await divide_op(10, 2)
        assert valid_result.is_ok()
        assert valid_result.default_value(None) == 5.0
        error_result = await divide_op(10, 0)
        assert error_result.is_error()
        assert isinstance(error_result.error, ZeroDivisionError)

class TestOverloading:
    @pytest.mark.asyncio
    async def test_overloading_forwarded_args(self):
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        @operation
        def add_one(a: int) -> int:
            return a + 1
        pipeline = add(1, 2) >> add_one
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 4
        pipeline = add(1,2) >> add_one
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 2

    @pytest.mark.asyncio
    async def test_operation_with_missing_arguments(self):
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
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        pipeline = subtract(10, 4).bind(lambda a: divide(a, 2))
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 3.0

    @pytest.mark.asyncio
    async def test_operation_with_complete_argument_override(self):
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        pipeline = subtract(20, 5) >> divide(30, 5)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 6.0

class TestPlaceholders:
    @pytest.mark.asyncio
    async def test_basic_placeholder_usage(self):
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        pipeline = subtract(10, 4) >> divide(_, 2)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 3.0

    @pytest.mark.asyncio
    async def test_placeholder_in_second_position(self):
        @operation
        def add(a: int, b: int) -> int:
            return a + b
        @operation
        def multiply(a: int, b: int) -> int:
            return a * b
        pipeline = add(5, 3) >> multiply(2, _)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 16

    @pytest.mark.asyncio
    async def test_placeholder_in_keyword_args(self):
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        @operation
        async def divide(dividend: int, divisor: int) -> float:
            return dividend / divisor
        pipeline = subtract(10, 4) >> divide(dividend=_, divisor=2)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 3.0
        pipeline2 = subtract(10, 4) >> divide(dividend=12, divisor=_)
        result2 = await pipeline2()
        assert result2.is_ok()
        assert result2.default_value(None) == 2.0

    @pytest.mark.asyncio
    async def test_multiple_placeholders(self):
        @operation
        def add(a: int, b: int) -> int:
            return a + b
        @operation
        def multiply_by_itself(x: int) -> int:
            return x * x
        @operation
        def complex_op(a: int, b: int, c: int) -> int:
            return a * b + c
        pipeline = add(5, 3) >> complex_op(_, _, 2)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 66
        pipeline2 = multiply_by_itself(4) >> complex_op(_, 2, _)
        result2 = await pipeline2()
        assert result2.is_ok()
        assert result2.default_value(None) == 48

    @pytest.mark.asyncio
    async def test_placeholders_in_collections(self):
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
        pipeline1 = get_number() >> sum_list([1, 2, _, 4])
        result1 = await pipeline1()
        assert result1.is_ok()
        assert result1.default_value(None) == 12
        pipeline2 = get_number() >> dict_values_product({"a": 2, "b": _, "c": 3})
        result2 = await pipeline2()
        assert result2.is_ok()
        assert result2.default_value(None) == 30
        pipeline3 = get_number() >> sum_tuple((1, _, 3, _))
        result3 = await pipeline3()
        assert result3.is_ok()
        assert result3.default_value(None) == 14

    @pytest.mark.asyncio
    async def test_nested_placeholders(self):
        @operation
        async def get_number() -> int:
            return 5
        @operation
        async def complex_structure(data: Dict[str, Any]) -> int:
            return data["outer"]["inner"][1] * data["multiplier"]
        pipeline = get_number() >> complex_structure({
            "outer": {
                "inner": [1, _, 3]
            },
            "multiplier": _
        })
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 25

    @pytest.mark.asyncio
    async def test_pipeline_composition(self):
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        @operation
        async def multiply(a: int, b: int) -> int:
            return a * b
        @operation
        async def square(x: int) -> int:
            return x * x
        pipeline = add(3, 4) >> multiply(_, 2) >> square(_)
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 196
        pipeline2 = add(3, 4) >> square(_) >> multiply(_, 2)
        result2 = await pipeline2()
        assert result2.is_ok()
        assert result2.default_value(None) == 98

    @pytest.mark.asyncio
    async def test_error_handling_with_placeholders(self):
        @operation
        async def divide(a: int, b: int) -> float:
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        pipeline = divide(10, 0) >> add(_, 5)
        result = await pipeline()
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert "Division by zero" in str(result.error)
        pipeline2 = divide(10, 2) >> divide(_, 0)
        result2 = await pipeline2()
        assert result2.is_error()
        assert isinstance(result2.error, ValueError)

    @pytest.mark.asyncio
    async def test_placeholder_with_bound_operations(self):
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        @operation
        async def multiply(a: int, b: int) -> int:
            return a * b
        double = multiply(_, 2)
        pipeline = add(5, 3) >> double
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 16
        pipeline2 = add(7, 1) >> double
        result2 = await pipeline2()
        assert result2.is_ok()
        assert result2.default_value(None) == 16

    @pytest.mark.asyncio
    async def test_identity_operation(self):
        @operation
        async def add(a: int, b: int) -> int:
            return a + b
        pipeline = add(5, 3) >> _
        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 8

    @pytest.mark.asyncio
    async def test_comparison_with_bind(self):
        @operation
        async def subtract(a: int, b: int) -> int:
            return a - b
        @operation
        async def divide(a: int, b: int) -> float:
            return a / b
        pipeline_bind = subtract(10, 4).bind(lambda a: divide(a, 2))
        result_bind = await pipeline_bind()
        pipeline_placeholder = subtract(10, 4) >> divide(_, 2)
        result_placeholder = await pipeline_placeholder()
        assert result_bind.is_ok()
        assert result_placeholder.is_ok()
        assert result_bind.default_value(None) == result_placeholder.default_value(None) == 3.0