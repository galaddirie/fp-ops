import pytest
import asyncio
import time
from typing import Any, Dict, List, Optional, Type
from unittest.mock import Mock, patch, AsyncMock

from fp_ops.context import BaseContext
from fp_ops.operator import Operation, operation
from expression import Result

# Import flow functions
from fp_ops.flow import (
    branch,
    attempt,
    fail,
    retry,
    tap,
    loop_until,
    wait,
)

# Helper functions for testing
async def async_identity(x):
    return x

def sync_identity(x):
    return x

async def async_error(x):
    raise ValueError("Async error")

def sync_error(x):
    raise ValueError("Sync error")

# Create a simple context for testing
class TestContext(BaseContext):
    value: str = "test"

#########################################
# Tests for branch
#########################################
@pytest.mark.asyncio
async def test_branch_true_condition():
    # Setup
    condition = lambda x: True
    true_op = operation(async_identity)
    false_op = operation(async_error)
    
    # Execute
    branch_op = branch(condition, true_op, false_op)
    result = await branch_op.execute(42)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42

@pytest.mark.asyncio
async def test_branch_false_condition():
    # Setup
    condition = lambda x: False
    true_op = operation(async_error)
    false_op = operation(async_identity)
    
    # Execute
    branch_op = branch(condition, true_op, false_op)
    result = await branch_op.execute(42)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42

@pytest.mark.asyncio
async def test_branch_async_condition():
    # Setup
    async def async_condition(x):
        await asyncio.sleep(0.01)
        return x > 5
    
    true_op = operation(lambda x: x * 2)
    false_op = operation(lambda x: x * 3)
    
    # Execute - true branch
    branch_op = branch(async_condition, true_op, false_op)
    result_true = await branch_op.execute(10)
    
    # Execute - false branch
    result_false = await branch_op.execute(3)
    
    # Assert
    assert result_true.is_ok() and result_true.default_value(None) == 20
    assert result_false.is_ok() and result_false.default_value(None) == 9

@pytest.mark.asyncio
async def test_branch_with_context():
    # Setup
    context = TestContext()
    
    # Define operations that use context
    @operation(context=True, context_type=TestContext)
    def context_condition(x, context=None):
        return context.value == "test" and x > 5
    
    @operation(context=True, context_type=TestContext)
    def true_op(x, context=None):
        return f"{x}_{context.value}_true"
    
    @operation(context=True, context_type=TestContext)
    def false_op(x, context=None):
        return f"{x}_{context.value}_false"
    
    # Execute - true branch
    branch_op = branch(context_condition, true_op, false_op)
    result_true = await branch_op.execute(10, context=context)
    
    # Execute - false branch
    result_false = await branch_op.execute(3, context=context)
    
    # Assert
    assert result_true.is_ok() and result_true.default_value(None) == "10_test_true"
    # The condition x > 5 is false for x=3, so false_op should run
    assert result_false.is_ok()
    assert result_false.default_value(None) == "3_test_false"

@pytest.mark.asyncio
async def test_branch_condition_error():
    # Setup
    def error_condition(x):
        raise ValueError("Condition error")
    
    true_op = operation(async_identity)
    false_op = operation(async_identity)
    
    # Execute
    branch_op = branch(error_condition, true_op, false_op)
    result = await branch_op.execute(42)
    
    # Assert
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Condition error"

#########################################
# Tests for attempt
#########################################
@pytest.mark.asyncio
async def test_attempt_success_sync():
    # Setup
    def success_func(x):
        return x * 2
    
    # Execute
    attempt_op = attempt(success_func)
    result = await attempt_op.execute(21)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42

@pytest.mark.asyncio
async def test_attempt_success_async():
    # Setup
    async def success_func(x):
        await asyncio.sleep(0.01)
        return Result.Ok(x * 2)  # Async funcs need to return Result objects
    
    # Execute
    attempt_op = attempt(success_func)
    result = await attempt_op.execute(21)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42

@pytest.mark.asyncio
async def test_attempt_error_sync():
    # Setup
    def error_func(x):
        raise ValueError(f"Error with {x}")
    
    # Execute
    attempt_op = attempt(error_func)
    result = await attempt_op.execute(42)
    
    # Assert
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Error with 42"

@pytest.mark.asyncio
async def test_attempt_error_async():
    # Setup
    async def error_func(x):
        await asyncio.sleep(0.01)
        return Result.Error(ValueError(f"Async error with {x}"))
    
    # Execute
    attempt_op = attempt(error_func)
    result = await attempt_op.execute(42)
    
    # Assert
    assert result.is_error(), f"Expected error, got {result}, {result.default_value(None)}"
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Async error with 42"

@pytest.mark.asyncio
async def test_attempt_with_context():
    # Setup
    context = TestContext()
    
    # For context functions, we need to ensure they properly return results
    def context_func(x, context=None):
        return Result.Ok(f"{x}_{context.value}")
    
    # Mark it as requiring context
    context_func.requires_context = True
    context_func.context_type = TestContext
    
    # Execute
    attempt_op = attempt(context_func, context=True, context_type=TestContext)
    result = await attempt_op.execute(42, context=context)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == "42_test"

@pytest.mark.asyncio
async def test_attempt_with_invalid_context():
    # Setup
    context = {"value": "not_a_context"}  # Dict instead of TestContext
    
    # For context functions, ensure they return result objects
    def context_func(x, context=None):
        return Result.Ok(f"{x}_{context.value}")
    
    # Mark it as requiring context
    context_func.requires_context = True
    context_func.context_type = TestContext
    
    # Execute - should try to convert the dict to TestContext
    attempt_op = attempt(context_func, context=True, context_type=TestContext)
    result = await attempt_op.execute(42, context=context)
    
    # Assert - should succeed with converted context
    assert result.is_ok()
    assert result.default_value(None) == "42_not_a_context"

#########################################
# Tests for fail
#########################################
@pytest.mark.asyncio
async def test_fail_with_string():
    # Setup & Execute
    fail_op = fail("Test failure")
    result = await fail_op.execute()
    
    # Assert
    assert result.is_error()
    assert isinstance(result.error, Exception)
    assert str(result.error) == "Test failure"

@pytest.mark.asyncio
async def test_fail_with_exception():
    # Setup & Execute
    custom_error = ValueError("Custom error")
    fail_op = fail(custom_error)
    result = await fail_op.execute()
    
    # Assert
    assert result.is_error()
    assert result.error is custom_error
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Custom error"

@pytest.mark.asyncio
async def test_fail_ignores_args():
    # Setup & Execute
    fail_op = fail("Ignores arguments")
    result = await fail_op.execute(1, 2, 3, keyword="value")
    
    # Assert
    assert result.is_error()
    assert str(result.error) == "Ignores arguments"

#########################################
# Tests for retry
#########################################
@pytest.mark.asyncio
async def test_retry_immediate_success():
    # Setup - using a real Operation instead of a Mock
    @operation
    def success_op(x=None):
        return 42  # Will be wrapped in Result.Ok by the operation decorator
    
    # Execute
    retry_op = retry(success_op, max_retries=3, delay=0.01)
    result = await retry_op.execute()
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42

@pytest.mark.asyncio
async def test_retry_succeeds_after_failures():
    # Setup - using a counter to fail twice then succeed
    counter = 0
    
    @operation
    def flaky_op(x=None):
        nonlocal counter
        counter += 1
        if counter < 3:
            return Result.Error(ValueError(f"Error attempt {counter}"))
        return 42
    
    # Execute with fast delay
    retry_op = retry(flaky_op, max_retries=3, delay=0.01)
    result = await retry_op.execute()
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42
    assert counter == 3  # Called 3 times

@pytest.mark.asyncio
async def test_retry_fails_after_max_attempts():
    # Setup - always fail
    counter = 0
    error_msg = "Persistent error"
    
    @operation
    def failing_op(x=None):
        nonlocal counter
        counter += 1
        return Result.Error(ValueError(error_msg))
    
    # Execute with fast delay
    retry_op = retry(failing_op, max_retries=3, delay=0.01)
    result = await retry_op.execute()
    
    # Assert
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == error_msg
    assert counter == 3  # Called 3 times

@pytest.mark.asyncio
async def test_retry_with_context():
    # Setup
    context = TestContext()
    captured_contexts = []
    
    @operation(context=True, context_type=TestContext)
    def context_op(x=None, context=None):
        captured_contexts.append(context)
        if len(captured_contexts) < 2:
            return Result.Error(ValueError(f"Error attempt {len(captured_contexts)}"))
        return 42
    
    # Execute with context
    retry_op = retry(context_op, max_retries=3, delay=0.01)
    result = await retry_op.execute(context=context)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42
    assert len(captured_contexts) == 2
    assert all(ctx is context for ctx in captured_contexts)

#########################################
# Tests for tap
#########################################
@pytest.mark.asyncio
async def test_tap_function():
    # Setup
    tap_value = None
    
    def side_effect(value):
        nonlocal tap_value
        tap_value = value * 2
    
    # Create base operation
    base_op = operation(async_identity)
    
    # Execute tap
    tap_op = tap(base_op, side_effect)
    result = await tap_op.execute(21)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 21  # Original value unchanged
    assert tap_value == 42  # Side effect happened

@pytest.mark.asyncio
async def test_tap_async_function():
    # Setup
    tap_value = None
    
    async def async_side_effect(value):
        nonlocal tap_value
        await asyncio.sleep(0.01)
        tap_value = value * 2
    
    # Create base operation
    base_op = operation(async_identity)
    
    # Execute tap
    tap_op = tap(base_op, async_side_effect)
    result = await tap_op.execute(21)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 21  # Original value unchanged
    assert tap_value == 42  # Side effect happened

@pytest.mark.asyncio
async def test_tap_function_error_is_ignored():
    # Setup
    def error_side_effect(value):
        raise ValueError("Side effect error")
    
    # Create base operation
    base_op = operation(async_identity)
    
    # Execute tap
    tap_op = tap(base_op, error_side_effect)
    result = await tap_op.execute(42)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42  # Operation still succeeds

@pytest.mark.asyncio
async def test_tap_with_context():
    # Setup
    context = TestContext()
    tap_result = None
    
    def context_side_effect(value, context=None):
        nonlocal tap_result
        tap_result = f"{value}_{context.value}"
        return None  # Side effect shouldn't return a value
    
    # Create base operation that returns a Result
    @operation
    def base_func(x, context=None):
        return x
    
    # Execute tap with context marked as required for the side effect
    tap_op = tap(base_func, context_side_effect, context=True, context_type=TestContext)
    
    # Use await to actually execute the operation
    result = await tap_op.execute(42, context=context)
    
    # Assert - wait briefly for any async side effects to complete
    await asyncio.sleep(0.05)
    
    assert result.is_ok()
    assert result.default_value(None) == 42  # Original value unchanged
    assert tap_result == "42_test"  # Side effect with context happened

#########################################
# Tests for loop_until
#########################################
@pytest.mark.asyncio
async def test_loop_until_condition_met():
    # Setup
    counter = 0
    
    def increment(x):
        nonlocal counter
        counter += 1
        return x + 1
    
    def condition(x):
        return x >= 5  # Stop when we reach 5
    
    body_op = operation(increment)
    
    # Execute - start at 1, should loop until 5
    loop_op = loop_until(condition, body_op, max_iterations=10, delay=0.01)
    result = await loop_op.execute(1)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 5
    assert counter == 4  # Incremented 4 times: 1→2→3→4→5

@pytest.mark.asyncio
async def test_loop_until_async_condition():
    # Setup
    counter = 0
    
    def increment(x):
        nonlocal counter
        counter += 1
        return x + 1
    
    async def async_condition(x):
        await asyncio.sleep(0.01)
        return x >= 3
    
    body_op = operation(increment)
    
    # Execute
    loop_op = loop_until(async_condition, body_op, max_iterations=10, delay=0.01)
    result = await loop_op.execute(1)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 3
    assert counter == 2  # Incremented twice: 1→2→3

@pytest.mark.asyncio
async def test_loop_until_max_iterations():
    # Setup
    counter = 0
    
    def increment(x):
        nonlocal counter
        counter += 1
        return x + 1
    
    def never_condition(x):
        return False  # Never stop
    
    body_op = operation(increment)
    
    # Execute with limited iterations
    loop_op = loop_until(never_condition, body_op, max_iterations=5, delay=0.01)
    result = await loop_op.execute(1)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 6  # Started at 1, incremented 5 times
    assert counter == 5  # Incremented 5 times

@pytest.mark.asyncio
async def test_loop_until_body_error():
    # Setup
    counter = 0
    
    @operation
    def increment_then_error(x):
        nonlocal counter
        counter += 1
        if x >= 3:
            return Result.Error(ValueError(f"Error at x={x}"))
        return x + 1
    
    def never_condition(x):
        return False  # Never stop
    
    # Execute
    loop_op = loop_until(never_condition, increment_then_error, max_iterations=10, delay=0.01)
    result = await loop_op.execute(1)
    
    # Assert
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Error at x=3"
    # Counter is 3 because: 1→2→3 (three calls, error on third call)
    assert counter == 3

@pytest.mark.asyncio
async def test_loop_until_with_context():
    # Setup
    context = TestContext()
    values = []
    
    @operation
    def context_operation(x, context=None):
        values.append(f"{x}_{context.value}")
        print(f"context_operation: {x}_{context.value}")
        return int(x) + 1
    
    def context_condition(x, context=None):
        # Return True when condition is met (to exit the loop)
        return x >= 3 and context.value == "test"
    
    # Mark the condition as requiring context
    context_condition.requires_context = True
    context_condition.context_type = TestContext
    
    # Execute
    loop_op = loop_until(context_condition, context_operation, context=True, context_type=TestContext)
    result = await loop_op.execute(1, context=context) 
    
    # Assert
    assert result.is_ok()
    # The final result should be the last value passed to the body operation
    assert result.default_value(None) == 3
    assert values == ["1_test", "2_test"]

#########################################
# Tests for wait
#########################################
@pytest.mark.asyncio
async def test_wait_immediate_success():
    # Setup
    result_value = Result.Ok(42)
    mock_func = AsyncMock(return_value=result_value)
    mock_op = Mock(spec=Operation)
    mock_op.execute = mock_func
    mock_op.context_type = None
    
    # Execute
    wait_op = wait(mock_op, timeout=1.0, delay=0.01)
    result = await wait_op.execute()
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42
    mock_func.assert_called_once()

@pytest.mark.asyncio
async def test_wait_success_after_failures():
    # Setup - fail twice then succeed
    side_effects = [
        Result.Error(ValueError("Error 1")),
        Result.Error(ValueError("Error 2")),
        Result.Ok(42)
    ]
    mock_func = AsyncMock(side_effect=side_effects)
    mock_op = Mock(spec=Operation)
    mock_op.execute = mock_func
    mock_op.context_type = None
    
    # Execute with fast delay
    wait_op = wait(mock_op, timeout=1.0, delay=0.01)
    result = await wait_op.execute()
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_wait_timeout():
    # Setup - always fail
    error = ValueError("Persistent error")
    
    # Simulate slow operation that times out
    async def slow_execute(*args, **kwargs):
        await asyncio.sleep(0.1)  # Slow operation
        return Result.Error(error)
    
    mock_op = Mock(spec=Operation)
    mock_op.execute = slow_execute
    mock_op.context_type = None
    
    # Execute with a short timeout
    wait_op = wait(mock_op, timeout=0.05, delay=0.01)
    result = await wait_op.execute()
    
    # Assert
    assert result.is_error()
    # Either we get the last error or a timeout error
    assert isinstance(result.error, (ValueError, TimeoutError))

@pytest.mark.asyncio
async def test_wait_with_context():
    # Setup
    context = TestContext()
    
    # Track contexts seen
    contexts_seen = []
    
    async def context_execute(*args, **kwargs):
        contexts_seen.append(kwargs.get('context'))
        if len(contexts_seen) < 2:
            return Result.Error(ValueError("Not ready yet"))
        return Result.Ok(42)
    
    mock_op = Mock(spec=Operation)
    mock_op.execute = context_execute
    mock_op.context_type = TestContext
    
    # Execute with context
    wait_op = wait(mock_op, timeout=1.0, delay=0.01)
    result = await wait_op.execute(context=context)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == 42
    assert len(contexts_seen) == 2
    assert all(ctx is context for ctx in contexts_seen)