import pytest
import asyncio
from typing import Any, Dict, List, Optional, Type, Tuple
from unittest.mock import Mock, patch, AsyncMock

from fp_ops.context import BaseContext
from fp_ops.operator import Operation, operation, identity
from expression import Result

# Import composition functions
from fp_ops.composition import (
    sequence,
    pipe,
    compose,
    parallel,
    fallback,
    map,
    filter,
    reduce,
    zip,
    flat_map,
    group_by,
    partition,
    first,
    last,
    gather_operations,
)

# Helper functions for testing
async def async_identity(x):
    return x

def sync_identity(x):
    return x

async def async_double(x):
    return x * 2

def sync_double(x):
    return x * 2

async def async_error(x):
    raise ValueError("Async error")

def sync_error(x):
    raise ValueError("Sync error")

# Create a simple context for testing
class TestContext(BaseContext):
    value: str = "test"

#########################################
# Tests for sequence
#########################################
@pytest.mark.asyncio
async def test_sequence_basic():
    # Setup
    op1 = operation(lambda x: x + 1) # 5 + 1 = 6
    op2 = operation(lambda x: x * 2) # 5 * 2 = 10
    op3 = operation(lambda x: x - 3) # 10 - 3 = 7 OR 5 - 3 = 2
    
    # Execute
    seq_op = sequence(op1, op2, op3)
    result = await seq_op.execute(5)
    
    # Assert - should collect all results
    assert result.is_ok()
    assert result.default_value(None) == [6, 10, 2]

@pytest.mark.asyncio
async def test_sequence_empty():
    # Setup & Execute
    seq_op = sequence()
    result = await seq_op.execute(42)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == []

@pytest.mark.asyncio
async def test_sequence_with_error():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)  # Will raise ZeroDivisionError
    op3 = operation(lambda x: x - 3)
    
    # Execute
    seq_op = sequence(op1, op2, op3)
    result = await seq_op.execute(5)
    
    # Assert - should stop at first error
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_sequence_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    
    # Execute
    seq_op = sequence(op1, op2, op3)
    result = await seq_op.execute(5, context=context)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) == ["5_test_1", "5_test_2", "5_test_3"]

@pytest.mark.asyncio
async def test_sequence_with_context_update():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    
    @operation(context=True, context_type=TestContext)
    def update_context(x, context=None):
        # Return a new context
        new_context = TestContext(value="updated")
        return new_context
    
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    
    # Execute
    seq_op = sequence(op1, update_context, op3)
    result = await seq_op.execute(5, context=context)
    
    # Assert - only op1 and op3 results should be in the list
    # The context update operation doesn't add a value
    assert result.is_ok()
    assert result.default_value(None) == ["5_test_1", "5_updated_3"]

#########################################
# Tests for pipe
#########################################
@pytest.mark.asyncio
async def test_pipe_basic():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    
    # Execute
    pipe_op = pipe(op1, op2, op3)
    result = await pipe_op.execute(5)
    
    # Assert - should apply operations sequentially
    # 5 + 1 = 6, 6 * 2 = 12, 12 - 3 = 9
    assert result.is_ok()
    assert result.default_value(None) == 9

@pytest.mark.asyncio
async def test_pipe_with_lambdas():
    # Setup - use lambdas to dynamically create operations
    op1 = operation(lambda x: x + 1)
    
    # Lambda that returns an operation based on input value
    def step2(x):
        if x > 5:
            return operation(lambda y: y * 2)
        else:
            return operation(lambda y: y + 5)
    
    op3 = operation(lambda x: x - 3)
    
    # Execute - first case where x > 5
    pipe_op = pipe(op1, step2, op3)
    result1 = await pipe_op.execute(5)
    
    # Execute - second case where x <= 5
    result2 = await pipe_op.execute(4)
    
    # Assert - should follow different paths
    # First case: 5 + 1 = 6, 6 * 2 = 12, 12 - 3 = 9
    # Second case: 4 + 1 = 5, 5 + 5 = 10, 10 - 3 = 7
    assert result1.is_ok() and result1.default_value(None) == 9
    assert result2.is_ok() and result2.default_value(None) == 7

@pytest.mark.asyncio
async def test_pipe_empty():
    # Setup & Execute
    pipe_op = pipe()
    result = await pipe_op.execute(42)
    
    # Assert
    assert result.is_ok()
    assert result.default_value(None) is None

@pytest.mark.asyncio
async def test_pipe_with_error():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)  # Will raise ZeroDivisionError
    op3 = operation(lambda x: x - 3)
    
    # Execute
    pipe_op = pipe(op1, op2, op3)
    result = await pipe_op.execute(5)
    
    # Assert - should stop at first error
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_pipe_with_lambda_error():
    # Setup
    op1 = operation(lambda x: x + 1)
    
    # Lambda that raises an error
    def step2(x):
        raise ValueError("Lambda error")
    
    op3 = operation(lambda x: x - 3)
    
    # Execute
    pipe_op = pipe(op1, step2, op3)
    result = await pipe_op.execute(5)
    
    # Assert - should stop at lambda error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Lambda error"

@pytest.mark.asyncio
async def test_pipe_with_invalid_lambda_return():
    # Setup
    op1 = operation(lambda x: x + 1)
    
    # Lambda that returns a non-Operation
    def step2(x):
        return "not an operation"
    
    op3 = operation(lambda x: x - 3)
    
    # Execute
    pipe_op = pipe(op1, step2, op3)
    result = await pipe_op.execute(5)
    
    # Assert - should error with type error
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_pipe_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    
    # Execute
    pipe_op = pipe(op1, op2, op3)
    result = await pipe_op.execute(5, context=context)
    
    # Assert - should apply operations sequentially with context
    assert result.is_ok()
    assert result.default_value(None) == "5_test_1_test_2_test_3"

@pytest.mark.asyncio
async def test_pipe_with_context_update():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}"
    
    # Update context in the middle
    @operation(context=True, context_type=TestContext)
    def update_context(x, context=None):
        new_context = TestContext(value="updated")
        return new_context
    
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}"
    
    # Execute
    pipe_op = pipe(op1, update_context, op3)
    result = await pipe_op.execute(5, context=context)
    
    # Assert - should use updated context in op3
    assert result.is_ok()
    # Note: The value passed to op3 is the context object itself
    # This behavior might be reviewed depending on intended usage
    assert isinstance(result.default_value(None), TestContext)
    assert result.default_value(None).value == "updated"

#########################################
# Tests for compose
#########################################
@pytest.mark.asyncio
async def test_compose_basic():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    
    # Execute
    # compose is right-to-left: op1 >> op2 >> op3
    compose_op = compose(op1, op2, op3)
    result = await compose_op.execute(5)
    
    # Assert - should apply operations sequentially
    # 5 + 1 = 6, 6 * 2 = 12, 12 - 3 = 9
    assert result.is_ok()
    assert result.default_value(None) == 9

@pytest.mark.asyncio
async def test_compose_empty():
    # Setup & Execute
    compose_op = compose()
    result = await compose_op.execute(42)
    
    # Assert - should return identity operation
    assert result.is_ok()
    assert result.default_value(None) == 42

@pytest.mark.asyncio
async def test_compose_single_operation():
    # Setup
    op = operation(lambda x: x * 2)
    
    # Execute
    compose_op = compose(op)
    result = await compose_op.execute(5)
    
    # Assert - should just apply the one operation
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_compose_with_error():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)  # Will raise ZeroDivisionError
    op3 = operation(lambda x: x - 3)
    
    # Execute
    compose_op = compose(op1, op2, op3)
    result = await compose_op.execute(5)
    
    # Assert - should stop at first error
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_compose_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    
    # Execute
    compose_op = compose(op1, op2, op3)
    result = await compose_op.execute(5, context=context)
    
    # Assert - should apply operations sequentially with context
    assert result.is_ok()
    assert result.default_value(None) == "5_test_1_test_2_test_3"

#########################################
# Tests for parallel
#########################################
@pytest.mark.asyncio
async def test_parallel_basic():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    
    # Execute
    parallel_op = parallel(op1, op2, op3)
    result = await parallel_op.execute(5)
    
    # Assert - should run all operations and return results as a tuple
    assert result.is_ok()
    assert result.default_value(None) == (6, 10, 2)

@pytest.mark.asyncio
async def test_parallel_empty():
    # Setup & Execute
    parallel_op = parallel()
    result = await parallel_op.execute(42)
    
    # Assert - should return empty tuple
    assert result.is_ok()
    assert result.default_value(None) == ()

@pytest.mark.asyncio
async def test_parallel_with_error():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)  # Will raise ZeroDivisionError
    op3 = operation(lambda x: x - 3)
    
    # Execute
    parallel_op = parallel(op1, op2, op3)
    result = await parallel_op.execute(5)
    
    # Assert - should return first error encountered
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_parallel_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    
    # Execute
    parallel_op = parallel(op1, op2, op3)
    result = await parallel_op.execute(5, context=context)
    
    # Assert - should run all operations with the same context
    assert result.is_ok()
    assert result.default_value(None) == ("5_test_1", "5_test_2", "5_test_3")

#########################################
# Tests for fallback
#########################################
@pytest.mark.asyncio
async def test_fallback_first_succeeds():
    # Setup
    op1 = operation(lambda x: x * 2)
    op2 = operation(sync_error)
    op3 = operation(sync_error)
    
    # Execute
    fallback_op = fallback(op1, op2, op3)
    result = await fallback_op.execute(5)
    
    # Assert - should use first operation's result
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_fallback_second_succeeds():
    # Setup
    op1 = operation(sync_error)
    op2 = operation(lambda x: x * 2)
    op3 = operation(sync_error)
    
    # Execute
    fallback_op = fallback(op1, op2, op3)
    result = await fallback_op.execute(5)
    
    # Assert - should use second operation's result
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_fallback_all_fail():
    # Setup
    op1 = operation(sync_error)
    op2 = operation(lambda x: x / 0)
    op3 = operation(async_error)
    
    # Execute
    fallback_op = fallback(op1, op2, op3)
    result = await fallback_op.execute(5)
    
    # Assert - should return last error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Async error"

@pytest.mark.asyncio
async def test_fallback_empty():
    # Setup & Execute
    fallback_op = fallback()
    result = await fallback_op.execute(42)
    
    # Assert - should return error for empty operations
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert "No operations provided" in str(result.error)

@pytest.mark.asyncio
async def test_fallback_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        if x < 10:
            raise ValueError("Value too small")
        return f"{x}_{context.value}"
    
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"fallback_{x}_{context.value}"
    
    # Execute - first operation fails
    fallback_op = fallback(op1, op2)
    result_fail = await fallback_op.execute(5, context=context)
    
    # Execute - first operation succeeds
    result_success = await fallback_op.execute(15, context=context)
    
    # Assert
    assert result_fail.is_ok()
    assert result_fail.default_value(None) == "fallback_5_test"
    
    assert result_success.is_ok()
    assert result_success.default_value(None) == "15_test"


#########################################
# Tests for map
#########################################
@pytest.mark.asyncio
async def test_map_basic():
    # Setup
    base_op = operation(lambda x: x + 1)
    mapper = lambda x: x * 2
    
    # Execute
    map_op = map(base_op, mapper)
    result = await map_op.execute(5)
    
    # Assert - should apply base operation then map the result
    # 5 + 1 = 6, 6 * 2 = 12
    assert result.is_ok()
    assert result.default_value(None) == 12

@pytest.mark.asyncio
async def test_map_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    mapper = lambda x: x * 2
    
    # Execute
    map_op = map(base_op, mapper)
    result = await map_op.execute(5)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_map_with_error_in_mapper():
    # Setup
    base_op = operation(lambda x: x + 1)
    
    def error_mapper(x):
        raise ValueError("Mapper error")
    
    # Execute
    map_op = map(base_op, error_mapper)
    result = await map_op.execute(5)
    
    # Assert - should fail with mapper's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Mapper error"

@pytest.mark.asyncio
async def test_map_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return f"{x}_{context.value}"
    
    def mapper(x):
        return f"mapped_{x}"
    
    # Execute
    map_op = map(base_op, mapper)
    result = await map_op.execute(5, context=context)
    
    # Assert - should apply mapper to base operation result
    assert result.is_ok()
    assert result.default_value(None) == "mapped_5_test"

#########################################
# Tests for filter
#########################################
@pytest.mark.asyncio
async def test_filter_pass():
    # Setup
    base_op = operation(lambda x: x + 5)
    predicate = lambda x: x > 7
    
    # Execute - should pass filter
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(3)
    
    # Assert - 3 + 5 = 8, 8 > 7 is True
    assert result.is_ok()
    assert result.default_value(None) == 8

@pytest.mark.asyncio
async def test_filter_fail():
    # Setup
    base_op = operation(lambda x: x + 5)
    predicate = lambda x: x > 10
    
    # Execute - should fail filter
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(3)
    
    # Assert - 3 + 5 = 8, 8 > 10 is False
    assert result.is_error()
    assert isinstance(result.error, ValueError)

@pytest.mark.asyncio
async def test_filter_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    predicate = lambda x: x > 7
    
    # Execute
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(3)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_filter_with_error_in_predicate():
    # Setup
    base_op = operation(lambda x: x + 5)
    
    def error_predicate(x):
        raise ValueError("Predicate error")
    
    # Execute
    filter_op = filter(base_op, error_predicate)
    result = await filter_op.execute(3)
    
    # Assert - should fail with predicate's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Predicate error"

@pytest.mark.asyncio
async def test_filter_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return f"{x}_{context.value}"
    
    # Predicate that checks if context value is in the result
    def predicate(x):
        return "test" in x
    
    # Execute
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(5, context=context)
    
    # Assert - should pass filter
    assert result.is_ok()
    assert result.default_value(None) == "5_test"

#########################################
# Tests for reduce
#########################################
@pytest.mark.asyncio
async def test_reduce_basic():
    # Setup
    # Operation that returns a list
    base_op = operation(lambda x: [1, 2, 3, 4, 5])
    # Reducer that sums elements
    reducer = lambda acc, item: acc + item
    
    # Execute
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    
    # Assert - should reduce [1,2,3,4,5] to 15
    assert result.is_ok()
    assert result.default_value(None) == 15

@pytest.mark.asyncio
async def test_reduce_empty_list():
    # Setup
    base_op = operation(lambda x: [])
    reducer = lambda acc, item: acc + item
    
    # Execute
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    
    # Assert - should return None for empty list
    assert result.is_ok()
    assert result.default_value(None) is None

@pytest.mark.asyncio
async def test_reduce_non_list_input():
    # Setup
    base_op = operation(lambda x: "not a list")
    reducer = lambda acc, item: acc + item
    
    # Execute
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    
    # Assert - should error for non-list input
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_reduce_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    reducer = lambda acc, item: acc + item
    
    # Execute
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_reduce_with_error_in_reducer():
    # Setup
    base_op = operation(lambda x: [1, 2, 3])
    
    def error_reducer(acc, item):
        raise ValueError("Reducer error")
    
    # Execute
    reduce_op = reduce(base_op, error_reducer)
    result = await reduce_op.execute(None)
    
    # Assert - should fail with reducer's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Reducer error"

@pytest.mark.asyncio
async def test_reduce_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        # Return a list with the context value
        return [1, 2, 3, context.value]
    
    # Custom reducer that handles strings and numbers
    def reducer(acc, item):
        if isinstance(acc, str) or isinstance(item, str):
            return str(acc) + str(item)
        return acc + item
    
    # Execute
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None, context=context)
    
    # Assert - should reduce [1,2,3,"test"] to "6test"
    assert result.is_ok()
    assert result.default_value(None) == "6test"

#########################################
# Tests for zip
#########################################
@pytest.mark.asyncio
async def test_zip_basic():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    
    # Execute
    zip_op = zip(op1, op2, op3)
    result = await zip_op.execute(5)
    
    # Assert - should return results as a tuple
    assert result.is_ok()
    assert result.default_value(None) == (6, 10, 2)

@pytest.mark.asyncio
async def test_zip_empty():
    # Setup & Execute
    zip_op = zip()
    result = await zip_op.execute(42)
    
    # Assert - should return empty list
    assert result.is_ok()
    assert result.default_value(None) == ()

@pytest.mark.asyncio
async def test_zip_with_error():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)  # Will raise ZeroDivisionError
    op3 = operation(lambda x: x - 3)
    
    # Execute
    zip_op = zip(op1, op2, op3)
    result = await zip_op.execute(5)
    
    # Assert - should return error
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_zip_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    
    # Execute
    zip_op = zip(op1, op2)
    result = await zip_op.execute(5, context=context)
    
    # Assert - should contain results from both operations
    assert result.is_ok()
    assert result.default_value(None) == ("5_test_1", "5_test_2")

#########################################
# Tests for flat_map
#########################################
@pytest.mark.asyncio
async def test_flat_map_basic():
    # Setup
    base_op = operation(lambda x: x + 1)
    
    def mapper(value):
        return [[value, value * 2], [value * 3, value * 4]]
    
    # Execute
    flat_map_op = flat_map(base_op, mapper)
    result = await flat_map_op.execute(5)
    
    # Assert - should flat map results
    # 5 + 1 = 6, then map to [[6, 12], [18, 24]], flattened to [6, 12, 18, 24]
    assert result.is_ok()
    assert result.default_value(None) == [6, 12, 18, 24]

@pytest.mark.asyncio
async def test_flat_map_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    mapper = lambda x: [[x, x * 2]]
    
    # Execute
    flat_map_op = flat_map(base_op, mapper)
    result = await flat_map_op.execute(5)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_flat_map_with_error_in_mapper():
    # Setup
    base_op = operation(lambda x: x + 1)
    
    def error_mapper(x):
        raise ValueError("Mapper error")
    
    # Execute
    flat_map_op = flat_map(base_op, error_mapper)
    result = await flat_map_op.execute(5)
    
    # Assert - should fail with mapper's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Mapper error"

@pytest.mark.asyncio
async def test_flat_map_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return f"{x}_{context.value}"
    
    def mapper(value):
        return [[f"{value}_a", f"{value}_b"], [f"{value}_c"]]
    
    # Execute
    flat_map_op = flat_map(base_op, mapper)
    result = await flat_map_op.execute(5, context=context)
    
    # Assert - should flat map with context
    assert result.is_ok()
    assert result.default_value(None) == ["5_test_a", "5_test_b", "5_test_c"]

#########################################
# Tests for group_by
#########################################
@pytest.mark.asyncio
async def test_group_by_basic():
    # Setup
    # Operation that returns a list
    base_op = operation(lambda x: [1, 2, 3, 4, 5, 6])
    # Group by even/odd
    grouper = lambda x: "even" if x % 2 == 0 else "odd"
    
    # Execute
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    
    # Assert - should group [1,2,3,4,5,6] into {"odd": [1,3,5], "even": [2,4,6]}
    assert result.is_ok()
    grouped = result.default_value(None)
    assert "odd" in grouped and "even" in grouped
    assert grouped["odd"] == [1, 3, 5]
    assert grouped["even"] == [2, 4, 6]

@pytest.mark.asyncio
async def test_group_by_empty_list():
    # Setup
    base_op = operation(lambda x: [])
    grouper = lambda x: x
    
    # Execute
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    
    # Assert - should return empty dict for empty list
    assert result.is_ok()
    assert result.default_value(None) == {}

@pytest.mark.asyncio
async def test_group_by_non_list_input():
    # Setup
    base_op = operation(lambda x: "not a list")
    grouper = lambda x: x
    
    # Execute
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    
    # Assert - should error for non-list input
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_group_by_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    grouper = lambda x: x
    
    # Execute
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_group_by_with_error_in_grouper():
    # Setup
    base_op = operation(lambda x: [1, 2, 3])
    
    def error_grouper(x):
        raise ValueError("Grouper error")
    
    # Execute
    group_by_op = group_by(base_op, error_grouper)
    result = await group_by_op.execute(None)
    
    # Assert - should fail with grouper's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Grouper error"

@pytest.mark.asyncio
async def test_group_by_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        # Return a list with strings that include context value
        return [f"a_{context.value}", f"b_{context.value}", f"a_other", f"c_{context.value}"]
    
    # Group by first character
    grouper = lambda s: s[0]
    
    # Execute
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None, context=context)
    
    # Assert - should group by first character
    assert result.is_ok()
    grouped = result.default_value(None)
    assert "a" in grouped and "b" in grouped and "c" in grouped
    assert len(grouped["a"]) == 2
    assert grouped["b"] == [f"b_{context.value}"]
    assert grouped["c"] == [f"c_{context.value}"]

#########################################
# Tests for partition
#########################################
@pytest.mark.asyncio
async def test_partition_basic():
    # Setup
    # Operation that returns a list
    base_op = operation(lambda x: [1, 2, 3, 4, 5, 6])
    # Partition by even/odd
    predicate = lambda x: x % 2 == 0
    
    # Execute
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    
    # Assert - should partition [1,2,3,4,5,6] into ([2,4,6], [1,3,5])
    assert result.is_ok()
    partitioned = result.default_value(None)
    assert isinstance(partitioned, tuple) and len(partitioned) == 2
    assert partitioned[0] == [2, 4, 6]  # truthy values
    assert partitioned[1] == [1, 3, 5]  # falsy values

@pytest.mark.asyncio
async def test_partition_empty_list():
    # Setup
    base_op = operation(lambda x: [])
    predicate = lambda x: True
    
    # Execute
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    
    # Assert - should return empty lists for empty input
    assert result.is_ok()
    partitioned = result.default_value(None)
    assert isinstance(partitioned, tuple) and len(partitioned) == 2
    assert partitioned[0] == []
    assert partitioned[1] == []

@pytest.mark.asyncio
async def test_partition_non_list_input():
    # Setup
    base_op = operation(lambda x: "not a list")
    predicate = lambda x: True
    
    # Execute
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    
    # Assert - should error for non-list input
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_partition_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    predicate = lambda x: True
    
    # Execute
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_partition_with_error_in_predicate():
    # Setup
    base_op = operation(lambda x: [1, 2, 3])
    
    def error_predicate(x):
        raise ValueError("Predicate error")
    
    # Execute
    partition_op = partition(base_op, error_predicate)
    result = await partition_op.execute(None)
    
    # Assert - should fail with predicate's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Predicate error"

@pytest.mark.asyncio
async def test_partition_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        # Return a list of strings that may contain the context value
        return [f"has_{context.value}", "no_match", f"also_has_{context.value}", "another_no_match"]
    
    # Predicate that checks if string contains context value
    def predicate(s):
        return "test" in s
    
    # Execute
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None, context=context)
    
    # Assert - should partition by presence of "test"
    assert result.is_ok()
    partitioned = result.default_value(None)
    assert isinstance(partitioned, tuple) and len(partitioned) == 2
    assert partitioned[0] == [f"has_{context.value}", f"also_has_{context.value}"]
    assert partitioned[1] == ["no_match", "another_no_match"]

#########################################
# Tests for first
#########################################
@pytest.mark.asyncio
async def test_first_basic():
    # Setup
    # Operation that returns a list
    base_op = operation(lambda x: [10, 20, 30, 40])
    
    # Execute
    first_op = first(base_op)
    result = await first_op.execute(None)
    
    # Assert - should return first element
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_first_empty_list():
    # Setup
    base_op = operation(lambda x: [])
    
    # Execute
    first_op = first(base_op)
    result = await first_op.execute(None)
    
    # Assert - should error for empty list
    assert result.is_error()
    assert isinstance(result.error, IndexError)
    assert "empty" in str(result.error).lower()

@pytest.mark.asyncio
async def test_first_non_list_input():
    # Setup
    base_op = operation(lambda x: "not a list")
    
    # Execute
    first_op = first(base_op)
    result = await first_op.execute(None)
    
    # Assert - should error for non-list input
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_first_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    
    # Execute
    first_op = first(base_op)
    result = await first_op.execute(None)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_first_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return [f"first_{context.value}", "second", "third"]
    
    # Execute
    first_op = first(base_op)
    result = await first_op.execute(None, context=context)
    
    # Assert - should return first element
    assert result.is_ok()
    assert result.default_value(None) == f"first_{context.value}"

#########################################
# Tests for last
#########################################
@pytest.mark.asyncio
async def test_last_basic():
    # Setup
    # Operation that returns a list
    base_op = operation(lambda x: [10, 20, 30, 40])
    
    # Execute
    last_op = last(base_op)
    result = await last_op.execute(None)
    
    # Assert - should return last element
    assert result.is_ok()
    assert result.default_value(None) == 40

@pytest.mark.asyncio
async def test_last_empty_list():
    # Setup
    base_op = operation(lambda x: [])
    
    # Execute
    last_op = last(base_op)
    result = await last_op.execute(None)
    
    # Assert - should error for empty list
    assert result.is_error()
    assert isinstance(result.error, IndexError)
    assert "empty" in str(result.error).lower()

@pytest.mark.asyncio
async def test_last_non_list_input():
    # Setup
    base_op = operation(lambda x: "not a list")
    
    # Execute
    last_op = last(base_op)
    result = await last_op.execute(None)
    
    # Assert - should error for non-list input
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_last_with_error_in_base():
    # Setup
    base_op = operation(sync_error)
    
    # Execute
    last_op = last(base_op)
    result = await last_op.execute(None)
    
    # Assert - should fail with base operation's error
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_last_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return ["first", "second", f"last_{context.value}"]
    
    # Execute
    last_op = last(base_op)
    result = await last_op.execute(None, context=context)
    
    # Assert - should return last element
    assert result.is_ok()
    assert result.default_value(None) == f"last_{context.value}"

#########################################
# Tests for gather_operations
#########################################
@pytest.mark.asyncio
async def test_gather_operations_basic():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    
    # Execute
    results = await gather_operations(op1, op2, op3, args=(5,))
    
    # Assert - should run all operations concurrently
    assert len(results) == 3
    assert all(isinstance(r, Result) for r in results)
    assert all(r.is_ok() for r in results)
    assert [r.default_value(None) for r in results] == [6, 10, 2]

@pytest.mark.asyncio
async def test_gather_operations_with_error():
    # Setup
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)  # Will raise ZeroDivisionError
    op3 = operation(lambda x: x - 3)
    
    # Execute
    results = await gather_operations(op1, op2, op3, args=(5,))
    
    # Assert - should return both successes and errors
    assert len(results) == 3
    assert results[0].is_ok() and results[0].default_value(None) == 6
    assert results[1].is_error() and isinstance(results[1].error, ZeroDivisionError)
    assert results[2].is_ok() and results[2].default_value(None) == 2

@pytest.mark.asyncio
async def test_gather_operations_with_context():
    # Setup
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    
    # Execute with context
    results = await gather_operations(op1, op2, args=(5,), kwargs={"context": context})
    
    # Assert - should pass context to all operations
    assert len(results) == 2
    assert all(r.is_ok() for r in results)
    assert results[0].default_value(None) == "5_test_1"
    assert results[1].default_value(None) == "5_test_2"

@pytest.mark.asyncio
async def test_gather_operations_without_args():
    # Setup - operations that don't need args
    op1 = operation(lambda: 10)
    op2 = operation(lambda: 20)
    
    # Execute without args
    results = await gather_operations(op1, op2)
    
    # Assert
    assert len(results) == 2
    assert all(r.is_ok() for r in results)
    assert results[0].default_value(None) == 10
    assert results[1].default_value(None) == 20

@pytest.mark.asyncio
async def test_gather_operations_with_bound_operations():
    # Setup - create bound operations
    op1 = operation(lambda x: x + 1)(5)
    op2 = operation(lambda x: x * 2)(10)
    
    # Execute without additional args (using bound values)
    results = await gather_operations(op1, op2)
    
    # Assert
    assert len(results) == 2
    assert all(r.is_ok() for r in results)
    assert results[0].default_value(None) == 6  # 5 + 1
    assert results[1].default_value(None) == 20  # 10 * 2