import pytest
import asyncio
from typing import Any, Dict, List, Optional, Type, Tuple, cast
from unittest.mock import Mock, patch, AsyncMock

from fp_ops.context import BaseContext
from fp_ops.operator import Operation, operation, identity, constant
from fp_ops.placeholder import _
from fp_ops.composition import flat_map
from expression import Result


class TestContext(BaseContext):
    value: str = "test"


@pytest.mark.asyncio
async def test_operation_returning_operation_basic():
    """Test basic case of an operation returning another operation."""
    
    @operation
    async def get_operation(x: int) -> Operation:
        # Return a different operation based on input
        if x > 5:
            return operation(lambda y: y * 2)
        else:
            return operation(lambda y: y + 5)
    
    # Execute with x > 5
    op_result1 = await get_operation(10)
    # Now execute the returned operation
    result1 = await op_result1.execute(3)
    
    # Execute with x <= 5
    op_result2 = await get_operation(5)
    # Now execute the returned operation
    result2 = await op_result2.execute(3)
    
    # Assert
    assert result1.is_ok() and result1.default_value(None) == 6  # 3 * 2 = 6
    assert result2.is_ok() and result2.default_value(None) == 8  # 3 + 5 = 8


@pytest.mark.asyncio
async def test_operation_returning_operation_with_context():
    """Test an operation returning another operation with context handling."""
    context = TestContext()
    
    @operation(context=True, context_type=TestContext)
    async def get_operation_with_context(x: int, context=None) -> Operation:
        # Return a different context-aware operation based on context
        if context.value == "test":
            @operation(context=True, context_type=TestContext)
            async def inner_op(y: int, context=None) -> str:
                return f"{y}_{context.value}_inner"
            return inner_op
        else:
            return operation(lambda y: f"{y}_default")
    
    # Execute with context
    op_result = await get_operation_with_context(10, context=context)
    
    # Now execute the returned operation with the same context
    operation = op_result.default_value(None)
    if isinstance(operation, Operation):
        print("Executing operation")
        result = await operation.execute(3, context=context)
    else:
        result = operation
    
    # Assert
    assert result.is_ok() and result.default_value(None) == "3_test_inner"


@pytest.mark.asyncio
async def test_operation_returning_operation_with_rshift():
    """Test composing an operation that returns another operation using >>."""
    
    @operation
    async def get_operation(x: int) -> Operation:
        # Return a different operation based on input
        if x > 5:
            return operation(lambda y: y * 2)
        else:
            return operation(lambda y: y + 5)
    
    # Create a pipeline: get_operation >> identity
    # This should:
    # 1. Execute get_operation to get an operation
    # 2. Execute that returned operation
    # 3. Pass the result to identity
    pipeline = get_operation >> identity
    
    # Execute pipeline with x > 5
    result1 = await pipeline.execute(10, 3)  # First arg to get_operation, second to returned op
    
    # Execute pipeline with x <= 5
    result2 = await pipeline.execute(5, 3)
    
    # Assert
    assert result1.is_ok() and result1.default_value(None) == 6  # 3 * 2 = 6
    assert result2.is_ok() and result2.default_value(None) == 8  # 3 + 5 = 8


@pytest.mark.asyncio
async def test_nested_operation_composition():
    """Test composing multiple operations that return operations."""
    
    @operation
    async def get_multiplier(x: int) -> Operation:
        return operation(lambda y: y * x)
    
    @operation
    async def get_adder(x: int) -> Operation:
        return operation(lambda y: y + x)
    
    # Compose operations: get_multiplier >> get_adder
    # This should:
    # 1. Get a multiplier operation based on first input
    # 2. Execute that multiplier with second input
    # 3. Get an adder operation based on the result
    # 4. Execute the adder with third input
    pipeline = get_multiplier >> get_adder
    
    # Execute pipeline
    # This is tricky because we need to provide args for multiple operations
    # First 2 for get_multiplier, then returned multiplier, then returned adder 
    result = await pipeline.execute(2, 3, 4)
    
    # Assert - should be: 3 * 2 = 6, then get adder(6), which adds 6, 
    # so final result is 4 + 6 = 10
    assert result.is_ok() and result.default_value(None) == 10


@pytest.mark.asyncio
async def test_placeholder_with_returned_operations():
    """Test using placeholders with operations that return operations."""
    
    @operation
    async def get_operation(x: int) -> Operation:
        return operation(lambda y: y * x)
    
    # Create a pipeline using placeholder
    # This should:
    # 1. Execute get_operation to get an operation (passing 2)
    # 2. Execute that returned operation with _ (the result of a previous operation)
    pipeline = constant(2) >> get_operation(_) >> (lambda x: x + 5)
    
    # Execute pipeline
    result = await pipeline.execute(3)  # This is used as input to constant(2)
    
    # Assert - constant(2) returns 2, get_operation(2) returns op that does y*2,
    # which is executed with 2 to get 4, then + 5 = 9
    assert result.is_ok() and result.default_value(None) == 9


@pytest.mark.asyncio
async def test_error_handling_with_returned_operations():
    """Test error handling when an operation returns another operation."""
    
    @operation
    async def get_operation(x: int) -> Operation:
        if x == 0:
            raise ValueError("Cannot process zero")
        return operation(lambda y: y * x)
    
    # Create a pipeline with error handling
    pipeline = get_operation.catch(lambda e: operation(lambda y: -1)) >> (lambda x: x + 5)
    
    # Execute pipeline with error
    result1 = await pipeline.execute(0, 3)
    
    # Execute pipeline without error
    result2 = await pipeline.execute(2, 3)
    
    # Assert
    # In error case, catch returns an operation that returns -1, then -1 + 5 = 4
    assert result1.is_ok() and result1.default_value(None) == 4
    # In success case, get_operation(2) returns an operation that does y*2,
    # which is executed with 3 to get 6, then 6 + 5 = 11
    assert result2.is_ok() and result2.default_value(None) == 11


@pytest.mark.asyncio
async def test_complex_composition_with_returned_operations():
    """Test complex composition with operations that return operations."""
    
    @operation
    async def get_operation_a(x: int) -> Operation:
        return operation(lambda y: y * x)
    
    @operation
    async def get_operation_b(x: int) -> Operation:
        return operation(lambda y: y + x)
    
    # Create a complex pipeline with parallel execution
    # (get_operation_a & get_operation_b) creates a tuple of operations
    # Then we need to apply both operations and combine the results
    pipeline = (get_operation_a & get_operation_b) >> (lambda ops_tuple: 
                ops_tuple[0].execute(5) & ops_tuple[1].execute(5)) >> (lambda results_tuple:
                results_tuple[0].default_value(0) + results_tuple[1].default_value(0))
    
    # Execute pipeline
    result = await pipeline.execute(2, 3)
    
    # Assert
    # get_operation_a(2) returns op that does y*2
    # get_operation_b(3) returns op that does y+3
    # Then execute both with 5: 5*2=10 and 5+3=8
    # Finally add results: 10+8=18
    assert result.is_ok() and result.default_value(None) == 18


@pytest.mark.asyncio
async def test_operation_returning_bound_operation():
    """Test an operation returning a bound operation."""
    
    @operation
    async def get_bound_operation(x: int) -> Operation:
        # Return an already bound operation
        base_op = operation(lambda a, b: a * b)
        # Bind the first argument
        return base_op(x)
    
    # Create a pipeline
    pipeline = get_bound_operation >> (lambda x: x + 5)
    
    # Execute pipeline
    result = await pipeline.execute(2, 3)  # 2 to get_bound_operation, 3 to the bound operation
    
    # Assert
    # get_bound_operation(2) returns operation(2, b) which computes 2*b
    # This is executed with b=3 to get 6, then 6+5=11
    assert result.is_ok() and result.default_value(None) == 11


@pytest.mark.asyncio
async def test_flat_map_with_operation_returning_operations():
    """Test flat_map with an operation that returns operations with nested results."""
    
    @operation
    async def get_operations_creator(x: int) -> Operation:
        # Return an operation that creates operations based on a list
        # This represents a "factory function" that creates multiple operations
        
        @operation
        async def create_operations(nums: List[int]) -> List[Operation]:
            # Create a list of operations based on input
            return [operation(lambda y, num=num: y * num) for num in nums]
        
        return create_operations
    
    # Create a pipeline using flat_map
    # First, get an operation that creates operations
    # Then use flat_map to transform the list of operations 
    # into a flattened list of results by executing each operation with a value
    
    # The "mapper" function for flat_map executes each operation with value 2
    # and returns a list of results for each operation
    async def execute_operations(ops_list: List[Operation]) -> List[List[Any]]:
        results = []
        for op in ops_list:
            result = await op.execute(2)
            results.append([result.default_value(None)])
        return results
    
    pipeline = get_operations_creator >> flat_map(_, execute_operations)
    
    # Execute pipeline with [1, 2, 3] as input to create_operations
    result = await pipeline.execute(5, [1, 2, 3])
    
    # Assert
    # create_operations([1, 2, 3]) creates 3 operations
    # Each operation multiplies by its respective number: 2*1=2, 2*2=4, 2*3=6
    # Results are nested lists: [[2], [4], [6]]
    # flat_map flattens to: [2, 4, 6]
    assert result.is_ok() 
    assert result.default_value(None) == [2, 4, 6]


@pytest.mark.asyncio
async def test_flat_map_with_operations_creating_operations():
    """Test flat_map with operations that create and execute other operations."""
    
    @operation
    async def create_factories(count: int) -> List[Operation]:
        # Create a list of "factory" operations
        factories = []
        for i in range(1, count + 1):
            # Each factory creates a specific type of operation
            @operation
            async def factory(x: int, multiplier=i) -> Operation:
                return operation(lambda y: y * multiplier)
            factories.append(factory)
        return factories
    
    # Mapper function that executes each factory with input 2
    # to get the operations, then executes those operations with input 3
    async def execute_factory_and_op(factory: Operation) -> List[Any]:
        # Execute the factory to get an operation
        op_result = await factory.execute(2)
        if op_result.is_error():
            return [op_result.error]
        
        # Get the operation from the result
        op = op_result.default_value(None)
        
        # Execute the operation
        result = await op.execute(3)
        if result.is_error():
            return [result.error]
            
        # Return the result in a list for flat_map
        return [result.default_value(None)]
    
    # Create pipeline using flat_map
    pipeline = create_factories >> flat_map(_, execute_factory_and_op)
    
    # Execute pipeline
    result = await pipeline.execute(3)  # Create 3 factories
    
    # Assert
    # 3 factories are created with multipliers 1, 2, 3
    # Each factory creates an operation that multiplies by its multiplier
    # Those operations are executed with input 3: 3*1=3, 3*2=6, 3*3=9
    # Results come back as [[3], [6], [9]] and are flattened to [3, 6, 9]
    assert result.is_ok()
    assert result.default_value(None) == [3, 6, 9] 