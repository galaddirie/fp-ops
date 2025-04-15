import pytest
import asyncio
from typing import Any, Callable, List, Dict, Optional, Union, TypeVar, cast
from expression import Result

from fp_ops.operator import Operation, operation, identity, constant
from fp_ops.context import BaseContext
from fp_ops.placeholder import Placeholder, _ 
from fp_ops.composition import pipe, sequence, compose, parallel, fallback

# Type variables for better typing
T = TypeVar("T")
S = TypeVar("S")

# Basic operations for testing
@operation
async def double(x: int) -> int:
    """Double the input value"""
    return x * 2

@operation
async def add_n(n: int) -> Callable[[int], int]:
    """Return an operation that adds n to a value"""
    @operation
    async def add(x: int) -> int:
        return x + n
    return add

@operation
async def multiply_by(n: int) -> Operation:
    """Return an operation that multiplies by n"""
    @operation
    async def multiply(x: int) -> int:
        return x * n
    return multiply

@operation
async def apply_func(func: Callable[[int], int]) -> Operation:
    """Return an operation that applies the given function"""
    @operation
    async def apply(x: int) -> int:
        return func(x)
    return apply

@operation
async def conditional_operation(condition: bool) -> Operation:
    """Return different operations based on a condition"""
    if condition:
        return double
    else:
        return add_n(5)

@operation
async def operation_factory(op_type: str, param: int = 1) -> Operation:
    """Factory that returns different types of operations based on input"""
    if op_type == "add":
        return add_n(param)
    elif op_type == "multiply":
        return multiply_by(param)
    else:
        # Default case - return identity
        return identity

@operation
async def map_values(mapper: Callable[[int], int]) -> Operation:
    """Higher-order operation that returns an operation mapping values in a list"""
    @operation
    async def mapper_op(values: List[int]) -> List[int]:
        return [mapper(val) for val in values]
    return mapper_op

# Custom context class for testing
class TestContext(BaseContext):
    value: int = 0
    name: str = ""

@operation(context=True, context_type=TestContext)
async def increment_context(x: int, context: TestContext) -> TestContext:
    """Increment the context value"""
    context.value += x
    return context

@operation
async def context_operation_factory(multiply_factor: int) -> Operation:
    """Return a context-aware operation"""
    @operation(context=True, context_type=TestContext)
    async def multiply_context(context: TestContext) -> TestContext:
        context.value *= multiply_factor
        return context
    return multiply_context

# Test fixtures
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Test cases
class TestHigherOrderOperations:
    
    @pytest.mark.asyncio
    async def test_basic_operation_returns(self):
        """Test that operations can return other operations"""
        # Get an operation that adds 5
        add_5_op = await add_n(5)
        assert isinstance(add_5_op, Operation)
        
        # Execute the returned operation
        result = await add_5_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 15
        
        # Test with composition
        composed = add_5_op >> double
        result = await composed.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 30  # (10 + 5) * 2
    
    @pytest.mark.asyncio
    async def test_conditional_operation_creation(self):
        """Test operations that conditionally return other operations"""
        # Test with condition=True
        true_op = await conditional_operation(True)
        assert isinstance(true_op, Operation)
        result = await true_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 20  # double(10)
        
        # Test with condition=False
        false_op = await conditional_operation(False)
        assert isinstance(false_op, Operation)
        result = await false_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 15  # add_5(10)
    
    @pytest.mark.asyncio
    async def test_operation_factory(self):
        """Test a factory that creates different operations"""
        # Create an add operation
        add_op = await operation_factory("add", 3)
        assert isinstance(add_op, Operation)
        result = await add_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 13  # 10 + 3
        
        # Create a multiply operation
        mul_op = await operation_factory("multiply", 4)
        assert isinstance(mul_op, Operation)
        result = await mul_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 40  # 10 * 4
        
        # Test default case
        default_op = await operation_factory("unknown")
        assert isinstance(default_op, Operation)
        result = await default_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 10  # identity(10)
    
    @pytest.mark.asyncio
    async def test_pipe_with_higher_order_ops(self):
        """Test using pipe with higher-order operations"""
        # Create a pipeline: 
        # 1. Create an add_3 operation
        # 2. Execute it on the input
        # 3. Double the result
        pipeline = pipe(
            operation_factory("add", 3),
            lambda op: op >> double
        )
        
        result = await pipeline.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 26  # (10 + 3) * 2
    
    @pytest.mark.asyncio
    async def test_composition_of_higher_order_results(self):
        """Test composing operations returned by higher-order functions"""
        # Get operations
        add_7_op = await add_n(7)
        mul_3_op = await multiply_by(3)
        
        # Compose them
        composed = add_7_op >> mul_3_op
        
        # Test the composed operation
        result = await composed.execute(5)
        assert result.is_ok()
        assert result.default_value(None) == 36  # (5 + 7) * 3
        
        # Test with a more complex composition
        complex_composed = add_n(2) >> (lambda x: multiply_by(x))
        result = await complex_composed.execute(5)
        assert result.is_ok()
        # This should create an add_2 operation, then create a multiply_by_7 operation (5+2=7),
        # then return the multiply_by_7 operation (not execute it)
        assert isinstance(result.default_value(None), Operation)
        
        # Now execute the returned operation
        mul_op = result.default_value(None)
        mul_result = await mul_op.execute(10)
        assert mul_result.is_ok()
        assert mul_result.default_value(None) == 70  # 10 * 7
    
    @pytest.mark.asyncio
    async def test_function_to_operation_conversion(self):
        """Test converting regular functions to operations and composing them"""
        # Define a regular function
        def add_five(x: int) -> int:
            return x + 5
        
        # Convert it to an operation via the apply_func higher-order operation
        add_five_op = await apply_func(add_five)
        assert isinstance(add_five_op, Operation)
        
        # Test the operation
        result = await add_five_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 15
        
        # Compose with another operation
        composed = add_five_op >> double
        result = await composed.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 30  # (10 + 5) * 2
    
    @pytest.mark.asyncio
    async def test_mapping_with_higher_order_op(self):
        """Test higher-order operation that maps values"""
        # Create a mapper operation that squares numbers
        square_mapper = await map_values(lambda x: x * x)
        assert isinstance(square_mapper, Operation)
        
        # Test the mapper operation
        result = await square_mapper.execute([1, 2, 3, 4])
        assert result.is_ok()
        assert result.default_value(None) == [1, 4, 9, 16]
        
        # Create a mapper operation that adds 1
        add_one_mapper = await map_values(lambda x: x + 1)
        
        # Compose operations using sequence
        # First square the numbers, then add 1 to each result
        seq_op = square_mapper >> add_one_mapper
        result = await seq_op.execute([1, 2, 3])
        assert result.is_ok()
        assert result.default_value(None) == [2, 5, 10]  # [1²+1, 2²+1, 3²+1]
    
    @pytest.mark.asyncio
    async def test_context_with_higher_order_ops(self):
        """Test higher-order operations with context"""
        # Create a context
        context = TestContext(value=5, name="test")
        
        # Create a context operation that multiplies by 3
        multiply_3_op = await context_operation_factory(3)
        assert isinstance(multiply_3_op, Operation)
        
        # Execute the operation with context
        result = await multiply_3_op.execute(context=context)
        assert result.is_ok()
        assert result.default_value(None).value == 15  # 5 * 3
        
        # Create a pipeline that processes the context
        pipeline = pipe(
            context_operation_factory(2),  # Multiply by 2
            increment_context(10)          # Add 10
        )
        
        # Reset the context
        context.value = 5
        
        # Execute the pipeline
        result = await pipeline.execute(context=context)
        assert result.is_ok()
        assert result.default_value(None).value == 20  # (5 * 2) + 10
    
    @pytest.mark.asyncio
    async def test_chaining_higher_order_ops(self):
        """Test chaining multiple higher-order operations"""
        # Chain creation of operations
        # 1. Create an add_2 operation
        # 2. Create a multiply_by_3 operation
        # 3. Compose them
        pipeline = pipe(
            add_n(2),
            lambda add_op: pipe(
                multiply_by(3),
                lambda mul_op: add_op >> mul_op
            )
        )
        
        # Execute the pipeline to get the composed operation
        result = await pipeline.execute()
        assert result.is_ok()
        composed_op = result.default_value(None)
        assert isinstance(composed_op, Operation)
        
        # Test the composed operation
        result = await composed_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 36  # (10 + 2) * 3
    
    @pytest.mark.asyncio
    async def test_nested_higher_order_operations(self):
        """Test nesting higher-order operations"""
        # Create a factory that returns other factories
        @operation
        async def factory_factory(factory_type: str) -> Operation:
            if factory_type == "add":
                return add_n
            elif factory_type == "multiply":
                return multiply_by
            else:
                return lambda x: identity  # Default factory returns identity
        
        # Get an add_n factory
        add_factory = await factory_factory("add")
        assert isinstance(add_factory, Operation)
        
        # Use the factory to create an add_5 operation
        add_5_op = await add_factory.execute(5)
        assert isinstance(add_5_op, Operation)
        
        # Test the created operation
        result = await add_5_op.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 15  # 10 + 5
        
        # Get a multiply_by factory
        mul_factory = await factory_factory("multiply")
        
        # Use the factory to create a multiply_by_3 operation
        mul_3_op = await mul_factory.execute(3)
        
        # Compose the operations
        composed = add_5_op >> mul_3_op
        result = await composed.execute(10)
        assert result.is_ok()
        assert result.default_value(None) == 45  # (10 + 5) * 3
    
    @pytest.mark.asyncio
    async def test_partial_application(self):
        """Test partial application with operations"""
        # Create a curried operation
        @operation
        async def add_three_nums(a: int, b: int, c: int) -> int:
            return a + b + c
        
        # Partially apply arguments
        add_with_10 = add_three_nums(10)
        assert isinstance(add_with_10, Operation)
        
        # Further partial application
        add_with_10_and_20 = add_with_10(20)
        assert isinstance(add_with_10_and_20, Operation)
        
        # Final application
        result = await add_with_10_and_20.execute(30)
        assert result.is_ok()
        assert result.default_value(None) == 60  # 10 + 20 + 30
        
        # Test with a higher-order operation
        @operation
        async def curry_operation(a: int) -> Operation:
            @operation
            async def inner(b: int) -> Operation:
                @operation
                async def innermost(c: int) -> int:
                    return a + b + c
                return innermost
            return inner
        
        # Get the first operation
        first_op = await curry_operation(10)
        assert isinstance(first_op, Operation)
        
        # Get the second operation
        second_op = await first_op.execute(20)
        assert isinstance(second_op, Operation)
        
        # Get the final result
        result = await second_op.execute(30)
        assert result.is_ok()
        assert result.default_value(None) == 60  # 10 + 20 + 30