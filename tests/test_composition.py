"""
Comprehensive test suite for fp_ops.composition module.
Tests all composition operations including sequence, pipe, compose, parallel, and fallback.
Covers edge cases, context propagation, error handling, and complex scenarios.
"""
import pytest
import asyncio
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

# Assuming these imports based on the provided code
from fp_ops import operation, Operation
from fp_ops.composition import  pipe, compose, parallel, fallback
from fp_ops.objects import get, build, merge
from fp_ops.context import BaseContext
from expression import Ok, Error, Result


# Test fixtures and helper classes
class TestContext(BaseContext):
    """Test context for context propagation tests."""
    value: int = 0
    name: str = "test"
    
    def increment(self) -> "TestContext":
        return TestContext(value=self.value + 1, name=self.name)


class ExtendedContext(TestContext):
    """Extended context for testing context type inference."""
    extra: str = "extra"


# Helper operations
@operation
def add_one(x: int) -> int:
    """Add one to input."""
    return x + 1


@operation
def multiply_by_two(x: int) -> int:
    """Multiply input by two."""
    return x * 2


@operation
def to_string(x: int) -> str:
    """Convert to string."""
    return str(x)


@operation
def failing_op(x: Any) -> Any:
    """Always fails."""
    raise ValueError("Intentional failure")


@operation(context=True)
def context_aware_add(x: int, **kwargs) -> int:
    """Add context value to input."""
    context = kwargs.get("context")
    return x + context.value


@operation(context=True)
def update_context(x: Any, **kwargs) -> TestContext:
    """Update and return context."""
    context = kwargs["context"]
    # Return the new context so the executor propagates it
    new_context = context.increment()
    return new_context


# Fixtures
@pytest.fixture
def simple_ops():
    """Simple operations for testing."""
    return [add_one, multiply_by_two, to_string]


@pytest.fixture
def test_context():
    """Test context instance."""
    return TestContext(value=10, name="test")


# Test pipe operation
class TestPipeOperation:
    """Test suite for the pipe operation."""
    
    @pytest.mark.asyncio
    async def test_pipe_basic(self):
        """Test basic pipe with operations."""
        pipeline = pipe(add_one, multiply_by_two, to_string)
        result = await pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value("") == "12"
    
    @pytest.mark.asyncio
    async def test_pipe_with_lambdas(self):
        """Test pipe with lambda functions returning operations."""
        from fp_ops.flow import branch
        pipeline = pipe(
            add_one,
            branch(lambda x: x > 5, multiply_by_two, add_one),
            to_string
        )
        
        # Test path where x > 5
        result1 = await pipeline.execute(5)
        assert result1.is_ok()
        assert result1.default_value("") == "12"
        
        # Test path where x <= 5
        result2 = await pipeline.execute(3)
        assert result2.is_ok()
        assert result2.default_value("") == "5"
    
    @pytest.mark.asyncio
    async def test_pipe_empty(self):
        """Test pipe with no steps."""
        pipeline = pipe()
        result = await pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value(None) == 5
    
    @pytest.mark.asyncio
    async def test_pipe_single_step(self):
        """Test pipe with single step."""
        pipeline = pipe(add_one)
        result = await pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 6
    
    @pytest.mark.asyncio
    async def test_pipe_error_in_operation(self):
        """Test pipe error handling in operations."""
        pipeline = pipe(add_one, failing_op, multiply_by_two)
        result = await pipeline.execute(5)
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_pipe_error_in_function(self):
        """Test pipe error handling in lambda functions."""
        def failing_function(x: Any) -> Any:
            raise ValueError("Intentional failure")
        
        pipeline = pipe(
            add_one,
            failing_function,
            multiply_by_two
        )
        result = await pipeline.execute(5)
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_pipe_lambda_returns_non_operation(self):
        """Test pipe with lambda that doesn't return an Operation."""
        pipeline = pipe(
            add_one,
            lambda x: x * 2  # Returns int, not Operation
        )
        # bind wraps non-Operations, so this should work
        result = await pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 12
    
    @pytest.mark.asyncio
    async def test_pipe_context_propagation(self, test_context):
        """Test context propagation through pipe."""
        pipeline = pipe(
            update_context,  # Returns updated context
            context_aware_add,  # Should use updated context
            to_string
        )
        
        result = await pipeline.execute(5, context=test_context)
        assert result.is_ok()
        # 5 + 11 (updated context value) = 16
        assert result.default_value("") == "16"
    
    @pytest.mark.asyncio
    async def test_pipe_with_bound_operations(self):
        """Test pipe with bound operations."""
        from fp_ops.placeholder import _
        @operation
        def add(a: int, b: int) -> int:
            return a + b
        
        # Use a lambda to create partial application
        pipeline = pipe(
            add_one,
            add(_,10),  # Partial application via lambda
            to_string
        )
        
        result = await pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value("") == "16"
    
   

# Test compose operation
class TestComposeOperation:
    """Test suite for the compose operation."""
    
    @pytest.mark.asyncio
    async def test_compose_basic(self):
        """Test basic composition."""
        comp = compose(to_string, multiply_by_two, add_one)
        result = await comp.execute(5)
        assert result.is_ok()
        assert result.default_value("") == "12"
    
    @pytest.mark.asyncio
    async def test_compose_empty(self):
        """Test compose with no operations returns identity."""
        comp = compose()
        result = await comp.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 5
    
    @pytest.mark.asyncio
    async def test_compose_single_operation(self):
        """Test compose with single operation."""
        comp = compose(add_one)
        result = await comp.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 6
    
    @pytest.mark.asyncio
    async def test_compose_order(self):
        """Test that compose maintains correct order."""
        # compose is right-to-left, so compose(a,b,c) = c >> b >> a
        comp1 = compose(to_string, multiply_by_two, add_one)
        comp2 = compose(to_string, multiply_by_two, add_one)
        
        result1 = await comp1.execute(5)
        result2 = await comp2.execute(5)
        
        assert result1.is_ok() and result2.is_ok()
        assert result1.default_value("") == result2.default_value("")
    
    @pytest.mark.asyncio
    async def test_compose_with_context(self, test_context):
        """Test compose with context-aware operations."""
        # Since update_context returns the context, we need a different approach
        @operation(context=True)
        def passthrough_with_update(x: Any, **kwargs) -> Any:
            """Pass through value while updating context."""
            context = kwargs["context"]
            # This will trigger context propagation when update_context is called
            return x
        
        # First call update_context to get new context, then restore the value and add
        comp = compose(
            to_string, 
            context_aware_add,
            operation(lambda _: 5),  # Restore original value after update_context
            update_context
        )
        result = await comp.execute(5, context=test_context)
        assert result.is_ok()
        assert result.default_value("") == "16"


# Test parallel operation
class TestParallelOperation:
    """Test suite for the parallel operation."""
    
    @pytest.mark.asyncio
    async def test_parallel_basic(self):
        """Test basic parallel execution."""
        par = parallel(add_one, multiply_by_two, to_string)
        result = await par.execute(5)
        assert result.is_ok()
        values = result.default_value(())
        assert values == (6, 10, "5")
    
    @pytest.mark.asyncio
    async def test_parallel_empty(self):
        """Test parallel with no operations."""
        par = parallel()
        result = await par.execute(5)
        assert result.is_ok()
        assert result.default_value(()) == ()
    
    @pytest.mark.asyncio
    async def test_parallel_single_operation(self):
        """Test parallel with single operation."""
        par = parallel(add_one)
        result = await par.execute(5)
        assert result.is_ok()
        assert result.default_value(()) == (6,)
    
    @pytest.mark.asyncio
    async def test_parallel_error_handling(self):
        """Test parallel stops on any error."""
        par = parallel(add_one, failing_op, multiply_by_two)
        result = await par.execute(5)
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_parallel_truly_concurrent(self):
        """Test that operations run concurrently."""
        execution_order = []
        
        @operation
        async def slow_op1(x: int) -> str:
            execution_order.append("start1")
            await asyncio.sleep(0.1)
            execution_order.append("end1")
            return "op1"
        
        @operation
        async def slow_op2(x: int) -> str:
            execution_order.append("start2")
            await asyncio.sleep(0.05)
            execution_order.append("end2")
            return "op2"
        
        par = parallel(slow_op1, slow_op2)
        result = await par.execute(5)
        assert result.is_ok()
        assert result.default_value(()) == ("op1", "op2")
        
        # Check execution was interleaved (concurrent)
        assert execution_order == ["start1", "start2", "end2", "end1"]
    
    @pytest.mark.asyncio
    async def test_parallel_with_context(self, test_context):
        """Test parallel with context-aware operations."""
        @operation(context=True)
        def get_context_value(x: Any, **kwargs) -> int:
            context = kwargs.get("context")
            return context.value
        
        @operation(context=True)
        def get_context_name(x: Any, **kwargs) -> str:
            context = kwargs.get("context")
            return context.name
        
        par = parallel(
            context_aware_add,  # x + context.value
            get_context_value,  # context.value
            get_context_name    # context.name
        )
        
        result = await par.execute(5, context=test_context)
        assert result.is_ok()
        assert result.default_value(()) == (15, 10, "test")
    
    @pytest.mark.asyncio
    async def test_parallel_different_return_types(self):
        """Test parallel with operations returning different types."""
        @operation
        def to_list(x: int) -> List[int]:
            return [x, x+1, x+2]
        
        @operation
        def to_dict(x: int) -> Dict[str, int]:
            return {"value": x, "doubled": x*2}
        
        par = parallel(add_one, to_list, to_dict)
        result = await par.execute(5)
        assert result.is_ok()
        one, lst, dct = result.default_value((0, [], {}))
        assert one == 6
        assert lst == [5, 6, 7]
        assert dct == {"value": 5, "doubled": 10}


# Test fallback operation
class TestFallbackOperation:
    """Test suite for the fallback operation."""
    
    @pytest.mark.asyncio
    async def test_fallback_first_succeeds(self):
        """Test fallback when first operation succeeds."""
        fb = fallback(add_one, multiply_by_two, to_string)
        result = await fb.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 6
    
    @pytest.mark.asyncio
    async def test_fallback_first_fails(self):
        """Test fallback when first operation fails."""
        fb = fallback(failing_op, multiply_by_two, to_string)
        result = await fb.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 10
    
    @pytest.mark.asyncio
    async def test_fallback_multiple_failures(self):
        """Test fallback with multiple failures."""
        @operation
        def also_fails(x: Any) -> Any:
            raise RuntimeError("Also fails")
        
        fb = fallback(failing_op, also_fails, add_one)
        result = await fb.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 6
    
    @pytest.mark.asyncio
    async def test_fallback_all_fail(self):
        """Test fallback when all operations fail."""
        @operation
        def also_fails(x: Any) -> Any:
            raise RuntimeError("Also fails")
        
        fb = fallback(failing_op, also_fails)
        result = await fb.execute(5)
        assert result.is_error()
        assert isinstance(result.error, RuntimeError)
    
    @pytest.mark.asyncio
    async def test_fallback_empty(self):
        """Test fallback with no operations."""
        fb = fallback()
        result = await fb.execute(5)
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_fallback_single_operation(self):
        """Test fallback with single operation."""
        fb = fallback(add_one)
        result = await fb.execute(5)
        assert result.is_ok()
        assert result.default_value(0) == 6
    
    @pytest.mark.asyncio
    async def test_fallback_with_context(self, test_context):
        """Test fallback with context-aware operations."""
        @operation(context=True)
        def failing_context_op(x: Any, **kwargs) -> Any:
            context = kwargs.get("context")
            raise ValueError("Context op fails")
        
        fb = fallback(failing_context_op, context_aware_add)
        result = await fb.execute(5, context=test_context)
        assert result.is_ok()
        assert result.default_value(0) == 15
    
    @pytest.mark.asyncio
    async def test_fallback_different_types(self):
        """Test fallback with operations returning different types."""
        @operation
        def maybe_int(x: str) -> int:
            # Only works if x is numeric
            return int(x)
        
        @operation
        def string_length(x: str) -> int:
            return len(x)
        
        fb = fallback(maybe_int, string_length)
        
        # Test with numeric string
        result1 = await fb.execute("42")
        assert result1.is_ok()
        assert result1.default_value(0) == 42
        
        # Test with non-numeric string
        result2 = await fb.execute("hello")
        assert result2.is_ok()
        assert result2.default_value(0) == 5


# Test complex compositions
class TestComplexCompositions:
    """Test suite for complex composition scenarios."""
    
    @pytest.mark.asyncio
    async def test_nested_compositions(self):
        """Test compositions of compositions."""
        # Create sub-pipelines
        preprocess = pipe(add_one, multiply_by_two)
        postprocess = pipe(to_string, operation(lambda s: f"Result: {s}"))
        
        # Combine them
        full_pipeline = pipe(preprocess, postprocess)
        
        result = await full_pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value("") == "Result: 12"
    

    @pytest.mark.asyncio
    async def test_fallback_in_pipe(self):
        """Test using fallback within a pipe."""
        # Fallback that tries to parse as int, then returns 0
        parse_or_zero = fallback(
            operation(lambda x: int(x)),
            operation(lambda x: 0)
        )
        
        pipeline = pipe(
            parse_or_zero,
            add_one,
            to_string
        )
        
        # Test with valid int string
        result1 = await pipeline.execute("41")
        assert result1.is_ok()
        assert result1.default_value("") == "42"
        
        # Test with invalid int string
        result2 = await pipeline.execute("not a number")
        assert result2.is_ok()
        assert result2.default_value("") == "1"
    
    @pytest.mark.asyncio
    async def test_parallel_with_fallbacks(self):
        """Test parallel execution of fallback operations."""
        fb1 = fallback(failing_op, add_one)
        fb2 = fallback(multiply_by_two, failing_op)  # This succeeds first
        
        par = parallel(fb1, fb2)
        result = await par.execute(5)
        assert result.is_ok()
        assert result.default_value(()) == (6, 10)
    
    @pytest.mark.asyncio
    async def test_dynamic_pipeline_construction(self):
        """Test building pipelines dynamically based on data."""
        def build_pipeline(config: Dict[str, Any]) -> Operation:
            steps = []
            
            if config.get("add"):
                steps.append(add_one)
            if config.get("multiply"):
                steps.append(multiply_by_two)
            if config.get("stringify"):
                steps.append(to_string)
                
            return pipe(*steps) if steps else operation(lambda x: x)
        
        # Test different configurations
        config1 = {"add": True, "multiply": True, "stringify": True}
        pipeline1 = build_pipeline(config1)
        result1 = await pipeline1.execute(5)
        assert result1.default_value("") == "12"
        
        config2 = {"multiply": True}
        pipeline2 = build_pipeline(config2)
        result2 = await pipeline2.execute(5)
        assert result2.default_value(0) == 10
    
    @pytest.mark.asyncio
    async def test_context_flow_through_compositions(self, test_context):
        """Test context flowing through nested compositions."""
        # Create an operation that updates context but passes through the value
        @operation(context=True)
        def update_context_passthrough(x: Any, **kwargs) -> Any:
            """Update context and return the input value."""
            context = kwargs["context"]
            # Return the input value, context propagation happens via executor
            # when we return a BaseContext from update_context
            return x
        
        # First update context, then use it
        update_then_use = pipe(
            update_context,  # Returns new context
            operation(lambda ctx: 5),  # Ignore context, return original value
            context_aware_add  # Add with updated context
        )
        
        # Just use the original context
        just_use = context_aware_add
        
        parallel_with_context = parallel(
            update_then_use,
            just_use  # Uses original context
        )

        
        # Replace sequence with pipe
        pipeline = pipe(
            parallel_with_context,
            operation(lambda results: sum(results))  # Sum the parallel results
        )
        
        result = await pipeline.execute(5, context=test_context)
        assert result.is_ok()
        value = result.default_value(0)  # Updated to expect single value
        # First operation: context incremented to 11, then 5 + 11 = 16
        # Second operation: just add with original context -> 5 + 10 = 15
        # Sum = 31
        assert value == 31
    
    @pytest.mark.asyncio
    async def test_real_world_data_pipeline(self):
        """Test a realistic data processing pipeline."""
        # Simulate a data processing pipeline
        raw_data = {
            "users": [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35}
            ],
            "threshold": 28
        }
        
        # Operations for the pipeline
        extract_users = get("users")
        extract_threshold = get("threshold")
        
        @operation
        def filter_by_age(data: Tuple[List[Dict], int]) -> List[Dict]:
            users, threshold = data
            return [u for u in users if u["age"] >= threshold]
        
        @operation
        def extract_names(users: List[Dict]) -> List[str]:
            return [u["name"] for u in users]
        
        @operation
        def format_output(names: List[str]) -> str:
            return f"Users over threshold: {', '.join(names)}"
        
        # Build the pipeline
        pipeline = pipe(
            # First, extract both pieces of data in parallel
            parallel(extract_users, extract_threshold),
            # Then filter users by age
            filter_by_age,
            # Extract just the names
            extract_names,
            # Format for output
            format_output
        )
        
        result = await pipeline.execute(raw_data)
        assert result.is_ok()
        assert result.default_value("") == "Users over threshold: Alice, Charlie"


# Test edge cases and error scenarios
class TestEdgeCases:
    """Test suite for edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_deeply_nested_pipes(self):
        """Test very deep nesting of pipe operations."""
        # Create a deep pipeline programmatically
        depth = 50
        
        def create_deep_pipe(n: int) -> Operation:
            if n == 0:
                return add_one
            else:
                return pipe(
                    add_one,
                    create_deep_pipe(n - 1)
                )
        
        deep_pipe = create_deep_pipe(depth)
        result = await deep_pipe.execute(0)
        assert result.is_ok()
        assert result.default_value(0) == depth + 1
    
    @pytest.mark.asyncio
    async def test_error_in_context_update(self, test_context):
        """Test handling errors in context operations."""
        @operation(context=True)
        def failing_context_update(x: Any, **kwargs) -> Any:
            context = kwargs.get("context")
            raise ValueError("Context update failed")
        
        # Replace sequence with pipe
        pipeline = pipe(
            failing_context_update,
            context_aware_add  # Should not be reached
        )
        
        result = await pipeline.execute(5, context=test_context)
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_operations(self):
        """Test mixing sync and async operations."""
        @operation
        async def async_add_one(x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1
        
        # Mix sync and async operations
        pipeline = pipe(
            add_one,  # sync
            async_add_one,  # async
            multiply_by_two,  # sync
            to_string  # sync
        )
        
        result = await pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value("") == "14"
    
    @pytest.mark.asyncio
    async def test_operations_with_side_effects(self):
        """Test operations with side effects in different compositions."""
        side_effects = []
        
        @operation
        def record_value(x: int) -> int:
            side_effects.append(x)
            return x
        
        # Test pipe - should record each intermediate value
        side_effects.clear()
        pipeline = pipe(
            record_value,
            add_one,
            record_value,
            multiply_by_two,
            record_value
        )
        await pipeline.execute(5)
        assert side_effects == [5, 6, 12]
        
        # Test parallel - should only record initial value
        side_effects.clear()
        par = parallel(
            record_value,
            record_value,
            record_value
        )
        await par.execute(5)
        assert side_effects == [5, 5, 5]
    
    @pytest.mark.asyncio
    async def test_type_safety_in_compositions(self):
        """Test that type mismatches are handled properly."""
        # Operation expecting int but receiving string
        @operation
        def expects_int(x: int) -> int:
            return x / 2  # Will fail if x is string
        
        pipeline = pipe(
            to_string,  # Converts to string
            expects_int  # Expects int
        )
        
        result = await pipeline.execute(5)
        assert result.is_error()
        assert isinstance(result.error, TypeError)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_parallel(self):
        """Test parallel with many operations doesn't cause issues."""
        # Create many lightweight operations
        ops = [
            operation(lambda x, i=i: x + i)
            for i in range(100)
        ]
        
        par = parallel(*ops)
        result = await par.execute(0)
        assert result.is_ok()
        values = result.default_value(())
        assert len(values) == 100
        assert values[50] == 50
    
    @pytest.mark.asyncio
    async def test_recursive_operations_in_pipe(self):
        """Test recursive operation definitions in pipe."""
        
        def factorial(n: int) -> int:
            if n <= 1:
                return 1
            else:
                return n * factorial(n - 1)
        
        # This won't work directly as recursive composition,
        # but we can test operations that internally recurse
        pipeline = pipe(
            operation(factorial),
            to_string
        )
        
        result = await pipeline.execute(5)
        assert result.is_ok()
        assert result.default_value("") == "120"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])