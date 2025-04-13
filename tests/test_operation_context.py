import asyncio
import pytest
from typing import Dict, List, Any, Optional, Tuple, Type
from pydantic import BaseModel, Field
from expression import Result

# Import the Operation class and utilities
from fp_ops.operator import (
    Operation,
    identity,
    constant
)
from fp_ops.flow import branch, attempt, fail
from fp_ops.composition import gather_operations
from fp_ops.decorators import operation
from fp_ops.context import BaseContext
from fp_ops.placeholder import _

# -----------------------------------------------
# CONTEXT MODEL DEFINITIONS
# -----------------------------------------------

class TestContext(BaseContext):
    """Basic test context"""
    value: str = "default"
    counter: int = 0
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        """Merge another context into this one"""
        if isinstance(other, TestContext):
            return TestContext(
                value=other.value,
                counter=self.counter + other.counter,
                metadata={**self.metadata, **other.metadata}
            )
        return super().merge(other)

class BrowserContext(BaseContext):
    """Context for browser operations"""
    url: Optional[str] = None
    title: Optional[str] = None
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        """Merge another context into this one"""
        if isinstance(other, BrowserContext):
            return BrowserContext(
                url=other.url or self.url,
                title=other.title or self.title,
                metadata={**self.metadata, **other.metadata}
            )
        return super().merge(other)

class DataContext(BaseContext):
    """Context for data operations"""
    data: List[Dict[str, Any]] = Field(default_factory=list)
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        """Merge another context into this one"""
        if isinstance(other, DataContext):
            # Combine data lists
            return DataContext(
                data=self.data + other.data,
                metadata={**self.metadata, **other.metadata}
            )
        return super().merge(other)

class AppContext(BaseContext):
    """Combined context for application"""
    browser: Optional[BrowserContext] = None
    data: Optional[DataContext] = None
    app_version: str = "1.0"
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        """Merge another context into this one"""
        if isinstance(other, AppContext):
            # Merge browser and data contexts if present
            browser = self.browser
            if self.browser and other.browser:
                browser = self.browser.merge(other.browser)
            elif other.browser:
                browser = other.browser
                
            data = self.data
            if self.data and other.data:
                data = self.data.merge(other.data)
            elif other.data:
                data = other.data
                
            return AppContext(
                browser=browser,
                data=data,
                app_version=other.app_version or self.app_version,
                metadata={**self.metadata, **other.metadata}
            )
        return super().merge(other)

# -----------------------------------------------
# FIXTURES
# -----------------------------------------------

@pytest.fixture
def event_loop():
    """Create an event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_context():
    """Create a basic test context"""
    return TestContext(value="test", counter=1)

@pytest.fixture
def browser_context():
    """Create a browser context"""
    return BrowserContext(url="https://example.com", title="Example")

@pytest.fixture
def data_context():
    """Create a data context"""
    return DataContext(data=[{"id": 1, "name": "Item 1"}])

@pytest.fixture
def app_context(browser_context, data_context):
    """Create a combined app context"""
    return AppContext(browser=browser_context, data=data_context, app_version="1.0")

@pytest.fixture
def context_aware_op():
    """Create an operation that requires context"""
    @operation(context=True, context_type=TestContext)
    async def increment_counter(**kwargs):
        context = kwargs["context"]
        context.counter += 1
        return context
    
    return increment_counter

@pytest.fixture
def browser_op():
    """Create an operation that uses browser context"""
    @operation(context=True, context_type=AppContext)  # Change to AppContext
    async def navigate(url: str, **kwargs):
        context = kwargs["context"]
        # Update the browser in the AppContext
        if context.browser:
            updated_browser = BrowserContext(url=url, title=context.browser.title)
            return AppContext(
                browser=updated_browser,
                data=context.data,
                app_version=context.app_version
            )
        else:
            # Create a new browser context if none exists
            new_browser = BrowserContext(url=url)
            return AppContext(
                browser=new_browser,
                data=context.data,
                app_version=context.app_version
            )
    
    return navigate

# Fix the data_op test fixture to properly handle context
@pytest.fixture
def data_op():
    """Create an operation that uses data context"""
    @operation(context=True, context_type=AppContext)  # Change to AppContext
    async def add_item(item: Dict[str, Any], **kwargs):
        context = kwargs["context"]
        # Add an item to the data context within AppContext
        if context.data:
            new_data = DataContext(data=context.data.data + [item])
            return AppContext(
                browser=context.browser,
                data=new_data,
                app_version=context.app_version
            )
        else:
            # Create a new data context if none exists
            new_data = DataContext(data=[item])
            return AppContext(
                browser=context.browser,
                data=new_data,
                app_version=context.app_version
            )
    
    return add_item

@pytest.fixture
def app_op():
    """Create an operation that uses app context"""
    @operation(context=True, context_type=AppContext)
    async def update_app(**kwargs):
        context = kwargs["context"]
        # Update the app version
        return AppContext(
            browser=context.browser, 
            data=context.data,
            app_version="2.0"
        )
    
    return update_app

@pytest.fixture
def context_transform_op():
    """Operation that transforms a value but preserves context"""
    @operation(context=True)
    async def double_value(value: int, **kwargs):
        # Simply double the value, context passes through
        return value * 2
    
    return double_value

@pytest.fixture
def context_factory():
    """Function that creates a context"""
    async def create_context():
        await asyncio.sleep(0.01)  # Simulate async initialization
        return TestContext(value="factory", counter=5)
    
    return create_context

# -----------------------------------------------
# BASIC CONTEXT TESTS
# -----------------------------------------------

class TestBasicContext:
    """Test basic context functionality."""
    
    @pytest.mark.asyncio
    async def test_context_creation(self, test_context):
        """Test creating and initializing a context."""
        # Create an operation that initializes context
        init_context = Operation.with_context(lambda: test_context, context_type=TestContext)
        
        # Execute and check result
        result = await init_context()
        
        assert result.is_ok()
        context = result.default_value(None)
        assert isinstance(context, TestContext)
        assert context.value == "test"
        assert context.counter == 1
    
    @pytest.mark.asyncio
    async def test_context_factory(self, context_factory):
        """Test creating context with a factory function."""
        # Create an operation that initializes context with a factory
        init_context = Operation.with_context(context_factory, context_type=TestContext)
        
        # Execute and check result
        result = await init_context()
        
        assert result.is_ok()
        context = result.default_value(None)
        assert isinstance(context, TestContext)
        assert context.value == "factory"
        
        assert context.counter == 5
    
    @pytest.mark.asyncio
    async def test_context_chaining_with_factory(self, context_factory):
        """Test chaining operations with a context factory."""
        # Create an operation that initializes context with a factory
        init_context = Operation.with_context(context_factory, context_type=TestContext)
        
        # Create operations that use and modify the context
        @operation(context=True, context_type=TestContext)
        async def increment_counter(**kwargs):
            context = kwargs["context"]
            new_context = TestContext(value=context.value, counter=context.counter + 1)
            return new_context
        
        @operation(context=True, context_type=TestContext)
        async def get_context_values(**kwargs):
            context = kwargs["context"]
            return {"value": context.value, "counter": context.counter}
        
        # Chain operations: initialize context -> modify context -> read context
        pipeline = init_context >> increment_counter >> increment_counter >> get_context_values
        
        # Execute the pipeline
        result = await pipeline()
        
        # Verify results
        assert result.is_ok()
        values = result.default_value(None)
        assert values["value"] == "factory"
        assert values["counter"] == 7  # Initial 5 + increment 1
    
    @pytest.mark.asyncio
    async def test_context_chaining_with_initial_context(self, test_context):
        """Test chaining operations with an initial context."""
        # Create operations that use and modify the context
        @operation(context=True, context_type=TestContext)
        async def append_to_value(**kwargs):
            context = kwargs["context"]
            new_context = TestContext(value=context.value + "_modified", counter=context.counter)
            return new_context
        
        @operation(context=True, context_type=TestContext)
        async def double_counter(**kwargs):
            context = kwargs["context"]
            new_context = TestContext(value=context.value, counter=context.counter * 2)
            return new_context
        
        @operation(context=True, context_type=TestContext)
        async def get_context_state(**kwargs):
            context = kwargs["context"]
            return f"{context.value}:{context.counter}"
        
        # Chain operations: modify value -> modify counter -> read context
        pipeline = append_to_value >> double_counter >> get_context_state
        
        # Execute with initial context
        result = await pipeline(context=test_context)
        
        # Verify results
        assert result.is_ok()
        state = result.default_value(None)
        assert state == "test_modified:2"  # Modified value and doubled counter (1*2)
    @pytest.mark.asyncio
    async def test_context_required_missing(self, context_aware_op):
        """Test operation that requires context but none is provided."""
        # Execute without providing context
        result = await context_aware_op()
        
        # Should fail because context is required
        assert result.is_error()
        assert "requires a context" in str(result.error)
    
    @pytest.mark.asyncio
    async def test_context_type_validation(self, data_context):
        """Test context type validation."""
        # Create an operation that requires TestContext
        @operation(context=True, context_type=TestContext)
        async def requires_test_context(**kwargs):
            context = kwargs["context"]
            return context.value
        
        # Execute with wrong context type
        result = await requires_test_context(context=data_context)
        
        # Should handle the type mismatch by attempting conversion
        assert result.is_error()
        error_msg = str(result.error)
        assert "Invalid context" in error_msg
        assert "Missing fields" in error_msg or "Could not convert" in error_msg
# -----------------------------------------------
# CONTEXT PROPAGATION TESTS
# -----------------------------------------------

class TestContextPropagation:
    """Test context propagation through operation chains."""
    
    @pytest.mark.asyncio
    async def test_basic_propagation(self, test_context, context_aware_op):
        """Test basic context propagation through a chain."""
        # Create a second operation
        @operation(context=True, context_type=TestContext)
        async def get_counter(**kwargs):
            context = kwargs["context"]
            return context.counter
        
        # Create a pipeline that modifies then reads context
        pipeline = context_aware_op >> get_counter
        
        # Execute with context
        result = await pipeline(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None) == 2  # Original 1 + increment
    
    @pytest.mark.asyncio
    async def test_context_update_propagation(self, test_context):
        """Test that context updates propagate through the chain."""
        # Create operations that update context in sequence
        @operation(context=True, context_type=TestContext)
        async def update_value(**kwargs):
            context = kwargs["context"]
            return TestContext(value="updated", counter=context.counter)
        
        @operation(context=True, context_type=TestContext)
        async def check_context(**kwargs):
            context = kwargs["context"]
            return {
                "value": context.value,
                "counter": context.counter
            }
        
        # Create the pipeline
        pipeline = update_value >> check_context
        
        # Execute with context
        result = await pipeline(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None)["value"] == "updated"
        assert result.default_value(None)["counter"] == 1  # Original value is preserved
    
    @pytest.mark.asyncio
    async def test_operation_returns_context(self, test_context):
        """Test operations that return context objects."""
        # Create operations that return context
        @operation(context=True, context_type=TestContext)
        async def modify_and_return(**kwargs):
            context = kwargs["context"]
            # Return the context directly
            return TestContext(value="modified", counter=context.counter + 1)
        
        @operation(context=True, context_type=TestContext)
        async def check_context(**kwargs):
            context = kwargs["context"]
            return {
                "value": context.value,
                "counter": context.counter
            }
        
        # Chain them
        pipeline = modify_and_return >> check_context
        
        # Execute
        result = await pipeline(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None)["value"] == "modified"
        assert result.default_value(None)["counter"] == 2  # Original 1 + increment
    
    @pytest.mark.asyncio
    async def test_mixed_return_values(self, test_context):
        """Test mixing operations that return context and regular values."""
        @operation(context=True, context_type=TestContext)
        async def return_context(**kwargs):
            context = kwargs["context"]
            return TestContext(value="from op", counter=context.counter + 1)
        
        @operation(context=True, context_type=TestContext)
        async def return_value(**kwargs):
            context = kwargs["context"]
            return f"Value from context: {context.value}-{context.counter}"
        
        # Chain them
        pipeline = return_context >> return_value
        
        # Execute
        result = await pipeline(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None) == "Value from context: from op-2"  # Original 1 + increment

# -----------------------------------------------
# CONTEXT WITH OPERATORS TESTS
# -----------------------------------------------

class TestContextWithOperators:
    """Test context behavior with different operators."""
    
    @pytest.mark.asyncio
    async def test_context_with_parallel(self, test_context):
        """Test context with parallel operations."""
        # Create operations that use context
        @operation(context=True, context_type=TestContext)
        async def get_value(**kwargs):
            context = kwargs["context"]
            return context.value
        
        @operation(context=True, context_type=TestContext)
        async def get_counter(**kwargs):
            context = kwargs["context"]
            return context.counter
        
        # Chain with parallel operations
        parallel_op = get_value & get_counter
        
        # Execute
        result = await parallel_op(context=test_context)
        
        assert result.is_ok()
        value, counter = result.default_value(None)
        assert value == "test"
        assert counter == 1
    
    @pytest.mark.asyncio
    async def test_context_with_fallback(self, test_context):
        """Test context with fallback operations."""
        # Create operations that use context
        @operation(context=True, context_type=TestContext)
        async def fail_with_context(**kwargs):
            context = kwargs["context"]
            raise ValueError(f"Error with context: {context.value}")
        
        @operation(context=True, context_type=TestContext)
        async def backup_op(**kwargs):
            context = kwargs["context"]
            return f"Backup with context: {context.value}"
        
        # Chain with fallback
        fallback_op = fail_with_context | backup_op
        
        # Execute
        result = await fallback_op(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None) == "Backup with context: test"
    
    @pytest.mark.asyncio
    async def test_context_with_map(self, test_context):
        """Test context with map transformation."""
        # Create an operation that uses context
        @operation(context=True, context_type=TestContext)
        async def get_value(**kwargs):
            context = kwargs["context"]
            return context.value
        
        # Map the result
        mapped_op = get_value.map(lambda v: f"Mapped: {v}")
        
        # Execute
        result = await mapped_op(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None) == "Mapped: test"
    
    @pytest.mark.asyncio
    async def test_context_with_bind(self, test_context):
        """Test context with bind transformation."""
        # Create operations that use context
        @operation(context=True, context_type=TestContext)
        async def get_counter(**kwargs):
            context = kwargs["context"]
            return context.counter
        
        # Create a bind function
        @operation(context=True, context_type=TestContext)
        async def multiply_by_value(counter, **kwargs):
            context = kwargs["context"]
            return counter * (len(context.value) if context.value else 1)
        
        # Bind them
        bound_op = get_counter.bind(lambda c: multiply_by_value(c))
        
        # Execute
        result = await bound_op(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None) == 4  # counter(1) * len("test")(4)
    
    @pytest.mark.asyncio
    async def test_context_with_filter(self, test_context):
        """Test context with filter."""
        # Create an operation that uses context
        @operation(context=True, context_type=TestContext)
        async def get_counter(**kwargs):
            context = kwargs["context"]
            return context.counter
        
        # Create a filter that passes
        filtered_pass = get_counter.filter(lambda c: c > 0, "Counter must be positive")
        
        # Create a filter that fails
        filtered_fail = get_counter.filter(lambda c: c > 10, "Counter must be > 10")
        
        # Execute passing filter
        result_pass = await filtered_pass(context=test_context)
        assert result_pass.is_ok()
        assert result_pass.default_value(None) == 1
        
        # Execute failing filter
        result_fail = await filtered_fail(context=test_context)
        assert result_fail.is_error()
        assert "Counter must be > 10" in str(result_fail.error)

# -----------------------------------------------
# CONTEXT ERROR HANDLING TESTS
# -----------------------------------------------

class TestContextErrorHandling:
    """Test context behavior with error handling."""
    
    @pytest.mark.asyncio
    async def test_context_with_catch(self, test_context):
        """Test context with catch error handler."""
        # Create an operation that fails
        @operation(context=True, context_type=TestContext)
        async def failing_op(**kwargs):
            context = kwargs["context"]
            raise ValueError(f"Error with context: {context.value}")
        
        # Create an error handler that uses context
        def error_handler(error):
            return f"Caught: {str(error)}"
        
        # Apply catch
        with_recovery = failing_op.catch(error_handler)
        
        # Execute
        result = await with_recovery(context=test_context)
        
        assert result.is_ok()
        assert "Caught: Error with context: test" in result.default_value(None)
    
    @pytest.mark.asyncio
    async def test_context_with_default_value(self, test_context):
        """Test context with default_value."""
        # Create an operation that fails
        @operation(context=True, context_type=TestContext)
        async def failing_op(**kwargs):
            context = kwargs["context"]
            raise ValueError(f"Error with context: {context.value}")
        
        # Apply default_value
        with_default = failing_op.default_value("Default result")
        
        # Execute
        result = await with_default(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None) == "Default result"
    
    @pytest.mark.asyncio
    async def test_context_with_retry(self, test_context):
        """Test context with retry."""
        # Create a flaky operation that succeeds on second attempt
        call_count = 0
        
        @operation(context=True, context_type=TestContext)
        async def flaky_op(**kwargs):
            nonlocal call_count
            context = kwargs["context"]
            call_count += 1
            
            if call_count == 1:
                raise ConnectionError(f"Failed with {context.value}")
            
            return f"Success on attempt {call_count} with {context.value}"
        
        # Apply retry
        with_retry = flaky_op.retry(attempts=3, delay=0.01)
        
        # Execute
        result = await with_retry(context=test_context)
        
        assert result.is_ok()
        assert "Success on attempt 2" in result.default_value(None)
        assert "with test" in result.default_value(None)
    
    @pytest.mark.asyncio
    async def test_context_with_tap(self, test_context):
        """Test context with tap for side effects."""
        side_effect_results = []
        
        @operation(context=True, context_type=TestContext)
        async def get_value(**kwargs):
            context = kwargs["context"]
            return context.value
        
        # Apply tap
        with_tap = get_value.tap(
            lambda value: side_effect_results.append(f"Tapped: {value}")
        )
        
        # Execute
        result = await with_tap(context=test_context)
        
        assert result.is_ok()
        assert result.default_value(None) == "test"
        assert len(side_effect_results) == 1
        assert side_effect_results[0] == "Tapped: test"

# -----------------------------------------------
# COMPLEX CONTEXT SCENARIOS TESTS
# -----------------------------------------------

class TestComplexContextScenarios:
    """Test more complex context usage patterns."""
    
    @pytest.mark.asyncio
    async def test_context_hierarchy(self, app_context):
        """Test operations that work with complex nested contexts."""
        # Create operations that use different parts of the context
        @operation(context=True, context_type=AppContext)
        async def get_browser_url(**kwargs):
            context = kwargs["context"]
            if context.browser:
                return context.browser.url
            return None
        
        @operation(context=True, context_type=AppContext)
        async def get_data_count(**kwargs):
            context = kwargs["context"]
            if context.data:
                return len(context.data.data)
            return 0
        
        # Modified get_app_details to not call other operations directly
        @operation(context=True, context_type=AppContext)
        async def get_app_details(**kwargs):
            context = kwargs["context"]
            return {
                "url": context.browser.url if context.browser else None,
                "data_count": len(context.data.data) if context.data else 0,
                "version": context.app_version
            }
        
        # Execute
        result = await get_app_details(context=app_context)
        
        assert result.is_ok()
        details = result.default_value(None)
        assert details["url"] == "https://example.com"
        assert details["data_count"] == 1
        assert details["version"] == "1.0"
        
    @pytest.mark.asyncio
    async def test_context_merging(self, browser_context, data_context):
        """Test merging different context types."""
        # Create operations that return different context types
        @operation(context=True, context_type=BrowserContext)
        async def update_browser(**kwargs):
            context = kwargs["context"]
            return BrowserContext(url=context.url, title="Updated Title")
        
        @operation(context=True, context_type=DataContext)
        async def add_data_item(**kwargs):
            context = kwargs["context"]
            new_data = context.data + [{"id": 2, "name": "Item 2"}]
            return DataContext(data=new_data)
        
        # Create an app context factory
        async def create_app_context():
            # Start with an empty app context
            return AppContext(app_version="1.0")
        
        # Create a pipeline that uses different contexts
        init_app = Operation.with_context(create_app_context, context_type=AppContext)
        
        # Execute and check the result
        app_result = await init_app()
        assert app_result.is_ok()
        
        # Add browser context
        browser_result = await update_browser(context=browser_context)
        assert browser_result.is_ok()
        
        # Add data context
        data_result = await add_data_item(context=data_context)
        assert data_result.is_ok()
        
        # Manually combine contexts
        combined_context = AppContext(
            browser=browser_result.default_value(None),
            data=data_result.default_value(None),
            app_version=app_result.default_value(None).app_version
        )
        
        # Verify combined context has all parts
        assert combined_context.browser.title == "Updated Title"
        assert len(combined_context.data.data) == 2
        assert combined_context.app_version == "1.0"
    
    @pytest.mark.asyncio
    async def test_context_in_complex_chain(self, app_context, browser_op, data_op, app_op):
        """Test context in a complex operation chain."""
        # Create operations that get information
        @operation(context=True, context_type=AppContext)
        async def get_summary(**kwargs):
            context = kwargs["context"]
            return {
                "url": context.browser.url if context.browser else None,
                "data_count": len(context.data.data) if context.data else 0,
                "version": context.app_version
            }
        
        # Create a pipeline with multiple transformations
        pipeline = (
            app_op 
            >> browser_op("https://new-example.com") 
            >> data_op({"id": 2, "name": "New Item"})
            >> data_op({"id": 3, "name": "New Item 2"})
            >> get_summary
        )
        
        # Execute with app context
        result = await pipeline(context=app_context)
        
        assert result.is_ok()
        summary = result.default_value(None)
        assert summary["url"] == "https://new-example.com"
        assert summary["data_count"] == 3  # Original item + new items
        assert summary["version"] == "2.0"  # Updated by app_op
    
    @pytest.mark.asyncio
    async def test_gather_with_context(self, app_context):
        """Test gathering multiple operations with shared context."""
        # Create operations that use different parts of the context
        @operation(context=True, context_type=AppContext)
        async def get_url(**kwargs):
            context = kwargs["context"]
            return context.browser.url if context.browser else None
        
        @operation(context=True, context_type=AppContext)
        async def get_data(**kwargs):
            context = kwargs["context"]
            return context.data.data if context.data else []
        
        @operation(context=True, context_type=AppContext)
        async def get_version(**kwargs):
            context = kwargs["context"]
            return context.app_version
        
        # Gather operations
        results = await gather_operations(
            get_url, get_data, get_version,
            kwargs={"context": app_context}
        )
        
        # Check results
        assert len(results) == 3
        assert all(r.is_ok() for r in results)
        assert results[0].default_value(None) == "https://example.com"
        assert len(results[1].default_value(None)) == 1
        assert results[2].default_value(None) == "1.0"
    
    @pytest.mark.asyncio
    async def test_context_with_placeholders(self, test_context):
        """Test using placeholders with context-aware operations."""
        # Create operations that use placeholders and context
        @operation(context=True, context_type=TestContext)
        async def multiply_by_counter(value: int, **kwargs):
            context = kwargs["context"]
            return value * context.counter
        
        @operation(context=True, context_type=TestContext)
        async def add_to_result(value: int, **kwargs):
            context = kwargs["context"]
            return value + len(context.value)
        
        # Create a pipeline with placeholders
        pipeline = constant(5) >> multiply_by_counter(_) >> add_to_result(_)
        
        # Execute with context
        result = await pipeline(context=test_context)
        
        assert result.is_ok()
        # 5 * counter(1) + len("test")(4) = 9
        assert result.default_value(None) == 9
    
    @pytest.mark.asyncio
    async def test_branching_with_context(self, test_context):
        """Test branching based on context values."""
        # Create a condition that checks context
        def check_counter_positive(**kwargs):
            context = kwargs.get("context")
            return context and context.counter > 0
        
        # Create operations for each branch
        @operation(context=True, context_type=TestContext)
        async def positive_branch(**kwargs):
            context = kwargs["context"]
            return f"Positive counter: {context.counter}"
        
        @operation(context=True, context_type=TestContext)
        async def negative_branch(**kwargs):
            context = kwargs["context"]
            return f"Non-positive counter: {context.counter}"
        
        # Create the branch operation
        branch_op = branch(check_counter_positive, positive_branch, negative_branch)
        
        # Test with positive counter
        positive_result = await branch_op(context=test_context)
        assert positive_result.is_ok()
        assert positive_result.default_value(None) == "Positive counter: 1"
        
        # Test with non-positive counter
        negative_context = TestContext(value="test", counter=0)
        negative_result = await branch_op(context=negative_context)
        assert negative_result.is_ok()
        assert negative_result.default_value(None) == "Non-positive counter: 0"