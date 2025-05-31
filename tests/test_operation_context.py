import asyncio
import pytest
from typing import Dict, List, Any, Optional, Tuple, Type
from pydantic import BaseModel, Field
from expression import Result

from fp_ops.operator import (
    Operation,
    constant,
)
from fp_ops.flow import branch, attempt, fail
from fp_ops.collections import map, filter
from fp_ops.decorators import operation
from fp_ops.context import BaseContext
from fp_ops.placeholder import _

class TestContext(BaseContext):
    value: str = "default"
    counter: int = 0
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        if isinstance(other, TestContext):
            return TestContext(
                value=other.value,
                counter=self.counter + other.counter,
                metadata={**self.metadata, **other.metadata}
            )
        return super().merge(other)

class BrowserContext(BaseContext):
    url: Optional[str] = None
    title: Optional[str] = None
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        if isinstance(other, BrowserContext):
            return BrowserContext(
                url=other.url or self.url,
                title=other.title or self.title,
                metadata={**self.metadata, **other.metadata}
            )
        return super().merge(other)

class DataContext(BaseContext):
    data: List[Dict[str, Any]] = Field(default_factory=list)
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        if isinstance(other, DataContext):
            return DataContext(
                data=self.data + other.data,
                metadata={**self.metadata, **other.metadata}
            )
        return super().merge(other)

class AppContext(BaseContext):
    browser: Optional[BrowserContext] = None
    data: Optional[DataContext] = None
    app_version: str = "1.0"
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        if isinstance(other, AppContext):
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

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_context():
    return TestContext(value="test", counter=1)

@pytest.fixture
def browser_context():
    return BrowserContext(url="https://example.com", title="Example")

@pytest.fixture
def data_context():
    return DataContext(data=[{"id": 1, "name": "Item 1"}])

@pytest.fixture
def app_context(browser_context, data_context):
    return AppContext(browser=browser_context, data=data_context, app_version="1.0")

@pytest.fixture
def context_aware_op():
    @operation(context=True, context_type=TestContext)
    async def increment_counter(**kwargs):
        context = kwargs["context"]
        context.counter += 1
        return context
    
    return increment_counter

@pytest.fixture
def browser_op():
    @operation(context=True, context_type=AppContext)
    async def navigate(url: str, **kwargs):
        context = kwargs["context"]
        if context.browser:
            updated_browser = BrowserContext(url=url, title=context.browser.title)
            return AppContext(
                browser=updated_browser,
                data=context.data,
                app_version=context.app_version
            )
        else:
            new_browser = BrowserContext(url=url)
            return AppContext(
                browser=new_browser,
                data=context.data,
                app_version=context.app_version
            )
    
    return navigate

@pytest.fixture
def data_op():
    @operation(context=True, context_type=AppContext)
    async def add_item(item: Dict[str, Any], **kwargs):
        context = kwargs["context"]
        if context.data:
            new_data = DataContext(data=context.data.data + [item])
            return AppContext(
                browser=context.browser,
                data=new_data,
                app_version=context.app_version
            )
        else:
            new_data = DataContext(data=[item])
            return AppContext(
                browser=context.browser,
                data=new_data,
                app_version=context.app_version
            )
    
    return add_item

@pytest.fixture
def app_op():
    @operation(context=True, context_type=AppContext)
    async def update_app(**kwargs):
        context = kwargs["context"]
        return AppContext(
            browser=context.browser, 
            data=context.data,
            app_version="2.0"
        )
    
    return update_app

@pytest.fixture
def context_transform_op():
    @operation(context=True)
    async def double_value(value: int, **kwargs):
        return value * 2
    
    return double_value

@pytest.fixture
def context_factory():
    async def create_context():
        await asyncio.sleep(0.01)
        return TestContext(value="factory", counter=5)
    
    return create_context

class TestBasicContext:
    
    @pytest.mark.asyncio
    async def test_context_creation(self, test_context):
        init_context = Operation.with_context(lambda: test_context, context_type=TestContext)
        result = await init_context()
        assert result.is_ok()
        context = result.default_value(None)
        assert isinstance(context, TestContext)
        assert context.value == "test"
        assert context.counter == 1
    
    @pytest.mark.asyncio
    async def test_context_factory(self, context_factory):
        init_context = Operation.with_context(context_factory, context_type=TestContext)
        result = await init_context()
        assert result.is_ok()
        context = result.default_value(None)
        assert isinstance(context, TestContext)
        assert context.value == "factory"
        assert context.counter == 5
    
    @pytest.mark.asyncio
    async def test_context_chaining_with_factory(self, context_factory):
        init_context = Operation.with_context(context_factory, context_type=TestContext)
        @operation(context=True, context_type=TestContext)
        async def increment_counter(**kwargs):
            context = kwargs["context"]
            new_context = TestContext(value=context.value, counter=context.counter + 1)
            return new_context
        
        @operation(context=True, context_type=TestContext)
        async def get_context_values(**kwargs):
            context = kwargs["context"]
            return {"value": context.value, "counter": context.counter}
        
        pipeline = init_context >> increment_counter >> increment_counter >> get_context_values
        result = await pipeline()
        assert result.is_ok()
        values = result.default_value(None)
        assert values["value"] == "factory"
        assert values["counter"] == 7
    
    @pytest.mark.asyncio
    async def test_context_chaining_with_initial_context(self, test_context):
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
        
        pipeline = append_to_value >> double_counter >> get_context_state
        result = await pipeline(context=test_context)
        assert result.is_ok()
        state = result.default_value(None)
        assert state == "test_modified:2"
    @pytest.mark.asyncio
    async def test_context_required_missing(self, context_aware_op):
        result = await context_aware_op()
        assert result.is_error()
        assert "requires a context" in str(result.error)
    
    @pytest.mark.asyncio
    async def test_context_type_validation(self, data_context):
        @operation(context=True, context_type=TestContext)
        async def requires_test_context(**kwargs):
            context = kwargs["context"]
            return context.value
        
        result = await requires_test_context(context=data_context)
        assert result.is_error()
        error_msg = str(result.error)
        assert "Invalid context" in error_msg
        assert "Missing fields" in error_msg or "Could not convert" in error_msg

class TestContextPropagation:
    
    @pytest.mark.asyncio
    async def test_basic_propagation(self, test_context, context_aware_op):
        @operation(context=True, context_type=TestContext)
        async def get_counter(**kwargs):
            context = kwargs["context"]
            return context.counter
        
        pipeline = context_aware_op >> get_counter
        result = await pipeline(context=test_context)
        assert result.is_ok()
        assert result.default_value(None) == 2
    
    @pytest.mark.asyncio
    async def test_context_update_propagation(self, test_context):
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
        
        pipeline = update_value >> check_context
        result = await pipeline(context=test_context)
        assert result.is_ok()
        assert result.default_value(None)["value"] == "updated"
        assert result.default_value(None)["counter"] == 1
    
    @pytest.mark.asyncio
    async def test_operation_returns_context(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def modify_and_return(**kwargs):
            context = kwargs["context"]
            return TestContext(value="modified", counter=context.counter + 1)
        
        @operation(context=True, context_type=TestContext)
        async def check_context(**kwargs):
            context = kwargs["context"]
            return {
                "value": context.value,
                "counter": context.counter
            }
        
        pipeline = modify_and_return >> check_context
        result = await pipeline(context=test_context)
        assert result.is_ok()
        assert result.default_value(None)["value"] == "modified"
        assert result.default_value(None)["counter"] == 2
    
    @pytest.mark.asyncio
    async def test_mixed_return_values(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def return_context(**kwargs):
            context = kwargs["context"]
            return TestContext(value="from op", counter=context.counter + 1)
        
        @operation(context=True, context_type=TestContext)
        async def return_value(**kwargs):
            context = kwargs["context"]
            return f"Value from context: {context.value}-{context.counter}"
        
        pipeline = return_context >> return_value
        result = await pipeline(context=test_context)
        assert result.is_ok()
        assert result.default_value(None) == "Value from context: from op-2"

class TestContextWithOperators:
    
    @pytest.mark.asyncio
    async def test_context_with_parallel(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def get_value(**kwargs):
            context = kwargs["context"]
            return context.value
        
        @operation(context=True, context_type=TestContext)
        async def get_counter(**kwargs):
            context = kwargs["context"]
            return context.counter
        
        parallel_op = get_value & get_counter
        result = await parallel_op(context=test_context)
        assert result.is_ok()
        value, counter = result.default_value(None)
        assert value == "test"
        assert counter == 1
    
    @pytest.mark.asyncio
    async def test_context_with_fallback(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def fail_with_context(**kwargs):
            context = kwargs["context"]
            raise ValueError(f"Error with context: {context.value}")
        
        @operation(context=True, context_type=TestContext)
        async def backup_op(**kwargs):
            context = kwargs["context"]
            return f"Backup with context: {context.value}"
        
        fallback_op = fail_with_context | backup_op
        result = await fallback_op(context=test_context)
        assert result.is_ok()
        assert result.default_value(None) == "Backup with context: test"
    

    @pytest.mark.asyncio
    async def test_context_with_bind(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def get_counter(**kwargs):
            context = kwargs["context"]
            return context.counter
        
        @operation(context=True, context_type=TestContext)
        async def multiply_by_value(counter, **kwargs):
            context = kwargs["context"]
            return counter * (len(context.value) if context.value else 1)
        
        bound_op = get_counter.bind(lambda c: multiply_by_value(c))
        result = await bound_op(context=test_context)
        assert result.is_ok()
        assert result.default_value(None) == 4


class TestContextErrorHandling:
    
    @pytest.mark.asyncio
    async def test_context_with_catch(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def failing_op(**kwargs):
            context = kwargs["context"]
            raise ValueError(f"Error with context: {context.value}")
        
        def error_handler(error):
            return f"Caught: {str(error)}"
        
        with_recovery = failing_op.catch(error_handler)
        result = await with_recovery(context=test_context)
        assert result.is_ok()
        assert "Caught: Error with context: test" in result.default_value(None)
    
    @pytest.mark.asyncio
    async def test_context_with_default_value(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def failing_op(**kwargs):
            context = kwargs["context"]
            raise ValueError(f"Error with context: {context.value}")
        
        with_default = failing_op.default_value("Default result")
        result = await with_default(context=test_context)
        assert result.is_ok()
        assert result.default_value(None) == "Default result"
    
    @pytest.mark.asyncio
    async def test_context_with_retry(self, test_context):
        call_count = 0
        
        @operation(context=True, context_type=TestContext)
        async def flaky_op(**kwargs):
            nonlocal call_count
            context = kwargs["context"]
            call_count += 1
            
            if call_count == 1:
                raise ConnectionError(f"Failed with {context.value}")
            
            return f"Success on attempt {call_count} with {context.value}"
        
        with_retry = flaky_op.retry(attempts=3, delay=0.01)
        result = await with_retry(context=test_context)
        assert result.is_ok()
        assert "Success on attempt 2" in result.default_value(None)
        assert "with test" in result.default_value(None)
    
    @pytest.mark.asyncio
    async def test_context_with_tap(self, test_context):
        side_effect_results = []
        
        @operation(context=True, context_type=TestContext)
        async def get_value(**kwargs):
            context = kwargs["context"]
            return context.value
        
        with_tap = get_value.tap(
            lambda value: side_effect_results.append(f"Tapped: {value}")
        )
        result = await with_tap(context=test_context)
        assert result.is_ok()
        assert result.default_value(None) == "test"
        assert len(side_effect_results) == 1
        assert side_effect_results[0] == "Tapped: test"

class TestComplexContextScenarios:
    
    @pytest.mark.asyncio
    async def test_context_hierarchy(self, app_context):
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
        
        @operation(context=True, context_type=AppContext)
        async def get_app_details(**kwargs):
            context = kwargs["context"]
            return {
                "url": context.browser.url if context.browser else None,
                "data_count": len(context.data.data) if context.data else 0,
                "version": context.app_version
            }
        
        result = await get_app_details(context=app_context)
        assert result.is_ok()
        details = result.default_value(None)
        assert details["url"] == "https://example.com"
        assert details["data_count"] == 1
        assert details["version"] == "1.0"
        
    @pytest.mark.asyncio
    async def test_context_merging(self, browser_context, data_context):
        @operation(context=True, context_type=BrowserContext)
        async def update_browser(**kwargs):
            context = kwargs["context"]
            return BrowserContext(url=context.url, title="Updated Title")
        
        @operation(context=True, context_type=DataContext)
        async def add_data_item(**kwargs):
            context = kwargs["context"]
            new_data = context.data + [{"id": 2, "name": "Item 2"}]
            return DataContext(data=new_data)
        
        async def create_app_context():
            return AppContext(app_version="1.0")
        
        init_app = Operation.with_context(create_app_context, context_type=AppContext)
        app_result = await init_app()
        assert app_result.is_ok()
        browser_result = await update_browser(context=browser_context)
        assert browser_result.is_ok()
        data_result = await add_data_item(context=data_context)
        assert data_result.is_ok()
        combined_context = AppContext(
            browser=browser_result.default_value(None),
            data=data_result.default_value(None),
            app_version=app_result.default_value(None).app_version
        )
        assert combined_context.browser.title == "Updated Title"
        assert len(combined_context.data.data) == 2
        assert combined_context.app_version == "1.0"
    
    @pytest.mark.asyncio
    async def test_context_in_complex_chain(self, app_context, browser_op, data_op, app_op):
        @operation(context=True, context_type=AppContext)
        async def get_summary(**kwargs):
            context = kwargs["context"]
            return {
                "url": context.browser.url if context.browser else None,
                "data_count": len(context.data.data) if context.data else 0,
                "version": context.app_version
            }
        
        pipeline = (
            app_op 
            >> browser_op("https://new-example.com") 
            >> data_op({"id": 2, "name": "New Item"})
            >> data_op({"id": 3, "name": "New Item 2"})
            >> get_summary
        )
        result = await pipeline(context=app_context)
        assert result.is_ok()
        summary = result.default_value(None)
        assert summary["url"] == "https://new-example.com"
        assert summary["data_count"] == 3
        assert summary["version"] == "2.0"
    
   
    @pytest.mark.asyncio
    async def test_context_with_placeholders(self, test_context):
        @operation(context=True, context_type=TestContext)
        async def multiply_by_counter(value: int, **kwargs):
            context = kwargs["context"]
            return value * context.counter
        
        @operation(context=True, context_type=TestContext)
        async def add_to_result(value: int, **kwargs):
            context = kwargs["context"]
            return value + len(context.value)
        
        pipeline = constant(5) >> multiply_by_counter(_) >> add_to_result(_)
        result = await pipeline(context=test_context)
        assert result.is_ok()
        assert result.default_value(None) == 9
    
    @pytest.mark.asyncio
    async def test_branching_with_context(self, test_context):
        def check_counter_positive(**kwargs):
            context = kwargs.get("context")
            return context and context.counter > 0
        
        @operation(context=True, context_type=TestContext)
        async def positive_branch(**kwargs):
            context = kwargs["context"]
            return f"Positive counter: {context.counter}"
        
        @operation(context=True, context_type=TestContext)
        async def negative_branch(**kwargs):
            context = kwargs["context"]
            return f"Non-positive counter: {context.counter}"
        
        branch_op = branch(check_counter_positive, positive_branch, negative_branch)
        positive_result = await branch_op(context=test_context)
        assert positive_result.is_ok()
        assert positive_result.default_value(None) == "Positive counter: 1"
        negative_context = TestContext(value="test", counter=0)
        negative_result = await branch_op(context=negative_context)
        assert negative_result.is_ok()
        assert negative_result.default_value(None) == "Non-positive counter: 0"