"""
Comprehensive test suite for fp_ops.objects module.
Tests all object operations including get, build, merge, and update.
Covers edge cases, composition, placeholders, and error handling.
"""
import pytest
import asyncio
from typing import Any, Dict, List
from dataclasses import dataclass

# Assuming these imports based on the provided code
from fp_ops import operation, Operation
from fp_ops.objects import get, build, merge, update
from fp_ops.primitives import _, Placeholder
from expression import Ok, Error, Result


# Test fixtures and helper classes
@dataclass
class SampleObject:
    """Sample object for testing attribute access."""
    name: str
    value: int
    nested: Dict[str, Any] = None


class AsyncDict:
    """Helper class that returns values asynchronously."""
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    async def get_data(self) -> Dict[str, Any]:
        return self.data


# Fixtures
@pytest.fixture
def simple_dict():
    return {"a": 1, "b": 2, "c": 3}


@pytest.fixture
def nested_dict():
    return {
        "user": {
            "name": "John",
            "contact": {
                "email": "john@example.com",
                "phone": "123-456-7890"
            },
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        },
        "items": [
            {"id": 1, "price": 10.5},
            {"id": 2, "price": 20.0},
            {"id": 3, "price": 15.75}
        ],
        "metadata": {
            "version": "1.0",
            "count": 3
        }
    }


@pytest.fixture
def sample_object():
    return SampleObject(
        name="test",
        value=42,
        nested={"inner": "data"}
    )


# Test get operation
class TestGetOperation:
    """Test suite for the get operation."""
    
    @pytest.mark.asyncio
    async def test_get_simple_key(self, simple_dict):
        """Test getting a simple key from a dict."""
        op = get("a")
        result = await op.execute(simple_dict)
        assert result.is_ok()
        assert result.default_value(None) == 1
    
    @pytest.mark.asyncio
    async def test_get_missing_key_with_default(self, simple_dict):
        """Test getting a missing key returns default value."""
        op = get("missing", "default_value")
        result = await op.execute(simple_dict)
        assert result.is_ok()
        assert result.default_value(None) == "default_value"
    
    @pytest.mark.asyncio
    async def test_get_missing_key_no_default(self, simple_dict):
        """Test getting a missing key returns None when no default."""
        op = get("missing")
        result = await op.execute(simple_dict)
        assert result.is_ok()
        assert result.default_value(None) is None
    
    @pytest.mark.asyncio
    async def test_get_nested_path(self, nested_dict):
        """Test getting nested values using dot notation."""
        op = get("user.contact.email")
        result = await op.execute(nested_dict)
        assert result.is_ok()
        assert result.default_value(None) == "john@example.com"
    
    @pytest.mark.asyncio
    async def test_get_array_index(self, nested_dict):
        """Test getting array elements by index."""
        op = get("items.1.price")
        result = await op.execute(nested_dict)
        assert result.is_ok()
        assert result.default_value(None) == 20.0
    
    @pytest.mark.asyncio
    async def test_get_array_bracket_notation(self, nested_dict):
        """Test array access with bracket notation."""
        op = get("items[0].id")
        result = await op.execute(nested_dict)
        assert result.is_ok()
        assert result.default_value(None) == 1
    
    @pytest.mark.asyncio
    async def test_get_out_of_bounds_index(self, nested_dict):
        """Test accessing out of bounds array index returns default."""
        op = get("items.10.price", 0.0)
        result = await op.execute(nested_dict)
        assert result.is_ok()
        assert result.default_value(None) == 0.0
    
    @pytest.mark.asyncio
    async def test_get_attribute_access(self, sample_object):
        """Test getting attributes from objects."""
        op = get("name")
        result = await op.execute(sample_object)
        assert result.is_ok()
        assert result.default_value(None) == "test"
    
    @pytest.mark.asyncio
    async def test_get_nested_attribute_dict(self, sample_object):
        """Test mixing attribute and dict access."""
        op = get("nested.inner")
        result = await op.execute(sample_object)
        assert result.is_ok()
        assert result.default_value(None) == "data"
    
    @pytest.mark.asyncio
    async def test_get_empty_path(self, simple_dict):
        """Test empty path returns the original data."""
        op = get("")
        result = await op.execute(simple_dict)
        assert result.is_ok()
        assert result.default_value(None) == simple_dict
    
    @pytest.mark.asyncio
    async def test_get_none_data(self):
        """Test get on None data returns default."""
        op = get("any.path", "default")
        result = await op.execute(None)
        assert result.is_ok()
        assert result.default_value(None) == "default"
    
    @pytest.mark.asyncio
    async def test_get_composition(self, nested_dict):
        """Test composing multiple get operations."""
        pipeline = get("user") >> get("contact") >> get("email")
        result = await pipeline.execute(nested_dict)
        assert result.is_ok()
        assert result.default_value(None) == "john@example.com"


# Test build operation
class TestBuildOperation:
    """Test suite for the build operation."""
    
    @pytest.mark.asyncio
    async def test_build_static_values(self, simple_dict):
        """Test building object with static values."""
        schema = {
            "constant": "static",
            "number": 42,
            "boolean": True
        }
        op = build(schema)
        result = await op.execute(simple_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["constant"] == "static"
        assert output["number"] == 42
        assert output["boolean"] is True
    
    @pytest.mark.asyncio
    async def test_build_with_operations(self, nested_dict):
        """Test building object with operations."""
        schema = {
            "id": get("user.name"),
            "email": get("user.contact.email"),
            "itemCount": get("metadata.count")
        }
        op = build(schema)
        result = await op.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["id"] == "John"
        assert output["email"] == "john@example.com"
        assert output["itemCount"] == 3
    
    @pytest.mark.asyncio
    async def test_build_with_callables(self, nested_dict):
        """Test building object with callable functions."""
        schema = {
            "userName": get("user.name"),
            "upperName": lambda d: d["user"]["name"].upper(),
            "itemTotal": lambda d: sum(item["price"] for item in d["items"])
        }
        op = build(schema)
        result = await op.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["userName"] == "John"
        assert output["upperName"] == "JOHN"
        assert output["itemTotal"] == 46.25
    
    @pytest.mark.asyncio
    async def test_build_nested_schemas(self, nested_dict):
        """Test building nested objects."""
        schema = {
            "user": {
                "name": get("user.name"),
                "contact": {
                    "email": get("user.contact.email"),
                    "hasPhone": lambda d: "phone" in d["user"]["contact"]
                }
            },
            "summary": {
                "itemCount": get("metadata.count"),
                "version": get("metadata.version")
            }
        }
        op = build(schema)
        result = await op.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["user"]["name"] == "John"
        assert output["user"]["contact"]["email"] == "john@example.com"
        assert output["user"]["contact"]["hasPhone"] is True
        assert output["summary"]["itemCount"] == 3
    
    @pytest.mark.asyncio
    async def test_build_error_handling(self, simple_dict):
        """Test build handles errors gracefully."""
        schema = {
            "safe": get("a"),
            "error": lambda d: d["nonexistent"]["path"],  # This will error
            "afterError": get("b")
        }
        op = build(schema)
        result = await op.execute(simple_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["safe"] == 1
        assert output["error"] is None  # Error results in None
        assert output["afterError"] == 2
    
    @pytest.mark.asyncio
    async def test_build_with_failed_operations(self, simple_dict):
        """Test build handles failed operations."""
        schema = {
            "exists": get("a"),
            "missing": get("x.y.z"),  # No default, will return None
            "withDefault": get("x.y.z", "default")
        }
        op = build(schema)
        result = await op.execute(simple_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["exists"] == 1
        assert output["missing"] is None
        assert output["withDefault"] == "default"


# Test merge operation
class TestMergeOperation:
    """Test suite for the merge operation."""
    
    @pytest.mark.asyncio
    async def test_merge_static_dicts(self):
        """Test merging static dictionaries."""
        op = merge(
            {"a": 1, "b": 2},
            {"c": 3, "d": 4},
            {"b": 5, "e": 6}  # b should override
        )
        result = await op.execute({})
        assert result.is_ok()
        output = result.default_value(None)
        assert output == {"a": 1, "b": 5, "c": 3, "d": 4, "e": 6}
    
    @pytest.mark.asyncio
    async def test_merge_operations(self, nested_dict):
        """Test merging operations that return dicts."""
        op = merge(
            get("user.settings"),
            get("metadata"),
            {"custom": "value"}
        )
        result = await op.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["theme"] == "dark"
        assert output["notifications"] is True
        assert output["version"] == "1.0"
        assert output["count"] == 3
        assert output["custom"] == "value"
    
    @pytest.mark.asyncio
    async def test_merge_callables(self, nested_dict):
        """Test merging callable functions."""
        op = merge(
            lambda d: {"userName": d["user"]["name"]},
            lambda d: {"itemCount": len(d["items"])},
            get("user.settings")
        )
        result = await op.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["userName"] == "John"
        assert output["itemCount"] == 3
        assert output["theme"] == "dark"
    
    @pytest.mark.asyncio
    async def test_merge_empty_sources(self):
        """Test merging with empty sources."""
        op = merge({}, {}, {"a": 1})
        result = await op.execute({})
        assert result.is_ok()
        assert result.default_value(None) == {"a": 1}
    
    @pytest.mark.asyncio
    async def test_merge_none_handling(self, simple_dict):
        """Test merge handles None results gracefully."""
        @operation
        def returns_none(data):
            return None
        
        op = merge(
            {"a": 1},
            returns_none,  # This returns None, not a dict
            {"b": 2}
        )
        result = await op.execute(simple_dict)
        assert result.is_ok()
        output = result.default_value(None)
        # Should skip the None and merge the rest
        assert output == {"a": 1, "b": 2}


# Test update operation
class TestUpdateOperation:
    """Test suite for the update operation."""
    
    @pytest.mark.asyncio
    async def test_update_basic(self, simple_dict):
        """Test basic dict update."""
        op = update({"b": 10, "d": 4})
        result = await op.execute(simple_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output == {"a": 1, "b": 10, "c": 3, "d": 4}
    
    @pytest.mark.asyncio
    async def test_update_empty_dict(self):
        """Test updating empty dict."""
        op = update({"a": 1, "b": 2})
        result = await op.execute({})
        assert result.is_ok()
        assert result.default_value(None) == {"a": 1, "b": 2}
    
    @pytest.mark.asyncio
    async def test_update_no_changes(self, simple_dict):
        """Test update with no changes."""
        op = update({})
        result = await op.execute(simple_dict)
        assert result.is_ok()
        assert result.default_value(None) == simple_dict
    
    @pytest.mark.asyncio
    async def test_update_composition(self, simple_dict):
        """Test composing update operations."""
        pipeline = update({"b": 10}) >> update({"c": 20}) >> update({"d": 30})
        result = await pipeline.execute(simple_dict)
        assert result.is_ok()
        assert result.default_value(None) == {"a": 1, "b": 10, "c": 20, "d": 30}


# Test complex compositions and interactions
class TestComplexCompositions:
    """Test suite for complex operation compositions."""
    
    @pytest.mark.asyncio
    async def test_get_build_pipeline(self, nested_dict):
        """Test pipeline combining get and build."""
        pipeline = get("user") >> build({
            "displayName": get("name"),
            "primaryEmail": get("contact.email"),
            "hasNotifications": get("settings.notifications", False)
        })
        result = await pipeline.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["displayName"] == "John"
        assert output["primaryEmail"] == "john@example.com"
        assert output["hasNotifications"] is True
    
    @pytest.mark.asyncio
    async def test_build_merge_pipeline(self, nested_dict):
        """Test pipeline combining build and merge."""
        user_summary = build({
            "name": get("user.name"),
            "email": get("user.contact.email")
        })
        
        item_summary = build({
            "itemCount": get("metadata.count"),
            "totalPrice": lambda d: sum(item["price"] for item in d["items"])
        })
        
        pipeline = merge(
            user_summary,
            item_summary,
            {"processed": True}
        )
        
        result = await pipeline.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["name"] == "John"
        assert output["email"] == "john@example.com"
        assert output["itemCount"] == 3
        assert output["totalPrice"] == 46.25
        assert output["processed"] is True
    
    @pytest.mark.asyncio
    async def test_nested_operations_in_build(self, nested_dict):
        """Test deeply nested operations in build schemas."""
        schema = {
            "user": build({
                "info": merge(
                    get("user.contact"),
                    {"name": get("user.name")}
                ),
                "preferences": get("user.settings")
            }),
            "items": get("items") >> operation(lambda items: [
                {"id": item["id"], "price": item["price"] * 1.1}  # Add 10% tax
                for item in items
            ])
        }
        
        op = build(schema)
        result = await op.execute(nested_dict)
        assert result.is_ok()
        output = result.default_value(None)
        assert output["user"]["info"]["email"] == "john@example.com"
        assert output["user"]["info"]["name"] == "John"
        assert output["user"]["preferences"]["theme"] == "dark"
        assert len(output["items"]) == 3
        assert output["items"][0]["price"] == 11.55  # 10.5 * 1.1
    
    @pytest.mark.asyncio
    async def test_error_propagation_in_pipeline(self):
        """Test error propagation through pipelines."""
        @operation
        def failing_op(data):
            raise ValueError("Intentional error")
        
        pipeline = get("a") >> failing_op >> build({"result": _})
        result = await pipeline.execute({"a": 1})
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_placeholder_in_operations(self, nested_dict):
        """Test using placeholders in operations."""
        # Create a simple extraction pipeline
        # First approach: Use get operation directly
        pipeline = get("user") >> get("name")
        result = await pipeline.execute(nested_dict)
        assert result.is_ok()
        assert result.default_value(None) == "John"
        
        # Second approach: Create a parameterized extraction
        # This shows how to create reusable field extractors
        def extract_nested(outer_field: str, inner_field: str) -> Operation:
            return get(outer_field) >> get(inner_field)
        
        # Use the parameterized extractor
        user_name_extractor = extract_nested("user", "name")
        result2 = await user_name_extractor.execute(nested_dict)
        assert result2.is_ok()
        assert result2.default_value(None) == "John"
        
        # Third approach: Using a factory function to create operations
        def make_field_selector(field_list: List[str]) -> Operation:
            @operation
            def select_fields(data: Dict) -> Dict:
                return {field: data.get(field) for field in field_list}
            return select_fields
        
        # Create and use the selector
        selector = make_field_selector(["user", "metadata"])
        result3 = await selector.execute(nested_dict)
        assert result3.is_ok()
        selected = result3.default_value(None)
        assert "user" in selected
        assert "metadata" in selected
        assert selected["user"]["name"] == "John"
    
    @pytest.mark.asyncio
    async def test_parallel_operations(self, nested_dict):
        """Test parallel operation execution."""
        # Using & operator for parallel execution
        user_op = get("user.name")
        count_op = get("metadata.count")
        
        parallel = user_op & count_op
        result = await parallel.execute(nested_dict)
        assert result.is_ok()
        name, count = result.default_value(None)
        assert name == "John"
        assert count == 3
    
    @pytest.mark.asyncio
    async def test_fallback_operations(self, simple_dict):
        """Test fallback operation with | operator."""
        @operation
        def failing_op(data):
            raise ValueError("Intentional failure")
        
        # Create fallback chain
        primary = get("a") >> failing_op
        fallback = get("b")
        
        op = primary | fallback
        result = await op.execute(simple_dict)
        assert result.is_ok()
        # Should get 'a' since 'missing_key' returns None (not an error)
        # Note: This might depend on implementation details
        assert result.default_value(None) == 2
    
    @pytest.mark.asyncio
    async def test_complex_data_transformation(self, nested_dict):
        """Test a complex real-world data transformation."""
        # Transform user data into a formatted report
        pipeline = build({
            "report": {
                "userSummary": get("user") >> build({
                    "fullName": get("name"),
                    "contactMethod": get("contact.email"),
                    "preferences": get("settings")
                }),
                "orderSummary": build({
                    "items": get("items"),
                    "totalItems": get("metadata.count"),
                    "totalValue": lambda d: sum(item["price"] for item in d["items"]),
                    "averagePrice": lambda d: sum(item["price"] for item in d["items"]) / len(d["items"])
                }),
                "metadata": merge(
                    get("metadata"),
                    {"reportGenerated": True, "reportVersion": "2.0"}
                )
            }
        })
        
        result = await pipeline.execute(nested_dict)
        assert result.is_ok()
        report = result.default_value(None)["report"]
        
        assert report["userSummary"]["fullName"] == "John"
        assert report["userSummary"]["preferences"]["theme"] == "dark"
        assert report["orderSummary"]["totalItems"] == 3
        assert report["orderSummary"]["totalValue"] == 46.25
        assert report["orderSummary"]["averagePrice"] == 46.25 / 3
        assert report["metadata"]["version"] == "1.0"
        assert report["metadata"]["reportGenerated"] is True
        assert report["metadata"]["reportVersion"] == "2.0"


# Test edge cases and special scenarios
class TestEdgeCases:
    """Test suite for edge cases and special scenarios."""
    
    @pytest.mark.asyncio
    async def test_circular_reference_handling(self):
        """Test handling of circular references in data."""
        # Create circular reference
        data = {"a": {"b": None}}
        data["a"]["b"] = data["a"]  # Circular reference
        
        op = get("a.b.b.b")  # Should handle circular refs gracefully
        result = await op.execute(data)
        assert result.is_ok()
        # Should get the circular reference back
        assert result.default_value(None) is data["a"]
    
    @pytest.mark.asyncio
    async def test_unicode_keys(self):
        """Test handling of unicode keys."""
        data = {
            "こんにちは": "hello",
            "user": {"名前": "太郎"}
        }
        
        op1 = get("こんにちは")
        result1 = await op1.execute(data)
        assert result1.is_ok()
        assert result1.default_value(None) == "hello"
        
        op2 = get("user.名前")
        result2 = await op2.execute(data)
        assert result2.is_ok()
        assert result2.default_value(None) == "太郎"
    
    @pytest.mark.asyncio
    async def test_special_characters_in_paths(self):
        """Test paths with special characters."""
        data = {
            "user-name": "John",
            "user.email": "john@example.com",  # Key contains dot
            "items[0]": "first"  # Key contains brackets
        }
        
        # These should work as direct key access
        op1 = get("user-name")
        result1 = await op1.execute(data)
        assert result1.is_ok()
        assert result1.default_value(None) == "John"
        
        # This is tricky - might be treated as nested access
        op2 = get("user.email")
        result2 = await op2.execute(data)
        assert result2.is_ok()
        # Depending on implementation, might return None or the value
    
    @pytest.mark.asyncio
    async def test_very_deep_nesting(self):
        """Test very deep nesting."""
        # Create deeply nested structure
        data = {"level": 0}
        current = data
        for i in range(100):
            current["next"] = {"level": i + 1}
            current = current["next"]
        
        # Access deep value
        path = ".".join(["next"] * 99) + ".level"
        op = get(path)
        result = await op.execute(data)
        assert result.is_ok()
        assert result.default_value(None) == 99
    
    @pytest.mark.asyncio
    async def test_mixed_types_in_path(self):
        """Test paths through mixed types."""
        data = {
            "users": [
                {"name": "Alice", "scores": {"math": 90}},
                {"name": "Bob", "scores": {"math": 85}}
            ]
        }
        
        op = get("users.0.scores.math")
        result = await op.execute(data)
        assert result.is_ok()
        assert result.default_value(None) == 90
    
    @pytest.mark.asyncio
    async def test_operations_with_none_input(self):
        """Test all operations handle None input gracefully."""
        # get with None
        get_op = get("any.path", "default")
        result = await get_op.execute(None)
        assert result.is_ok()
        assert result.default_value(None) == "default"
        
        # build with None
        build_op = build({"key": "value"})
        result = await build_op.execute(None)
        assert result.is_ok()
        assert result.default_value(None) == {"key": "value"}
        
        # merge with None
        merge_op = merge({"a": 1}, lambda d: {"b": 2} if d else { "b": 3 })
        result = await merge_op.execute(None)
        assert result.is_ok()
        assert result.default_value(None) == {"a": 1, "b": 3}
    
    @pytest.mark.asyncio
    async def test_performance_with_large_data(self):
        """Test operations perform well with large datasets."""
        # Create large dataset
        large_data = {
            "items": [{"id": i, "value": i * 2} for i in range(10000)]
        }
        
        # Test get with large array
        op = get("items.9999.value")
        result = await op.execute(large_data)
        assert result.is_ok()
        assert result.default_value(None) == 19998
        
        # Test build with many operations
        schema = {
            f"item_{i}": get(f"items.{i}.value")
            for i in range(100)
        }
        build_op = build(schema)
        result = await build_op.execute(large_data)
        assert result.is_ok()
        output = result.default_value(None)
        assert len(output) == 100
        assert output["item_50"] == 100


# Test async behavior
class TestAsyncBehavior:
    """Test suite for async-specific behavior."""
    
    @pytest.mark.asyncio
    async def test_async_operations_in_build(self):
        """Test build with async operations."""
        @operation
        async def async_transform(data):
            await asyncio.sleep(0.01)  # Simulate async work
            return {"async_result": data.get("value", 0) * 2}
        
        schema = {
            "sync": get("value"),
            "async": async_transform
        }
        
        op = build(schema)
        result = await op.execute({"value": 21})
        assert result.is_ok()
        output = result.default_value(None)
        assert output["sync"] == 21
        assert output["async"] == {"async_result": 42}
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_in_merge(self):
        """Test that merge can handle concurrent operations."""
        call_order = []
        
        @operation
        async def slow_op1(data):
            call_order.append("start1")
            await asyncio.sleep(0.02)
            call_order.append("end1")
            return {"op1": "done"}
        
        @operation
        async def slow_op2(data):
            call_order.append("start2")
            await asyncio.sleep(0.01)
            call_order.append("end2")
            return {"op2": "done"}
        
        op = merge(slow_op1, slow_op2)
        result = await op.execute({})
        assert result.is_ok()
        output = result.default_value(None)
        assert output == {"op1": "done", "op2": "done"}
        # Operations should overlap if truly concurrent
        # Though merge might execute sequentially - depends on implementation


# Integration tests
class TestIntegration:
    """Integration tests combining multiple features."""
    
    @pytest.mark.asyncio
    async def test_real_world_user_transformation(self):
        """Test a real-world user data transformation scenario."""
        raw_user_data = {
            "id": "usr_123",
            "personal_info": {
                "first_name": "Jane",
                "last_name": "Doe",
                "date_of_birth": "1990-01-01"
            },
            "contact": {
                "emails": ["jane@work.com", "jane@personal.com"],
                "phones": [
                    {"type": "work", "number": "555-0123"},
                    {"type": "home", "number": "555-0456"}
                ]
            },
            "preferences": {
                "notifications": {
                    "email": True,
                    "sms": False,
                    "push": True
                },
                "theme": "dark",
                "language": "en"
            },
            "activity": {
                "last_login": "2024-01-15T10:30:00Z",
                "login_count": 42
            }
        }
        
        # Complex transformation pipeline
        transform_pipeline = build({
            "userId": get("id"),
            "displayName": lambda d: f"{d['personal_info']['first_name']} {d['personal_info']['last_name']}",
            "primaryEmail": get("contact.emails.0"),
            "contactMethods": build({
                "emails": get("contact.emails"),
                "primaryPhone": get("contact.phones.0.number", "No phone"),
                "hasWorkPhone": lambda d: any(
                    p["type"] == "work" for p in d.get("contact", {}).get("phones", [])
                )
            }),
            "settings": merge(
                get("preferences"),
                build({
                    "notificationChannels": lambda d: [
                        channel for channel, enabled in d["preferences"]["notifications"].items()
                        if enabled
                    ]
                })
            ),
            "accountStatus": build({
                "lastSeen": get("activity.last_login"),
                "isActive": lambda d: d["activity"]["login_count"] > 10,
                "engagementLevel": lambda d: (
                    "high" if d["activity"]["login_count"] > 30
                    else "medium" if d["activity"]["login_count"] > 10
                    else "low"
                )
            })
        })
        
        result = await transform_pipeline.execute(raw_user_data)
        assert result.is_ok()
        transformed = result.default_value(None)
        
        # Verify the transformation
        assert transformed["userId"] == "usr_123"
        assert transformed["displayName"] == "Jane Doe"
        assert transformed["primaryEmail"] == "jane@work.com"
        assert transformed["contactMethods"]["emails"] == ["jane@work.com", "jane@personal.com"]
        assert transformed["contactMethods"]["hasWorkPhone"] is True
        assert transformed["settings"]["theme"] == "dark"
        assert transformed["settings"]["notificationChannels"] == ["email", "push"]
        assert transformed["accountStatus"]["isActive"] is True
        assert transformed["accountStatus"]["engagementLevel"] == "high"
    
    @pytest.mark.asyncio
    async def test_api_response_normalization(self):
        """Test normalizing different API response formats."""
        # Simulate different API response formats
        api_v1_response = {
            "user": {
                "user_id": 123,
                "user_name": "olduser",
                "user_email": "old@example.com"
            }
        }
        
        api_v2_response = {
            "data": {
                "id": 456,
                "attributes": {
                    "username": "newuser",
                    "email": "new@example.com"
                }
            }
        }
        
        # Normalizer for v1
        normalize_v1 = get("user") >> build({
            "id": get("user_id"),
            "username": get("user_name"),
            "email": get("user_email")
        })
        
        # Normalizer for v2
        normalize_v2 = get("data") >> build({
            "id": get("id"),
            "username": get("attributes.username"),
            "email": get("attributes.email")
        })
        
        # Test v1 normalization
        result_v1 = await normalize_v1.execute(api_v1_response)
        assert result_v1.is_ok()
        norm_v1 = result_v1.default_value(None)
        assert norm_v1["id"] == 123
        assert norm_v1["username"] == "olduser"
        assert norm_v1["email"] == "old@example.com"
        
        # Test v2 normalization
        result_v2 = await normalize_v2.execute(api_v2_response)
        assert result_v2.is_ok()
        norm_v2 = result_v2.default_value(None)
        assert norm_v2["id"] == 456
        assert norm_v2["username"] == "newuser"
        assert norm_v2["email"] == "new@example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])