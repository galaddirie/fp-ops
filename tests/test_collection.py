"""
Comprehensive test suite for fp_ops.collections module.
Tests all collection operations including filter, map, reduce, zip, and utility functions.
Covers edge cases, composition, async operations, and error handling.
"""
import pytest
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from fp_ops import operation, Operation
from fp_ops.collections import (
    filter, map, reduce, zip,
    contains, not_contains,
    flatten, flatten_deep, unique, reverse,
    length, keys, values, items,

)
from fp_ops.objects import get, build
from expression import Ok, Error, Result


# Test fixtures and helper classes
@dataclass
class User:
    """Sample user class for testing."""
    id: int
    name: str
    age: int
    active: bool = True
    email: str = ""


@dataclass
class Product:
    """Sample product class for testing."""
    id: int
    name: str
    price: float
    category: str
    in_stock: bool = True


# Helper operations for testing
@operation
async def is_adult(user: Dict[str, Any]) -> bool:
    """Check if user is an adult."""
    age = user.get("age", 0)
    return age >= 18


@operation
async def double_value(x: int) -> int:
    """Double a value."""
    return x * 2


@operation
async def add_values(a: int, b: int) -> int:
    """Add two values."""
    return a + b


@operation
def to_upper(s: str) -> str:
    """Convert string to uppercase."""
    return s.upper()


@operation
def count_vowels(s: str) -> int:
    """Count vowels in a string."""
    return sum(1 for c in s.lower() if c in 'aeiou')


# Fixtures
@pytest.fixture
def simple_numbers():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def users_data():
    return [
        {"id": 1, "name": "Alice", "age": 25, "active": True, "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "age": 17, "active": True, "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "age": 30, "active": False, "email": "charlie@example.com"},
        {"id": 4, "name": "Diana", "age": 22, "active": True, "email": "diana@example.com"},
        {"id": 5, "name": "Eve", "age": 16, "active": True, "email": "eve@example.com"}
    ]


@pytest.fixture
def products_data():
    return [
        {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics", "in_stock": True},
        {"id": 2, "name": "Mouse", "price": 29.99, "category": "Electronics", "in_stock": True},
        {"id": 3, "name": "Desk", "price": 199.99, "category": "Furniture", "in_stock": False},
        {"id": 4, "name": "Chair", "price": 149.99, "category": "Furniture", "in_stock": True},
        {"id": 5, "name": "Monitor", "price": 299.99, "category": "Electronics", "in_stock": True}
    ]


@pytest.fixture
def nested_lists():
    return [[1, 2], [3, 4], [5, 6, 7]]


@pytest.fixture
def deeply_nested():
    return [1, [2, [3, [4, [5]]], 6], 7, [8, 9]]


# Test filter operation
class TestFilterOperation:
    """Test suite for the filter operation."""
    
    @pytest.mark.asyncio
    async def test_filter_with_callable(self, simple_numbers):
        """Test filter with a simple callable predicate."""
        op = filter(lambda x: x > 3)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == [4, 5]
    
    @pytest.mark.asyncio
    async def test_filter_with_operation(self, users_data):
        """Test filter with an Operation predicate."""
        op = filter(is_adult)
        result = await op.execute(users_data)
        assert result.is_ok()
        filtered = result.default_value(None)
        assert len(filtered) == 3
        assert all(user["age"] >= 18 for user in filtered)
    
    @pytest.mark.asyncio
    async def test_filter_with_dict_matching(self, users_data):
        """Test filter with dictionary matching."""
        op = filter({"active": True})
        result = await op.execute(users_data)
        assert result.is_ok()
        filtered = result.default_value(None)
        assert len(filtered) == 4
        assert all(user["active"] for user in filtered)
    
    @pytest.mark.asyncio
    async def test_filter_with_multiple_dict_criteria(self, users_data):
        """Test filter with multiple dictionary criteria."""
        op = filter({"active": True, "age": 22})
        result = await op.execute(users_data)
        assert result.is_ok()
        filtered = result.default_value(None)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "Diana"
    
    @pytest.mark.asyncio
    async def test_filter_empty_list(self):
        """Test filter on empty list."""
        op = filter(lambda x: x > 0)
        result = await op.execute([])
        assert result.is_ok()
        assert result.default_value(None) == []
    
    @pytest.mark.asyncio
    async def test_filter_none_match(self, simple_numbers):
        """Test filter when no items match."""
        op = filter(lambda x: x > 10)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == []
    
    @pytest.mark.asyncio
    async def test_filter_all_match(self, simple_numbers):
        """Test filter when all items match."""
        op = filter(lambda x: x > 0)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == simple_numbers
    
    @pytest.mark.asyncio
    async def test_filter_with_nested_dict_matching(self, products_data):
        """Test filter with nested paths in dict matching."""
        # Add nested data
        for product in products_data:
            product["details"] = {"warehouse": {"location": "A" if product["in_stock"] else "B"}}
        
        op = filter({"details.warehouse.location": "A"})
        result = await op.execute(products_data)
        assert result.is_ok()
        filtered = result.default_value(None)
        assert len(filtered) == 4  # Only in-stock items are in warehouse A


# Test map operation
class TestMapOperation:
    """Test suite for the map operation."""
    
    @pytest.mark.asyncio
    async def test_map_with_callable(self, simple_numbers):
        """Test map with a simple callable."""
        op = map(lambda x: x * 2)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_map_with_operation(self, simple_numbers):
        """Test map with an Operation."""
        op = map(double_value)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == [2, 4, 6, 8, 10]
    
    @pytest.mark.asyncio
    async def test_map_extract_field(self, users_data):
        """Test map to extract a field from objects."""
        op = map(lambda user: user["name"])
        result = await op.execute(users_data)
        assert result.is_ok()
        assert result.default_value(None) == ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    
    @pytest.mark.asyncio
    async def test_map_with_transformation(self, users_data):
        """Test map with complex transformation."""
        op = map(lambda user: {
            "display_name": f"{user['name']} (ID: {user['id']})",
            "can_vote": user["age"] >= 18
        })
        result = await op.execute(users_data)
        assert result.is_ok()
        transformed = result.default_value(None)
        assert len(transformed) == 5
        assert transformed[0]["display_name"] == "Alice (ID: 1)"
        assert transformed[0]["can_vote"] is True
        assert transformed[1]["can_vote"] is False
    
    @pytest.mark.asyncio
    async def test_map_empty_list(self):
        """Test map on empty list."""
        op = map(lambda x: x * 2)
        result = await op.execute([])
        assert result.is_ok()
        assert result.default_value(None) == []
    
    @pytest.mark.asyncio
    async def test_map_with_error_in_operation(self):
        """Test map when operation raises an error."""
        @operation
        async def failing_transform(x):
            if x == 3:
                raise ValueError("Cannot process 3")
            return x * 2
        
        op = map(failing_transform)
        result = await op.execute([1, 2, 3, 4, 5])
        assert result.is_error()
        assert isinstance(result.error, ValueError)


# Test reduce operation
class TestReduceOperation:
    """Test suite for the reduce operation."""
    
    @pytest.mark.asyncio
    async def test_reduce_sum(self, simple_numbers):
        """Test reduce to sum numbers."""
        op = reduce(lambda a, b: a + b)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == 15
    
    @pytest.mark.asyncio
    async def test_reduce_with_initial(self, simple_numbers):
        """Test reduce with initial value."""
        op = reduce(lambda a, b: a + b, 10)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == 25
    
    @pytest.mark.asyncio
    async def test_reduce_with_operation(self, simple_numbers):
        """Test reduce with an Operation."""
        op = reduce(add_values)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == 15
    
    @pytest.mark.asyncio
    async def test_reduce_product(self, simple_numbers):
        """Test reduce to calculate product."""
        op = reduce(lambda a, b: a * b, 1)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == 120
    
    @pytest.mark.asyncio
    async def test_reduce_max(self, simple_numbers):
        """Test reduce to find maximum."""
        op = reduce(lambda a, b: a if a > b else b)
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == 5
    
    @pytest.mark.asyncio
    async def test_reduce_complex_objects(self, users_data):
        """Test reduce with complex objects."""
        op = reduce(
            lambda acc, user: acc + user["age"],
            0
        )
        result = await op.execute(users_data)
        assert result.is_ok()
        assert result.default_value(None) == sum(user["age"] for user in users_data)
    
    @pytest.mark.asyncio
    async def test_reduce_empty_list_with_initial(self):
        """Test reduce on empty list with initial value."""
        op = reduce(lambda a, b: a + b, 42)
        result = await op.execute([])
        assert result.is_ok()
        assert result.default_value(None) == 42
    
    @pytest.mark.asyncio
    async def test_reduce_empty_list_no_initial(self):
        """Test reduce on empty list without initial value."""
        op = reduce(lambda a, b: a + b)
        result = await op.execute([])
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_reduce_single_item(self):
        """Test reduce with single item."""
        op = reduce(lambda a, b: a + b)
        result = await op.execute([42])
        assert result.is_ok()
        assert result.default_value(None) == 42


# Test zip operation
class TestZipOperation:
    """Test suite for the zip operation."""
    
    @pytest.mark.asyncio
    async def test_zip_multiple_operations(self, users_data):
        """Test zip with multiple operations."""
        op = zip(
            get("id"),
            get("name"),
            get("email")
        )
        result = await op.execute(users_data)
        assert result.is_ok()
        zipped = result.default_value(None)
        assert len(zipped) == 5
        assert zipped[0] == (1, "Alice", "alice@example.com")
        assert zipped[1] == (2, "Bob", "bob@example.com")
    
    @pytest.mark.asyncio
    async def test_zip_with_callables(self, simple_numbers):
        """Test zip with callable functions."""
        op = zip(
            lambda x: x * 2,
            lambda x: x ** 2,
            lambda x: x + 10
        )
        result = await op.execute(simple_numbers)
        assert result.is_ok()
        zipped = result.default_value(None)
        assert zipped[0] == (2, 1, 11)
        assert zipped[1] == (4, 4, 12)
        assert zipped[2] == (6, 9, 13)
    
    @pytest.mark.asyncio
    async def test_zip_mixed_operations_and_callables(self):
        """Test zip with mix of operations and callables."""
        strings = ["hello", "world", "test"]
        op = zip(
            to_upper,              # Operation
            lambda s: len(s),      # Callable
            count_vowels          # Operation
        )
        result = await op.execute(strings)
        assert result.is_ok()
        zipped = result.default_value(None)
        assert zipped[0] == ("HELLO", 5, 2)
        assert zipped[1] == ("WORLD", 5, 1)
        assert zipped[2] == ("TEST", 4, 1)
    
    @pytest.mark.asyncio
    async def test_zip_empty_list(self):
        """Test zip on empty list."""
        op = zip(get("a"), get("b"))
        result = await op.execute([])
        assert result.is_ok()
        assert result.default_value(None) == []
    
    @pytest.mark.asyncio
    async def test_zip_no_operations(self):
        """Test zip with no operations."""
        op = zip()
        result = await op.execute([1, 2, 3])
        assert result.is_ok()
        assert result.default_value(None) == [(), (), ()]
    
    @pytest.mark.asyncio
    async def test_zip_with_failed_operations(self, users_data):
        """Test zip handles failed operations gracefully."""
        @operation
        async def may_fail(user):
            if user["id"] == 3:
                raise ValueError("Cannot process user 3")
            return user["name"]
        
        op = zip(
            get("id"),
            may_fail,
            get("active")
        )
        result = await op.execute(users_data)
        assert result.is_ok()
        zipped = result.default_value(None)
        # Failed operations should return None
        assert zipped[2][1] is None  # User 3's name should be None


# Test utility operations
class TestUtilityOperations:
    """Test suite for utility operations."""
    
    @pytest.mark.asyncio
    async def test_contains(self):
        """Test contains operation."""
        # List
        result = await contains.execute(["hello", "world"], "hello")
        assert result.is_ok() and result.default_value(None) is True
        
        result = await contains.execute(["hello", "world"], "foo")
        assert result.is_ok() and result.default_value(None) is False
        
        # String
        result = await contains.execute("hello", "ell")
        assert result.is_ok() and result.default_value(None) is True
        
        # Dict
        result = await contains.execute({"a": 1, "b": 2}, "a")
        assert result.is_ok() and result.default_value(None) is True
        
        # Set
        result = await contains.execute({1, 2, 3}, 2)
        assert result.is_ok() and result.default_value(None) is True
    
    @pytest.mark.asyncio
    async def test_not_contains(self):
        """Test not_contains operation."""
        result = await not_contains.execute(["hello", "world"], "foo")
        assert result.is_ok() and result.default_value(None) is True
        
        result = await not_contains.execute(["hello", "world"], "hello")
        assert result.is_ok() and result.default_value(None) is False
    
    @pytest.mark.asyncio
    async def test_flatten(self, nested_lists):
        """Test flatten operation."""
        result = await flatten.execute(nested_lists)
        assert result.is_ok()
        assert result.default_value(None) == [1, 2, 3, 4, 5, 6, 7]
    
    @pytest.mark.asyncio
    async def test_flatten_mixed_items(self):
        """Test flatten with mixed items."""
        data = [[1, 2], 3, [4, 5], 6]
        result = await flatten.execute(data)
        assert result.is_ok()
        assert result.default_value(None) == [1, 2, 3, 4, 5, 6]
    
    @pytest.mark.asyncio
    async def test_flatten_deep(self, deeply_nested):
        """Test flatten_deep operation."""
        result = await flatten_deep.execute(deeply_nested)
        assert result.is_ok()
        assert result.default_value(None) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    @pytest.mark.asyncio
    async def test_unique(self):
        """Test unique operation."""
        # Simple values
        result = await unique.execute([1, 2, 2, 3, 1, 4, 3, 5])
        assert result.is_ok()
        assert result.default_value(None) == [1, 2, 3, 4, 5]
        
        # Strings
        result = await unique.execute(["a", "b", "a", "c", "b"])
        assert result.is_ok()
        assert result.default_value(None) == ["a", "b", "c"]
    
    @pytest.mark.asyncio
    async def test_unique_unhashable(self):
        """Test unique with unhashable types."""
        data = [{"a": 1}, {"b": 2}, {"a": 1}, {"c": 3}]
        result = await unique.execute(data)
        assert result.is_ok()
        # Should preserve all dicts as they're not hashable
        assert len(result.default_value(None)) == 4
    
    @pytest.mark.asyncio
    async def test_reverse(self):
        """Test reverse operation."""
        # List
        result = await reverse.execute([1, 2, 3, 4, 5])
        assert result.is_ok()
        assert result.default_value(None) == [5, 4, 3, 2, 1]
        
        # String
        result = await reverse.execute("hello")
        assert result.is_ok()
        assert result.default_value(None) == "olleh"
    
    @pytest.mark.asyncio
    async def test_length(self):
        """Test length operation."""
        # List
        result = await length.execute([1, 2, 3])
        assert result.is_ok() and result.default_value(None) == 3
        
        # String
        result = await length.execute("hello")
        assert result.is_ok() and result.default_value(None) == 5
        
        # Dict
        result = await length.execute({"a": 1, "b": 2})
        assert result.is_ok() and result.default_value(None) == 2
        
        # Empty
        result = await length.execute([])
        assert result.is_ok() and result.default_value(None) == 0


# Test dictionary operations
class TestDictionaryOperations:
    """Test suite for dictionary-specific operations."""
    
    @pytest.mark.asyncio
    async def test_keys(self):
        """Test keys operation."""
        data = {"a": 1, "b": 2, "c": 3}
        result = await keys.execute(data)
        assert result.is_ok()
        k = result.default_value(None)
        assert set(k) == {"a", "b", "c"}
    
    @pytest.mark.asyncio
    async def test_values(self):
        """Test values operation."""
        data = {"a": 1, "b": 2, "c": 3}
        result = await values.execute(data)
        assert result.is_ok()
        v = result.default_value(None)
        assert set(v) == {1, 2, 3}
    
    @pytest.mark.asyncio
    async def test_items(self):
        """Test items operation."""
        data = {"a": 1, "b": 2}
        result = await items.execute(data)
        assert result.is_ok()
        i = result.default_value(None)
        assert set(i) == {("a", 1), ("b", 2)}
    

# Test complex compositions
class TestComplexCompositions:
    """Test suite for complex operation compositions."""
    
    @pytest.mark.asyncio
    async def test_filter_map_pipeline(self, users_data):
        """Test pipeline combining filter and map."""
        pipeline = (
            filter(lambda u: u["age"] >= 18) >>
            map(lambda u: {
                "name": u["name"],
                "email": u["email"],
                "years_can_vote": u["age"] - 18
            })
        )
        result = await pipeline.execute(users_data)
        assert result.is_ok()
        adults = result.default_value(None)
        assert len(adults) == 3
        assert all(a["years_can_vote"] >= 0 for a in adults)
    
    @pytest.mark.asyncio
    async def test_map_filter_reduce_pipeline(self, simple_numbers):
        """Test pipeline with map, filter, and reduce."""
        pipeline = (
            map(lambda x: x * 2) >>      # Double each number
            filter(lambda x: x > 5) >>    # Keep only > 5
            reduce(lambda a, b: a + b, 0) # Sum them
        )
        result = await pipeline.execute(simple_numbers)
        assert result.is_ok()
        assert result.default_value(None) == 24  # 6 + 8 + 10
    
    @pytest.mark.asyncio
    async def test_nested_operations(self, products_data):
        """Test nested operations in map."""
        # Group products by category and calculate stats
        @operation
        async def group_by_category(products):
            grouped = {}
            for product in products:
                cat = product["category"]
                if cat not in grouped:
                    grouped[cat] = []
                grouped[cat].append(product)
            return grouped
        
        pipeline = (
            group_by_category >>
            map(lambda items: {
                "count": len(items),
                "total_value": sum(p["price"] for p in items),
                "in_stock_count": sum(1 for p in items if p["in_stock"])
            })
        )
        
        result = await pipeline.execute(products_data)
        assert result.is_ok()
        stats = result.default_value(None)
        
        # Should have stats for each category
        assert "Electronics" in stats
        assert "Furniture" in stats
        assert stats["Electronics"]["count"] == 3
        assert stats["Furniture"]["count"] == 2
    
    @pytest.mark.asyncio
    async def test_zip_with_filter_map(self, users_data):
        """Test zip after filter and map operations."""
        # First filter active users, then extract multiple fields
        pipeline = (
            filter({"active": True}) >>
            zip(
                get("id"),
                map(lambda u: u["name"].upper()),
                map(lambda u: f"{u['email'].split('@')[0]}@masked.com")
            )
        )
        
        result = await pipeline.execute(users_data)
        assert result.is_ok()
        transformed = result.default_value(None)
        assert len(transformed) == 4  # 4 active users
        # Check first active user
        assert transformed[0] == (
            1,
            ["ALICE", "BOB", "DIANA", "EVE"],
            ["alice@masked.com", "bob@masked.com", "diana@masked.com", "eve@masked.com"]
        )
    
    @pytest.mark.asyncio
    async def test_complex_data_transformation(self, users_data, products_data):
        """Test complex real-world data transformation."""
        # Create purchase data
        purchases = [
            {"user_id": 1, "product_id": 1, "quantity": 1},
            {"user_id": 1, "product_id": 2, "quantity": 2},
            {"user_id": 2, "product_id": 3, "quantity": 1},
            {"user_id": 3, "product_id": 1, "quantity": 1},
            {"user_id": 3, "product_id": 4, "quantity": 3},
        ]
        
        # Create lookup operations
        @operation
        async def enrich_purchase(purchase):
            user = next((u for u in users_data if u["id"] == purchase["user_id"]), None)
            product = next((p for p in products_data if p["id"] == purchase["product_id"]), None)
            
            if not user or not product:
                return None
                
            return {
                "purchase_id": f"{purchase['user_id']}-{purchase['product_id']}",
                "user_name": user["name"],
                "user_age": user["age"],
                "product_name": product["name"],
                "product_category": product["category"],
                "quantity": purchase["quantity"],
                "total_price": product["price"] * purchase["quantity"],
                "eligible_for_warranty": user["age"] >= 18 and product["category"] == "Electronics"
            }
        
        # Transform pipeline
        pipeline = (
            map(enrich_purchase) >>
            filter(lambda x: x is not None) >>  # Remove failed enrichments
            filter(lambda x: x["eligible_for_warranty"])  # Only warranty-eligible
        )
        
        result = await pipeline.execute(purchases)
        assert result.is_ok()
        eligible = result.default_value(None)
        
        # Should have purchases by adults for electronics
        assert len(eligible) == 2
        assert all(p["eligible_for_warranty"] for p in eligible)
        assert all(p["user_age"] >= 18 for p in eligible)
        assert all(p["product_category"] == "Electronics" for p in eligible)


# Test edge cases
class TestEdgeCases:
    """Test suite for edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_operations_with_none_values(self):
        """Test operations handle None values gracefully."""
        data = [1, None, 3, None, 5]
        
        # Filter should handle None
        op = filter(lambda x: x is not None and x > 2)
        result = await op.execute(data)
        assert result.is_ok()
        assert result.default_value(None) == [3, 5]
        
        # Map should preserve None
        op = map(lambda x: x * 2 if x is not None else None)
        result = await op.execute(data)
        assert result.is_ok()
        assert result.default_value(None) == [2, None, 6, None, 10]
    
    @pytest.mark.asyncio
    async def test_very_large_lists(self):
        """Test operations with large datasets."""
        large_list = list(range(10000))
        
        # Filter
        op = filter(lambda x: x % 1000 == 0)
        result = await op.execute(large_list)
        assert result.is_ok()
        assert len(result.default_value(None)) == 10
        
        # Map
        op = map(lambda x: x * 2)
        result = await op.execute(large_list[:1000])  # Use smaller subset for map
        assert result.is_ok()
        assert len(result.default_value(None)) == 1000
        
        # Reduce
        op = reduce(lambda a, b: a + b, 0)
        result = await op.execute(list(range(100)))
        assert result.is_ok()
        assert result.default_value(None) == 4950
    
    @pytest.mark.asyncio
    async def test_mixed_type_collections(self):
        """Test operations on collections with mixed types."""
        mixed = [1, "hello", 3.14, True, {"a": 1}, [1, 2], None]
        
        # Filter by type
        op = filter(lambda x: isinstance(x, (int, float)) and x is not None)
        result = await op.execute(mixed)
        assert result.is_ok()
        assert result.default_value(None) == [1, 3.14, True]  # True is instance of int
        
        # Map with type checking
        op = map(lambda x: str(x).upper() if isinstance(x, str) else str(x))
        result = await op.execute(mixed)
        assert result.is_ok()
        mapped = result.default_value(None)
        assert mapped[1] == "HELLO"
    
    @pytest.mark.asyncio
    async def test_deeply_nested_data_access(self):
        """Test operations on deeply nested data structures."""
        data = [
            {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "value": i
                            }
                        }
                    }
                }
            }
            for i in range(5)
        ]
        
        # Use get operation in filter
        op = filter({"level1.level2.level3.level4.value": 3})
        result = await op.execute(data)
        assert result.is_ok()
        assert len(result.default_value(None)) == 1
        
        # Extract deeply nested values
        extract_deep = map(get("level1.level2.level3.level4.value"))
        result = await extract_deep.execute(data)
        assert result.is_ok()
        assert result.default_value(None) == [0, 1, 2, 3, 4]
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self):
        """Test operations with unicode and special characters."""
        data = ["hello", "world", "こんにちは", "🌍", "test123", "!@#$%"]
        
        op = map(to_upper)
        result = await op.execute(data)
        assert result.is_ok()
        assert result.default_value(None)[0] == "HELLO"
        assert result.default_value(None)[2] == "こんにちは"  # Japanese doesn't have uppercase
        
        op = filter(lambda s: len(s) > 3)
        result = await op.execute(data)
        assert result.is_ok()
        filtered = result.default_value(None)
        assert "hello" in filtered
        assert "こんにちは" in filtered
        assert "test123" in filtered
    
    @pytest.mark.asyncio
    async def test_empty_operations(self):
        """Test edge cases with empty inputs."""
        # Empty zip
        op = zip()
        result = await op.execute([1, 2, 3])
        assert result.is_ok()
        assert result.default_value(None) == [(), (), ()]
        
        # Flatten empty nested lists
        op = flatten([[]])
        result = await op.execute()
        assert result.is_ok()
        assert result.default_value(None) == []
        
        # Unique on empty list
        result = await unique.execute([])
        assert result.is_ok()
        assert result.default_value(None) == []


# Test async behavior
class TestAsyncBehavior:
    """Test suite for async-specific behavior."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_in_zip(self):
        """Test that zip operations can run concurrently."""
        call_log = []
        
        @operation
        async def slow_op1(x):
            call_log.append(f"start1-{x}")
            await asyncio.sleep(0.01)
            call_log.append(f"end1-{x}")
            return f"op1-{x}"
        
        @operation
        async def slow_op2(x):
            call_log.append(f"start2-{x}")
            await asyncio.sleep(0.005)
            call_log.append(f"end2-{x}")
            return f"op2-{x}"
        
        op = zip(slow_op1, slow_op2)
        result = await op.execute([1, 2])
        assert result.is_ok()
        assert result.default_value(None) == [("op1-1", "op2-1"), ("op1-2", "op2-2")]
        
        # Operations should be sequential per item, but could be optimized
        # to run concurrently across items in future implementations
    
    @pytest.mark.asyncio
    async def test_async_predicate_in_filter(self):
        """Test filter with async predicate operation."""
        @operation
        async def async_check(x):
            await asyncio.sleep(0.001)
            return x % 2 == 0
        
        op = filter(async_check)
        result = await op.execute([1, 2, 3, 4, 5, 6])
        assert result.is_ok()
        assert result.default_value(None) == [2, 4, 6]
    
    @pytest.mark.asyncio
    async def test_async_transform_in_map(self):
        """Test map with async transformation."""
        @operation
        async def async_transform(x):
            await asyncio.sleep(0.001)
            return {"original": x, "squared": x ** 2}
        
        op = map(async_transform)
        result = await op.execute([1, 2, 3])
        assert result.is_ok()
        transformed = result.default_value(None)
        assert len(transformed) == 3
        assert transformed[0] == {"original": 1, "squared": 1}
        assert transformed[2] == {"original": 3, "squared": 9}


# Integration tests
class TestIntegration:
    """Integration tests combining multiple features."""
    
    @pytest.mark.asyncio
    async def test_data_pipeline_etl(self):
        """Test a complete ETL-style data pipeline."""
        # Raw log data
        logs = [
            {"timestamp": "2024-01-01T10:00:00", "user_id": 1, "action": "login", "success": True},
            {"timestamp": "2024-01-01T10:05:00", "user_id": 2, "action": "login", "success": False},
            {"timestamp": "2024-01-01T10:10:00", "user_id": 1, "action": "purchase", "amount": 99.99},
            {"timestamp": "2024-01-01T10:15:00", "user_id": 3, "action": "login", "success": True},
            {"timestamp": "2024-01-01T10:20:00", "user_id": 2, "action": "login", "success": True},
            {"timestamp": "2024-01-01T10:25:00", "user_id": 3, "action": "purchase", "amount": 49.99},
            {"timestamp": "2024-01-01T10:30:00", "user_id": 1, "action": "logout", "success": True},
        ]
        
        # Extract successful logins
        successful_logins = await (
            filter({"action": "login", "success": True}) >>
            map(lambda log: {
                "user_id": log["user_id"],
                "login_time": log["timestamp"]
            })
        ).execute(logs)
        
        assert successful_logins.is_ok()
        assert len(successful_logins.default_value(None)) == 2
        
        # Extract purchases and calculate total
        purchase_pipeline = (
            filter({"action": "purchase"}) >>
            map(lambda log: log.get("amount", 0)) >>
            reduce(lambda a, b: a + b, 0)
        )
        
        total_purchases = await purchase_pipeline.execute(logs)
        assert total_purchases.is_ok()
        assert total_purchases.default_value(None) == 149.98
        
        # User activity summary
        @operation
        async def summarize_by_user(logs):
            summary = {}
            for log in logs:
                user_id = log["user_id"]
                if user_id not in summary:
                    summary[user_id] = {
                        "actions": [],
                        "successful_actions": 0,
                        "total_spent": 0
                    }
                
                summary[user_id]["actions"].append(log["action"])
                if log.get("success", False):
                    summary[user_id]["successful_actions"] += 1
                if log["action"] == "purchase":
                    summary[user_id]["total_spent"] += log.get("amount", 0)
            
            return list(summary.values())
        
        user_summaries = await summarize_by_user.execute(logs)
        assert user_summaries.is_ok()
        summaries = user_summaries.default_value(None)
        assert len(summaries) == 3
        
        # Find high-value users
        high_value_users = await (
            filter(lambda s: s["total_spent"] > 50)
        ).execute(summaries)
        
        assert high_value_users.is_ok()
        assert len(high_value_users.default_value(None)) == 1
        assert high_value_users.default_value(None)[0]["total_spent"] == 99.99
    
    @pytest.mark.asyncio
    async def test_complex_aggregation_pipeline(self):
        """Test complex aggregation scenarios."""
        # Sales data with multiple dimensions
        sales_data = [
            {"date": "2024-01-01", "product": "A", "region": "North", "quantity": 10, "price": 100},
            {"date": "2024-01-01", "product": "B", "region": "North", "quantity": 5, "price": 200},
            {"date": "2024-01-01", "product": "A", "region": "South", "quantity": 15, "price": 100},
            {"date": "2024-01-02", "product": "A", "region": "North", "quantity": 8, "price": 100},
            {"date": "2024-01-02", "product": "B", "region": "South", "quantity": 12, "price": 200},
            {"date": "2024-01-02", "product": "C", "region": "North", "quantity": 20, "price": 50},
        ]
        
        # Calculate revenue for each sale
        with_revenue = await map(
            lambda sale: {**sale, "revenue": sale["quantity"] * sale["price"]}
        ).execute(sales_data)
        
        assert with_revenue.is_ok()
        sales_with_revenue = with_revenue.default_value(None)
        
        # Group by product and calculate totals
        @operation
        async def group_and_aggregate(sales, group_key):
            groups = {}
            for sale in sales:
                key = sale[group_key]
                if key not in groups:
                    groups[key] = {
                        group_key: key,
                        "total_quantity": 0,
                        "total_revenue": 0,
                        "sale_count": 0
                    }
                
                groups[key]["total_quantity"] += sale["quantity"]
                groups[key]["total_revenue"] += sale["revenue"]
                groups[key]["sale_count"] += 1
            
            return list(groups.values())
        
        # Aggregate by product
        product_summary = await group_and_aggregate.execute(
            sales_with_revenue, "product"
        )
        assert product_summary.is_ok()
        product_stats = product_summary.default_value(None)
        
        # Find best-selling product
        best_product = await (
            reduce(
                lambda a, b: a if a["total_revenue"] > b["total_revenue"] else b
            )
        ).execute(product_stats)
        
        assert best_product.is_ok()
        assert best_product.default_value(None)["product"] == "B"
        assert best_product.default_value(None)["total_revenue"] == 3400
        
        # Multi-level pipeline with various operations
        complex_pipeline = (
            filter(lambda s: s["quantity"] > 5) >>
            map(lambda s: {
                **s,
                "revenue": s["quantity"] * s["price"],
                "high_value": s["quantity"] * s["price"] > 1000
            }) >>
            zip(
                get("product"),
                get("region"),
                get("revenue"),
                get("high_value")
            )
        )
        
        result = await complex_pipeline.execute(sales_data)
        assert result.is_ok()
        processed = result.default_value(None)
        assert len(processed) > 0
        assert all(len(item) == 4 for item in processed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])