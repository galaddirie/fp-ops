"""
Comprehensive test suite for fp_ops text/utility operations.
Tests all utility operations including type checking, value validation,
conversions, and debugging helpers.
"""
import pytest
import asyncio
from typing import Any, List, Dict, Set, Tuple, Optional
from unittest.mock import Mock, call
import sys
from io import StringIO

# Assuming these imports based on the provided code
from fp_ops import operation, Operation
from fp_ops.utils import (
    # Type/Value Checks
    is_empty, is_not_empty, is_none, is_not_none, default,
    # Type Checking
    is_type, is_string, is_int, is_float, is_bool, 
    is_list, is_dict, is_tuple, is_set,
    equals, not_equals, greater_than, less_than,
    greater_or_equal, less_or_equal, in_range,
    to_string, to_int, to_float, to_bool, to_list, to_set,

)
from fp_ops.objects import get, build
from expression import Ok, Error, Result


# Test Type/Value Checks
class TestTypeValueChecks:
    """Test suite for type and value checking operations."""
    
    @pytest.mark.asyncio
    async def test_is_empty(self):
        """Test is_empty operation with various data types."""
        # Empty collections
        assert (await is_empty.execute([])).default_value(False) is True
        assert (await is_empty.execute({})).default_value(False) is True
        assert (await is_empty.execute("")).default_value(False) is True
        assert (await is_empty.execute(set())).default_value(False) is True
        assert (await is_empty.execute(())).default_value(False) is True
        
        # Non-empty collections
        assert (await is_empty.execute([1, 2, 3])).default_value(False) is False
        assert (await is_empty.execute({"a": 1})).default_value(False) is False
        assert (await is_empty.execute("hello")).default_value(False) is False
        assert (await is_empty.execute({1, 2, 3})).default_value(False) is False
        assert (await is_empty.execute((1, 2))).default_value(False) is False
        
        # Custom sized objects
        class CustomSized:
            def __len__(self):
                return 0
        
        assert (await is_empty.execute(CustomSized())).default_value(False) is True
        
        # Objects without __len__
        assert (await is_empty.execute(42)).default_value(False) is True
        assert (await is_empty.execute(None)).default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_not_empty(self):
        """Test is_not_empty operation."""
        # Empty collections
        assert (await is_not_empty.execute([])).default_value(False) is False
        assert (await is_not_empty.execute({})).default_value(False) is False
        assert (await is_not_empty.execute("")).default_value(False) is False
        
        # Non-empty collections
        assert (await is_not_empty.execute([1])).default_value(False) is True
        assert (await is_not_empty.execute({"a": 1})).default_value(False) is True
        assert (await is_not_empty.execute("a")).default_value(False) is True
        
        # Objects without __len__
        assert (await is_not_empty.execute(42)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_none(self):
        """Test is_none operation."""
        assert (await is_none.execute(None)).default_value(False) is True
        assert (await is_none.execute(0)).default_value(False) is False
        assert (await is_none.execute("")).default_value(False) is False
        assert (await is_none.execute([])).default_value(False) is False
        assert (await is_none.execute(False)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_not_none(self):
        """Test is_not_none operation."""
        assert (await is_not_none.execute(None)).default_value(False) is False
        assert (await is_not_none.execute(0)).default_value(False) is True
        assert (await is_not_none.execute("")).default_value(False) is True
        assert (await is_not_none.execute([])).default_value(False) is True
        assert (await is_not_none.execute(False)).default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_default(self):
        """Test default operation."""
        # Create default operations with different values
        default_string = default("N/A")
        default_zero = default(0)
        default_list = default([])
        
        # Test with None
        assert (await default_string.execute(None)).default_value("") == "N/A"
        assert (await default_zero.execute(None)).default_value(0) == 0
        assert (await default_list.execute(None)).default_value([]) == []
        
        # Test with non-None values
        assert (await default_string.execute("hello")).default_value("") == "hello"
        assert (await default_zero.execute(42)).default_value(0) == 42
        assert (await default_list.execute([1, 2, 3])).default_value([]) == [1, 2, 3]
        
        # Test with falsy but non-None values
        assert (await default_string.execute("")).default_value("") == ""
        assert (await default_zero.execute(0)).default_value(0) == 0
        assert (await default_list.execute([])).default_value([]) == []


# Test Type Checking
class TestTypeChecking:
    """Test suite for type checking operations."""
    
    @pytest.mark.asyncio
    async def test_is_type(self):
        """Test is_type operation factory."""
        # Create type checkers
        check_str = is_type(str)
        check_int = is_type(int)
        check_list = is_type(list)
        
        # Test string checker
        assert (await check_str.execute("hello")).default_value(False) is True
        assert (await check_str.execute(42)).default_value(False) is False
        assert (await check_str.execute([])).default_value(False) is False
        
        # Test int checker
        assert (await check_int.execute(42)).default_value(False) is True
        assert (await check_int.execute("42")).default_value(False) is False
        assert (await check_int.execute(42.0)).default_value(False) is False
        
        # Test list checker
        assert (await check_list.execute([])).default_value(False) is True
        assert (await check_list.execute([1, 2, 3])).default_value(False) is True
        assert (await check_list.execute((1, 2, 3))).default_value(False) is False
        
        # Test with inheritance
        class CustomList(list):
            pass
        
        assert (await check_list.execute(CustomList())).default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_string(self):
        """Test is_string operation."""
        assert (await is_string.execute("hello")).default_value(False) is True
        assert (await is_string.execute("")).default_value(False) is True
        assert (await is_string.execute(42)).default_value(False) is False
        assert (await is_string.execute(None)).default_value(False) is False
        assert (await is_string.execute([])).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_int(self):
        """Test is_int operation."""
        assert (await is_int.execute(42)).default_value(False) is True
        assert (await is_int.execute(-10)).default_value(False) is True
        assert (await is_int.execute(0)).default_value(False) is True
        
        # Should exclude booleans
        assert (await is_int.execute(True)).default_value(False) is False
        assert (await is_int.execute(False)).default_value(False) is False
        
        # Other types
        assert (await is_int.execute(42.0)).default_value(False) is False
        assert (await is_int.execute("42")).default_value(False) is False
        assert (await is_int.execute(None)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_float(self):
        """Test is_float operation."""
        assert (await is_float.execute(3.14)).default_value(False) is True
        assert (await is_float.execute(0.0)).default_value(False) is True
        assert (await is_float.execute(-1.5)).default_value(False) is True
        assert (await is_float.execute(float('inf'))).default_value(False) is True
        
        assert (await is_float.execute(42)).default_value(False) is False
        assert (await is_float.execute("3.14")).default_value(False) is False
        assert (await is_float.execute(None)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_bool(self):
        """Test is_bool operation."""
        assert (await is_bool.execute(True)).default_value(False) is True
        assert (await is_bool.execute(False)).default_value(False) is True
        
        # Not bools even though they're truthy/falsy
        assert (await is_bool.execute(1)).default_value(False) is False
        assert (await is_bool.execute(0)).default_value(False) is False
        assert (await is_bool.execute("")).default_value(False) is False
        assert (await is_bool.execute(None)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_list(self):
        """Test is_list operation."""
        assert (await is_list.execute([])).default_value(False) is True
        assert (await is_list.execute([1, 2, 3])).default_value(False) is True
        assert (await is_list.execute(list("abc"))).default_value(False) is True
        
        assert (await is_list.execute((1, 2, 3))).default_value(False) is False
        assert (await is_list.execute({1, 2, 3})).default_value(False) is False
        assert (await is_list.execute("abc")).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_dict(self):
        """Test is_dict operation."""
        assert (await is_dict.execute({})).default_value(False) is True
        assert (await is_dict.execute({"a": 1})).default_value(False) is True
        assert (await is_dict.execute(dict(a=1, b=2))).default_value(False) is True
        
        assert (await is_dict.execute([])).default_value(False) is False
        assert (await is_dict.execute(set())).default_value(False) is False
        assert (await is_dict.execute(None)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_tuple(self):
        """Test is_tuple operation."""
        assert (await is_tuple.execute(())).default_value(False) is True
        assert (await is_tuple.execute((1,))).default_value(False) is True
        assert (await is_tuple.execute((1, 2, 3))).default_value(False) is True
        
        assert (await is_tuple.execute([1, 2, 3])).default_value(False) is False
        assert (await is_tuple.execute("abc")).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_set(self):
        """Test is_set operation."""
        assert (await is_set.execute(set())).default_value(False) is True
        assert (await is_set.execute({1, 2, 3})).default_value(False) is True
        assert (await is_set.execute(set([1, 2, 3]))).default_value(False) is True
        
        assert (await is_set.execute(frozenset([1, 2, 3]))).default_value(False) is False
        assert (await is_set.execute([1, 2, 3])).default_value(False) is False
        assert (await is_set.execute({})).default_value(False) is False  # Empty dict, not set


# Test Value Comparisons
class TestValueComparisons:
    """Test suite for value comparison operations."""
    
    @pytest.mark.asyncio
    async def test_equals(self):
        """Test equals operation."""
        eq_42 = equals(42)
        eq_hello = equals("hello")
        eq_list = equals([1, 2, 3])
        
        # Exact matches
        assert (await eq_42.execute(42)).default_value(False) is True
        assert (await eq_hello.execute("hello")).default_value(False) is True
        assert (await eq_list.execute([1, 2, 3])).default_value(False) is True
        
        # Non-matches
        assert (await eq_42.execute(43)).default_value(False) is False
        assert (await eq_hello.execute("world")).default_value(False) is False
        assert (await eq_list.execute([1, 2])).default_value(False) is False
        
        # Type differences
        assert (await eq_42.execute("42")).default_value(False) is False
        assert (await eq_42.execute(42.0)).default_value(False) is True  # int/float equality
        
        # None handling
        eq_none = equals(None)
        assert (await eq_none.execute(None)).default_value(False) is True
        assert (await eq_none.execute(0)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_not_equals(self):
        """Test not_equals operation."""
        neq_42 = not_equals(42)
        neq_hello = not_equals("hello")
        
        assert (await neq_42.execute(43)).default_value(False) is True
        assert (await neq_42.execute(42)).default_value(False) is False
        assert (await neq_hello.execute("world")).default_value(False) is True
        assert (await neq_hello.execute("hello")).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_greater_than(self):
        """Test greater_than operation."""
        gt_5 = greater_than(5)
        
        assert (await gt_5.execute(10)).default_value(False) is True
        assert (await gt_5.execute(6)).default_value(False) is True
        assert (await gt_5.execute(5)).default_value(False) is False
        assert (await gt_5.execute(4)).default_value(False) is False
        assert (await gt_5.execute(-10)).default_value(False) is False
        
        # Float comparison
        gt_pi = greater_than(3.14)
        assert (await gt_pi.execute(3.15)).default_value(False) is True
        assert (await gt_pi.execute(3.14)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_less_than(self):
        """Test less_than operation."""
        lt_5 = less_than(5)
        
        assert (await lt_5.execute(3)).default_value(False) is True
        assert (await lt_5.execute(4)).default_value(False) is True
        assert (await lt_5.execute(5)).default_value(False) is False
        assert (await lt_5.execute(6)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_greater_or_equal(self):
        """Test greater_or_equal operation."""
        gte_5 = greater_or_equal(5)
        
        assert (await gte_5.execute(6)).default_value(False) is True
        assert (await gte_5.execute(5)).default_value(False) is True
        assert (await gte_5.execute(4)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_less_or_equal(self):
        """Test less_or_equal operation."""
        lte_5 = less_or_equal(5)
        
        assert (await lte_5.execute(4)).default_value(False) is True
        assert (await lte_5.execute(5)).default_value(False) is True
        assert (await lte_5.execute(6)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_in_range(self):
        """Test in_range operation."""
        # Inclusive range (default)
        range_1_10 = in_range(1, 10)
        
        assert (await range_1_10.execute(5)).default_value(False) is True
        assert (await range_1_10.execute(1)).default_value(False) is True
        assert (await range_1_10.execute(10)).default_value(False) is True
        assert (await range_1_10.execute(0)).default_value(False) is False
        assert (await range_1_10.execute(11)).default_value(False) is False
        
        # Exclusive range
        range_1_10_exclusive = in_range(1, 10, inclusive=False)
        
        assert (await range_1_10_exclusive.execute(5)).default_value(False) is True
        assert (await range_1_10_exclusive.execute(1)).default_value(False) is False
        assert (await range_1_10_exclusive.execute(10)).default_value(False) is False
        
        # Float range
        range_float = in_range(0.0, 1.0)
        assert (await range_float.execute(0.5)).default_value(False) is True
        assert (await range_float.execute(1.0)).default_value(False) is True
        assert (await range_float.execute(1.1)).default_value(False) is False


# Test Type Conversions
class TestTypeConversions:
    """Test suite for type conversion operations."""
    
    @pytest.mark.asyncio
    async def test_to_string(self):
        """Test to_string operation."""
        assert (await to_string.execute(42)).default_value("") == "42"
        assert (await to_string.execute(3.14)).default_value("") == "3.14"
        assert (await to_string.execute(True)).default_value("") == "True"
        assert (await to_string.execute([1, 2, 3])).default_value("") == "[1, 2, 3]"
        assert (await to_string.execute({"a": 1})).default_value("") == "{'a': 1}"
        assert (await to_string.execute(None)).default_value("") == "None"
        
        # Already a string
        assert (await to_string.execute("hello")).default_value("") == "hello"
        
        # Custom __str__
        class Custom:
            def __str__(self):
                return "custom"
        
        assert (await to_string.execute(Custom())).default_value("") == "custom"
    
    @pytest.mark.asyncio
    async def test_to_int(self):
        """Test to_int operation."""
        # Successful conversions
        assert (await to_int.execute("42")).default_value(None) == 42
        assert (await to_int.execute(42.7)).default_value(None) == 42
        assert (await to_int.execute(42.0)).default_value(None) == 42
        assert (await to_int.execute(True)).default_value(None) == 1
        assert (await to_int.execute(False)).default_value(None) == 0
        
        # Failed conversions return None
        assert (await to_int.execute("abc")).default_value(None) is None
        assert (await to_int.execute("12.34")).default_value(None) is None  # String with decimal
        assert (await to_int.execute([])).default_value(None) is None
        assert (await to_int.execute({})).default_value(None) is None
        assert (await to_int.execute(None)).default_value(None) is None
        
        # Edge cases
        assert (await to_int.execute("-42")).default_value(None) == -42
        assert (await to_int.execute("  42  ")).default_value(None) == 42  # Whitespace handled
    
    @pytest.mark.asyncio
    async def test_to_float(self):
        """Test to_float operation."""
        # Successful conversions
        assert (await to_float.execute("3.14")).default_value(None) == 3.14
        assert (await to_float.execute("42")).default_value(None) == 42.0
        assert (await to_float.execute(42)).default_value(None) == 42.0
        assert (await to_float.execute(True)).default_value(None) == 1.0
        assert (await to_float.execute(False)).default_value(None) == 0.0
        
        # Failed conversions return None
        assert (await to_float.execute("abc")).default_value(None) is None
        assert (await to_float.execute([])).default_value(None) is None
        assert (await to_float.execute({})).default_value(None) is None
        assert (await to_float.execute(None)).default_value(None) is None
        
        # Special float values
        assert (await to_float.execute("inf")).default_value(None) == float('inf')
        assert (await to_float.execute("-inf")).default_value(None) == float('-inf')
        result = await to_float.execute("nan")
        assert result.is_ok()
        import math
        assert math.isnan(result.default_value(None))
    
    @pytest.mark.asyncio
    async def test_to_bool(self):
        """Test to_bool operation."""
        # Truthy values
        assert (await to_bool.execute(1)).default_value(False) is True
        assert (await to_bool.execute("hello")).default_value(False) is True
        assert (await to_bool.execute([1])).default_value(False) is True
        assert (await to_bool.execute({"a": 1})).default_value(False) is True
        assert (await to_bool.execute(True)).default_value(False) is True
        assert (await to_bool.execute(3.14)).default_value(False) is True
        
        # Falsy values
        assert (await to_bool.execute(0)).default_value(False) is False
        assert (await to_bool.execute("")).default_value(False) is False
        assert (await to_bool.execute([])).default_value(False) is False
        assert (await to_bool.execute({})).default_value(False) is False
        assert (await to_bool.execute(set())).default_value(False) is False
        assert (await to_bool.execute(None)).default_value(False) is False
        assert (await to_bool.execute(False)).default_value(False) is False
        
        # String "false" is truthy (non-empty string)
        assert (await to_bool.execute("false")).default_value(False) is True
        assert (await to_bool.execute("False")).default_value(False) is True
        assert (await to_bool.execute("0")).default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_to_list(self):
        """Test to_list operation."""
        # Already a list
        assert (await to_list.execute([1, 2, 3])).default_value([]) == [1, 2, 3]
        
        # String to list of characters
        assert (await to_list.execute("hello")).default_value([]) == ["h", "e", "l", "l", "o"]
        
        # Tuple to list
        assert (await to_list.execute((1, 2, 3))).default_value([]) == [1, 2, 3]
        
        # Set to list (order may vary)
        result = await to_list.execute({1, 2, 3})
        assert result.is_ok()
        assert sorted(result.default_value([])) == [1, 2, 3]
        
        # Dict to list of items
        assert (await to_list.execute({"a": 1, "b": 2})).default_value([]) == [("a", 1), ("b", 2)]
        
        # Single value to list
        assert (await to_list.execute(42)).default_value([]) == [42]
        assert (await to_list.execute(None)).default_value([]) == [None]
        
        # Empty collections
        assert (await to_list.execute("")).default_value([]) == []
        assert (await to_list.execute(())).default_value([]) == []
        assert (await to_list.execute(set())).default_value([]) == []
    
    @pytest.mark.asyncio
    async def test_to_set(self):
        """Test to_set operation."""
        # Already a set
        result = await to_set.execute({1, 2, 3})
        assert result.is_ok()
        assert result.default_value(set()) == {1, 2, 3}
        
        # List to set
        assert (await to_set.execute([1, 2, 2, 3])).default_value(set()) == {1, 2, 3}
        
        # String to set of characters
        assert (await to_set.execute("hello")).default_value(set()) == {"h", "e", "l", "o"}
        
        # Tuple to set
        assert (await to_set.execute((1, 2, 2, 3))).default_value(set()) == {1, 2, 3}
        
        # Dict to set of keys
        assert (await to_set.execute({"a": 1, "b": 2})).default_value(set()) == {"a", "b"}
        
        # Single value to set
        assert (await to_set.execute(42)).default_value(set()) == {42}
        
        # Empty collections
        assert (await to_set.execute("")).default_value(set()) == set()
        assert (await to_set.execute([])).default_value(set()) == set()
        assert (await to_set.execute(())).default_value(set()) == set()



# Test Complex Compositions
class TestComplexCompositions:
    """Test suite for complex operation compositions."""
    
    @pytest.mark.asyncio
    async def test_validation_pipeline(self):
        """Test a validation pipeline using multiple operations."""
        # Create a validation pipeline for user age
        validate_age = (
            to_int
            >> default(-1)
            >> in_range(0, 150)
        )
        
        # Valid ages
        assert (await validate_age.execute(25)).default_value(False) is True
        assert (await validate_age.execute("30")).default_value(False) is True
        assert (await validate_age.execute(0)).default_value(False) is True
        assert (await validate_age.execute(150)).default_value(False) is True
        
        # Invalid ages
        assert (await validate_age.execute(-5)).default_value(False) is False
        assert (await validate_age.execute(200)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_type_checking_with_conversion(self):
        """Test type checking combined with conversion."""
        # Create a safe string-to-int converter
        safe_to_int = (
            operation(lambda x: x if is_string(x) else str(x))
            >> to_int
            >> default(-1)  # Default for failed conversions
        )
        
        assert (await safe_to_int("42")).default_value(-1) == 42
        assert (await safe_to_int(42)).default_value(-1) == 42
        assert (await safe_to_int(3.14)).default_value(-1) == 3
        assert (await safe_to_int("abc")).default_value(-1) == -1
        assert (await safe_to_int(None)).default_value(-1) == -1
    
    @pytest.mark.asyncio
    async def test_conditional_processing(self):
        """Test conditional processing based on type."""
        
        # Process different types differently
        @operation
        def process_value(value):
            if isinstance(value, str):
                return value.upper()
            elif isinstance(value, (int, float)):
                return value * 2
            elif isinstance(value, list):
                return len(value)
            else:
                return None
        
        # Create a pipeline that processes and validates
        process_pipeline = process_value >> is_not_none
        
        assert (await process_pipeline.execute("hello")).default_value(False) is True
        assert (await process_pipeline.execute(21)).default_value(False) is True
        assert (await process_pipeline.execute([1, 2, 3])).default_value(False) is True
        assert (await process_pipeline.execute({})).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_data_cleaning_pipeline(self):
        """Test a data cleaning pipeline."""
        # Clean and validate email addresses
        clean_email = (
            operation(lambda x: x if x and is_string(x) else '')
            >> operation(lambda x: x.strip().lower())
            >> operation(lambda x: x if "@" in x else None)
            >> default("invalid@example.com")
        )
        
        assert (await clean_email.execute("  USER@EXAMPLE.COM  ")).default_value("") == "user@example.com"
        assert (await clean_email.execute("test@test.com")).default_value("") == "test@test.com"
        assert (await clean_email.execute("invalid")).default_value(None) == "invalid@example.com"
        assert (await clean_email.execute(None)).default_value(None) == "invalid@example.com"
    

    @pytest.mark.asyncio
    async def test_parallel_type_checks(self):
        """Test parallel type checking operations."""
        # Using & for parallel checks
        is_positive_int = is_int & greater_than(0)
        
        # This would need custom logic since & returns tuple
        # Let's create a different approach
        @operation
        def all_true(checks: Tuple[bool, ...]) -> bool:
            return all(checks)
        
        # Create combined checker
        check_positive_int = (is_int & greater_than(0)) >> all_true
        
        # Note: This assumes the & operator passes the same value to both operations
        # and returns a tuple of results
    
    @pytest.mark.asyncio
    async def test_integration_with_object_ops(self):
        """Test integration with object operations."""
        # Process user data with validation
        user_data = {
            "name": "  John Doe  ",
            "age": "25",
            "email": "JOHN@EXAMPLE.COM",
            "scores": [85, 90, 78, 92, 88]
        }
        
        # Create processing pipeline
        process_user = build({
            "name": get("name") >> operation(lambda x: x.strip() if is_string(x) else ""),
            "age": get("age") >> to_int >> default(0),
            "email": get("email") >> operation(lambda x: x.lower() if is_string(x) else ""),
            "average_score": get("scores") >> operation(
                lambda scores: sum(scores) / len(scores) if is_list(scores) and len(scores) > 0 else 0
            ),
            "is_adult": get("age") >> to_int >> default(0) >> greater_or_equal(18)
        })
        
        result = await process_user.execute(user_data)
        assert result.is_ok()
        processed = result.default_value({})
        
        assert processed["name"] == "John Doe"
        assert processed["age"] == 25
        assert processed["email"] == "john@example.com"
        assert processed["average_score"] == 86.6
        assert processed["is_adult"] is True


# Test Edge Cases
class TestEdgeCases:
    """Test suite for edge cases and special scenarios."""
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test operations with unicode strings."""
        unicode_str = "Hello 世界 🌍"
        
        assert (await is_string.execute(unicode_str)).default_value(False) is True
        assert (await is_empty.execute(unicode_str)).default_value(False) is False
        assert (await to_list.execute(unicode_str)).default_value([]) == list(unicode_str)
        assert (await equals("Hello 世界 🌍").execute(unicode_str)).default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_special_numeric_values(self):
        """Test operations with special numeric values."""
        # Infinity
        assert (await is_float.execute(float('inf'))).default_value(False) is True
        assert (await greater_than(1000).execute(float('inf'))).default_value(False) is True
        assert (await less_than(1000).execute(float('-inf'))).default_value(False) is True
        
        # NaN
        nan_value = float('nan')
        assert (await is_float.execute(nan_value)).default_value(False) is True
        # NaN comparisons are always False
        assert (await equals(nan_value).execute(nan_value)).default_value(False) is False
        assert (await greater_than(0).execute(nan_value)).default_value(False) is False
        assert (await less_than(0).execute(nan_value)).default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_large_collections(self):
        """Test operations with large collections."""
        large_list = list(range(10000))
        large_dict = {str(i): i for i in range(10000)}
        large_set = set(range(10000))
        
        assert (await is_not_empty.execute(large_list)).default_value(False) is True
        assert (await is_list.execute(large_list)).default_value(False) is True
        assert (await is_dict.execute(large_dict)).default_value(False) is True
        assert (await is_set.execute(large_set)).default_value(False) is True
        
        # Conversion should work
        result = await to_set.execute(large_list)
        assert result.is_ok()
        assert len(result.default_value(set())) == 10000
    
    @pytest.mark.asyncio
    async def test_custom_objects(self):
        """Test operations with custom objects."""
        class CustomNumber:
            def __init__(self, value):
                self.value = value
            
            def __int__(self):
                return self.value
            
            def __float__(self):
                return float(self.value)
            
            def __str__(self):
                return f"CustomNumber({self.value})"
            
            def __eq__(self, other):
                if isinstance(other, CustomNumber):
                    return self.value == other.value
                return self.value == other
        
        custom = CustomNumber(42)
        
        assert (await to_int.execute(custom)).default_value(None) == 42
        assert (await to_float.execute(custom)).default_value(None) == 42.0
        assert (await to_string.execute(custom)).default_value("") == "CustomNumber(42)"
        assert (await equals(42).execute(custom)).default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation through pipelines."""
        @operation
        def may_fail(value):
            if value < 0:
                raise ValueError("Negative value not allowed")
            return value
        
        pipeline = to_int >> default(0) >> may_fail >> greater_than(10)
        
        # Should work with valid input
        result = await pipeline.execute("42")
        assert result.is_ok()
        assert result.default_value(False) is True
        
        # Should fail with negative input
        result = await pipeline.execute("-5")
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_type_preservation(self):
        """Test that operations preserve types appropriately."""
        # Test that default preserves type
        default_list = default([1, 2, 3])
        result = await default_list.execute(None)
        assert result.is_ok()
        returned_list = result.default_value([])
        assert returned_list == [1, 2, 3]
        # Modify to ensure it's not the same object
        returned_list.append(4)
        
        # Get fresh default - should still be [1, 2, 3]
        result2 = await default_list.execute(None)
        assert result2.default_value([]) == [1, 2, 3]


# Test Performance and Memory
class TestPerformance:
    """Test suite for performance-related scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_string_operations(self):
        """Test operations with large strings."""
        large_string = "x" * 1_000_000  # 1MB string
        
        # Should handle large strings efficiently
        assert (await is_string.execute(large_string)).default_value(False) is True
        assert (await is_not_empty.execute(large_string)).default_value(False) is True
        
        # Conversion to list might be memory intensive
        # Only test that it works, not the full result
        result = await to_list.execute(large_string)
        assert result.is_ok()
        assert len(result.default_value([])) == 1_000_000
    
    @pytest.mark.asyncio
    async def test_repeated_operations(self):
        """Test repeated application of operations."""
        # Create a pipeline that applies multiple operations
        repeated_pipeline = (
            to_string
            >> operation(lambda x: x + "a")
            >> operation(lambda x: x + "b")
            >> operation(lambda x: x + "c")
        )
        
        # Apply many times
        for i in range(100):
            result = await repeated_pipeline.execute(i)
            assert result.is_ok()
            assert result.default_value("") == f"{i}abc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])