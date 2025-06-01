"""
Comprehensive test suite for fp_ops.text module.
Tests all text operations including string manipulation, pattern matching, and validation.
Covers edge cases, composition, unicode, and error handling.
"""
import pytest
import re
from typing import List, Optional

# Assuming these imports based on the provided code
from fp_ops import operation, Operation
from fp_ops.text import (
    # Basic String Operations
    split, join, replace, to_lower, to_upper,
    strip, lstrip, rstrip,
    # Case Conversions
    capitalize, title,
    # String Checks
    starts_with, ends_with, contains,
    # Pattern Matching
    match, search, find_all, sub,
    # String Validation
    is_alpha, is_numeric, is_alphanumeric,
    is_whitespace, is_upper, is_lower
)
from expression import Ok, Error, Result


# Test fixtures
@pytest.fixture
def sample_text():
    return "Hello World"


@pytest.fixture
def multiline_text():
    return """Line 1
Line 2
Line 3"""


@pytest.fixture
def unicode_text():
    return "Hello 世界 🌍"


@pytest.fixture
def mixed_case_text():
    return "HeLLo WoRLd 123"


# Test Basic String Operations
class TestBasicStringOperations:
    """Test suite for basic string operations."""
    
    # Test split operation
    @pytest.mark.asyncio
    async def test_split_default_delimiter(self, sample_text):
        """Test splitting with default space delimiter."""
        op = split()
        result = await op.execute(sample_text)
        assert result.is_ok()
        assert result.default_value([]) == ["Hello", "World"]
    
    @pytest.mark.asyncio
    async def test_split_custom_delimiter(self):
        """Test splitting with custom delimiter."""
        op = split(",")
        result = await op.execute("apple,banana,orange")
        assert result.is_ok()
        assert result.default_value([]) == ["apple", "banana", "orange"]
    
    @pytest.mark.asyncio
    async def test_split_empty_string(self):
        """Test splitting empty string."""
        op = split()
        result = await op.execute("")
        assert result.is_ok()
        assert result.default_value([]) == [""]
    
    @pytest.mark.asyncio
    async def test_split_no_delimiter_found(self):
        """Test splitting when delimiter not found."""
        op = split(",")
        result = await op.execute("hello world")
        assert result.is_ok()
        assert result.default_value([]) == ["hello world"]
    
    @pytest.mark.asyncio
    async def test_split_non_string_input(self):
        """Test split with non-string input."""
        op = split()
        result = await op.execute(123)
        assert result.is_ok()
        assert result.default_value([]) == []
    
    @pytest.mark.asyncio
    async def test_split_multiline(self, multiline_text):
        """Test splitting multiline text."""
        op = split("\n")
        result = await op.execute(multiline_text)
        assert result.is_ok()
        assert result.default_value([]) == ["Line 1", "Line 2", "Line 3"]
    
    # Test join operation
    @pytest.mark.asyncio
    async def test_join_default_delimiter(self):
        """Test joining with default space delimiter."""
        op = join()
        result = await op.execute(["Hello", "World"])
        assert result.is_ok()
        assert result.default_value("") == "Hello World"
    
    @pytest.mark.asyncio
    async def test_join_custom_delimiter(self):
        """Test joining with custom delimiter."""
        op = join(", ")
        result = await op.execute(["apple", "banana", "orange"])
        assert result.is_ok()
        assert result.default_value("") == "apple, banana, orange"
    
    @pytest.mark.asyncio
    async def test_join_mixed_types(self):
        """Test joining mixed type items."""
        op = join("-")
        result = await op.execute([1, 2, 3, "four"])
        assert result.is_ok()
        assert result.default_value("") == "1-2-3-four"
    
    @pytest.mark.asyncio
    async def test_join_empty_list(self):
        """Test joining empty list."""
        op = join(",")
        result = await op.execute([])
        assert result.is_ok()
        assert result.default_value("") == ""
    
    @pytest.mark.asyncio
    async def test_join_single_item(self):
        """Test joining single item list."""
        op = join(",")
        result = await op.execute(["alone"])
        assert result.is_ok()
        assert result.default_value("") == "alone"
    
    # Test replace operation
    @pytest.mark.asyncio
    async def test_replace_basic(self, sample_text):
        """Test basic string replacement."""
        op = replace("World", "Universe")
        result = await op.execute(sample_text)
        assert result.is_ok()
        assert result.default_value("") == "Hello Universe"
    
    @pytest.mark.asyncio
    async def test_replace_multiple_occurrences(self):
        """Test replacing multiple occurrences."""
        op = replace("o", "0")
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value("") == "Hell0 W0rld"
    
    @pytest.mark.asyncio
    async def test_replace_with_count(self):
        """Test replace with count limit."""
        op = replace("o", "0", count=1)
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value("") == "Hell0 World"
    
    @pytest.mark.asyncio
    async def test_replace_not_found(self, sample_text):
        """Test replace when substring not found."""
        op = replace("xyz", "abc")
        result = await op.execute(sample_text)
        assert result.is_ok()
        assert result.default_value("") == sample_text
    
    @pytest.mark.asyncio
    async def test_replace_non_string_input(self):
        """Test replace with non-string input."""
        op = replace("a", "b")
        result = await op.execute(123)
        assert result.is_ok()
        assert result.default_value(123) == 123
    
    # Test case conversion operations
    @pytest.mark.asyncio
    async def test_to_lower(self, mixed_case_text):
        """Test converting to lowercase."""
        result = await to_lower.execute(mixed_case_text)
        assert result.is_ok()
        assert result.default_value("") == "hello world 123"
    
    @pytest.mark.asyncio
    async def test_to_upper(self, mixed_case_text):
        """Test converting to uppercase."""
        result = await to_upper.execute(mixed_case_text)
        assert result.is_ok()
        assert result.default_value("") == "HELLO WORLD 123"
    
    @pytest.mark.asyncio
    async def test_to_lower_unicode(self, unicode_text):
        """Test lowercase conversion with unicode."""
        result = await to_lower.execute(unicode_text)
        assert result.is_ok()
        assert result.default_value("") == "hello 世界 🌍"
    
    @pytest.mark.asyncio
    async def test_case_conversion_non_string(self):
        """Test case conversion with non-string input."""
        result = await to_lower.execute(123)
        assert result.is_ok()
        assert result.default_value(123) == 123
    
    # Test strip operations
    @pytest.mark.asyncio
    async def test_strip_whitespace(self):
        """Test stripping whitespace."""
        result = await strip.execute("  hello world  ")
        assert result.is_ok()
        assert result.default_value("") == "hello world"
    
    @pytest.mark.asyncio
    async def test_strip_custom_chars(self):
        """Test stripping custom characters."""
        result = await strip.execute("__hello__", "_")
        assert result.is_ok()
        assert result.default_value("") == "hello"
    
    @pytest.mark.asyncio
    async def test_lstrip(self):
        """Test left strip."""
        result = await lstrip.execute("  hello  ")
        assert result.is_ok()
        assert result.default_value("") == "hello  "
    
    @pytest.mark.asyncio
    async def test_rstrip(self):
        """Test right strip."""
        result = await rstrip.execute("  hello  ")
        assert result.is_ok()
        assert result.default_value("") == "  hello"
    
    @pytest.mark.asyncio
    async def test_strip_newlines(self):
        """Test stripping newlines."""
        result = await strip.execute("\nhello\n")
        assert result.is_ok()
        assert result.default_value("") == "hello"
    
    @pytest.mark.asyncio
    async def test_strip_empty_string(self):
        """Test stripping empty string."""
        result = await strip.execute("")
        assert result.is_ok()
        assert result.default_value("") == ""


# Test Case Conversions
class TestCaseConversions:
    """Test suite for case conversion operations."""
    
    @pytest.mark.asyncio
    async def test_capitalize(self):
        """Test capitalize operation."""
        result = await capitalize.execute("hello world")
        assert result.is_ok()
        assert result.default_value("") == "Hello world"
    
    @pytest.mark.asyncio
    async def test_capitalize_already_capitalized(self):
        """Test capitalize on already capitalized string."""
        result = await capitalize.execute("Hello World")
        assert result.is_ok()
        assert result.default_value("") == "Hello world"
    
    @pytest.mark.asyncio
    async def test_capitalize_all_caps(self):
        """Test capitalize on all caps string."""
        result = await capitalize.execute("HELLO WORLD")
        assert result.is_ok()
        assert result.default_value("") == "Hello world"
    
    @pytest.mark.asyncio
    async def test_title(self):
        """Test title case conversion."""
        result = await title.execute("hello world")
        assert result.is_ok()
        assert result.default_value("") == "Hello World"
    
    @pytest.mark.asyncio
    async def test_title_with_apostrophes(self):
        """Test title case with apostrophes."""
        result = await title.execute("it's a beautiful day")
        assert result.is_ok()
        assert result.default_value("") == "It'S A Beautiful Day"  # Note: Python's title() behavior
    
    @pytest.mark.asyncio
    async def test_title_mixed_case(self):
        """Test title case on mixed case input."""
        result = await title.execute("hELLo WoRLd")
        assert result.is_ok()
        assert result.default_value("") == "Hello World"


# Test String Checks
class TestStringChecks:
    """Test suite for string check operations."""
    
    @pytest.mark.asyncio
    async def test_starts_with_true(self):
        """Test starts_with when true."""
        op = starts_with("Hello")
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_starts_with_false(self):
        """Test starts_with when false."""
        op = starts_with("World")
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_starts_with_substring(self):
        """Test starts_with with substring range."""
        op = starts_with("World", start=6)
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_ends_with_true(self):
        """Test ends_with when true."""
        op = ends_with("World")
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_ends_with_false(self):
        """Test ends_with when false."""
        op = ends_with("Hello")
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_contains_true(self):
        """Test contains when substring exists."""
        op = contains("lo Wo")
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_contains_false(self):
        """Test contains when substring doesn't exist."""
        op = contains("xyz")
        result = await op.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_contains_empty_string(self):
        """Test contains with empty string."""
        op = contains("")
        result = await op.execute("Hello")
        assert result.is_ok()
        assert result.default_value(False) is True  # Empty string is in every string
    
    @pytest.mark.asyncio
    async def test_checks_non_string_input(self):
        """Test string checks with non-string input."""
        op = starts_with("Hello")
        result = await op.execute(123)
        assert result.is_ok()
        assert result.default_value(False) is False


# Test Pattern Matching
class TestPatternMatching:
    """Test suite for pattern matching operations."""
    
    @pytest.mark.asyncio
    async def test_match_success(self):
        """Test successful pattern match."""
        op = match(r'^[A-Z][a-z]+')
        result = await op.execute("Hello")
        assert result.is_ok()
        match_obj = result.default_value(None)
        assert match_obj is not None
        assert match_obj.group() == "Hello"
    
    @pytest.mark.asyncio
    async def test_match_failure(self):
        """Test failed pattern match."""
        op = match(r'^\d+')
        result = await op.execute("Hello")
        assert result.is_ok()
        assert result.default_value(None) is None
    
    @pytest.mark.asyncio
    async def test_match_with_flags(self):
        """Test match with regex flags."""
        op = match(r'^hello', flags=re.IGNORECASE)
        result = await op.execute("HELLO world")
        assert result.is_ok()
        match_obj = result.default_value(None)
        assert match_obj is not None
        assert match_obj.group() == "HELLO"
    
    @pytest.mark.asyncio
    async def test_match_compiled_pattern(self):
        """Test match with compiled pattern."""
        pattern = re.compile(r'^\w+')
        op = match(pattern)
        result = await op.execute("Hello123")
        assert result.is_ok()
        match_obj = result.default_value(None)
        assert match_obj is not None
        assert match_obj.group() == "Hello123"
    
    @pytest.mark.asyncio
    async def test_search_anywhere(self):
        """Test search finding pattern anywhere in string."""
        op = search(r'\d+')
        result = await op.execute("abc123def456")
        assert result.is_ok()
        match_obj = result.default_value(None)
        assert match_obj is not None
        assert match_obj.group() == "123"
    
    @pytest.mark.asyncio
    async def test_search_not_found(self):
        """Test search when pattern not found."""
        op = search(r'\d+')
        result = await op.execute("abcdef")
        assert result.is_ok()
        assert result.default_value(None) is None
    
    @pytest.mark.asyncio
    async def test_find_all_multiple_matches(self):
        """Test finding all matches."""
        op = find_all(r'\d+')
        result = await op.execute("abc123def456ghi789")
        assert result.is_ok()
        assert result.default_value([]) == ["123", "456", "789"]
    
    @pytest.mark.asyncio
    async def test_find_all_no_matches(self):
        """Test find_all with no matches."""
        op = find_all(r'\d+')
        result = await op.execute("abcdef")
        assert result.is_ok()
        assert result.default_value([]) == []
    
    @pytest.mark.asyncio
    async def test_find_all_groups(self):
        """Test find_all with groups."""
        op = find_all(r'([a-z]+)(\d+)')
        result = await op.execute("abc123def456")
        assert result.is_ok()
        # findall returns tuples when there are groups
        assert result.default_value([]) == [("abc", "123"), ("def", "456")]
    
    @pytest.mark.asyncio
    async def test_sub_basic(self):
        """Test basic pattern substitution."""
        op = sub(r'\d+', 'X')
        result = await op.execute("abc123def456")
        assert result.is_ok()
        assert result.default_value("") == "abcXdefX"
    
    @pytest.mark.asyncio
    async def test_sub_with_backreferences(self):
        """Test substitution with backreferences."""
        op = sub(r'(\w+)@(\w+)', r'\2@\1')
        result = await op.execute("user@domain")
        assert result.is_ok()
        assert result.default_value("") == "domain@user"
    
    @pytest.mark.asyncio
    async def test_sub_with_count(self):
        """Test substitution with count limit."""
        op = sub(r'\d+', 'X', count=1)
        result = await op.execute("abc123def456")
        assert result.is_ok()
        assert result.default_value("") == "abcXdef456"
    
    @pytest.mark.asyncio
    async def test_pattern_ops_non_string_input(self):
        """Test pattern operations with non-string input."""
        op = match(r'\d+')
        result = await op.execute(123)
        assert result.is_ok()
        assert result.default_value(None) is None


# Test String Validation
class TestStringValidation:
    """Test suite for string validation operations."""
    
    @pytest.mark.asyncio
    async def test_is_alpha_true(self):
        """Test is_alpha with alphabetic string."""
        result = await is_alpha.execute("HelloWorld")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_alpha_false(self):
        """Test is_alpha with non-alphabetic characters."""
        result = await is_alpha.execute("Hello123")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_alpha_with_space(self):
        """Test is_alpha with space."""
        result = await is_alpha.execute("Hello World")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_numeric_true(self):
        """Test is_numeric with numeric string."""
        result = await is_numeric.execute("12345")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_numeric_unicode(self):
        """Test is_numeric with unicode numbers."""
        result = await is_numeric.execute("①②③")  # Unicode numeric characters
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_numeric_false(self):
        """Test is_numeric with non-numeric characters."""
        result = await is_numeric.execute("123.45")  # Period is not numeric
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_alphanumeric_true(self):
        """Test is_alphanumeric with alphanumeric string."""
        result = await is_alphanumeric.execute("Hello123")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_alphanumeric_false(self):
        """Test is_alphanumeric with special characters."""
        result = await is_alphanumeric.execute("Hello_123")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_whitespace_true(self):
        """Test is_whitespace with whitespace string."""
        result = await is_whitespace.execute("   \t\n")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_whitespace_false(self):
        """Test is_whitespace with non-whitespace."""
        result = await is_whitespace.execute(" hello ")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_upper_true(self):
        """Test is_upper with uppercase string."""
        result = await is_upper.execute("HELLO WORLD")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_upper_mixed(self):
        """Test is_upper with mixed case."""
        result = await is_upper.execute("Hello WORLD")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_is_lower_true(self):
        """Test is_lower with lowercase string."""
        result = await is_lower.execute("hello world")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_is_lower_with_numbers(self):
        """Test is_lower with numbers (should be true)."""
        result = await is_lower.execute("hello123")
        assert result.is_ok()
        assert result.default_value(False) is True
    
    @pytest.mark.asyncio
    async def test_validation_empty_string(self):
        """Test validation operations with empty string."""
        # Empty strings return False for all validations
        result = await is_alpha.execute("")
        assert result.is_ok()
        assert result.default_value(False) is False
        
        result = await is_numeric.execute("")
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_validation_non_string(self):
        """Test validation operations with non-string input."""
        result = await is_alpha.execute(123)
        assert result.is_ok()
        assert result.default_value(False) is False


# Test Complex Compositions
class TestComplexCompositions:
    """Test suite for complex operation compositions."""
    
    @pytest.mark.asyncio
    async def test_split_transform_join(self):
        """Test split -> transform -> join pipeline."""
        pipeline = (
            split(",") >> 
            operation(lambda items: [item.strip().upper() for item in items]) >>
            join(" | ")
        )
        result = await pipeline.execute("apple, banana, orange")
        assert result.is_ok()
        assert result.default_value("") == "APPLE | BANANA | ORANGE"
    
    @pytest.mark.asyncio
    async def test_normalize_whitespace(self):
        """Test normalizing whitespace in text."""
        pipeline = (
            strip >>
            operation(lambda s: re.sub(r'\s+', ' ', s)) >>
            to_lower
        )
        result = await pipeline.execute("  Hello    World\n\t  ")
        assert result.is_ok()
        assert result.default_value("") == "hello world"
    
    @pytest.mark.asyncio
    async def test_extract_and_validate_emails(self):
        """Test extracting and validating email-like patterns."""
        text = "Contact us at john@example.com or jane@company.org"
        
        # Extract email-like patterns
        extract_emails = find_all(r'\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b')
        
        result = await extract_emails.execute(text)
        assert result.is_ok()
        emails = result.default_value([])
        assert len(emails) == 2
        assert "john@example.com" in emails
        assert "jane@company.org" in emails
    
    @pytest.mark.asyncio
    async def test_conditional_text_processing(self):
        """Test conditional text processing based on content."""
        @operation
        def process_based_on_content(text: str) -> str:
            if text.startswith("ERROR:"):
                return text.upper()
            elif text.startswith("WARNING:"):
                return text.title()
            else:
                return text.lower()
        
        pipeline = strip >> process_based_on_content
        
        # Test error message
        result = await pipeline.execute("ERROR: Something went wrong")
        assert result.is_ok()
        assert result.default_value("") == "ERROR: SOMETHING WENT WRONG"
        
        # Test warning message
        result = await pipeline.execute("WARNING: Check this out")
        assert result.is_ok()
        assert result.default_value("") == "Warning: Check This Out"
        
        # Test normal message
        result = await pipeline.execute("Normal message")
        assert result.is_ok()
        assert result.default_value("") == "normal message"
    
    @pytest.mark.asyncio
    async def test_text_statistics_pipeline(self):
        """Test pipeline that computes text statistics."""
        @operation
        def compute_stats(text: str) -> dict:
            words = text.split()
            return {
                "char_count": len(text),
                "word_count": len(words),
                "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
                "uppercase_words": sum(1 for w in words if w.isupper()),
                "lowercase_words": sum(1 for w in words if w.islower())
            }
        
        pipeline = strip >> compute_stats
        result = await pipeline.execute("  Hello WORLD this IS a TEST  ")
        assert result.is_ok()
        stats = result.default_value({})
        assert stats["word_count"] == 6
        assert stats["uppercase_words"] == 3  # WORLD, IS, TEST
        assert stats["lowercase_words"] == 2  # this, a
    
    @pytest.mark.asyncio
    async def test_url_slug_generator(self):
        """Test generating URL slugs from text."""
        slug_pipeline = (
            to_lower >>
            strip >>
            operation(lambda s: re.sub(r'[^\w\s-]', '', s)) >>  # Remove special chars
            operation(lambda s: re.sub(r'[-\s]+', '-', s)) >>    # Replace spaces/hyphens with single hyphen
            operation(lambda s: s.strip('-'))                    # Remove leading/trailing hyphens
        )
        
        test_cases = [
            ("Hello World!", "hello-world"),
            ("  Multiple   Spaces  ", "multiple-spaces"),
            ("Special@#$Characters", "specialcharacters"),
            ("Already-Hyphenated-Text", "already-hyphenated-text"),
            ("Numbers 123 and Letters", "numbers-123-and-letters")
        ]
        
        for input_text, expected in test_cases:
            result = await slug_pipeline.execute(input_text)
            assert result.is_ok()
            assert result.default_value("") == expected


# Test Unicode and Special Characters
class TestUnicodeAndSpecialCharacters:
    """Test suite for Unicode and special character handling."""
    
    @pytest.mark.asyncio
    async def test_unicode_operations(self, unicode_text):
        """Test various operations with Unicode text."""
        # Test basic operations
        result = await to_upper.execute(unicode_text)
        assert result.is_ok()
        assert result.default_value("") == "HELLO 世界 🌍"
        
        # Test split
        op = split(" ")
        result = await op.execute(unicode_text)
        assert result.is_ok()
        assert result.default_value([]) == ["Hello", "世界", "🌍"]
        
        # Test join
        op = join("-")
        result = await op.execute(["Hello", "世界", "🌍"])
        assert result.is_ok()
        assert result.default_value("") == "Hello-世界-🌍"
    
    @pytest.mark.asyncio
    async def test_regex_with_unicode(self):
        """Test regex operations with Unicode."""
        # Test Unicode word matching
        op = find_all(r'\w+')
        result = await op.execute("Hello 世界 123")
        assert result.is_ok()
        matches = result.default_value([])
        assert "Hello" in matches
        assert "123" in matches
        # Note: \w behavior with Unicode depends on Python version and flags
    
    @pytest.mark.asyncio
    async def test_special_regex_characters(self):
        """Test handling of special regex characters."""
        # Test literal special characters
        text = "Price: $10.99 (on sale!)"
        
        # Find price pattern
        op = search(r'\$(\d+\.\d{2})')
        result = await op.execute(text)
        assert result.is_ok()
        match_obj = result.default_value(None)
        assert match_obj is not None
        assert match_obj.group(1) == "10.99"
    
    @pytest.mark.asyncio
    async def test_newline_handling(self, multiline_text):
        """Test operations with newline characters."""
        # Test replace newlines
        op = replace("\n", " | ")
        result = await op.execute(multiline_text)
        assert result.is_ok()
        assert result.default_value("") == "Line 1 | Line 2 | Line 3"
        
        # Test multiline regex
        op = find_all(r'^Line \d+$', flags=re.MULTILINE)
        result = await op.execute(multiline_text)
        assert result.is_ok()
        assert result.default_value([]) == ["Line 1", "Line 2", "Line 3"]


# Test Error Handling and Edge Cases
class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_operations_with_none(self):
        """Test all operations handle None gracefully."""
        # Test basic operations
        op = split()
        result = await op.execute(None)
        assert result.is_ok()
        assert result.default_value([]) == []
        
        result = await to_lower.execute(None)
        assert result.is_ok()
        assert result.default_value(None) is None
        
        # Test validation operations
        result = await is_alpha.execute(None)
        assert result.is_ok()
        assert result.default_value(False) is False
    
    @pytest.mark.asyncio
    async def test_empty_pattern_edge_cases(self):
        """Test edge cases with empty patterns."""
        # Empty pattern in split
        op = split("")
        result = await op.execute("hello")
        assert not result.is_ok()
      
    
    @pytest.mark.asyncio
    async def test_very_long_strings(self):
        """Test operations with very long strings."""
        # Create a long string
        long_text = ("word " * 9999) + "word"  # No trailing space

        # Test split performance
        op = split()
        result = await op.execute(long_text)
        assert result.is_ok()
        assert len(result.default_value([])) == 10000
        
        # Test regex performance
        op = find_all(r'\w+')
        result = await op.execute(long_text)
        assert result.is_ok()
        assert len(result.default_value([])) == 10000
    
    @pytest.mark.asyncio
    async def test_composition_error_propagation(self):
        """Test error propagation through pipelines."""
        @operation
        def failing_transform(text: str) -> str:
            if "error" in text.lower():
                raise ValueError("Found error keyword!")
            return text.upper()
        
        pipeline = strip >> failing_transform >> to_lower
        
        # Test normal flow
        result = await pipeline.execute("  hello  ")
        assert result.is_ok()
        assert result.default_value("") == "hello"
        
        # Test error propagation
        result = await pipeline.execute("  error message  ")
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert "Found error keyword!" in str(result.error)


# Test Performance and Memory
class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations."""
    
    @pytest.mark.asyncio
    async def test_chained_operations_efficiency(self):
        """Test efficiency of chained operations."""
        # Create a long chain of operations
        pipeline = (
            strip >>
            to_lower >>
            replace("a", "A") >>
            replace("e", "E") >>
            replace("i", "I") >>
            replace("o", "O") >>
            replace("u", "U") >>
            capitalize
        )
        
        result = await pipeline.execute("  Hello Beautiful World  ")
        assert result.is_ok()
        assert result.default_value("") == "Hello beautiful world"
    
    @pytest.mark.asyncio
    async def test_regex_compilation_reuse(self):
        """Test that regex patterns can be efficiently reused."""
        # Create operation with compiled pattern
        pattern = re.compile(r'\b\w{5}\b')  # 5-letter words
        op = find_all(pattern)
        
        # Use it multiple times
        texts = [
            "Hello world from Python",
            "These words have different sizes",
            "Short and sweet texts here"
        ]
        
        for text in texts:
            result = await op.execute(text)
            assert result.is_ok()
            # Check that 5-letter words are found
            words = result.default_value([])
            assert all(len(word) == 5 for word in words)


# Integration tests
class TestIntegration:
    """Integration tests combining multiple text operations."""
    
    @pytest.mark.asyncio
    async def test_log_parser_pipeline(self):
        """Test parsing and processing log entries."""
        log_entry = "2024-01-15 10:30:45 [ERROR] Database connection failed: timeout"
        
        # Extract components
        @operation
        def parse_log_entry(entry: str) -> dict:
            pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)'
            match = re.match(pattern, entry)
            if match:
                return {
                    "timestamp": match.group(1),
                    "level": match.group(2),
                    "message": match.group(3)
                }
            return {}
        
        # Process based on log level
        @operation
        def format_by_level(log_dict: dict) -> str:
            if not log_dict:
                return "Invalid log entry"
            
            level = log_dict.get("level", "")
            message = log_dict.get("message", "")
            
            if level == "ERROR":
                return f"🔴 {message.upper()}"
            elif level == "WARNING":
                return f"🟡 {message}"
            else:
                return f"ℹ️ {message}"
        
        pipeline = parse_log_entry >> format_by_level
        result = await pipeline.execute(log_entry)
        assert result.is_ok()
        assert result.default_value("") == "🔴 DATABASE CONNECTION FAILED: TIMEOUT"
    
    @pytest.mark.asyncio
    async def test_markdown_to_plain_text(self):
        """Test converting markdown to plain text."""
        markdown = """
# Hello World

This is a **bold** text and this is *italic*.

Here's a [link](https://example.com) and some `code`.

- Item 1
- Item 2
"""
        
        # Simple markdown stripper
        markdown_pipeline = (
            strip >>
            # Remove headers
            operation(lambda s: re.sub(r'^#+\s+', '', s, flags=re.MULTILINE)) >>
            # Remove bold
            operation(lambda s: re.sub(r'\*\*(.+?)\*\*', r'\1', s)) >>
            # Remove italic
            operation(lambda s: re.sub(r'\*(.+?)\*', r'\1', s)) >>
            # Remove links
            operation(lambda s: re.sub(r'\[(.+?)\]\(.+?\)', r'\1', s)) >>
            # Remove code
            operation(lambda s: re.sub(r'`(.+?)`', r'\1', s)) >>
            # Remove list markers
            operation(lambda s: re.sub(r'^-\s+', '', s, flags=re.MULTILINE)) >>
            # Clean up multiple newlines
            operation(lambda s: re.sub(r'\n\n+', '\n\n', s)) >>
            strip
        )
        
        result = await markdown_pipeline.execute(markdown)
        assert result.is_ok()
        plain_text = result.default_value("")
        assert "**" not in plain_text
        assert "[link]" not in plain_text
        assert "Hello World" in plain_text
        assert "bold" in plain_text
        assert "italic" in plain_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])