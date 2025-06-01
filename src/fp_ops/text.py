"""
String and text manipulation operations for FP-Ops.
"""
from typing import Any, List, Optional, Pattern, Union
import re
from fp_ops import operation, Operation

# ============================================================================
# Basic String Operations
# ============================================================================

def split(delimiter: str = " ") -> Operation[[str], List[str]]:
    """
    Split string by delimiter.
    
    Example:
        split(",")({"apple,banana,orange"})  # ["apple", "banana", "orange"]
        split()({"hello world"})  # ["hello", "world"]
    """
    @operation
    def _split(text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        # Handle empty delimiter case
        if delimiter == "":
            raise ValueError("empty separator")
        return text.split(delimiter)
    return _split


def join(delimiter: str = " ") -> Operation[[List[Any]], str]:
    """
    Join list of items into a string.
    
    Example:
        join(", ")(["apple", "banana", "orange"])  # "apple, banana, orange"
        join("-")([1, 2, 3])  # "1-2-3"
    """
    @operation
    def _join(items: List[Any]) -> str:
        return delimiter.join(str(item) for item in items)
    return _join


def replace(old: str, new: str, count: int = -1) -> Operation[[str], str]:
    """
    Replace occurrences of substring in string.
    
    Args:
        old: Substring to replace
        new: Replacement string
        count: Maximum number of replacements (-1 for all)
    
    Example:
        replace("world", "universe")("hello world")  # "hello universe"
        replace(" ", "_")("hello world")  # "hello_world"
    """
    @operation
    def _replace(text: str) -> str:
        return text.replace(old, new, count) if isinstance(text, str) else text
    return _replace


@operation
def to_lower(text: str) -> str:
    """
    Convert string to lowercase.
    
    Example:
        to_lower("Hello WORLD")  # "hello world"
    """
    return text.lower() if isinstance(text, str) else text


@operation
def to_upper(text: str) -> str:
    """
    Convert string to uppercase.
    
    Example:
        to_upper("hello world")  # "HELLO WORLD"
    """
    return text.upper() if isinstance(text, str) else text


@operation
def strip(text: str, chars: Optional[str] = None) -> str:
    """
    Strip whitespace (or specified characters) from both ends of string.
    
    Example:
        strip("  hello  ")  # "hello"
        strip("__hello__", "_")  # "hello"
    """
    if isinstance(text, str):
        return text.strip(chars) if chars else text.strip()
    return text


@operation
def lstrip(text: str, chars: Optional[str] = None) -> str:
    """
    Strip whitespace (or specified characters) from left side of string.
    
    Example:
        lstrip("  hello  ")  # "hello  "
        lstrip("__hello", "_")  # "hello"
    """
    if isinstance(text, str):
        return text.lstrip(chars) if chars else text.lstrip()
    return text


@operation
def rstrip(text: str, chars: Optional[str] = None) -> str:
    """
    Strip whitespace (or specified characters) from right side of string.
    
    Example:
        rstrip("  hello  ")  # "  hello"
        rstrip("hello__", "_")  # "hello"
    """
    if isinstance(text, str):
        return text.rstrip(chars) if chars else text.rstrip()
    return text


# ============================================================================
# Case Conversions
# ============================================================================

@operation
def capitalize(text: str) -> str:
    """
    Capitalize the first character of the string.
    
    Example:
        capitalize("hello world")  # "Hello world"
    """
    return text.capitalize() if isinstance(text, str) else text


@operation
def title(text: str) -> str:
    """
    Convert string to title case.
    
    Example:
        title("hello world")  # "Hello World"
    """
    return text.title() if isinstance(text, str) else text


# =====================================================================
# String Checks
# ============================================================================

def starts_with(prefix: str, start: int = 0, end: Optional[int] = None) -> Operation[[str], bool]:
    """
    Check if string starts with prefix.
    
    Example:
        starts_with("Hello")("Hello World")  # True
        starts_with("World")("Hello World")  # False
    """
    @operation
    def _starts_with(text: str) -> bool:
        if isinstance(text, str):
            return text.startswith(prefix, start, end)
        return False
    return _starts_with


def ends_with(suffix: str, start: int = 0, end: Optional[int] = None) -> Operation[[str], bool]:
    """
    Check if string ends with suffix.
    
    Example:
        ends_with("World")("Hello World")  # True
        ends_with("Hello")("Hello World")  # False
    """
    @operation
    def _ends_with(text: str) -> bool:
        if isinstance(text, str):
            return text.endswith(suffix, start, end)
        return False
    return _ends_with


def contains(substring: str) -> Operation[[str], bool]:
    """
    Check if string contains substring.
    
    Example:
        contains("World")("Hello World")  # True
        contains("xyz")("Hello World")  # False
    """
    @operation
    def _contains(text: str) -> bool:
        return substring in text if isinstance(text, str) else False
    return _contains


# ============================================================================
# Pattern Matching
# ============================================================================

def match(pattern: Union[str, Pattern], flags: int = 0) -> Operation[[str], Optional[re.Match]]:
    """
    Match string against regex pattern.
    
    Args:
        pattern: Regex pattern string or compiled pattern
        flags: Regex flags (e.g., re.IGNORECASE)
    
    Example:
        match(r'^[A-Z][a-z]+')("Hello")  # Match object
        match(r'^\d+')("Hello")  # None
    """
    @operation
    def _match(text: str) -> Optional[re.Match]:
        if not isinstance(text, str):
            return None
            
        if isinstance(pattern, str):
            return re.match(pattern, text, flags)
        else:
            return pattern.match(text)
    return _match


def search(pattern: Union[str, Pattern], flags: int = 0) -> Operation[[str], Optional[re.Match]]:
    """
    Search for pattern anywhere in string.
    
    Example:
        search(r'\d+')("abc123def")  # Match object for "123"
        search(r'xyz')("abc123def")  # None
    """
    @operation
    def _search(text: str) -> Optional[re.Match]:
        if not isinstance(text, str):
            return None
            
        if isinstance(pattern, str):
            return re.search(pattern, text, flags)
        else:
            return pattern.search(text)
    return _search


def find_all(pattern: Union[str, Pattern], flags: int = 0) -> Operation[[str], List[str]]:
    """
    Find all non-overlapping matches of pattern.
    
    Example:
        find_all(r'\d+')("abc123def456")  # ["123", "456"]
        find_all(r'[aeiou]')("hello")  # ["e", "o"]
    """
    @operation
    def _find_all(text: str) -> List[str]:
        if not isinstance(text, str):
            return []
            
        if isinstance(pattern, str):
            return re.findall(pattern, text, flags)
        else:
            return pattern.findall(text)
    return _find_all


def sub(pattern: Union[str, Pattern], replacement: str, count: int = 0, flags: int = 0) -> Operation[[str], str]:
    """
    Replace pattern matches with replacement string.
    
    Args:
        pattern: Regex pattern to match
        replacement: Replacement string (can use backreferences)
        count: Maximum number of replacements (0 for all)
        flags: Regex flags
    
    Example:
        sub(r'\d+', 'X')("abc123def456")  # "abcXdefX"
        sub(r'(\w+)@(\w+)', r'\2@\1')("user@domain")  # "domain@user"
    """
    @operation
    def _sub(text: str) -> str:
        if not isinstance(text, str):
            return text
            
        if isinstance(pattern, str):
            return re.sub(pattern, replacement, text, count, flags)
        else:
            return pattern.sub(replacement, text, count)
    return _sub



# ============================================================================
# String Validation
# ============================================================================

@operation
def is_alpha(text: str) -> bool:
    """Check if string contains only alphabetic characters."""
    return text.isalpha() if isinstance(text, str) and text else False


@operation
def is_numeric(text: str) -> bool:
    """Check if string contains only numeric characters."""
    return text.isnumeric() if isinstance(text, str) and text else False


@operation
def is_alphanumeric(text: str) -> bool:
    """Check if string contains only alphanumeric characters."""
    return text.isalnum() if isinstance(text, str) and text else False


@operation
def is_whitespace(text: str) -> bool:
    """Check if string contains only whitespace."""
    return text.isspace() if isinstance(text, str) and text else False


@operation
def is_upper(text: str) -> bool:
    """Check if string is all uppercase."""
    return text.isupper() if isinstance(text, str) and text else False


@operation
def is_lower(text: str) -> bool:
    """Check if string is all lowercase."""
    return text.islower() if isinstance(text, str) and text else False