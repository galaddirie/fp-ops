"""
Utility operations for FP-Ops providing common helpers for
type checking, value validation, and data introspection.
"""
from typing import Union, List, Dict, Any, Sized, Optional, TypeVar, Tuple, Callable
from fp_ops import operation, Operation

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ============================================================================
# Type/Value Checks
# ============================================================================

@operation
def is_empty(data: Union[List, Dict, str, Sized]) -> bool:
    """
    Check if list, dict, string, or any sized container is empty.
    
    Examples:
        is_empty([])  # True
        is_empty("")  # True
        is_empty({})  # True
        is_empty([1, 2, 3])  # False
    """
    if hasattr(data, '__len__'):
        return len(data) == 0
    return True


@operation
def is_not_empty(data: Union[List, Dict, str, Sized]) -> bool:
    """
    Check if list, dict, string, or any sized container is not empty.
    
    Examples:
        is_not_empty([1, 2, 3])  # True
        is_not_empty("hello")  # True
        is_not_empty({})  # False
    """
    if hasattr(data, '__len__'):
        return len(data) > 0
    return False


@operation
def is_none(value: Any) -> bool:
    """
    Check if value is None.
    
    Example:
        is_none(None)  # True
        is_none(0)  # False
        is_none("")  # False
    """
    return value is None


@operation
def is_not_none(value: Any) -> bool:
    """
    Check if value is not None.
    
    Example:
        is_not_none("hello")  # True
        is_not_none(0)  # True
        is_not_none(None)  # False
    """
    return value is not None

R = TypeVar('R')
def default(fallback: R) -> Operation[[R | None], R]:
    """
    Wrap the pipeline’s running value and replace `None` with `fallback`.
    Works even when the upstream stage produced no positional argument.
    """
    import copy

    @operation
    def _default(value: R | None = None) -> R:
        return value if value is not None else copy.deepcopy(fallback)

    return _default



# ============================================================================
# Type Checking
# ============================================================================

def is_type(*types: type) -> Operation[[Any], bool]:
    """
    Check if value is of any of the expected types.
    
    Example:
        is_type(str, int)(42)  # True
        is_type(str, int)(3.14)  # False
    """
    @operation
    def _is_type(value: Any) -> bool:
        return isinstance(value, types)
    return _is_type


@operation
def is_string(value: Any) -> bool:
    """Check if value is a string."""
    return isinstance(value, str)


@operation
def is_int(value: Any) -> bool:
    """Check if value is an integer."""
    return isinstance(value, int) and not isinstance(value, bool)


@operation
def is_float(value: Any) -> bool:
    """Check if value is a float."""
    return isinstance(value, float)


@operation
def is_bool(value: Any) -> bool:
    """Check if value is a boolean."""
    return isinstance(value, bool)


@operation
def is_list(value: Any) -> bool:
    """Check if value is a list."""
    return isinstance(value, list)


@operation
def is_dict(value: Any) -> bool:
    """Check if value is a dictionary."""
    return isinstance(value, dict)


@operation
def is_tuple(value: Any) -> bool:
    """Check if value is a tuple."""
    return isinstance(value, tuple)


@operation
def is_set(value: Any) -> bool:
    """Check if value is a set."""
    return isinstance(value, set)


# ============================================================================
# Value Comparisons
# ============================================================================

def equals(expected: Any) -> Operation[[Any], bool]:
    """
    Check if value equals expected value.
    
    Example:
        equals(42)(42)  # True
        equals("hello")("hello")  # True
        equals([1, 2, 3])([1, 2, 3])  # True
    """
    @operation
    def _equals(value: Any) -> bool:
        return bool(value == expected)
    return _equals


def not_equals(expected: Any) -> Operation[[Any], bool]:
    """
    Check if value does not equal expected value.
    
    Example:
        not_equals(42)(43)  # True
        not_equals("hello")("world")  # True
    """
    @operation
    def _not_equals(value: Any) -> bool:
        return bool(value != expected)
    return _not_equals


def greater_than(threshold: Union[int, float]) -> Operation[[Union[int, float]], bool]:
    """
    Check if value is greater than threshold.
    
    Example:
        greater_than(5)(10)  # True
        greater_than(5)(3)  # False
    """
    @operation
    def _greater_than(value: Union[int, float]) -> bool:
        return value > threshold
    return _greater_than


def less_than(threshold: Union[int, float]) -> Operation[[Union[int, float]], bool]:
    """
    Check if value is less than threshold.
    
    Example:
        less_than(5)(3)  # True
        less_than(5)(10)  # False
    """
    @operation
    def _less_than(value: Union[int, float]) -> bool:
        return value < threshold
    return _less_than


def greater_or_equal(threshold: Union[int, float]) -> Operation[[Union[int, float]], bool]:
    """
    Check if value is greater than or equal to threshold.
    
    Example:
        greater_or_equal(5)(5)  # True
        greater_or_equal(5)(6)  # True
        greater_or_equal(5)(4)  # False
    """
    @operation
    def _greater_or_equal(value: Union[int, float]) -> bool:
        return value >= threshold
    return _greater_or_equal


def less_or_equal(threshold: Union[int, float]) -> Operation[[Union[int, float]], bool]:
    """
    Check if value is less than or equal to threshold.
    
    Example:
        less_or_equal(5)(5)  # True
        less_or_equal(5)(4)  # True
        less_or_equal(5)(6)  # False
    """
    @operation
    def _less_or_equal(value: Union[int, float]) -> bool:
        return value <= threshold
    return _less_or_equal


def in_range(min_val: Union[int, float], max_val: Union[int, float], inclusive: bool = True) -> Operation[[Union[int, float]], bool]:
    """
    Check if value is within range.
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        inclusive: Whether range is inclusive (default: True)
    
    Example:
        in_range(1, 10)(5)  # True
        in_range(1, 10)(10)  # True
        in_range(1, 10, inclusive=False)(10)  # False
    """
    @operation
    def _in_range(value: Union[int, float]) -> bool:
        if inclusive:
            return min_val <= value <= max_val
        else:
            return min_val < value < max_val
    return _in_range


# ============================================================================
# Type Conversions
# ============================================================================

@operation
def to_string(value: Any) -> str:
    """Convert value to string."""
    return str(value)


@operation
def to_int(value: Any) -> Optional[int]:
    """
    Convert value to integer if possible.
    
    Example:
        to_int("42")  # 42
        to_int(3.14)  # 3
        to_int("abc")  # None
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


@operation
def to_float(value: Any) -> Optional[float]:
    """
    Convert value to float if possible.
    
    Example:
        to_float("3.14")  # 3.14
        to_float(42)  # 42.0
        to_float("abc")  # None
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


@operation
def to_bool(value: Any) -> bool:
    """
    Convert value to boolean.
    
    Falsy values: None, False, 0, "", [], {}, set()
    Everything else is True.
    
    Example:
        to_bool(1)  # True
        to_bool("")  # False
        to_bool("false")  # True (non-empty string)
    """
    return bool(value)


@operation
def to_list(value: Any) -> List[Any]:
    """
    Convert value to list.
    
    Example:
        to_list("hello")  # ["h", "e", "l", "l", "o"]
        to_list((1, 2, 3))  # [1, 2, 3]
        to_list(42)  # [42]
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, (tuple, set, str)):
        return list(value)
    elif isinstance(value, dict):
        return list(value.items())
    else:
        return [value]


@operation
def to_set(value: Any) -> set:
    """
    Convert value to set.
    
    Example:
        to_set([1, 2, 2, 3])  # {1, 2, 3}
        to_set("hello")  # {"h", "e", "l", "o"}
    """
    if isinstance(value, set):
        return value
    elif isinstance(value, (list, tuple, str)):
        return set(value)
    elif isinstance(value, dict):
        return set(value.keys())
    else:
        return {value}


