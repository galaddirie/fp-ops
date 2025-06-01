from typing import Union, Callable, List, Dict, Tuple, Sized, Any, Optional, TypeVar, cast, Iterable
from fp_ops import Operation, operation
from fp_ops.objects import get

T = TypeVar("T")
A = TypeVar("A")
R = TypeVar("R")

K = TypeVar("K")
V = TypeVar("V")


def filter(
    predicate: Union[Callable[[T], bool], Operation[[T], bool], Dict[str, Any]]
) -> Operation[[List[T]], List[T]]:
    """
    Filter items based on a predicate.
    
    Args:
        predicate: Can be:
            - A callable: filter(lambda x: x > 5)
            - An Operation: filter(is_valid_check)
            - A dict for matching: filter({"status": "active"})
    
    Returns:
        Operation that filters a list based on the predicate
        
    Examples:
        filter(lambda x: x["age"] > 18)(users)
        filter(is_adult_check)(users)  # Operation predicate
        filter({"status": "active", "verified": True})(users)  # Dict matching
    """
    @operation
    async def _filter(items: List[T]) -> List[T]:
        if isinstance(predicate, dict):
            # Dict matching
            results = []
            for item in items:
                match = True
                for key, value in predicate.items():
                    item_value = await get(key).execute(item)
                    if item_value.is_ok():
                        if item_value.default_value(None) != value:
                            match = False
                            break
                    else:
                        match = False
                        break
                if match:
                    results.append(item)
            return results
        elif isinstance(predicate, Operation):
            # Operation predicate
            results = []
            for item in items:
                res = await predicate.execute(item)
                if res.is_ok() and res.default_value(False):  # ✅ Fixed: use default_value() instead of .value
                    results.append(item)
            return results
        else:
            # Callable predicate
            return [item for item in items if predicate(item)]
    
    return _filter


def map(
    fn: Union[Callable[[T], R], Operation[[T], R]]
) -> Operation[[Union[List[T], Dict[str, T]]], Union[List[R], Dict[str, R]]]:
    """
    Transform each item in a list or each value in a dictionary.
    
    Args:
        fn: Can be:
            - A callable: map(lambda x: x * 2)
            - An Operation: map(transform_op)
    
    Returns:
        Operation that transforms each item in a list or each value in a dict
        
    Examples:
        map(lambda x: x["name"].upper())(users)  # List → List
        map(enrich_user)(users)  # Operation transform
        map(lambda items: len(items))({"cat1": [...], "cat2": [...]})  # Dict → Dict
    """
    @operation
    async def _map(items: Union[List[T], Dict[str, T]]) -> Union[List[R], Dict[str, R]]:
        if isinstance(items, dict):
            # Handle dictionary - map over values, preserve keys
            if isinstance(fn, Operation):
                results = {}
                for key, value in items.items():
                    res = await fn.execute(value)
                    if res.is_ok():
                        results[key] = res.default_value(cast(R, None))
                    elif res.is_error():
                        raise res.error
                return results
            else:
                return {key: fn(value) for key, value in items.items()}
        else:
            # Handle list - original behavior
            if isinstance(fn, Operation):
                results = []
                for item in items:
                    res = await fn.execute(item)
                    if res.is_ok():
                        results.append(res.default_value(cast(R, None)))
                    elif res.is_error():
                        raise res.error
                return results
            else:
                return [fn(item) for item in items]
    
    return _map


def reduce(
    fn: Union[Callable[[A, T], A], Operation[[A, T], A]],
    initial: Optional[A] = None,
) -> Operation[[Iterable[T]], A]:
    """
    Reduce a list to a single value.
    
    Args:
        fn: A binary function (or `Operation`) that combines the
            running accumulator (`A`) with the next element (`T`)
            and returns the new accumulator.
        items: The sequence to fold over.
        initial: Optional starting value for the accumulator.
    
    Examples:
        reduce(lambda a, b: a + b, numbers)
        reduce(lambda a, b: a + b, numbers, 0)
        reduce(combine_op, items)  # Operation reducer
    """
    if not isinstance(fn, Operation):
        fn_op = operation(fn)
    else:
        fn_op = fn
    
    has_initial = initial is not None
    
    @operation
    async def _reduce(items: Optional[Iterable[Any]] = None) -> Any:
        items_iter = iter(items)
        
        # Handle initial value
        if has_initial:
            accumulator = initial
        else:
            try:
                accumulator = next(items_iter)
            except StopIteration:
                raise ValueError("reduce() of empty sequence with no initial value")
        
        # Reduce over remaining items
        for item in items_iter:
            result = await fn_op.execute(accumulator, item)
            if result.is_error():
                raise result.error
            accumulator = cast(Any, result.default_value(None))
        
        return accumulator
    
    return _reduce


def zip(
    *operations: Operation[[T], R]
) -> Operation[[List[T]], List[Tuple[Any, ...]]]:
    """
    Apply multiple operations to each item in a list and return tuples of results.
    
    This is like a parallel map - for each item, all operations are applied
    and their results are collected into a tuple.
    
    Note: This is different from compose.parallel() which runs operations concurrently
    on the same input. zip() applies operations to each item in a list sequentially.
    
    Args:
        *operations: Operations to apply to each item
    
    Returns:
        Operation that returns a list of tuples, where each tuple contains
        the results of applying all operations to that item
    
    Examples:
        # Extract multiple fields from each user
        user_data = await zip(
            get("id"),
            get("name"),
            get("email")
        )(users)
        # Result: [(1, "Alice", "alice@example.com"), (2, "Bob", "bob@example.com"), ...]
        
        # Apply different transformations
        transformed = await zip(
            lambda x: x * 2,
            lambda x: x ** 2,
            lambda x: x + 10
        )([1, 2, 3, 4, 5])
        # Result: [(2, 1, 11), (4, 4, 12), (6, 9, 13), (8, 16, 14), (10, 25, 15)]
        
        # Mix operations and functions
        results = await zip(
            to_upper_op,                    # Operation
            lambda s: len(s),              # Function  
            count_vowels_op                # Another Operation
        )(["hello", "world"])
        # Result: [("HELLO", 5, 2), ("WORLD", 5, 1)]
    """
    @operation
    async def _zip(items: List[T]) -> List[Tuple[Any, ...]]:
        if not operations:
            return [()] * len(items)
            
        result = []
        
        for item in items:
            # Apply all operations to this item
            item_results = []
            
            for op in operations:
                if isinstance(op, Operation):
                    res = await op.execute(item)
                    if res.is_ok():
                        item_results.append(res.default_value(None))  # ✅ Fixed: use default_value() instead of .value
                    else:
                        # Include None for failed operations
                        item_results.append(None)
                elif callable(op):
                    try:
                        item_results.append(op(item))
                    except Exception:
                        item_results.append(None)
                else:
                    # Non-callable, non-operation - just include as-is
                    item_results.append(op)
                    
            result.append(tuple(item_results))
            
        return result
    
    return _zip

@operation
def contains(collection: Union[List, Dict, str, set], item: Any) -> bool:
    """
    Check if collection contains item.
    
    Example:
        contains(["hello", "world"], "hello")  # True
        contains(["hello", "world"], "foo")  # False
        contains("hello", "l")  # True
    """
    if hasattr(collection, '__contains__'):
        return item in collection
    return False


@operation
def not_contains(collection: Union[List, Dict, str, set], item: Any) -> bool:
    """
    Check if collection does not contain item.
    
    Example:
        not_contains(["hello", "world"], "foo")  # True
        not_contains(["hello", "world"], "hello")  # False
    """
    if hasattr(collection, '__contains__'):
        return item not in collection
    return True


@operation
def flatten(data: List[List[T]]) -> List[T]:
    """
    Flatten a list of lists one level deep.
    
    Example:
        flatten([[1, 2], [3, 4], [5]])  # [1, 2, 3, 4, 5]
    """
    result = []
    for item in data:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


@operation
def flatten_deep(data: List[Any]) -> List[Any]:
    """
    Recursively flatten nested lists.
    
    Example:
        flatten_deep([1, [2, [3, [4]], 5]])  # [1, 2, 3, 4, 5]
    """
    result = []
    
    def _flatten_recursive(lst: List[Any]) -> None:
        for item in lst:
            if isinstance(item, list):
                _flatten_recursive(item)
            else:
                result.append(item)
    
    _flatten_recursive(data)
    return result


@operation
def unique(data: List[T]) -> List[T]:
    """
    Get unique values from list while preserving order.
    For unhashable types, preserves all items (no deduplication).
    
    Example:
        unique([1, 2, 2, 3, 1, 4])  # [1, 2, 3, 4]
        unique([{"a": 1}, {"b": 2}, {"a": 1}])  # [{"a": 1}, {"b": 2}, {"a": 1}] (preserves all)
    """
    seen = set()
    result = []
    for item in data:
        # Handle unhashable types
        try:
            if item not in seen:
                seen.add(item)
                result.append(item)
        except TypeError:
            # For unhashable types, preserve all items (don't deduplicate)
            result.append(item)
    return result


@operation
def reverse(data: Union[List[T], str]) -> Union[List[T], str]:
    """
    Reverse a list or string.
    
    Example:
        reverse([1, 2, 3])  # [3, 2, 1]
        reverse("hello")  # "olleh"
    """
    if isinstance(data, str):
        return data[::-1]
    elif isinstance(data, list):
        return data[::-1]
    return data


@operation
def length(data: Sized) -> int:
    """
    Get length of list, dict, string, or any sized container.
    
    Examples:
        length([1, 2, 3])  # 3
        length("hello")  # 5
        length({"a": 1, "b": 2})  # 2
    """
    return len(data) if hasattr(data, '__len__') else 0


@operation
def keys(data: Dict[K, V]) -> List[K]:
    """
    Get dictionary keys as a list.
    
    Example:
        keys({"a": 1, "b": 2, "c": 3})  # ["a", "b", "c"]
    """
    return list(data.keys()) if isinstance(data, dict) else []


@operation
def values(data: Dict[K, V]) -> List[V]:
    """
    Get dictionary values as a list.
    
    Example:
        values({"a": 1, "b": 2, "c": 3})  # [1, 2, 3]
    """
    return list(data.values()) if isinstance(data, dict) else []


@operation
def items(data: Dict[K, V]) -> List[Tuple[K, V]]:
    """
    Get dictionary items as a list of tuples.
    
    Example:
        items({"a": 1, "b": 2})  # [("a", 1), ("b", 2)]
    """
    return list(data.items()) if isinstance(data, dict) else []