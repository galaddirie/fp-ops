from typing import (
    Union,
    Callable,
    List,
    Dict,
    Tuple,
    Sized,
    Any,
    Optional,
    TypeVar,
    cast,
    Iterable,
)
from fp_ops import Operation, operation
from fp_ops.objects import get

T = TypeVar("T")
A = TypeVar("A")
R = TypeVar("R")

K = TypeVar("K")
V = TypeVar("V")


def filter(
    predicate: Union[Callable[[T], bool], Operation[[T], bool], Dict[str, Any]],
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

    async def _filter(items: List[T]) -> List[T]:
        if isinstance(predicate, dict):
            # Dict matching
            filtered_items: List[T] = []
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
                    filtered_items.append(item)
            return filtered_items
        elif isinstance(predicate, Operation):
            # Operation predicate
            filtered_items = []
            for item in items:
                res = await predicate.execute(item)
                if res.is_ok() and res.default_value(False):
                    filtered_items.append(item)
            return filtered_items
        else:
            # Callable predicate
            return [item for item in items if predicate(item)]

    return operation(_filter)


def map(
    fn: Union[Callable[[T], R], Operation[[T], R]],
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

    async def _map(items: Union[List[T], Dict[str, T]], **op_kwargs: Any) -> Union[List[R], Dict[str, R]]:
        if isinstance(items, dict):
            # Handle dictionary - map over values, preserve keys
            if isinstance(fn, Operation):
                out_dict: Dict[str, R] = {}
                for key, value in items.items():
                    res = await fn.execute(value, **op_kwargs)
                    if res.is_ok():
                        out_dict[key] = res.default_value(cast(R, None))
                    else:
                        raise res.error
                return out_dict
            return {key: fn(value) for key, value in items.items()}

        # Handle list - original behavior
        if isinstance(fn, Operation):
            out_list: List[R] = []
            for item in items:  # type: ignore[arg-type]
                res = await fn.execute(item, **op_kwargs)
                if res.is_ok():
                    out_list.append(res.default_value(cast(R, None)))
                else:
                    raise res.error
            return out_list
        return [fn(item) for item in items]  # type: ignore[arg-type]

    return operation(_map)


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
    fn_op = fn if isinstance(fn, Operation) else operation(fn)
    has_initial = initial is not None

    async def _reduce(items: Iterable[T]) -> A:
        it = iter(items)
        # Explicitly handle the "empty sequence, no initial" case
        if has_initial:
            if initial is None:  # This should never happen due to has_initial check
                raise ValueError(
                    "Initial value cannot be None when has_initial is True"
                )
            acc = initial
        else:
            try:
                first = next(it)
                acc = cast(A, first)  # Cast first item to accumulator type
            except StopIteration:
                raise ValueError("reduce() of empty sequence with no initial value")

        for item in it:
            res = await fn_op.execute(acc, item)
            if res.is_error():
                raise res.error
            acc = res.default_value(cast(A, None))  # Cast result to accumulator type

        return acc

    return operation(_reduce)


def zip(*operations: Operation[[T], R]) -> Operation[[List[T]], List[Tuple[Any, ...]]]:
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

    async def _zip(items: List[T]) -> List[Tuple[Any, ...]]:
        if not operations:
            return [()] * len(items)

        result: List[Tuple[Any, ...]] = []
        for item in items:
            item_results: List[Any] = []
            for op in operations:
                if isinstance(op, Operation):
                    res = await op.execute(item)
                    if res.is_ok():
                        item_results.append(res.default_value(None))
                    else:
                        item_results.append(None)
                elif callable(op):
                    try:
                        item_results.append(op(item))
                    except Exception:
                        item_results.append(None)
                else:
                    item_results.append(op)
            result.append(tuple(item_results))
        return result

    return operation(_zip)


@operation
def contains(collection: Union[List, Dict, str, set], item: Any) -> bool:
    """
    Check if collection contains item.

    Example:
        contains(["hello", "world"], "hello")  # True
        contains(["hello", "world"], "foo")  # False
        contains("hello", "l")  # True
    """
    if hasattr(collection, "__contains__"):
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
    if hasattr(collection, "__contains__"):
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
    return len(data) if hasattr(data, "__len__") else 0


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
