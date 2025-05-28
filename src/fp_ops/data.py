"""
Core data operations for FP-Ops that provide ergonomic data handling
without expanding the DSL significantly.
"""
from __future__ import annotations
from typing import Any, Dict, List, Callable, TypeVar, Union, Tuple, Optional, cast, AsyncGenerator
from functools import reduce
from fp_ops import operation, Operation

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# Path-based access operations
# ============================================================================

def get(path: str, default: Any = None) -> Operation[[Any], Any]:
    """
    Access nested data using dot notation or dict keys.
    Configured with a path and an optional default value.
    The returned operation takes the data object as input.
    
    Examples:
        # Assuming 'data' is the dict or object to access
        # name_op = get("user.name")
        # user_name = await name_op.execute(data)
        
        # price_op = get("items.0.price", 0.0) # With default
        # item_price = await price_op.execute(data)

        # Can be used in 'build' or 'pipe':
        # pipe(
        #   get_data_source_op,
        #   get("user.profile.email", "notfound@example.com")
        # )
    """
    @operation
    def _get_inner(data: Any) -> Any:
        if not path: # path and default are from the outer scope
            return data
            
        parts = path.replace('[', '.').replace(']', '').split('.')
        current = data
        
        for part in parts:
            if current is None:
                return default
                
            # Try dict access first
            if isinstance(current, dict):
                current = current.get(part, None)
            # Then try numeric index for sequences
            elif isinstance(current, (list, tuple)) and part.isdigit():
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            # Finally try attribute access
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
                
        return current if current is not None else default
    return _get_inner


def pick(*keys: str) -> Operation[[Any], Dict[str, Any]]:
    """
    Pick specific keys from a dict or object.
    
    Example:
        pick("id", "name", "email")(user_dict)
        # Returns: {"id": 1, "name": "John", "email": "john@example.com"}
    """
    @operation  # type: ignore[arg-type]
    async def _pick(data: Any) -> Dict[str, Any]:
        result = {}
        
        for key in keys:
            if '.' in key:
                # Handle nested paths
                res = await get(key).execute(data)
                if res.is_ok():
                    value = res.default_value(None)
                    if value is not None:
                        # Use the last part of the path as the key
                        result_key = key.split('.')[-1]
                        result[result_key] = value
            elif isinstance(data, dict) and key in data:
                result[key] = data[key]
            elif hasattr(data, key):
                result[key] = getattr(data, key)
                
        return result
    
    return _pick


def pluck(key: str) -> Operation[[List[Any]], List[Any]]:
    """
    Extract a single key from a list of objects.
    
    Example:
        # users_list = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        # pluck_name = pluck("name")
        # names = await pluck_name.execute(users_list)  # Result: ["Alice", "Bob"]
    """
    @operation  # type: ignore[arg-type]
    async def _pluck(items: List[Any]) -> List[Any]:
        get_op = get(key)
        result = []
        for item in items:
            res = await get_op.execute(item)
            result.append(res.default_value(None) if res.is_ok() else None)
        return result
    
    return _pluck


# ============================================================================
# Object construction operations
# ============================================================================

def build(schema: Dict[str, Any]) -> Operation[[Any], Dict[str, Any]]:
    """
    Build an object from a schema. Values can be static, callables, or operations.
    
    Example:
        build({
            "id": get("user_id"),
            "fullName": lambda d: f"{d['first_name']} {d['last_name']}",
            "email": get("contact.email"),
            "isActive": True
        })(data)
    """
    @operation  # type: ignore[arg-type]
    async def _build(data: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for key, value in schema.items():
            if isinstance(value, Operation):
                # Handle operations
                res = await value.execute(data)
                # Always create the field; use None on error
                result[key] = res.default_value(None) if res.is_ok() else None
            elif callable(value) and not isinstance(value, (type, bool, int, float, str)):
                try:
                    result[key] = value(data)
                except Exception:
                    result[key] = None
            elif isinstance(value, dict):
                nested = await build(value).execute(data)
                result[key] = nested.default_value({})
            else:
                result[key] = value

        return result
    
    return _build


def merge(*sources: Union[Dict[str, Any],
                          Callable[[Any], Dict[str, Any]],
                          Operation[[Any], Dict[str, Any]]]
          ) -> Operation[[Any], Dict[str, Any]]:
    """
    Merge multiple dicts or dict-returning functions.
    
    Example:
        merge(
            get("basics"),
            get("profile"),
            {"processed": True}
        )(data)
    """
    @operation  # type: ignore[arg-type]
    async def _merge(data: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for src in sources:
            if isinstance(src, Operation):
                res = await src.execute(data)
                update = res.default_value({}) if res.is_ok() else {}
            elif callable(src):
                update = src(data)
            else:
                update = src

            if isinstance(update, dict):
                out.update(update)

        return out
    
    return _merge


def update(update_values: Dict[str, Any]) -> Operation[[Dict[str, Any]], Dict[str, Any]]:
    """
    Update a dict with another dict (the update_values).
    The outer function takes the update_values as configuration and returns
    an inner function that takes the source dictionary.

    Example:
        # source_dict = {"a": 1, "b": 2}
        # updater = update({"b": 3, "c": 4})
        # updated_dict = await updater.execute(source_dict)
        # Result: {"a": 1, "b": 3, "c": 4}
    """
    @operation  # type: ignore[arg-type]
    def _update_inner(source: Dict[str, Any]) -> Dict[str, Any]:
        return {**source, **update_values}
    return _update_inner



# ============================================================================
# List/Collection operations
# ============================================================================

def filter_by(predicate: Union[Callable[[Any], bool], Dict[str, Any]]) -> Operation[[List[Any]], List[Any]]:
    """
    Filter a list by a predicate function or dict match.
    
    Examples:
        filter_by(lambda x: x["age"] > 18)(users)
        filter_by({"status": "active"})(users)
    """
    @operation  # type: ignore[arg-type]
    async def _filter_by(items: List[Any]) -> List[Any]:
        if isinstance(predicate, dict):
            # Dict matching
            async def dict_match(item: Any) -> bool:
                for key, value in predicate.items():
                    res = await get(key).execute(item)
                    if res.default_value(None) != value:
                        return False
                return True
            return [item async for item in _aiter(items) if await dict_match(item)]
        else:
            # Function predicate
            return [item for item in items if predicate(item)]
    
    return _filter_by


def group_by(key: Union[str, Callable[[Any], Any]]) -> Operation[[List[Any]], Dict[Any, List[Any]]]:
    """
    Group items by a key or key function.
    
    Examples:
        # products = [{"name": "apple", "category": "fruit"}, {"name": "banana", "category": "fruit"}]
        # group_by_category = group_by("category")
        # grouped_products = await group_by_category.execute(products)
        # group_by_year = group_by(lambda x: x["date"].year)
        # grouped_events = await group_by_year.execute(events)
    """
    @operation  # type: ignore[arg-type]
    async def _group_by(items: List[Any]) -> Dict[Any, List[Any]]:
        groups: Dict[Any, List[Any]] = {}
        get_op = get(key) if isinstance(key, str) else None
        
        for item in items:
            if callable(key) and not isinstance(key, str):
                group_key = key(item)
            elif get_op:
                res = await get_op.execute(item)
                group_key = res.default_value(None) if res.is_ok() else None
            else:
                group_key = item.get(key) if isinstance(item, dict) else getattr(item, key, None)

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
            
        return groups
    
    return _group_by


def sort_by(key: Union[str, Callable[[Any], Any]], *, reverse: bool = False
            ) -> Operation[[List[Any]], List[Any]]:
    """
    Sort items by a key or key function.
    
    Examples:
        # users_list = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        # sort_by_name = sort_by("name")
        # sorted_users = await sort_by_name.execute(users_list)
    """
    @operation  # type: ignore[arg-type]
    async def _sort_by(items: List[Any]) -> List[Any]:
        if callable(key) and not isinstance(key, str):
            return sorted(items, key=key, reverse=reverse)

        g = get(key)
        enriched = []
        for it in items:
            res = await g.execute(it)
            val = res.default_value(None) if res.is_ok() else None
            enriched.append((val, it))

        # keep None values at the end regardless of direction
        non_null = [p for p in enriched if p[0] is not None]
        nulls    = [p for p in enriched if p[0] is None]
        non_null.sort(key=lambda p: cast(Any, p[0]), reverse=reverse)
        return [p[1] for p in (*non_null, *nulls)]
    return _sort_by

def unique_by(key: Union[str, Callable[[Any], Any]]) -> Operation[[List[Any]], List[Any]]:
    """
    Get unique items based on a key or key function.
    
    Example:
        # users_list = [{"email": "a@a.com"}, {"email": "b@b.com"}, {"email": "a@a.com"}]
        # unique_by_email_op = unique_by("email")
        # unique_users = await unique_by_email_op.execute(users_list) # Remove duplicate users by email
    """
    @operation  # type: ignore[arg-type]
    async def _unique_by(items: List[Any]) -> List[Any]:
        seen = set()
        result = []
        get_op = get(key) if isinstance(key, str) else None
        
        for item in items:
            if callable(key) and not isinstance(key, str):
                key_value = key(item)
            elif get_op:
                res = await get_op.execute(item)
                key_value = res.default_value(None) if res.is_ok() else None
            else:
                key_value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
                
            if key_value not in seen:
                seen.add(key_value)
                result.append(item)
                
        return result
    
    return _unique_by


# ============================================================================
# Transformation operations
# ============================================================================

def map_values(fn: Callable[[Any], Any]) -> Operation[[Dict[str, Any]], Dict[str, Any]]:
    """
    Transform all values in a dict.
    
    Example:
        map_values(str.upper)({"a": "hello", "b": "world"})
        # Returns: {"a": "HELLO", "b": "WORLD"}
    """
    @operation  # type: ignore[arg-type]
    async def _map_values(data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in data.items():
            if isinstance(fn, Operation):
                res = await fn.execute(v)
                result[k] = res.default_value(None) if res.is_ok() else None
            else:
                result[k] = fn(v)
        return result
    
    return _map_values


def map_keys(fn: Callable[[str], str]) -> Operation[[Dict[str, Any]], Dict[str, Any]]:
    """
    Transform all keys in a dict.
    
    Example:
        map_keys(str.upper)({"name": "John", "age": 30})
        # Returns: {"NAME": "John", "AGE": 30}
    """
    @operation  # type: ignore[arg-type]
    async def _map_keys(data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in data.items():
            if isinstance(fn, Operation):
                res = await fn.execute(k)
                new_k = res.default_value(None) if res.is_ok() else None
            else:
                new_k = fn(k)
            if new_k is not None:  # Skip None keys
                result[new_k] = v
        return result
    
    return _map_keys


def rename(mapping: Dict[str, str]) -> Operation[[Dict[str, Any]], Dict[str, Any]]:
    """
    Rename keys in a dict according to a mapping.
    
    Example:
        rename({"user_id": "id", "user_name": "name"})(data)
    """
    @operation  # type: ignore[arg-type]
    def _rename(data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        
        for old_key, value in data.items():
            new_key = mapping.get(old_key, old_key)
            result[new_key] = value
            
        return result
    
    return _rename


def omit(*keys: str) -> Operation[[Dict[str, Any]], Dict[str, Any]]:
    """
    Return a dict without specified keys.
    
    Example:
        omit("password", "secret")(user_data)
    """
    @operation  # type: ignore[arg-type]
    def _omit(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if k not in keys}
    
    return _omit


# ============================================================================
# Aggregation operations
# ============================================================================

def count_by(key: Union[str, Callable[[Any], Any]]) -> Operation[[List[Any]], Dict[Any, int]]:
    """
    Count occurrences of each unique value.
    
    Example:
        # orders_list = [{"status": "pending"}, {"status": "completed"}, {"status": "pending"}]
        # count_by_status_op = count_by("status")
        # counts = await count_by_status_op.execute(orders_list)
        # Returns: {"pending": 2, "completed": 1}
    """
    @operation  # type: ignore[arg-type]
    async def _count_by(items: List[Any]) -> Dict[Any, int]: 
        counts: Dict[Any, int] = {}
        get_op = get(key) if isinstance(key, str) else None
        
        for item in items:
            if callable(key) and not isinstance(key, str):
                value = key(item)
            elif get_op:
                res = await get_op.execute(item)
                value = res.default_value(None) if res.is_ok() else None
            else:
                value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)

            counts[value] = counts.get(value, 0) + 1
            
        return counts
    
    return _count_by


def sum_by(key: Union[str, Callable[[Any], float]]) -> Operation[[List[Any]], float]:
    """
    Sum values extracted by key.
    
    Example:
        # transactions_list = [{"amount": 10.0}, {"amount": 20.5}, {"amount": 5.0}]
        # sum_by_amount_op = sum_by("amount")
        # total_amount = await sum_by_amount_op.execute(transactions_list)
    """
    @operation  # type: ignore[arg-type]
    async def _sum_by(items: List[Any]) -> float:
        total = 0.0
        get_op = get(key) if isinstance(key, str) else None
        
        for item in items:
            if callable(key) and not isinstance(key, str):
                value: Any = key(item)
            elif get_op:
                res = await get_op.execute(item)
                value = res.default_value(None) if res.is_ok() else None
            else:
                value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)

            if value is not None:
                try:
                    total += float(value)
                except (TypeError, ValueError):
                    # silently ignore non-numeric values
                    continue
                
        return total
    
    return _sum_by


# ============================================================================
# Common transformations as operations
# ============================================================================

@operation  # type: ignore[arg-type]
def to_lower(text: str) -> str:
    """Convert string to lowercase."""
    return text.lower() if isinstance(text, str) else text


@operation  # type: ignore[arg-type]
def to_upper(text: str) -> str:
    """Convert string to uppercase."""
    return text.upper() if isinstance(text, str) else text


@operation  # type: ignore[arg-type]
def strip(text: str) -> str:
    """Strip whitespace from string."""
    return text.strip() if isinstance(text, str) else text


def split(delimiter: str = " ") -> Operation[[str], List[str]]:
    """Split string by delimiter."""
    @operation  # type: ignore[arg-type]
    def _split(text: str) -> List[str]:
        return text.split(delimiter) if isinstance(text, str) else []
    return _split


def join(delimiter: str = " ") -> Operation[[List[Any]], str]:
    """Join list of strings."""
    @operation  # type: ignore[arg-type]
    def _join(items: List[Any]) -> str:
        return delimiter.join(str(item) for item in items)
    return _join


@operation  # type: ignore[arg-type]
def keys(data: Dict[str, Any]) -> List[str]:
    """Get dictionary keys."""
    return list(data.keys()) if isinstance(data, dict) else []


@operation  # type: ignore[arg-type]
def values(data: Dict[str, Any]) -> List[Any]:
    """Get dictionary values."""
    return list(data.values()) if isinstance(data, dict) else []


@operation  # type: ignore[arg-type]
def length(data: Union[List, Dict, str]) -> int:
    """Get length of list, dict, or string."""
    return len(data) if hasattr(data, '__len__') else 0


@operation  # type: ignore[arg-type]
def is_empty(data: Union[List, Dict, str]) -> bool:
    """Check if list, dict, or string is empty."""
    return len(data) == 0 if hasattr(data, '__len__') else True


@operation  # type: ignore[arg-type]
def is_not_empty(data: Union[List, Dict, str]) -> bool:
    """Check if list, dict, or string is not empty."""
    return len(data) > 0 if hasattr(data, '__len__') else False


# Helper for async iteration
async def _aiter(seq: List[Any]) -> AsyncGenerator[Any, None]:
    """Helper for async iteration over sequences."""
    for item in seq:
        yield item