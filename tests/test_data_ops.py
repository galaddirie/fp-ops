import pytest
import pytest_asyncio # For async tests

from fp_ops.data import (
    get, pick, pluck, build, merge, update,
    filter_by, group_by, sort_by, unique_by,
    map_values, map_keys, rename, omit,
    count_by, sum_by,
    to_lower, to_upper, strip, split, join, keys, values, length,
    is_empty, is_not_empty
)
from fp_ops import Operation # For type hinting or checking if needed
from expression import Result, Ok, Error # For checking results more directly if needed
from fp_ops import constant
from fp_ops.composition import compose


# --- Test Data ---

TEST_DATA_SIMPLE_DICT = {"name": "Alice", "age": 30, "city": "New York"}
TEST_DATA_NESTED_DICT = {
    "user": {
        "id": 1,
        "name": "Bob",
        "profile": {"email": "bob@example.com", "active": True},
        "tags": ["developer", "python"],
    },
    "items": [
        {"id": "a", "price": 100, "qty": 2},
        {"id": "b", "price": 200, "qty": 1},
    ],
    "settings": None,
    "attributes": {"color": "blue", "size": "M"}
}

class MockAddress:
    def __init__(self, street):
        self.street = street

class MockUser:
    def __init__(self, username, age):
        self.username = username
        self.age = age
        self.address = MockAddress("123 Main St")

TEST_DATA_OBJECT = MockUser(username="Carol", age=25)

TEST_DATA_LIST_OF_DICTS = [
    {"id": 1, "name": "Alice", "role": "admin", "score": 90, "country": "USA"},
    {"id": 2, "name": "Bob", "role": "user", "score": 75, "country": "Canada"},
    {"id": 3, "name": "Charlie", "role": "user", "score": 85, "country": "USA"},
    {"id": 4, "name": "Alice", "role": "editor", "score": 90, "country": "UK"},
    {"id": 5, "name": "David", "role": "user", "score": None, "country": "Canada"}, # Score is None
]

# Fixtures from test_data_2.py
@pytest.fixture
def user_dict():
    return {
        "id": 1,
        "profile": {
            "name": "Galad",
            "email": "galad@example.com",
            "stats": {"posts": 7},
        },
        "orders": [
            {"id": "o1", "amount": 10.5, "status": "pending"},
            {"id": "o2", "amount": 20.0, "status": "completed"},
            {"id": "o3", "amount": 10.5, "status": "pending"},
        ],
    }


@pytest.fixture
def users_list(user_dict):
    bob = {
        "id": 2,
        "profile": {
            "name": "Bob",
            "email": "bob@example.com",
            "stats": {"posts": 3},
        },
        "orders": [],
    }
    return [user_dict, bob]

# --- Tests for get ---

@pytest.mark.asyncio
async def test_get_simple_path():
    get_name = get("name")
    result = await get_name.execute(TEST_DATA_SIMPLE_DICT)
    assert result.is_ok()
    assert result.default_value(None) == "Alice"

@pytest.mark.asyncio
async def test_get_nested_path():
    get_email = get("user.profile.email")
    result = await get_email.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value(None) == "bob@example.com"

@pytest.mark.asyncio
async def test_get_list_index():
    get_first_item_id = get("items.0.id")
    result = await get_first_item_id.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value(None) == "a"

    get_second_item_price = get("items[1].price") # Test bracket notation
    result = await get_second_item_price.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value(None) == 200


@pytest.mark.asyncio
async def test_get_attribute_access():
    get_username = get("username")
    result = await get_username.execute(TEST_DATA_OBJECT)
    assert result.is_ok()
    assert result.default_value(None) == "Carol"

    get_street = get("address.street")
    result = await get_street.execute(TEST_DATA_OBJECT)
    assert result.is_ok()
    assert result.default_value(None) == "123 Main St"

@pytest.mark.asyncio
async def test_get_path_not_found_with_default():
    get_non_existent = get("user.profile.non_existent_key", "default_val")
    result = await get_non_existent.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value("ERROR") == "default_val"

@pytest.mark.asyncio
async def test_get_path_not_found_without_default():
    get_non_existent = get("user.profile.non_existent_key")
    result = await get_non_existent.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value("ERROR") is None # Default for get() is None

@pytest.mark.asyncio
async def test_get_intermediate_path_is_none():
    get_setting_detail = get("settings.detail", "default_val")
    result = await get_setting_detail.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value("ERROR") == "default_val"

@pytest.mark.asyncio
async def test_get_empty_path():
    get_data = get("")
    result = await get_data.execute(TEST_DATA_SIMPLE_DICT)
    assert result.is_ok()
    assert result.default_value(None) == TEST_DATA_SIMPLE_DICT

@pytest.mark.asyncio
async def test_get_out_of_bounds_index():
    get_invalid_index = get("items.5.id", "default_index")
    result = await get_invalid_index.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value("ERROR") == "default_index"

# --- Tests for pick ---

@pytest.mark.asyncio
async def test_pick_existing_keys():
    pick_op = pick("name", "age")
    result = await pick_op.execute(TEST_DATA_SIMPLE_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {"name": "Alice", "age": 30}

@pytest.mark.asyncio
async def test_pick_some_non_existing_keys():
    pick_op = pick("name", "non_existent")
    result = await pick_op.execute(TEST_DATA_SIMPLE_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {"name": "Alice"} # non_existent is ignored

@pytest.mark.asyncio
async def test_pick_all_non_existing_keys():
    pick_op = pick("foo", "bar")
    result = await pick_op.execute(TEST_DATA_SIMPLE_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {}

@pytest.mark.asyncio
async def test_pick_from_object():
    pick_op = pick("username", "age")
    result = await pick_op.execute(TEST_DATA_OBJECT)
    assert result.is_ok()
    assert result.default_value(None) == {"username": "Carol", "age": 25}

@pytest.mark.asyncio
async def test_pick_with_nested_paths():
    pick_op = pick("user.name", "items.0.price", "user.profile.email")
    result = await pick_op.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    # The keys in the result dict are the last part of the path
    assert result.default_value(None) == {"name": "Bob", "price": 100, "email": "bob@example.com"}

@pytest.mark.asyncio
async def test_pick_nested_path_not_found():
    pick_op = pick("user.non_existent_attr", "user.name")
    result = await pick_op.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {"name": "Bob"}

# --- Tests for pluck ---

@pytest.mark.asyncio
async def test_pluck_existing_key():
    pluck_name = pluck("name")
    result = await pluck_name.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    assert result.default_value(None) == ["Alice", "Bob", "Charlie", "Alice", "David"]

@pytest.mark.asyncio
async def test_pluck_non_existing_key():
    pluck_non_existent = pluck("non_existent_key")
    result = await pluck_non_existent.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    # \\`get\\` defaults to None for non-existent keys
    assert result.default_value("ERROR") == [None, None, None, None, None]

@pytest.mark.asyncio
async def test_pluck_from_empty_list():
    pluck_name = pluck("name")
    result = await pluck_name.execute([])
    assert result.is_ok()
    assert result.default_value(None) == []

@pytest.mark.asyncio
async def test_pluck_nested_key():
    # Example data for this test
    data = [
        {"item": {"id": 1, "details": {"value": 10}}},
        {"item": {"id": 2, "details": {"value": 20}}},
        {"item": {"id": 3, "details": None}}, # Intermediate None
        {"item": {"id": 4}}, # Missing 'details'
    ]
    pluck_value = pluck("item.details.value")
    result = await pluck_value.execute(data)
    assert result.is_ok()
    assert result.default_value("ERROR") == [10, 20, None, None]

# --- Tests for update ---

@pytest.mark.asyncio
async def test_update_existing_and_new_keys():
    update_op = update({"age": 31, "city": "San Francisco", "new_key": "new_value"})
    result = await update_op.execute(TEST_DATA_SIMPLE_DICT.copy()) # Use copy to avoid modifying original
    assert result.is_ok()
    assert result.default_value(None) == {"name": "Alice", "age": 31, "city": "San Francisco", "new_key": "new_value"}

@pytest.mark.asyncio
async def test_update_empty_dict():
    update_op = update({"a": 1})
    result = await update_op.execute({})
    assert result.is_ok()
    assert result.default_value(None) == {"a": 1}

@pytest.mark.asyncio
async def test_update_with_empty_update_values():
    update_op = update({})
    original_copy = TEST_DATA_SIMPLE_DICT.copy()
    result = await update_op.execute(original_copy)
    assert result.is_ok()
    assert result.default_value(None) == original_copy

# --- Tests for build ---

@pytest.mark.asyncio
async def test_build_static_values():
    schema = {
        "greeting": "Hello",
        "count": 42,
        "active": True,
        "meta": {"version": "1.0"}
    }
    build_op = build(schema)
    result = await build_op.execute(TEST_DATA_SIMPLE_DICT) # Data can be anything if not used by schema
    assert result.is_ok()
    assert result.default_value(None) == schema

@pytest.mark.asyncio
async def test_build_with_get_operations():
    schema = {
        "user_name": get("user.name"),
        "user_email": get("user.profile.email", "N/A"),
        "first_item_id": get("items.0.id")
    }
    build_op = build(schema)
    result = await build_op.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {
        "user_name": "Bob",
        "user_email": "bob@example.com",
        "first_item_id": "a"
    }

@pytest.mark.asyncio
async def test_build_with_lambdas():
    schema = {
        "full_name": lambda d: f"{d['user']['name']} ({d['user']['id']})",
        "total_items": lambda d: len(d['items'])
    }
    build_op = build(schema)
    result = await build_op.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {
        "full_name": "Bob (1)",
        "total_items": 2
    }

@pytest.mark.asyncio
async def test_build_with_nested_schema():
    schema = {
        "id": get("user.id"),
        "contact": build({ # Nested build
            "email": get("user.profile.email"),
            "is_active": get("user.profile.active")
        }),
        "first_tag": get("user.tags.0")
    }
    build_op = build(schema)
    result = await build_op.execute(TEST_DATA_NESTED_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {
        "id": 1,
        "contact": {
            "email": "bob@example.com",
            "is_active": True
        },
        "first_tag": "developer"
    }

@pytest.mark.asyncio
async def test_build_lambda_receives_correct_data():
    schema = {"name_from_data": lambda data: data["name"]}
    build_op = build(schema)
    result = await build_op.execute(TEST_DATA_SIMPLE_DICT)
    assert result.is_ok()
    assert result.default_value(None) == {"name_from_data": "Alice"}

@pytest.mark.asyncio
async def test_build_operation_error_handling():
    # Test with a lambda that raises an error.
    result_direct_lambda_fail = await build({
        "a": lambda d: d["name"],
        "b": lambda d: 1/0 # raises ZeroDivisionError
    }).execute(TEST_DATA_SIMPLE_DICT)
    assert result_direct_lambda_fail.is_ok()
    assert result_direct_lambda_fail.default_value(None) == {"a": "Alice", "b": None}

    # Test how build handles an Operation that itself might return an Error.
    # Based on build's logic: if res.is_ok(): result[key] = res.default_value(None)
    # This means if the operation fails and res.is_ok() is false, the key might not be set,
    # or if Operation.default_value(None) is called on Error, it might raise.
    # Let's simulate an operation that returns an error.
    
    # Create a simple failing operation for testing purposes
    @Operation._from_function 
    async def failing_op_func(_data):
        return Error(ValueError("Simulated failure"))

    build_with_failing_op = build({
        "good_val": get("name"),
        "failed_val": failing_op_func # This is an Operation instance
    })
    result = await build_with_failing_op.execute(TEST_DATA_SIMPLE_DICT)
    assert result.is_ok()
    # The build operation sets the value to None if the sub-operation doesn't return Ok.
    assert result.default_value(None) == {"good_val": "Alice", "failed_val": None}


# --- Tests for merge ---

@pytest.mark.asyncio
async def test_merge_multiple_dicts():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    dict3 = {"d": 5}
    merge_op = merge(dict1, dict2, dict3)
    # Merge doesn't take input data if all sources are static dicts
    result = await merge_op.execute(None) # Or any data, it won't be used
    assert result.is_ok()
    assert result.default_value(None) == {"a": 1, "b": 3, "c": 4, "d": 5}

@pytest.mark.asyncio
async def test_merge_with_callables_and_get_ops():
    data_for_merge = {"val1": 10, "user_profile": {"name": "Eve", "id": 101}}
    
    merge_op = merge(
        {"static_a": 1},
        get("user_profile"), # This will return {"name": "Eve", "id": 101}
        lambda d: {"computed_b": d["val1"] * 2},
        {"static_a": 99} # Override
    )
    result = await merge_op.execute(data_for_merge)
    assert result.is_ok()
    assert result.default_value(None) == {
        "static_a": 99,
        "name": "Eve", 
        "id": 101,
        "computed_b": 20
    }

@pytest.mark.asyncio
async def test_merge_callable_returns_non_dict():
    data = {"val": 5}
    merge_op = merge(
        {"a": 1},
        lambda d: "not a dict" # This should be skipped by merge
    )
    result = await merge_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"a": 1}


@pytest.mark.asyncio
async def test_merge_with_no_args():
    merge_op = merge()
    result = await merge_op.execute(None)
    assert result.is_ok()
    assert result.default_value(None) == {}

@pytest.mark.asyncio
async def test_merge_with_get_op_that_returns_none():
    data = {"profile": None} # 'settings' in TEST_DATA_NESTED_DICT is None
    merge_op = merge(
        get("profile"), # This will evaluate to None using `data`
        {"fallback": True}
    )
    result = await merge_op.execute(data)
    assert result.is_ok()
    # Since get("profile")(data) is None, it's not a dict and won't be merged.
    assert result.default_value(None) == {"fallback": True}

# --- Tests for filter_by ---

@pytest.mark.asyncio
async def test_filter_by_lambda():
    filter_op = filter_by(lambda x: x["score"] is not None and x["score"] >= 85)
    result = await filter_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    expected = [
        {"id": 1, "name": "Alice", "role": "admin", "score": 90, "country": "USA"},
        {"id": 3, "name": "Charlie", "role": "user", "score": 85, "country": "USA"},
        {"id": 4, "name": "Alice", "role": "editor", "score": 90, "country": "UK"},
    ]
    assert result.default_value(None) == expected

@pytest.mark.asyncio
async def test_filter_by_dict_predicate():
    filter_op = filter_by({"role": "user", "country": "USA"})
    result = await filter_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    expected = [
        {"id": 3, "name": "Charlie", "role": "user", "score": 85, "country": "USA"},
    ]
    assert result.default_value(None) == expected

@pytest.mark.asyncio
async def test_filter_by_dict_predicate_nested():
    data = [
        {"user": {"status": "active", "id": 1}},
        {"user": {"status": "inactive", "id": 2}},
        {"user": {"status": "active", "id": 3}},
    ]
    filter_op = filter_by({"user.status": "active"})
    result = await filter_op.execute(data)
    assert result.is_ok()
    expected = [
        {"user": {"status": "active", "id": 1}},
        {"user": {"status": "active", "id": 3}},
    ]
    assert result.default_value(None) == expected

@pytest.mark.asyncio
async def test_filter_by_empty_list():
    filter_op = filter_by(lambda x: x["score"] > 0)
    result = await filter_op.execute([])
    assert result.is_ok()
    assert result.default_value(None) == []

@pytest.mark.asyncio
async def test_filter_by_no_matches():
    filter_op = filter_by({"role": "superuser"})
    result = await filter_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    assert result.default_value(None) == []

# --- Tests for group_by ---

@pytest.mark.asyncio
async def test_group_by_string_key():
    group_op = group_by("role")
    result = await group_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    grouped = result.default_value(None)
    assert "admin" in grouped and len(grouped["admin"]) == 1
    assert "user" in grouped and len(grouped["user"]) == 3
    assert "editor" in grouped and len(grouped["editor"]) == 1
    assert grouped["admin"][0]["name"] == "Alice"

@pytest.mark.asyncio
async def test_group_by_lambda():
    group_op = group_by(lambda x: "high_score" if x.get("score", 0) is not None and x["score"] >= 85 else "lower_score")
    result = await group_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    grouped = result.default_value(None)
    assert "high_score" in grouped and len(grouped["high_score"]) == 3
    assert "lower_score" in grouped and len(grouped["lower_score"]) == 2
    # Check one item from high_score
    assert any(item["name"] == "Alice" for item in grouped["high_score"])
     # Check one item from lower_score (Bob or David)
    assert any(item["name"] == "Bob" for item in grouped["lower_score"]) or \
           any(item["name"] == "David" for item in grouped["lower_score"])


@pytest.mark.asyncio
async def test_group_by_nested_string_key():
    data = [
        {"id": 1, "data": {"category": "A"}},
        {"id": 2, "data": {"category": "B"}},
        {"id": 3, "data": {"category": "A"}},
    ]
    group_op = group_by("data.category")
    result = await group_op.execute(data)
    assert result.is_ok()
    grouped = result.default_value(None)
    assert "A" in grouped and len(grouped["A"]) == 2
    assert "B" in grouped and len(grouped["B"]) == 1

@pytest.mark.asyncio
async def test_group_by_key_not_present():
    # If key is not present, \\`get\\` will return None, so they'll be grouped under None
    group_op = group_by("non_existent_key")
    result = await group_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    grouped = result.default_value(None)
    assert None in grouped and len(grouped[None]) == len(TEST_DATA_LIST_OF_DICTS)

# --- Tests for sort_by ---

@pytest.mark.asyncio
async def test_sort_by_string_key_ascending():
    sort_op = sort_by("name")
    # Take a slice to have a defined initial order for testing sort
    data_slice = [item for item in TEST_DATA_LIST_OF_DICTS if item["name"] in ["Charlie", "Alice", "Bob"]]
    result = await sort_op.execute(data_slice) 
    assert result.is_ok()
    names = [item["name"] for item in result.default_value(None)]
    assert names == ["Alice", "Alice", "Bob", "Charlie"]

@pytest.mark.asyncio
async def test_sort_by_string_key_descending_with_none():
    sort_op = sort_by("score", reverse=True)
    result = await sort_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    scores = [item["score"] for item in result.default_value(None)]
    # Python's default sort: None is smaller than any number.
    # For reverse=True, largest numbers first, then smaller, then Nones at the end.
    expected_scores_order = [90, 90, 85, 75, None]
    assert scores == expected_scores_order


@pytest.mark.asyncio
async def test_sort_by_lambda():
    sort_op = sort_by(lambda x: len(x["name"])) # Sort by length of name
    data_slice = [item for item in TEST_DATA_LIST_OF_DICTS if item["name"] in ["Charlie", "Alice", "Bob"]]
    result = await sort_op.execute(data_slice) 
    assert result.is_ok()
    names = [item["name"] for item in result.default_value(None)]
    assert names == ["Bob", "Alice", "Alice", "Charlie"], f"Expected names to be sorted by length: {names}"


@pytest.mark.asyncio
async def test_sort_by_nested_key():
    data = [
        {"details": {"order": 3, "name": "C"}},
        {"details": {"order": 1, "name": "A"}},
        {"details": {"order": 2, "name": "B"}},
    ]
    sort_op = sort_by("details.order")
    result = await sort_op.execute(data)
    assert result.is_ok()
    names = [item["details"]["name"] for item in result.default_value(None)]
    assert names == ["A", "B", "C"]

# --- Tests for unique_by ---

@pytest.mark.asyncio
async def test_unique_by_string_key():
    unique_op = unique_by("name") # First Alice encountered will be kept
    result = await unique_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    unique_items = result.default_value(None)
    names = [item["name"] for item in unique_items]
    assert names == ["Alice", "Bob", "Charlie", "David"]
    assert len(unique_items) == 4
    first_alice_original = next(item for item in TEST_DATA_LIST_OF_DICTS if item["name"] == "Alice")
    kept_alice = next(item for item in unique_items if item["name"] == "Alice")
    assert kept_alice["role"] == first_alice_original["role"] 

@pytest.mark.asyncio
async def test_unique_by_lambda():
    # Unique by first letter of name
    unique_op = unique_by(lambda x: x["name"][0])
    result = await unique_op.execute(TEST_DATA_LIST_OF_DICTS) # Names: Alice, Bob, Charlie, Alice, David -> A, B, C, A, D
    assert result.is_ok()
    names = [item["name"] for item in result.default_value(None)]
    # Expected: First 'A' (Alice), 'B' (Bob), 'C' (Charlie), 'D' (David)
    assert names == ["Alice", "Bob", "Charlie", "David"] 

@pytest.mark.asyncio
async def test_unique_by_non_existent_key():
    # \\`get\\` returns None for all, so only the first item should be kept as None is a unique key seen once
    unique_op = unique_by("non_existent")
    result = await unique_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    assert len(result.default_value(None)) == 1
    assert result.default_value(None)[0] == TEST_DATA_LIST_OF_DICTS[0]

@pytest.mark.asyncio
async def test_unique_by_nested_key():
    data = [
        {"item": {"id": 1, "val": "x"}},
        {"item": {"id": 2, "val": "y"}},
        {"item": {"id": 3, "val": "x"}}, # Duplicate val "x"
    ]
    unique_op = unique_by("item.val")
    result = await unique_op.execute(data)
    assert result.is_ok()
    ids = [item["item"]["id"] for item in result.default_value(None)]
    assert ids == [1, 2] # Keeps first 'x' (id 1), then 'y' (id 2)

# --- Tests for map_values ---

@pytest.mark.asyncio
async def test_map_values_simple():
    data = {"a": 1, "b": 2, "c": 3}
    map_op = map_values(lambda x: x * 2)
    result = await map_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"a": 2, "b": 4, "c": 6}

@pytest.mark.asyncio
async def test_map_values_string_transform():
    data = {"a": "hello", "b": "world"}
    map_op = map_values(str.upper)
    result = await map_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"a": "HELLO", "b": "WORLD"}

# --- Tests for map_keys ---

@pytest.mark.asyncio
async def test_map_keys_simple():
    data = {"name": "John", "age": 30}
    map_op = map_keys(str.upper)
    result = await map_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"NAME": "John", "AGE": 30}

@pytest.mark.asyncio
async def test_map_keys_prefixing():
    data = {"id": 1, "value": 100}
    map_op = map_keys(lambda k: f"prefix_{k}")
    result = await map_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"prefix_id": 1, "prefix_value": 100}

# --- Tests for rename ---

@pytest.mark.asyncio
async def test_rename_keys():
    data = {"user_id": 123, "user_name": "Kate", "status": "active"}
    rename_map = {"user_id": "id", "user_name": "name"}
    rename_op = rename(rename_map)
    result = await rename_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"id": 123, "name": "Kate", "status": "active"}

@pytest.mark.asyncio
async def test_rename_no_matching_keys():
    data = {"a": 1, "b": 2}
    rename_map = {"c": "new_c"}
    rename_op = rename(rename_map)
    result = await rename_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"a": 1, "b": 2} # No change

# --- Tests for omit ---

@pytest.mark.asyncio
async def test_omit_keys():
    data = {"id": 1, "name": "Test", "secret": "xyz", "public": True}
    omit_op = omit("secret", "another_non_existent")
    result = await omit_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"id": 1, "name": "Test", "public": True}

@pytest.mark.asyncio
async def test_omit_all_keys():
    data = {"a": 1, "b": 2}
    omit_op = omit("a", "b")
    result = await omit_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {}

# --- Tests for count_by ---

@pytest.mark.asyncio
async def test_count_by_string_key():
    count_op = count_by("role")
    result = await count_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    assert result.default_value(None) == {"admin": 1, "user": 3, "editor": 1}

@pytest.mark.asyncio
async def test_count_by_lambda():
    def score_type(item):
        score = item.get("score")
        if score is None: return "unknown"
        return "even" if score % 2 == 0 else "odd"

    count_op = count_by(score_type)
    result = await count_op.execute(TEST_DATA_LIST_OF_DICTS)
    # Scores: 90 (even), 75 (odd), 85 (odd), 90 (even), None (unknown)
    assert result.is_ok()
    assert result.default_value(None) == {"even": 2, "odd": 2, "unknown": 1}

@pytest.mark.asyncio
async def test_count_by_nested_key():
    data = [
        {"data": {"type": "A"}}, {"data": {"type": "B"}},
        {"data": {"type": "A"}}, {"data": {"type": "A"}},
    ]
    count_op = count_by("data.type")
    result = await count_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == {"A": 3, "B": 1}

# --- Tests for sum_by ---

@pytest.mark.asyncio
async def test_sum_by_string_key():
    sum_op = sum_by("score") # Scores: 90, 75, 85, 90, None
    result = await sum_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    assert result.default_value(None) == 90.0 + 75.0 + 85.0 + 90.0 # 340.0

@pytest.mark.asyncio
async def test_sum_by_lambda():
    # Sum of scores for users only
    def user_score(item):
        if item["role"] == "user":
            return item.get("score") # Returns None if score is not present or is None
        return None # Other roles contribute None, which sum_by skips

    sum_op = sum_by(user_score)
    # User scores: 75, 85, None
    result = await sum_op.execute(TEST_DATA_LIST_OF_DICTS)
    assert result.is_ok()
    assert result.default_value(None) == 75.0 + 85.0 # 160.0

@pytest.mark.asyncio
async def test_sum_by_nested_key():
    data = [
        {"item": {"value": 10.5}},
        {"item": {"value": 20}},
        {"item": {}}, # No value, get returns None
        {"item": {"value": None}}, # Value is None
        {"item": {"value": 5.5}},
    ]
    sum_op = sum_by("item.value")
    result = await sum_op.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == 10.5 + 20.0 + 5.5 # 36.0

@pytest.mark.asyncio
async def test_sum_by_empty_list():
    sum_op = sum_by("score")
    result = await sum_op.execute([])
    assert result.is_ok()
    assert result.default_value(None) == 0.0

# --- Tests for common transformations (to_lower, to_upper, strip) ---

@pytest.mark.asyncio
async def test_to_lower():
    op = to_lower
    result = await op.execute("Hello World")
    assert result.is_ok() and result.default_value(None) == "hello world"
    result_non_string = await op.execute(123)
    assert result_non_string.is_ok() and result_non_string.default_value(None) == 123 # Returns input if not string

@pytest.mark.asyncio
async def test_to_upper():
    op = to_upper
    result = await op.execute("Hello World")
    assert result.is_ok() and result.default_value(None) == "HELLO WORLD"
    result_non_string = await op.execute(123)
    assert result_non_string.is_ok() and result_non_string.default_value(None) == 123

@pytest.mark.asyncio
async def test_strip():
    op = strip
    result = await op.execute("  Hello World  ")
    assert result.is_ok() and result.default_value(None) == "Hello World"
    result_non_string = await op.execute(123)
    assert result_non_string.is_ok() and result_non_string.default_value(None) == 123

# --- Tests for split and join ---

@pytest.mark.asyncio
async def test_split_default_delimiter():
    split_op = split() # Default delimiter is space
    result = await split_op.execute("hello world example")
    assert result.is_ok() and result.default_value(None) == ["hello", "world", "example"]
    result_non_string = await split().execute(123) # Test with non-string
    assert result_non_string.is_ok() and result_non_string.default_value(None) == []


@pytest.mark.asyncio
async def test_split_custom_delimiter():
    split_op = split(",")
    result = await split_op.execute("a,b,c")
    assert result.is_ok() and result.default_value(None) == ["a", "b", "c"]
    result_empty = await split_op.execute("") # Test with empty string
    assert result_empty.is_ok() and result_empty.default_value(None) == [""]


@pytest.mark.asyncio
async def test_join_default_delimiter():
    join_op = join() # Default delimiter is space
    result = await join_op.execute(["hello", "world", 123]) # Mix of types
    assert result.is_ok() and result.default_value(None) == "hello world 123"
    result_empty_list = await join().execute([])
    assert result_empty_list.is_ok() and result_empty_list.default_value(None) == ""


@pytest.mark.asyncio
async def test_join_custom_delimiter():
    join_op = join("-")
    result = await join_op.execute(["a", "b", "c"])
    assert result.is_ok() and result.default_value(None) == "a-b-c"

# --- Tests for keys and values ---

@pytest.mark.asyncio
async def test_keys_op():
    op = keys
    data = {"a": 1, "b": 2, "c": 3}
    result = await op.execute(data)
    assert result.is_ok()
    # Order of keys is insertion order for Python 3.7+
    # For robustness, convert to set for comparison or sort
    assert sorted(list(result.default_value(None))) == ["a", "b", "c"]
    
    result_empty_dict = await op.execute({})
    assert result_empty_dict.is_ok() and result_empty_dict.default_value(None) == []
    
    result_non_dict = await op.execute([1,2,3]) # Test with non-dict
    assert result_non_dict.is_ok() and result_non_dict.default_value(None) == []


@pytest.mark.asyncio
async def test_values_op():
    op = values
    data = {"a": 1, "b": 2, "c": 3}
    result = await op.execute(data)
    assert result.is_ok()
    assert sorted(list(result.default_value(None))) == [1, 2, 3]

    result_empty_dict = await op.execute({})
    assert result_empty_dict.is_ok() and result_empty_dict.default_value(None) == []

    result_non_dict = await op.execute([1,2,3]) # Test with non-dict
    assert result_non_dict.is_ok() and result_non_dict.default_value(None) == []


# --- Tests for length, is_empty, is_not_empty ---

@pytest.mark.asyncio
@pytest.mark.parametrize("data_input, expected_len", [
    ([1, 2, 3], 3),
    ("hello", 5),
    ({"a": 1, "b": 2}, 2),
    ([], 0),
    ("", 0),
    ({}, 0),
    (123, 0), # Non-len
    (None, 0) # Non-len
])
async def test_length_op(data_input, expected_len):
    op = length
    result = await op.execute(data_input)
    assert result.is_ok() and result.default_value(-1) == expected_len

@pytest.mark.asyncio
@pytest.mark.parametrize("data_input, expected_is_empty", [
    ([1, 2, 3], False),
    ("hello", False),
    ({"a": 1, "b": 2}, False),
    ([], True),
    ("", True),
    ({}, True),
    (123, True), # Non-len
    (None, True) # Non-len
])
async def test_is_empty_op(data_input, expected_is_empty):
    op = is_empty
    result = await op.execute(data_input)
    assert result.is_ok() and result.default_value("ERROR") == expected_is_empty

@pytest.mark.asyncio
@pytest.mark.parametrize("data_input, expected_is_not_empty", [
    ([1, 2, 3], True),
    ("hello", True),
    ({"a": 1, "b": 2}, True),
    ([], False),
    ("", False),
    ({}, False),
    (123, False), # Non-len
    (None, False) # Non-len
])
async def test_is_not_empty_op(data_input, expected_is_not_empty):
    op = is_not_empty
    result = await op.execute(data_input)
    assert result.is_ok() and result.default_value("ERROR") == expected_is_not_empty

@pytest.mark.asyncio
async def test_get_variants(user_dict):
    assert (await get("id").execute(user_dict)).default_value(None) == 1
    assert (await get("profile.name").execute(user_dict)).default_value(None) == "Galad"
    assert (await get("orders.0.amount").execute(user_dict)).default_value(None) == 10.5
    assert (await get("profile.age", 99).execute(user_dict)).default_value(None) == 99


@pytest.mark.asyncio
async def test_pick_and_pluck(user_dict, users_list):
    got = (await pick("id", "profile.email", "profile.stats.posts")
           .execute(user_dict)).default_value({})
    assert got == {"id": 1, "email": "galad@example.com", "posts": 7}



@pytest.mark.asyncio
async def test_build_merge_update(user_dict):
    schema = {
        "USER_ID": get("id"),
        "upper": get("profile.name") >> to_upper,
        "posts": get("profile.stats.posts"),
        "static": 42,
    }
    built = (await build(schema).execute(user_dict)).default_value({})
    assert built == {"USER_ID": 1, "upper": "GALAD", "posts": 7, "static": 42}

    merged = merge({"a": 1}, get("profile"), lambda d: {"posts": d["profile"]["stats"]["posts"]})
    updated = update({"extra": True})
    out = (await (merged >> updated).execute(user_dict)).default_value({})
    assert out["a"] == 1 and out["name"] == "Galad" and out["extra"] is True


@pytest.mark.asyncio
async def test_collection_ops(users_list):
    # filter-by predicate & dict pattern
    pred = filter_by(lambda u: u["profile"]["stats"]["posts"] > 5)
    assert len((await pred.execute(users_list)).default_value([])) == 1

    by_dict = filter_by({"profile.name": "Bob"})
    assert (await by_dict.execute(users_list)).default_value([])[0]["id"] == 2

    # unique
    dup = users_list + [users_list[0]]
    assert len((await unique_by("id").execute(dup)).default_value([])) == 2

    # sort
    sorted_list = (await sort_by(lambda u: u["profile"]["stats"]["posts"], reverse=True)
                   .execute(users_list)).default_value([])
    assert [u["id"] for u in sorted_list] == [1, 2]

    # group & aggregations on orders
    groups = (await group_by("profile.name").execute(users_list)).default_value({})
    assert set(groups.keys()) == {"Galad", "Bob"}

    orders = users_list[0]["orders"]
    counts = (await count_by("status").execute(orders)).default_value({})
    total = (await sum_by("amount").execute(orders)).default_value(0.0)
    assert counts == {"pending": 2, "completed": 1}
    assert pytest.approx(total) == 41.0



@pytest.mark.asyncio
async def test_map_values_keys_rename_omit(): # No fixtures used from test_data_2
    src = {"A": "hello ", "B": "World "}
    op = map_values(strip) >> map_values(to_lower) >> map_keys(to_lower)
    res = (await op.execute(src)).default_value({})
    assert res == {"a": "hello", "b": "world"}

    src2 = {"user_id": 1, "secret": "x", "keep": True}
    op2 = rename({"user_id": "id"}) >> omit("secret")
    res2 = (await op2.execute(src2)).default_value({})
    assert res2 == {"id": 1, "keep": True}

    text = "foo,bar,baz"
    joined = (await (split(",") >> join("|")).execute(text)).default_value("")
    assert joined == "foo|bar|baz"



@pytest.mark.asyncio
async def test_pipeline_integration(user_dict):
    # RHS bound
    pip1 = get("profile.name") >> to_upper
    # RHS with placeholder route
    pip2 = get("profile.name") >> constant >> to_upper

    for pip in (pip1, pip2):
        assert (await pip.execute(user_dict)).default_value(None) == "GALAD"

    # compose helper vs >>
    op1 = compose(get("id"), constant(10))
    op2 = get("id") >> constant(10)
    assert (await op1.execute(user_dict)).default_value(None) == 10
    assert (await op2.execute(user_dict)).default_value(None) == 10

@pytest.mark.asyncio
async def test_data_ops_with_placeholders():
    """Test data operations with placeholder primitives."""
    from fp_ops.primitives import _  # Import the placeholder
    
    # Test placeholders in function composition pipeline
    data = {"name": "Alice", "age": 30, "role": "admin"}
    
    # First get name, then lowercase it
    pipeline = get("name") >> to_lower
    result = await pipeline.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == "alice"
    
    # Test filtering with pipeline-passed criteria
    users = [
        {"name": "Alice", "score": 85},
        {"name": "Bob", "score": 75},
        {"name": "Charlie", "score": 95}
    ]
    
    # Set minimum score, then filter users who meet it
    min_score = 80
    pipeline = constant(min_score) >> filter_by(lambda x: x["score"] >= _)
    result = await pipeline.execute(users)
    assert result.is_ok()
    filtered = result.default_value(None)
    assert len(filtered) == 2
    assert filtered[0]["name"] == "Alice"
    assert filtered[1]["name"] == "Charlie"
    
    # Test sorting with pipeline-provided key
    pipeline = constant("score") >> sort_by(_)
    result = await pipeline.execute(users)
    assert result.is_ok()
    sorted_users = result.default_value(None)
    assert [u["name"] for u in sorted_users] == ["Bob", "Alice", "Charlie"]
    
    # Test build with a value from pipeline
    user = {"id": 123, "name": "Dave"}
    current_date = "2023-04-01"
    
    # Pass date through pipeline to build function
    pipeline = constant(current_date) >> build({
        "id": get("id"),
        "name": get("name"),
        "created_at": _  # Use the date passed from previous step
    })
    result = await pipeline.execute(user)
    assert result.is_ok()
    built = result.default_value(None)
    assert built == {"id": 123, "name": "Dave", "created_at": "2023-04-01"}

@pytest.mark.asyncio
async def test_more_data_ops_with_placeholders():
    """Test additional data operations with placeholder primitives."""
    from fp_ops.primitives import _  # Import the placeholder
    
    # Test map_values with multiplier from pipeline
    data = {"a": 1, "b": 2, "c": 3}
    multiplier = 10
    
    # Pass multiplier through pipeline to map_values
    pipeline = constant(multiplier) >> map_values(lambda x: x * _)
    result = await pipeline.execute(data)
    assert result.is_ok()
    mapped = result.default_value(None)
    assert mapped == {"a": 10, "b": 20, "c": 30}
    
    # Test group_by with key from pipeline
    users = [
        {"name": "Alice", "department": "Engineering"},
        {"name": "Bob", "department": "Sales"},
        {"name": "Charlie", "department": "Engineering"}
    ]
    
    # Pass grouping key through pipeline
    pipeline = constant("department") >> group_by(_)
    result = await pipeline.execute(users)
    assert result.is_ok()
    grouped = result.default_value(None)
    assert len(grouped["Engineering"]) == 2
    assert len(grouped["Sales"]) == 1
    
    # Test omit with field names from pipeline
    user = {"id": 123, "name": "Alice", "password": "secret", "api_key": "abc123"}
    
    # Pass list of fields to omit through pipeline
    pipeline = constant(["password", "api_key"]) >> omit(*_)
    result = await pipeline.execute(user)
    assert result.is_ok()
    sanitized = result.default_value(None)
    assert sanitized == {"id": 123, "name": "Alice"}

@pytest.mark.asyncio
async def test_chained_ops_with_placeholders():
    """Test chaining multiple operations with placeholders."""
    from fp_ops.primitives import _  # Import the placeholder
    
    # Create a data processing pipeline
    users = [
        {"id": 1, "name": "Alice", "level": 3, "active": True},
        {"id": 2, "name": "Bob", "level": 2, "active": False},
        {"id": 3, "name": "Charlie", "level": 4, "active": True},
        {"id": 4, "name": "Dave", "level": 1, "active": True}
    ]
    
    # Chain operations: filter active users, then those with level >= min_level
    pipeline = (
        filter_by({"active": True}) >>
        constant(2) >>  # min_level value
        filter_by(lambda u: u["level"] >= _) >>
        constant("level") >>  # sort key
        sort_by(_, reverse=True)  # sort by level descending
    )
    
    result = await pipeline.execute(users)
    assert result.is_ok()
    processed = result.default_value(None)
    
    # Should have filtered inactive users and those with level < 2,
    # then sorted by level (highest first)
    assert len(processed) == 2
    assert processed[0]["id"] == 3  # Charlie (level 4)
    assert processed[1]["id"] == 1  # Alice (level 3)

@pytest.mark.asyncio
async def test_error_handling_with_placeholders():
    """Test error handling with placeholders in pipelines."""
    from fp_ops.primitives import _  # Import the placeholder
    
    # Test providing a default value through the pipeline
    data = {"user": None}
    
    # Try to access a nested field that doesn't exist, with default from pipeline
    pipeline = (
        constant("Default User") >>  # default value
        get("user.name").default_value(_)  # use the default value from pipeline
    )
    
    result = await pipeline.execute(data)
    assert result.is_ok()
    assert result.default_value(None) == "Default User"
    
    # Test a more complex fallback pipeline
    items = [1, 2, 3]
    
    # This should fail because we try to access a dict attribute on a list
    failing_op = get("items")
    
    # Pass fallback value through pipeline
    pipeline = (
        constant(["fallback"]) >>
        failing_op.default_value(_)
    )
    
    result = await pipeline.execute(items)
    assert result.is_ok()
    assert result.default_value(None) == ["fallback"]

@pytest.mark.asyncio
async def test_simple_placeholder():
    """Test basic placeholder functionality with simple operations."""
    from fp_ops.primitives import _
    from fp_ops.operator import operation
    
    # Define a simple operation that doubles a number
    @operation
    async def double(x):
        return x * 2
    
    # Define an operation that adds a value
    @operation
    async def add(x, y):
        return x + y
    
    # Test with placeholders in pipeline
    data = 5
    
    # Pipeline 1: Double, then add 10 (using placeholder)
    # This makes the second operation receive the output of the first
    pipeline1 = double >> add(_, 10)
    result1 = await pipeline1.execute(data)
    assert result1.is_ok()
    assert result1.default_value(None) == 20  # (5*2) + 10 = 20
    
    # Pipeline 2: The same thing but using a lambda instead
    pipeline2 = double >> (lambda x: add(x, 10))
    result2 = await pipeline2.execute(data)
    assert result2.is_ok()
    assert result2.default_value(None) == 20  # (5*2) + 10 = 20

