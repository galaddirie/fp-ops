import pytest
import asyncio
from typing import Any, Dict, List, Optional, Type, Tuple
from unittest.mock import Mock, patch, AsyncMock

from fp_ops.context import BaseContext
from fp_ops.operator import operation
from expression import Result


from fp_ops.composition import (
    sequence,
    pipe,
    compose,
    parallel,
    fallback,
    transform,
    map,
    filter,
    reduce,
    zip,
    flat_map,
    group_by,
    partition,
    first,
    last,
    gather_operations,
)

async def async_identity(x):
    return x

def sync_identity(x):
    return x

async def async_double(x):
    return x * 2

def sync_double(x):
    return x * 2

async def async_error(x):
    raise ValueError("Async error")

def sync_error(x):
    raise ValueError("Sync error")


@operation
def inc(x: int) -> int:
    return x + 1


@operation
async def async_inc(x: int) -> int:
    await asyncio.sleep(0.01)       # tiny delay to exercise concurrency path
    return x + 1


@operation
def err_on_three(x: int) -> int:
    if x == 3:
        raise ValueError("boom")
    return x


class TestContext(BaseContext):
    value: str = "test"
    label: str = "lab"



@pytest.mark.asyncio
async def test_sequence_basic():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    seq_op = sequence(op1, op2, op3)
    result = await seq_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == [6, 10, 2]

@pytest.mark.asyncio
async def test_sequence_empty():
    seq_op = sequence()
    result = await seq_op.execute(42)
    assert result.is_ok()
    assert result.default_value(None) == []

@pytest.mark.asyncio
async def test_sequence_with_error():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)
    op3 = operation(lambda x: x - 3)
    seq_op = sequence(op1, op2, op3)
    result = await seq_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_sequence_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    seq_op = sequence(op1, op2, op3)
    result = await seq_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == ["5_test_1", "5_test_2", "5_test_3"]

@pytest.mark.asyncio
async def test_sequence_with_context_update():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    @operation(context=True, context_type=TestContext)
    def update_context(x, context=None):
        new_context = TestContext(value="updated")
        return new_context
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    seq_op = sequence(op1, update_context, op3)
    result = await seq_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == ["5_test_1", "5_updated_3"]

@pytest.mark.asyncio
async def test_pipe_basic():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    pipe_op = pipe(op1, op2, op3)
    result = await pipe_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == 9

@pytest.mark.asyncio
async def test_pipe_with_lambdas():
    op1 = operation(lambda x: x + 1)
    def step2(x):
        if x > 5:
            return operation(lambda y: y * 2)
        else:
            return operation(lambda y: y + 5)
    op3 = operation(lambda x: x - 3)
    pipe_op = pipe(op1, step2, op3)
    result1 = await pipe_op.execute(5)
    result2 = await pipe_op.execute(4)
    assert result1.is_ok() and result1.default_value(None) == 9
    assert result2.is_ok() and result2.default_value(None) == 7

@pytest.mark.asyncio
async def test_pipe_empty():
    pipe_op = pipe()
    result = await pipe_op.execute(42)
    assert result.is_ok()
    assert result.default_value(None) is None

@pytest.mark.asyncio
async def test_pipe_with_error():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)
    op3 = operation(lambda x: x - 3)
    pipe_op = pipe(op1, op2, op3)
    result = await pipe_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_pipe_with_lambda_error():
    op1 = operation(lambda x: x + 1)
    def step2(x):
        raise ValueError("Lambda error")
    op3 = operation(lambda x: x - 3)
    pipe_op = pipe(op1, step2, op3)
    result = await pipe_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Lambda error"

@pytest.mark.asyncio
async def test_pipe_with_invalid_lambda_return():
    op1 = operation(lambda x: x + 1)
    def step2(x):
        return "not an operation"
    op3 = operation(lambda x: x - 3)
    pipe_op = pipe(op1, step2, op3)
    result = await pipe_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_pipe_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    pipe_op = pipe(op1, op2, op3)
    result = await pipe_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == "5_test_1_test_2_test_3"

@pytest.mark.asyncio
async def test_pipe_with_context_update():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}"
    @operation(context=True, context_type=TestContext)
    def update_context(x, context=None):
        new_context = TestContext(value="updated")
        return new_context
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}"
    pipe_op = pipe(op1, update_context, op3)
    result = await pipe_op.execute(5, context=context)
    assert result.is_ok()
    assert isinstance(result.default_value(None), TestContext)
    assert result.default_value(None).value == "updated"

@pytest.mark.asyncio
async def test_compose_basic():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    compose_op = compose(op1, op2, op3)
    result = await compose_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == 9

@pytest.mark.asyncio
async def test_compose_empty():
    compose_op = compose()
    result = await compose_op.execute(42)
    assert result.is_ok()
    assert result.default_value(None) == 42

@pytest.mark.asyncio
async def test_compose_single_operation():
    op = operation(lambda x: x * 2)
    compose_op = compose(op)
    result = await compose_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_compose_with_error():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)
    op3 = operation(lambda x: x - 3)
    compose_op = compose(op1, op2, op3)
    result = await compose_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_compose_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    compose_op = compose(op1, op2, op3)
    result = await compose_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == "5_test_1_test_2_test_3"

@pytest.mark.asyncio
async def test_parallel_basic():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    parallel_op = parallel(op1, op2, op3)
    result = await parallel_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == (6, 10, 2)

@pytest.mark.asyncio
async def test_parallel_empty():
    parallel_op = parallel()
    result = await parallel_op.execute(42)
    assert result.is_ok()
    assert result.default_value(None) == ()

@pytest.mark.asyncio
async def test_parallel_with_error():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)
    op3 = operation(lambda x: x - 3)
    parallel_op = parallel(op1, op2, op3)
    result = await parallel_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_parallel_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    @operation(context=True, context_type=TestContext)
    def op3(x, context=None):
        return f"{x}_{context.value}_3"
    parallel_op = parallel(op1, op2, op3)
    result = await parallel_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == ("5_test_1", "5_test_2", "5_test_3")

@pytest.mark.asyncio
async def test_fallback_first_succeeds():
    op1 = operation(lambda x: x * 2)
    op2 = operation(sync_error)
    op3 = operation(sync_error)
    fallback_op = fallback(op1, op2, op3)
    result = await fallback_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_fallback_second_succeeds():
    op1 = operation(sync_error)
    op2 = operation(lambda x: x * 2)
    op3 = operation(sync_error)
    fallback_op = fallback(op1, op2, op3)
    result = await fallback_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_fallback_all_fail():
    op1 = operation(sync_error)
    op2 = operation(lambda x: x / 0)
    op3 = operation(async_error)
    fallback_op = fallback(op1, op2, op3)
    result = await fallback_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Async error"

@pytest.mark.asyncio
async def test_fallback_empty():
    fallback_op = fallback()
    result = await fallback_op.execute(42)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert "No operations provided" in str(result.error)

@pytest.mark.asyncio
async def test_fallback_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        if x < 10:
            raise ValueError("Value too small")
        return f"{x}_{context.value}"
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"fallback_{x}_{context.value}"
    fallback_op = fallback(op1, op2)
    result_fail = await fallback_op.execute(5, context=context)
    result_success = await fallback_op.execute(15, context=context)
    assert result_fail.is_ok()
    assert result_fail.default_value(None) == "fallback_5_test"
    assert result_success.is_ok()
    assert result_success.default_value(None) == "15_test"

@pytest.mark.asyncio
async def test_map_basic():
    base_op = operation(lambda x: x + 1)
    mapper = lambda x: x * 2
    map_op = transform(base_op, mapper)
    result = await map_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == 12

@pytest.mark.asyncio
async def test_map_with_error_in_base():
    base_op = operation(sync_error)
    mapper = lambda x: x * 2
    map_op = transform(base_op, mapper)
    result = await map_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_map_with_error_in_mapper():
    base_op = operation(lambda x: x + 1)
    def error_mapper(x):
        raise ValueError("Mapper error")
    map_op = transform(base_op, error_mapper)
    result = await map_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Mapper error"

@pytest.mark.asyncio
async def test_map_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return f"{x}_{context.value}"
    def mapper(x):
        return f"mapped_{x}"
    map_op = transform(base_op, mapper)
    result = await map_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == "mapped_5_test"

@pytest.mark.asyncio
async def test_filter_pass():
    base_op = operation(lambda x: x + 5)
    predicate = lambda x: x > 7
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(3)
    assert result.is_ok()
    assert result.default_value(None) == 8

@pytest.mark.asyncio
async def test_filter_fail():
    base_op = operation(lambda x: x + 5)
    predicate = lambda x: x > 10
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(3)
    assert result.is_error()
    assert isinstance(result.error, ValueError)

@pytest.mark.asyncio
async def test_filter_with_error_in_base():
    base_op = operation(sync_error)
    predicate = lambda x: x > 7
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(3)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_filter_with_error_in_predicate():
    base_op = operation(lambda x: x + 5)
    def error_predicate(x):
        raise ValueError("Predicate error")
    filter_op = filter(base_op, error_predicate)
    result = await filter_op.execute(3)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Predicate error"

@pytest.mark.asyncio
async def test_filter_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return f"{x}_{context.value}"
    def predicate(x):
        return "test" in x
    filter_op = filter(base_op, predicate)
    result = await filter_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == "5_test"

@pytest.mark.asyncio
async def test_reduce_basic():
    base_op = operation(lambda x: [1, 2, 3, 4, 5])
    reducer = lambda acc, item: acc + item
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    assert result.is_ok()
    assert result.default_value(None) == 15

@pytest.mark.asyncio
async def test_reduce_empty_list():
    base_op = operation(lambda x: [])
    reducer = lambda acc, item: acc + item
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    assert result.is_ok()
    assert result.default_value(None) is None

@pytest.mark.asyncio
async def test_reduce_non_list_input():
    base_op = operation(lambda x: "not a list")
    reducer = lambda acc, item: acc + item
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_reduce_with_error_in_base():
    base_op = operation(sync_error)
    reducer = lambda acc, item: acc + item
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_reduce_with_error_in_reducer():
    base_op = operation(lambda x: [1, 2, 3])
    def error_reducer(acc, item):
        raise ValueError("Reducer error")
    reduce_op = reduce(base_op, error_reducer)
    result = await reduce_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Reducer error"

@pytest.mark.asyncio
async def test_reduce_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return [1, 2, 3, context.value]
    def reducer(acc, item):
        if isinstance(acc, str) or isinstance(item, str):
            return str(acc) + str(item)
        return acc + item
    reduce_op = reduce(base_op, reducer)
    result = await reduce_op.execute(None, context=context)
    assert result.is_ok()
    assert result.default_value(None) == "6test"

@pytest.mark.asyncio
async def test_zip_basic():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    zip_op = zip(op1, op2, op3)
    result = await zip_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == (6, 10, 2)

@pytest.mark.asyncio
async def test_zip_empty():
    zip_op = zip()
    result = await zip_op.execute(42)
    assert result.is_ok()
    assert result.default_value(None) == ()

@pytest.mark.asyncio
async def test_zip_with_error():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)
    op3 = operation(lambda x: x - 3)
    zip_op = zip(op1, op2, op3)
    result = await zip_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ZeroDivisionError)

@pytest.mark.asyncio
async def test_zip_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    zip_op = zip(op1, op2)
    result = await zip_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == ("5_test_1", "5_test_2")

@pytest.mark.asyncio
async def test_flat_map_basic():
    base_op = operation(lambda x: x + 1)
    def mapper(value):
        return [[value, value * 2], [value * 3, value * 4]]
    flat_map_op = flat_map(base_op, mapper)
    result = await flat_map_op.execute(5)
    assert result.is_ok()
    assert result.default_value(None) == [6, 12, 18, 24]

@pytest.mark.asyncio
async def test_flat_map_with_error_in_base():
    base_op = operation(sync_error)
    mapper = lambda x: [[x, x * 2]]
    flat_map_op = flat_map(base_op, mapper)
    result = await flat_map_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_flat_map_with_error_in_mapper():
    base_op = operation(lambda x: x + 1)
    def error_mapper(x):
        raise ValueError("Mapper error")
    flat_map_op = flat_map(base_op, error_mapper)
    result = await flat_map_op.execute(5)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Mapper error"

@pytest.mark.asyncio
async def test_flat_map_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return f"{x}_{context.value}"
    def mapper(value):
        return [[f"{value}_a", f"{value}_b"], [f"{value}_c"]]
    flat_map_op = flat_map(base_op, mapper)
    result = await flat_map_op.execute(5, context=context)
    assert result.is_ok()
    assert result.default_value(None) == ["5_test_a", "5_test_b", "5_test_c"]

@pytest.mark.asyncio
async def test_group_by_basic():
    base_op = operation(lambda x: [1, 2, 3, 4, 5, 6])
    grouper = lambda x: "even" if x % 2 == 0 else "odd"
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    assert result.is_ok()
    grouped = result.default_value(None)
    assert "odd" in grouped and "even" in grouped
    assert grouped["odd"] == [1, 3, 5]
    assert grouped["even"] == [2, 4, 6]

@pytest.mark.asyncio
async def test_group_by_empty_list():
    base_op = operation(lambda x: [])
    grouper = lambda x: x
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    assert result.is_ok()
    assert result.default_value(None) == {}

@pytest.mark.asyncio
async def test_group_by_non_list_input():
    base_op = operation(lambda x: "not a list")
    grouper = lambda x: x
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_group_by_with_error_in_base():
    base_op = operation(sync_error)
    grouper = lambda x: x
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_group_by_with_error_in_grouper():
    base_op = operation(lambda x: [1, 2, 3])
    def error_grouper(x):
        raise ValueError("Grouper error")
    group_by_op = group_by(base_op, error_grouper)
    result = await group_by_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Grouper error"

@pytest.mark.asyncio
async def test_group_by_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return [f"a_{context.value}", f"b_{context.value}", f"a_other", f"c_{context.value}"]
    grouper = lambda s: s[0]
    group_by_op = group_by(base_op, grouper)
    result = await group_by_op.execute(None, context=context)
    assert result.is_ok()
    grouped = result.default_value(None)
    assert "a" in grouped and "b" in grouped and "c" in grouped
    assert len(grouped["a"]) == 2
    assert grouped["b"] == [f"b_{context.value}"]
    assert grouped["c"] == [f"c_{context.value}"]

@pytest.mark.asyncio
async def test_partition_basic():
    base_op = operation(lambda x: [1, 2, 3, 4, 5, 6])
    predicate = lambda x: x % 2 == 0
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    assert result.is_ok()
    partitioned = result.default_value(None)
    assert isinstance(partitioned, tuple) and len(partitioned) == 2
    assert partitioned[0] == [2, 4, 6]
    assert partitioned[1] == [1, 3, 5]

@pytest.mark.asyncio
async def test_partition_empty_list():
    base_op = operation(lambda x: [])
    predicate = lambda x: True
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    assert result.is_ok()
    partitioned = result.default_value(None)
    assert isinstance(partitioned, tuple) and len(partitioned) == 2
    assert partitioned[0] == []
    assert partitioned[1] == []

@pytest.mark.asyncio
async def test_partition_non_list_input():
    base_op = operation(lambda x: "not a list")
    predicate = lambda x: True
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_partition_with_error_in_base():
    base_op = operation(sync_error)
    predicate = lambda x: True
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_partition_with_error_in_predicate():
    base_op = operation(lambda x: [1, 2, 3])
    def error_predicate(x):
        raise ValueError("Predicate error")
    partition_op = partition(base_op, error_predicate)
    result = await partition_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Predicate error"

@pytest.mark.asyncio
async def test_partition_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return [f"has_{context.value}", "no_match", f"also_has_{context.value}", "another_no_match"]
    def predicate(s):
        return "test" in s
    partition_op = partition(base_op, predicate)
    result = await partition_op.execute(None, context=context)
    assert result.is_ok()
    partitioned = result.default_value(None)
    assert isinstance(partitioned, tuple) and len(partitioned) == 2
    assert partitioned[0] == [f"has_{context.value}", f"also_has_{context.value}"]
    assert partitioned[1] == ["no_match", "another_no_match"]

@pytest.mark.asyncio
async def test_first_basic():
    base_op = operation(lambda x: [10, 20, 30, 40])
    first_op = first(base_op)
    result = await first_op.execute(None)
    assert result.is_ok()
    assert result.default_value(None) == 10

@pytest.mark.asyncio
async def test_first_empty_list():
    base_op = operation(lambda x: [])
    first_op = first(base_op)
    result = await first_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, IndexError)
    assert "empty" in str(result.error).lower()

@pytest.mark.asyncio
async def test_first_non_list_input():
    base_op = operation(lambda x: "not a list")
    first_op = first(base_op)
    result = await first_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_first_with_error_in_base():
    base_op = operation(sync_error)
    first_op = first(base_op)
    result = await first_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_first_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return [f"first_{context.value}", "second", "third"]
    first_op = first(base_op)
    result = await first_op.execute(None, context=context)
    assert result.is_ok()
    assert result.default_value(None) == f"first_{context.value}"

@pytest.mark.asyncio
async def test_last_basic():
    base_op = operation(lambda x: [10, 20, 30, 40])
    last_op = last(base_op)
    result = await last_op.execute(None)
    assert result.is_ok()
    assert result.default_value(None) == 40

@pytest.mark.asyncio
async def test_last_empty_list():
    base_op = operation(lambda x: [])
    last_op = last(base_op)
    result = await last_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, IndexError)
    assert "empty" in str(result.error).lower()

@pytest.mark.asyncio
async def test_last_non_list_input():
    base_op = operation(lambda x: "not a list")
    last_op = last(base_op)
    result = await last_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, TypeError)

@pytest.mark.asyncio
async def test_last_with_error_in_base():
    base_op = operation(sync_error)
    last_op = last(base_op)
    result = await last_op.execute(None)
    assert result.is_error()
    assert isinstance(result.error, ValueError)
    assert str(result.error) == "Sync error"

@pytest.mark.asyncio
async def test_last_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def base_op(x, context=None):
        return ["first", "second", f"last_{context.value}"]
    last_op = last(base_op)
    result = await last_op.execute(None, context=context)
    assert result.is_ok()
    assert result.default_value(None) == f"last_{context.value}"

@pytest.mark.asyncio
async def test_gather_operations_basic():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x * 2)
    op3 = operation(lambda x: x - 3)
    results = await gather_operations(op1, op2, op3, args=(5,))
    assert len(results) == 3
    assert all(isinstance(r, Result) for r in results)
    assert all(r.is_ok() for r in results)
    assert [r.default_value(None) for r in results] == [6, 10, 2]

@pytest.mark.asyncio
async def test_gather_operations_with_error():
    op1 = operation(lambda x: x + 1)
    op2 = operation(lambda x: x / 0)
    op3 = operation(lambda x: x - 3)
    results = await gather_operations(op1, op2, op3, args=(5,))
    assert len(results) == 3
    assert results[0].is_ok() and results[0].default_value(None) == 6
    assert results[1].is_error() and isinstance(results[1].error, ZeroDivisionError)
    assert results[2].is_ok() and results[2].default_value(None) == 2

@pytest.mark.asyncio
async def test_gather_operations_with_context():
    context = TestContext()
    @operation(context=True, context_type=TestContext)
    def op1(x, context=None):
        return f"{x}_{context.value}_1"
    @operation(context=True, context_type=TestContext)
    def op2(x, context=None):
        return f"{x}_{context.value}_2"
    results = await gather_operations(op1, op2, args=(5,), kwargs={"context": context})
    assert len(results) == 2
    assert all(r.is_ok() for r in results)
    assert results[0].default_value(None) == "5_test_1"
    assert results[1].default_value(None) == "5_test_2"

@pytest.mark.asyncio
async def test_gather_operations_without_args():
    op1 = operation(lambda: 10)
    op2 = operation(lambda: 20)
    results = await gather_operations(op1, op2)
    assert len(results) == 2
    assert all(r.is_ok() for r in results)
    assert results[0].default_value(None) == 10
    assert results[1].default_value(None) == 20

@pytest.mark.asyncio
async def test_gather_operations_with_bound_operations():
    op1 = operation(lambda x: x + 1)(5)
    op2 = operation(lambda x: x * 2)(10)
    results = await gather_operations(op1, op2)
    assert len(results) == 2
    assert all(r.is_ok() for r in results)
    assert results[0].default_value(None) == 6
    assert results[1].default_value(None) == 20




@pytest.mark.asyncio
async def test_map_iter_basic():
    op = map(inc)
    res: Result[List[int], Exception] = await op.execute([1, 2, 3])
    assert res.is_ok()
    assert res.default_value(None) == [2, 3, 4]


@pytest.mark.asyncio
async def test_map_iter_generator_input():
    op = map(inc)
    res = await op.execute((i for i in range(4)))
    assert res.is_ok()
    assert res.default_value(None) == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_map_iter_async_inner():
    op = map(async_inc)
    res = await op.execute([0, 1])
    assert res.is_ok()
    assert res.default_value(None) == [1, 2]


@pytest.mark.asyncio
async def test_map_iter_error_propagates():
    op = map(err_on_three)
    res = await op.execute([1, 2, 3, 4])
    assert res.is_error()
    assert isinstance(res.error, ValueError)
    assert str(res.error) == "boom"


# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

@operation(context=True, context_type=TestContext)
def tag(x: int, **kwargs) -> str:
    ctx = kwargs.get("context")
    return f"{ctx.label}:{x}"


@pytest.mark.asyncio
async def test_map_iter_with_context():
    ctx = TestContext(label="lab")
    op = map(tag)
    res = await op.execute([1, 2, 3], context=ctx)
    assert res.is_ok()
    assert res.default_value(None) == ["lab:1", "lab:2", "lab:3"]


# ---------------------------------------------------------------------------
# Concurrency limiter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_iter_max_concurrency_respected():
    current = 0          # number of in-flight calls
    peak = 0
    lock = asyncio.Lock()

    @operation
    async def tracked(x: int) -> int:
        nonlocal current, peak
        async with lock:
            current += 1
            peak = max(peak, current)
        await asyncio.sleep(0.02)   # keep several coroutines alive together
        async with lock:
            current -= 1
        return x

    op = map(tracked, max_concurrency=2)
    res = await op.execute(range(6))
    assert res.is_ok()
    assert res.default_value(None) == list(range(6))
    assert peak <= 2, f"observed concurrency {peak} exceeds limit"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_iter_empty_iterable():
    op = map(inc)
    res = await op.execute([])
    assert res.is_ok()
    assert res.default_value(None) == []
