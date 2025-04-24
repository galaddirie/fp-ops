from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from fp_ops.operator import operation
from fp_ops.placeholder import _


@pytest.fixture
def event_loop():
    """Provide a fresh event-loop per test (pytest-asyncio legacy pattern)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@operation
async def add(a: int, b: int) -> int:
    return a + b


@operation
async def add_one(a: int) -> int:
    return a + 1


@operation
async def mul(x: int, y: int) -> int:
    return x * y


@operation
async def sub(x: int, y: int) -> int:
    return x - y


@operation
async def negate(x: int) -> int:
    return -x


@operation
async def div(x: int, y: int) -> float:
    return x / y


@operation
async def mod(x: int, y: int) -> int:
    return x % y


@operation
async def identity(value: Any) -> Any:
    return value




class TestSimpleChaining:

    @pytest.mark.asyncio
    async def test_chain_no_binding(self):
        pipeline = add >> add_one
        pipeline.validate()

        result = await pipeline.execute(a=1, b=2)
        assert result.is_ok()
        # (1 + 2) + 1 = 4
        assert result.default_value(None) == 4

    
    @pytest.mark.asyncio
    async def test_chain_first_bound_only(self):
        pipeline = add(1, 2) >> add_one  # (1+2)+1
        pipeline.validate()

        result = await pipeline.execute()
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_chain_both_bound(self):
        pipeline = add(1, 2) >> add_one(1)  # add_one(1) => 2
        pipeline.validate()

        result = await pipeline.execute()
        assert result.is_ok()
        assert result.default_value(None) == 2

    @pytest.mark.asyncio
    async def test_chain_first_unbound_second_bound(self):
        pipeline = add >> add_one(1)  # add_one(1) => 2
        pipeline.validate()

        result = await pipeline.execute()
        assert result.is_ok()
        assert result.default_value(None) == 2


class TestPlaceholderBinding:
    """`_` placeholder behaviour"""

    @pytest.mark.asyncio
    async def test_placeholder_explicit_propagates_previous_result(self):
        pipeline = add >> add_one(_)
        pipeline.validate()

        result = await pipeline.execute(a=1, b=2)
        assert result.is_ok()
        # (1 + 2) + 1 = 4
        assert result.default_value(None) == 4


    @pytest.mark.asyncio
    async def test_placeholder_first_pos(self):
        pipeline = add >> mul(_, 3)  # (1+2) * 3
        pipeline.validate()

        result = await pipeline.execute(a=1, b=2)
        assert result.is_ok()
        assert result.default_value(None) == 9

    @pytest.mark.asyncio
    async def test_placeholder_second_pos(self):
        pipeline = add >> mul(3, _)  # 3 * (1+2)
        pipeline.validate()

        result = await pipeline.execute(a=1, b=2)
        assert result.is_ok()
        assert result.default_value(None) == 9

    @pytest.mark.asyncio
    async def test_placeholder_in_first_position_with_first_op_bound(self):
        pipeline = add(1, 2) >> mul(_, 3)  # (1+2) * 3
        pipeline.validate()

        result = await pipeline.execute()
        assert result.is_ok()
        assert result.default_value(None) == 9

    @pytest.mark.asyncio
    async def test_placeholder_in_second_position_with_first_op_bound(self):
        pipeline = add(1, 2) >> mul(3, _)  # 3 * (1+2)
        pipeline.validate()

        result = await pipeline.execute()
        assert result.is_ok()
        assert result.default_value(None) == 9


class TestMultiLevelChaining:
    """Three-stage pipelines"""

    @pytest.mark.asyncio
    async def test_multi_level_with_no_binding_or_placeholders(self):
        pipeline = add >> add_one >> identity
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_multi_level_with_first_op_prebound(self):
        pipeline = add(1, 2) >> add_one >> identity
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_multi_level_with_second_op_prebound(self):
        pipeline = add >> add_one(1) >> identity
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 2 # add_one(1) => 2

    @pytest.mark.asyncio
    async def test_multi_level_with_third_op_prebound(self):
        pipeline = add >> add_one >> identity(7)
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 7

    @pytest.mark.asyncio
    async def test_multi_level_with_first_op_prebound_and_second_op_prebound(self):
        pipeline = add(1, 2) >> add_one(1) >> identity
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 2 # add_one(1) => 2

    @pytest.mark.asyncio
    async def test_multi_level_with_first_op_prebound_and_third_op_prebound(self):
        pipeline = add(1, 2) >> add_one >> identity(7)
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 7

    @pytest.mark.asyncio
    async def test_multi_level_with_second_op_prebound_and_third_op_prebound(self):
        pipeline = add >> add_one(1) >> identity(7)
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 7

    @pytest.mark.asyncio
    async def test_multi_level_with_first_op_prebound_and_second_op_prebound_and_third_op_prebound(
        self,
    ):
        pipeline = add(1, 2) >> add_one(1) >> identity(7)
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 7

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholder_in_first_op(self):
        pipeline = add(_, 2) >> add_one >> identity
        pipeline.validate()

        result = await pipeline(1).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4  # (1 + 2) + 1 = 4

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholder_in_second_op(self):
        pipeline = add >> add_one(_) >> identity
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4  # (1 + 2) + 1 = 4

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholder_in_third_op(self):
        pipeline = add >> add_one >> identity(_)
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4  # ((1 + 2) + 1) = 4

    @pytest.mark.asyncio
    async def test_multi_level_with_multiple_placeholders(self):
        pipeline = add(_, _) >> add_one >> identity
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4  # (1 + 2) + 1 = 4

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholders_and_prebound_ops(self):
        pipeline = add(1, _) >> add_one(_) >> identity
        pipeline.validate()

        result = await pipeline(2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4  # (1 + 2) + 1 = 4

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholders_in_all_ops(self):
        pipeline = add(_, _) >> add_one(_) >> identity(_)
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4  # ((1 + 2) + 1) = 4

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholders_mixed_with_first_op_prebound(self):
        pipeline = add(1, 2) >> add_one >> identity(_)
        pipeline.validate()

        result = await pipeline.execute()
        assert result.is_ok()
        # ((1+2)+1) = 4
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholders_mixed_with_second_op_prebound(self):
        pipeline = add >> add_one(1) >> identity(_)
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        # add_one(1) => 2
        assert result.default_value(None) == 2

    @pytest.mark.asyncio
    async def test_multi_level_with_placeholders_mixed_with_first_op_prebound_and_second_op_prebound(
        self,
    ):
        pipeline = add(1, 2) >> add_one(1) >> identity(_)
        pipeline.validate()

        result = await pipeline.execute()
        assert result.is_ok()
        # add_one(1) => 2
        assert result.default_value(None) == 2


# TODO: Implement this test but not yet
class TestMultiLevelChainingLarge:
    """
    Testing various large chained operations and how they bind.
    Ensuring associativity and proper order of operations.
    """
    ...


# TODO: Implement this test
class TestAssociativityAndOrderOfOperationsForBasicChains:
    ...

class TestExecutionPatterns:
    """Different ways of *running* a pipeline"""

    @pytest.mark.asyncio
    async def test_execute_method(self):
        pipeline = add >> add_one
        pipeline.validate()

        result = await pipeline.execute(a=1, b=2)
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_call_then_execute(self):
        pipeline = add >> add_one
        pipeline.validate()

        result = await pipeline(1, 2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_mixed_params(self):
        pipeline = add >> add_one
        pipeline.validate()

        # positional + keyword
        result = await pipeline(1, b=2).execute()
        assert result.is_ok()
        assert result.default_value(None) == 4

        # keyword after build
        result = await pipeline.execute(1, b=2)
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_first_arg_as_kwarg(self):
        pipeline = add >> add_one
        pipeline.validate()

        result = await pipeline(2, a=1)
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_kwarg_only(self):
        pipeline = add >> add_one
        pipeline.validate()

        result = await pipeline(a=1, b=2)
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_prebound_ops(self):
        pipeline = add(1, 2) >> add_one
        pipeline.validate()

        result = await pipeline()
        assert result.is_ok()
        assert result.default_value(None) == 4

    @pytest.mark.asyncio
    async def test_prebound_ops_override_with_kwargs(self):
        pipeline = add(1, 2) >> add_one
        pipeline.validate()

        # kwargs override only affect downstream, not already bound add(1,2)
        result = await pipeline(a=5, b=6)
        assert result.is_ok()
        # add(1,2)=3; add_one(??) uses kwargs (a ignored, the first positional)
        assert result.default_value(None) == 4, "we should ignore the params passed in if we have a prebound op"

    @pytest.mark.asyncio
    async def test_prebound_ops_override_with_args(self):
        pipeline = add(1, 2) >> add_one
        pipeline.validate()

        result = await pipeline(5, 6)
        assert result.is_ok()
        assert result.default_value(None) == 4, "we should ignore the params passed in if we have a prebound op"

    @pytest.mark.asyncio
    async def test_prebound_ops_override_with_args_mixed_params(self):
        pipeline = add(1, 2) >> add_one
        pipeline.validate()

        result = await pipeline(5, b=6)
        assert result.is_ok()
        assert result.default_value(None) == 4, "we should ignore the params passed in if we have a prebound op"
