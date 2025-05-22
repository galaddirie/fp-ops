"""
Extra tests that exercise the new "append" semantics together with
context handling, positional / keyword collision rules, and error paths.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any

from expression import Error
from fp_ops.context import BaseContext
from fp_ops.operator import operation
from fp_ops.placeholder import _


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #

class Ctx(BaseContext):
    tenant: str


# ── context-aware ops ------------------------------------------------- #

@operation(context=True, context_type=Ctx)
async def who_am_i(**kwargs) -> str:
    """Return the tenant name to prove ctx injection works."""
    context = kwargs.pop("context", None)
    return context.tenant


@operation(context=True, context_type=Ctx)
async def greet(name: str, loud: bool = False, **kwargs) -> str:
    """Greet a person, optionally in all caps."""
    context = kwargs.pop("context", None)
    out = f"hello {name} from {context.tenant}"
    return out.upper() if loud else out


@operation
async def add(a: int, b: int) -> int:
    return a + b

@operation
async def mul(x: int, y: int) -> int:
    return x * y

@operation
async def tri_sum(a: int, b: int, c: int) -> int:
    return a + b + c



pytestmark = pytest.mark.asyncio(scope="function")


class TestAppendSemanticsBasic:

    @pytest.mark.asyncio
    async def test_append_kwargs(self):
        """Test that kwargs are merged (not replaced) in single-step operations."""
        # Create operation with initial kwargs
        op = add(a=1)
        
        # Add more kwargs - should KEEP a=1 and ADD b=2
        op2 = op(b=2)
        
        result = await op2.execute()
        assert result.is_ok()
        assert result.default_value(None) == 3
        
        # Original operation should remain unchanged
        result = await op.execute(b=5)
        assert result.default_value(None) == 6
    
    @pytest.mark.asyncio
    async def test_append_args(self):
        """Test that positional args are appended in single-step operations."""
        # Create operation with initial arg
        op = add(1)
        
        # Add more args - should result in add(1, 2)
        op2 = op(2)
        
        result = await op2.execute()
        assert result.is_ok()
        assert result.default_value(None) == 3
        
        # Original operation should remain unchanged
        result = await op.execute(2)
        assert result.default_value(None) == 3
    
    @pytest.mark.asyncio
    async def test_append_mixed(self):
        """Test mixing positional and keyword args in append operations."""
        # Create operation with initial arg and kwarg
        op = mul(2, y=3)
        
        # Add kwargs that override existing ones
        op2 = op(y=5)
        
        result = await op2.execute()
        assert result.is_ok()
        assert result.default_value(None) == 10  # 2 * 5
    
    @pytest.mark.asyncio
    async def test_partial_method(self):
        """Test that the partial method behaves the same as direct calling."""
        op = add(a=1)
        
        # These should be equivalent
        op_call = op(b=2)
        op_partial = op.partial(b=2)
        
        result1 = await op_call.execute()
        result2 = await op_partial.execute()
        
        assert result1.is_ok() and result2.is_ok()
        assert result1.default_value(None) == result2.default_value(None) == 3


    @pytest.mark.asyncio
    async def test_multi_call_single_step(self):
        """Test multiple calls to a single-step operation."""
        # Start with one arg
        op = add(1)
        
        # Add a second arg
        op2 = op(2)
        
        # Adding a third arg would exceed the function signature
        op3 = op2(3)  # This won't fail yet, but execution will
        
        # The error occurs during execution when the function is actually called
        
        result = await op3.execute()
        assert result.is_error()
        assert "positional" in str(result.error)
            
        # But we can update existing args with kwargs
        op4 = op(b=10)
        result = await op4.execute()
        assert result.is_ok()
        assert result.default_value(None) == 11

    
class TestContextAwareSingleStep:

    async def test_constants_preserved_when_calling_with_only_context(self):
        op = greet(name="alice")            # bind const
        ctx = Ctx(tenant="ACME")

        # second call carries only context → should EXECUTE, not rebind
        result = await op(context=ctx)
        assert result.default_value(None) == "hello alice from ACME"

    async def test_append_kw_and_ctx(self):
        """
        First call binds a constant, second call appends/overrides *and*
        injects context.
        """
        op = greet(name="alice")            # name bound
        op2 = op(loud=True)                 # append kw
        ctx = Ctx(tenant="ACME")

        res = await op2(context=ctx)
        # loud=True should be respected, name still "alice"
        assert res.default_value(None) == "HELLO ALICE FROM ACME"

    async def test_placeholder_with_context(self):
        """
        Make sure append rules do **not** interfere with placeholder
        rendering in ctx-aware ops.
        """
        pipeline = who_am_i >> greet(_, loud=True)
        ctx = Ctx(tenant="ACME")

        out = await pipeline(context=ctx).execute()
        assert out.default_value(None) == "HELLO ACME FROM ACME"


class TestPositionalKwargEdgeCases:

    async def test_merge_pos_and_kw(self):
        """
        Append positional arguments; then override one of them with a kwarg.
        """
        op = tri_sum(1)             # a = 1 (pos)
        op2 = op(2)                 # b = 2 (append pos)
        op3 = op2(c=10)             # kw overrides c
        res = await op3.execute()
        assert res.default_value(None) == 13    # 1 + 2 + 10

    async def test_too_many_positionals_raises(self):
        """
        tri_sum expects 3 args; appending a 4th should fail *at execution*.
        """
        op = tri_sum(1, 2, 3)(4)    # 4th positional
        res = await op.execute()
        assert res.is_error()
        assert isinstance(res.error, TypeError)
        assert "positional" in str(res.error).lower()


class TestPipelineWithCtxFactory:

    async def test_ctx_factory_respected_after_append(self):
        """
        When the op is built with `Operation.with_context`, subsequent
        appends must keep the factory.
        """
        base = greet(context=Ctx(tenant="TENANT-1"))
        # bind one kw
        op = base(name="bob")
        # append new kw (no ctx supplied here)
        op2 = op(loud=True)

        res = await op2.execute()
        assert res.is_ok()
        assert res.default_value(None) == "HELLO BOB FROM TENANT-1"


class TestPlaceholderDeepMerging:

    async def test_deep_placeholder_preserved_after_append(self):
        """
        A template containing placeholders inside nested structures must
        keep them after we append new kwargs.
        """
        @operation
        async def wrap(data: Any, meta: dict[str, Any]) -> tuple[Any, dict]:
            return data, meta

        op     = wrap(meta={"src": "test", "payload": _})
        op_app = op(data="VALUE")           # append pos → keeps nested _

        res = await op_app.execute("INNER")  # fills deep placeholder
        res_val = res.default_value(None)
        assert res_val == ("INNER", {"src": "test", "payload": "INNER"}), f"res: {res_val}"
