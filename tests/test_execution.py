from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import pytest

# The module under test -----------------------------------------------
from fp_ops.execution import _merge_first_call, ExecutionPlan, Executor, _has_nested_placeholder  # add the import
from fp_ops.primitives import Placeholder, _


# ---------------------------------------------------------------------
# _has_nested_placeholder tests
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        # Simple placeholder
        (_, True),
        (Placeholder(), True),
        
        # Nested in dict
        ({"key": _}, True),
        ({"key": "value", "nested": {"deep": _}}, True),
        ({"key": "value"}, False),
        
        # Nested in list
        ([1, 2, _], True),
        ([1, 2, [3, _]], True),
        ([1, 2, 3], False),
        
        # Nested in tuple
        ((1, 2, _), True),
        ((1, 2, (3, _)), True),
        ((1, 2, 3), False),
        
        # Mixed nesting
        ({"key": [1, 2, _]}, True),
        ([{"key": _}, 2, 3], True),
        (({"key": [1, _, 3]},), True),
        
        # Non-container types
        (123, False),
        ("string", False),
        (None, False),
    ],
)
def test_has_nested_placeholder(obj: Any, expected: bool) -> None:
    """Test that _has_nested_placeholder correctly detects Placeholder instances."""
    assert _has_nested_placeholder(obj) is expected


# ---------------------------------------------------------------------
# _merge_first_call – behaviour contract
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "fn",
        "base_args",
        "base_kwargs",
        "rt_args",
        "rt_kwargs",
        "expected_args",
        "expected_kwargs",
    ),
    [
        # 1. pure *args – all positionals are forwarded verbatim
        (
            lambda *args: args,
            (1, 2),
            {},
            (3, 4),
            {},
            (1, 2, 3, 4),
            {},
        ),
        # 2. runtime keyword overrides template keyword
        (
            lambda a, b: (a, b),
            (),
            {"a": 10},
            (),
            {"a": 99, "b": 0},
            (),
            {"a": 99, "b": 0},
        ),
        # 3. precedence: template positional fills first, then runtime positional fills next
        (
            lambda a, b, c: (a, b, c),
            (0,),  # template positional for *a*
            {"c": 9},  # template kw for *c*
            (1,),  # runtime positional for *b*
            {},
            (),
            {"a": 0, "b": 1, "c": 9},
        ),
        # 4. empty runtime args – expect pristine template
        (
            lambda a, b=2: (a, b),
            (42,),
            {"b": 5},
            (),
            {},
            (42,),
            {"b": 5},
        ),
    ],
)
def test_merge_first_call_happy_path(
    fn: Callable[..., Any],
    base_args: Tuple[Any, ...],
    base_kwargs: Dict[str, Any],
    rt_args: Tuple[Any, ...],
    rt_kwargs: Dict[str, Any],
    expected_args: Tuple[Any, ...],
    expected_kwargs: Dict[str, Any],
) -> None:
    """Validate the correct precedence rules for argument merging."""

    signature = inspect.signature(fn)
    merged_args, merged_kwargs = _merge_first_call(
        signature,
        base_args,
        base_kwargs,
        rt_args,
        rt_kwargs,
    )

    assert merged_args == expected_args
    assert merged_kwargs == expected_kwargs


def test_merge_first_call_too_many_positional() -> None:
    """Supplying more positionals than parameters should be silently ignored."""

    def f(a):  # noqa: D401 – simple stub
        return a

    sig = inspect.signature(f)

    # Extra positional args (2, 3) are silently ignored
    args, kwargs = _merge_first_call(sig, (), {}, (1, 2, 3), {})
    assert kwargs == {"a": 1}


# ---------------------------------------------------------------------
# Executor – end-to-end happy-path run
# ---------------------------------------------------------------------
# Minimal stubs that emulate the subset of OpSpec / Template used by the
# ExecutionPlan builder.  They are intentionally *very* small to keep the test
# independent from the rest of fp_ops internals.
# ---------------------------------------------------------------------
class _StubTemplate:  # pylint: disable=too-few-public-methods
    def __init__(self, args: Tuple = (), kwargs: Dict | None = None):
        self.args = args
        self.kwargs = kwargs or {}

    # The plan builder branches on these helpers ----------------------
    def has_placeholders(self) -> bool:  # noqa: D401 – simple stub
        return False

    def render(self, _value: Any):  # pragma: no cover
        raise RuntimeError("render() should not be invoked in these tests")


@dataclass(slots=True, frozen=True)
class _StubSpec:  # noqa: D101 – internal helper
    id: str
    func: Callable[..., Any]
    template: _StubTemplate
    signature: inspect.Signature
    require_ctx: bool = False
    ctx_type: type | None = None


async def _double(x: int) -> int:  # noqa: D401 – simple stub
    return x * 2


async def _plus_one(x: int) -> int:  # noqa: D401 – simple stub
    return x + 1


@pytest.mark.asyncio
async def test_executor_happy_path() -> None:
    """Executor should respect topo-order and propagate the running value."""

    # Build a manual execution plan → double → plus_one
    spec1 = _StubSpec(
        id="double",
        func=_double,
        template=_StubTemplate(),
        signature=inspect.signature(_double),
    )
    spec2 = _StubSpec(
        id="plus_one",
        func=_plus_one,
        template=_StubTemplate(),
        signature=inspect.signature(_plus_one),
    )

    # arg_render rules mimic what ExecutionPlan.from_graph would have produced
    arg_render: Dict[str, Callable[[Any, Any | None], Tuple[Tuple, Dict]]] = {
        spec1.id: lambda _prev, _ctx: ((), {}),  # first op – takes first_args
        spec2.id: lambda prev, _ctx: ((prev,), {}),  # unary – pipe value fwd
    }

    plan = ExecutionPlan(order=(spec1, spec2), arg_render=arg_render, successors={})
    executor = Executor(plan)

    result = await executor.run(3)

    # ``default_value`` extracts the wrapped value from ``Ok``
    assert result.default_value(None) == 7  # (3 * 2) + 1
