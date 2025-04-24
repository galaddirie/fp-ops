from __future__ import annotations
from typing import (
    Callable,
    Awaitable,
    Any,
    Optional,
    Dict,
    List,
    TypeVar,
    Generic,
    overload,
    ParamSpec,
    Type,
    Union,
    Tuple,
    cast,
)
import inspect
import copy
from types import MappingProxyType
from dataclasses import replace

from fp_ops.graph import OpGraph
from fp_ops.primitives import HandleId, OpSpec
from fp_ops.execution import ExecutionPlan, _default_executor

from fp_ops.placeholder import _, Template
from fp_ops.context import BaseContext
from fp_ops.utils import _contains_ph

T = TypeVar("T")
S = TypeVar("S")
C = TypeVar("C", bound=Optional[BaseContext])
P = ParamSpec("P")
R = TypeVar("R")


class Operation(Generic[T, S, C]):

    def __init__(self, graph: OpGraph, spec: OpSpec[S, C]):
        self._g = graph
        self._s = spec
        graph.add_spec(spec)
        self._plan_cache: ExecutionPlan | None = None

    def _spec(self) -> OpSpec:
        return self._s

    def _clone(self, spec: Optional[OpSpec] = None) -> "Operation[T, S, C]":
        """
        Return an immutable clone of *this stage* together with a **copy of
        the whole graph built so far**.  That lets us build new pipelines
        without ever mutating the global `add`, `add_one`, … objects while
        still keeping every edge that was already wired inside the current
        expression.
        """
        g = OpGraph()
        # copy every node & edge that currently exists
        for sp in self._g._nodes.values():
            g.add_spec(sp)
        for edges in self._g._edges_from.values():
            for e in edges:
                g.connect(e.source, e.target)
        return Operation(g, spec or self._s)

    def _bind_template(self, tpl: Template) -> "Operation":
        new_spec = replace(
            self._s,
            id=self._g.new_id(f"{self._s.id}_tpl"),
            bound_template=tpl,
            bound_args=(),  # guarantee mutual exclusion
            bound_kwargs=MappingProxyType({}),
        )
        return self._clone(new_spec)

    def _bind_constants(self, tpl: Template) -> "Operation":
        new_spec = replace(
            self._s,
            id=self._g.new_id(f"{self._s.id}_const"),
            bound_template=None,  # guarantee mutual exclusion
            bound_args=tpl.args,
            bound_kwargs=MappingProxyType(tpl.kwargs),
        )
        return self._clone(new_spec)

    def _invoke_later(self, tpl: Template) -> "Operation":
        invoked = copy.copy(self)
        invoked._call_args = tpl.args  # type: ignore[attr-defined]
        invoked._call_kwargs = tpl.kwargs  # type: ignore[attr-defined]
        invoked._plan_cache = None
        return invoked

    def __rshift__(self, other: "Operation[T, S, C]") -> "Operation[T, S, C]":
        """
        Wire the result of *this* operation into the next one and return
        the tail of the chain so that further “>>” continue left-to-right.
        """
        self = self._clone()
        other = other._clone()
        # ── 1. make sure both operations share the same OpGraph ─────────
        if self._g is not other._g:
            # copy every spec & edge from the “other” graph into ours
            for sp in other._g._nodes.values():
                self._g.add_spec(sp)
            for edges in other._g._edges_from.values():
                for e in edges:
                    self._g.connect(e.source, e.target)
            other._g = self._g  # now both point to the same graph

        # ── 2. find the first input on `other` that can accept our output ─
        chosen_param: str | None = None
        for idx, pname in enumerate(other._s.params):
            if pname == "context":
                continue

            # already wired?
            if self._g._incoming_edges(HandleId(other._s.id, pname)):
                continue

            # already bound (constant or placeholder)?
            if pname in other._s.bound_kwargs:
                if _contains_ph(other._s.bound_kwargs[pname]):
                    chosen_param = pname
                    break
                continue
            if idx < len(other._s.bound_args):
                if _contains_ph(other._s.bound_args[idx]):
                    chosen_param = pname
                    break
                continue

            # free parameter – use it
            chosen_param = pname
            break

        # ── 3. add the edge if we found a home for our result ───────────
        if chosen_param is not None:
            self._g.connect(
                HandleId(self._s.id, "result"),
                HandleId(other._s.id, chosen_param),
            )

        return other

    def __call__(self, *args: Any, **kwargs: Any) -> "Operation[T, S, C]":
        tpl = Template.from_call(args, kwargs)

        # 1) deferred template bind   op(_, y=42)
        if any(a is _ for a in tpl.args) or any(v is _ for v in tpl.kwargs.values()):
            return self._bind_template(tpl)

        # 2) eager constant bind      op(10, y=42)
        if tpl.args or tpl.kwargs:
            return self._bind_constants(tpl)

        # 3) run-time invocation (no early binding)
        return self._invoke_later(tpl)

    def __await__(self) -> Any:
        return self.execute().__await__()

    def validate(self) -> None:
        """Ensure every required parameter in the upstream DAG is satisfied."""
        topo = self._g._upstream(self._s)
        for spec in topo:
            sig = spec.signature
            # collect all non-*args/**kwargs parameters
            param_names = [
                name
                for name, param in sig.parameters.items()
                if param.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]

            # --- skip “root” operations (no incoming edges at all) ---
            has_incoming = False
            for pname in param_names:
                if self._g._incoming_edges(HandleId(spec.id, pname)):
                    has_incoming = True
                    break
            if not has_incoming:
                continue

            # --- enforce that each parameter is satisfied downstream ---
            for idx, pname in enumerate(param_names):
                # already handled by a bound arg/kw?
                if pname in spec.bound_kwargs:
                    continue
                if idx < len(spec.bound_args):
                    continue
                # or by a default value?
                if sig.parameters[pname].default is not inspect.Parameter.empty:
                    continue

                # otherwise we need an incoming edge
                target_h = HandleId(spec.id, pname)
                satisfied = False
                for edges in self._g._edges_from.values():
                    for e in edges:
                        if e.target == target_h:
                            satisfied = True
                            break
                    if satisfied:
                        break

                if not satisfied:
                    raise ValueError(
                        f"Missing connection for required input '{pname}' on '{spec.id}'"
                    )

    async def execute(self, *args: Any, **kwargs: Any):
        """
        Merge any args/kwargs stored by a preceding ``pipeline(*args, **kw)``
        call with the args/kwargs supplied directly to ``execute``.
        """
        stored_args = getattr(self, "_call_args", ())
        stored_kwargs = getattr(self, "_call_kwargs", {})

        merged_args = (*stored_args, *args)
        merged_kwargs = {**stored_kwargs, **kwargs}

        key = (merged_args, tuple(sorted(merged_kwargs.items())))
        if self._plan_cache is None or getattr(self, "_plan_key", None) != key:
            self._plan_cache = self._g.compile(self._s.id, merged_args, merged_kwargs)
            self._plan_key = key
        return await _default_executor.run(self._plan_cache)


@overload
def operation(
    func: Callable[P, R],
    *,
    context: bool = False,
    context_type: Optional[Type[C]] = None,
) -> Operation[Callable[P, R], S, C]: ...


@overload
def operation(
    func: None,
    *,
    context: bool = False,
    context_type: Optional[Type[C]] = None,
) -> Callable[[Callable[P, R]], Operation[Callable[P, R], S, C]]: ...


def operation(
    func: Optional[Callable[..., Any]] = None,
    *,
    context: bool = False,
    context_type: Optional[Type[C]] = None,
) -> Union[
    Operation[Callable[P, R], S, C],
    Callable[[Callable[P, R]], Operation[Callable[P, R], S, C]],
]:
    """Decorator / factory lifting a coroutine into an Operation."""

    def _lift(f: Callable[..., Awaitable[Any]]):
        graph = OpGraph()
        spec = OpSpec(
            id=graph.new_id(f.__name__),
            func=f,
            signature=inspect.signature(f),
            requires_ctx="context" in inspect.signature(f).parameters,
            ctx_type=None,
        )
        return Operation(graph, spec)

    if func is None:
        return _lift
    return _lift(func)


@operation
async def identity(x: Any) -> Any:
    """
    Return the input value unchanged.
    """
    return x


def constant(value: Any) -> Operation[Any, Any, None]:
    """
    Return a constant value.
    """
    return Operation.unit(value)


# Helper functions for safe awaiting
async def safe_await(value: Any) -> Any:
    """Safely await a value, handling both awaitable and non-awaitable values."""
    if inspect.isawaitable(value):
        return await value
    return value
