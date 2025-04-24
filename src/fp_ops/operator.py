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

from fp_ops.graph import OpGraph
from fp_ops.primitives import HandleId, OpSpec
from fp_ops.execution import ExecutionPlan, _default_executor

from fp_ops.placeholder import Placeholder
from fp_ops.context import BaseContext
from fp_ops.utils import _contains_ph

T = TypeVar("T")
S = TypeVar("S")
C = TypeVar("C", bound=Optional[BaseContext])
P = ParamSpec("P")
R = TypeVar("R")


class Operation(Generic[S, C]):

    def __init__(self, graph: OpGraph, spec: OpSpec[S, C]):
        self._g = graph
        self._s = spec
        graph.add_spec(spec)
        self._plan_cache: ExecutionPlan | None = None

    # ------------------------------------------------------------------
    # DSL sugar
    # ------------------------------------------------------------------
    def __rshift__(self, other: "Operation") -> "Operation":
        # merge graphs if different
        if self._g is not other._g:
            for sp in other._g._nodes.values():
                self._g.add_spec(sp)
            for edges in other._g._edges_from.values():
                for e in edges:
                    self._g.connect(e.source, e.target)
            other._g = self._g  # type: ignore

        # choose first *unbound* or *placeholder* param
        chosen_param: str | None = None
        params = other._s.params
        for idx, pname in enumerate(params):
            if pname == "context":
                continue
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
            chosen_param = pname
            break
        if chosen_param is not None:
            self._g.connect(HandleId(self._s.id, "result"), HandleId(other._s.id, chosen_param))
        return other
    # ------------------------------------------------------------------
    # Binding / placeholders
    # ------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> "Operation":
        # ---------- 1. placeholder-driven binding ----------------------
        if any(_contains_ph(a) for a in args) or any(_contains_ph(v) for v in kwargs.values()):
            return self._bind_with_placeholders(args, kwargs)          # <-- helper kept below

        # ---------- 2. constant pre-binding (only if signature accepts)-
        try:
            self._s.signature.bind_partial(*args, **kwargs)
        except TypeError:                                              # too many / invalid args
            can_bind = False
        else:
            can_bind = True

        if can_bind and (args or kwargs):
            return self._bind_constants(args, kwargs)                  # <-- helper kept below

        # ---------- 3. runtime invocation -----------------------------
        if not args and not kwargs:        # nothing to do
            return self

        invoked = copy.copy(self)          # cheap shallow copy; graph/spec are shared
        invoked._call_args: Tuple[Any, ...] = tuple(args)
        invoked._call_kwargs: Dict[str, Any] = dict(kwargs)
        invoked._plan_cache = None
        # we purposely do *not* touch validate()/edges/build-plan logic
        return invoked

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def execute(self, *args: Any, **kwargs: Any):
        """
        Merge any args/kwargs stored by a preceding ``pipeline(*args, **kw)``
        call with the args/kwargs supplied directly to ``execute``.
        """
        stored_args  = getattr(self, "_call_args", ())
        stored_kwargs = getattr(self, "_call_kwargs", {})

        merged_args  = (*stored_args, *args)
        merged_kwargs = {**stored_kwargs, **kwargs}

        key = (merged_args, tuple(sorted(merged_kwargs.items())))
        if self._plan_cache is None or getattr(self, "_plan_key", None) != key:
            self._plan_cache = self._g.compile(self._s.id, merged_args, merged_kwargs)
            self._plan_key = key
        return await _default_executor.run(self._plan_cache)

    def _bind_with_placeholders(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> "Operation":
        sig = self._s.signature
        bound = sig.bind_partial(*args, **kwargs)
        new_kwargs = dict(self._s.bound_kwargs)
        new_args   = list(self._s.bound_args)
        for name, val in bound.arguments.items():
            idx = list(sig.parameters).index(name)
            while len(new_args) <= idx:
                new_args.append(None)
            new_args[idx] = val
            new_kwargs[name] = val
        new_spec = OpSpec(
            id=self._g.new_id(self._s.id + "_bind"),
            func=self._s.func,
            signature=self._s.signature,
            requires_ctx=self._s.requires_ctx,
            ctx_type=self._s.ctx_type,
            bound_args=tuple(new_args),
            bound_kwargs=MappingProxyType(new_kwargs),
        )
        return Operation(self._g, new_spec)

    def _bind_constants(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> "Operation":
        new_spec = OpSpec(
            id=self._g.new_id(self._s.id + "_const"),
            func=self._s.func,
            signature=self._s.signature,
            requires_ctx=self._s.requires_ctx,
            ctx_type=self._s.ctx_type,
            bound_args=args or self._s.bound_args,
            bound_kwargs=MappingProxyType({**self._s.bound_kwargs, **kwargs}),
        )
        return Operation(self._g, new_spec)
    
    def validate(self) -> None:
        """Ensure every required parameter in the upstream DAG is satisfied."""
        topo = self._g._upstream(self._s)
        for spec in topo:
            sig = spec.signature
            # collect all non-*args/**kwargs parameters
            param_names = [
                name for name, param in sig.parameters.items()
                if param.kind not in (
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



    # expose for tests ---------------------------------------------------
    def _spec(self) -> OpSpec:
        return self._s
    
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
