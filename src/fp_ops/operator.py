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
    Sequence,
)
import inspect
import asyncio
import copy
from types import MappingProxyType
from functools import wraps
from dataclasses import replace

from fp_ops.graph import OpGraph
from fp_ops.primitives import HandleId, OpSpec
from fp_ops.execution import ExecutionPlan, _default_executor

from fp_ops.placeholder import _, Template
from fp_ops.context import BaseContext
from fp_ops.utils import _contains_ph

from expression import Result

T = TypeVar("T")
S = TypeVar("S")
C = TypeVar("C", bound=Optional[BaseContext])
P = ParamSpec("P")
R = TypeVar("R")

def _lift_function(fn: Callable[..., Any]) -> "Operation":
    """
    Take an arbitrary (sync-or-async) callable and turn it into an Operation node
    that accepts exactly the *result* of the previous stage as its first arg.
    """
    is_async = inspect.iscoroutinefunction(fn)

    @wraps(fn)
    async def _wrapped(x, *_, **__):
        try:
            if is_async:
                return await fn(x)
            return await asyncio.to_thread(fn, x)
        except Exception as e:            # always wrap in Result to stay monadic
            return Result.Error(e)

    return operation(_wrapped)           # use the public decorator

def _merge_graphs(left: "Operation", right: "Operation") -> None:
    """
    Re-attach every node & edge from *right* into *left*'s graph, then make sure
    both Operation instances point at the *same* OpGraph object.
    """
    if left._g is right._g:
        return

    for sp in right._g._nodes.values():
        left._g.add_spec(sp)
    for edges in right._g._edges_from.values():
        for e in edges:
            left._g.connect(e.source, e.target)
    right._g = left._g                # important: share the same graph object

class Operation(Generic[T, S, C]):

    def __init__(self, graph: OpGraph, spec: OpSpec[S, C], context_type: Optional[Type[C]] = None):
        self._g = graph
        self._s = spec
        self._context_type = context_type # deprecated
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
            bound_args=(),
            bound_kwargs=MappingProxyType({}),
        )
        return self._clone(new_spec)

    def _bind_constants(self, tpl: Template) -> "Operation":
        new_spec = replace(
            self._s,
            id=self._g.new_id(f"{self._s.id}_const"),
            bound_template=None,
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
        Connect the *result* of this Operation to an appropriate input of
        the next Operation and return the *tail* (`other`) so that further
        chaining continues left-to-right.
        """
        self  = self._clone()
        other = other._clone()

        if self._g is not other._g:
            for sp in other._g._nodes.values():
                self._g.add_spec(sp)
            for edges in other._g._edges_from.values():
                for e in edges:
                    self._g.connect(e.source, e.target)
            other._g = self._g

        chosen_param: str | None = None
        tpl = other._s.bound_template

        for idx, pname in enumerate(other._s.params):
            if pname == "context":
                continue

            if self._g._incoming_edges(HandleId(other._s.id, pname)):
                continue

            if tpl is not None:
                tpl_val: Any | None = None
                if pname in tpl.kwargs:
                    tpl_val = tpl.kwargs[pname]
                elif idx < len(tpl.args):
                    tpl_val = tpl.args[idx]

                if tpl_val is _:
                    chosen_param = pname
                    break
                if tpl_val is not None:
                    continue

            if (
                pname in other._s.bound_kwargs
                and _contains_ph(other._s.bound_kwargs[pname])
            ):
                chosen_param = pname
                break
            if (
                idx < len(other._s.bound_args)
                and _contains_ph(other._s.bound_args[idx])
            ):
                chosen_param = pname
                break

            chosen_param = pname
            break

        if chosen_param is not None:
            self._g.connect(
                HandleId(self._s.id, "result"),
                HandleId(other._s.id, chosen_param),
            )

        return other

    def __and__(self, other: "Operation") -> "Operation":
        """Run *both* branches and return a `(left, right)` tuple."""
        self  = self._clone()
        other = other._clone()
        _merge_graphs(self, other)

        async def _join(a, b):
            return (a, b)

        join_spec = OpSpec(
            id=self._g.new_id("tuple"),
            func=_join,
            signature=inspect.signature(_join),
            requires_ctx=False,
            ctx_type=None,
        )
        self._g.add_spec(join_spec)

        # wire the two results into the new node
        self._g.connect(HandleId(self._s.id,   "result"),
                        HandleId(join_spec.id, "a"))
        self._g.connect(HandleId(other._s.id,  "result"),
                        HandleId(join_spec.id, "b"))

        return Operation(self._g, join_spec)

    def __or__(self, other: "Operation") -> "Operation":
        """
        Try the *left* branch; if it returns a `Result.Error`, produce the right.
        (Both branches are evaluated **sequentially**, mirroring the old API.)
        """
        self  = self._clone()
        other = other._clone()
        _merge_graphs(self, other)

        async def _either(a: Result, b_thunk):
            if a.is_ok():
                return a               # already a Result
            return await b_thunk()     # lazily evaluate right branch

        # Wrap the right branch inside a zero-arg thunk so we can
        # postpone its execution until the left fails.
        # TODO: Why do we call .execute() here?
        async def _thunk_wrapper():
            return await other.execute()

        either_spec = OpSpec(
            id=self._g.new_id("either"),
            func=_either,
            signature=inspect.signature(_either),
            requires_ctx=False,
            ctx_type=None,
        )
        self._g.add_spec(either_spec)

        self._g.connect(HandleId(self._s.id,   "result"),
                        HandleId(either_spec.id, "a"))
        # pass the *callable* (thunk) as a constant
        either_spec = replace(
            either_spec,
            bound_kwargs=MappingProxyType({"b_thunk": _thunk_wrapper}),
        )

        return Operation(self._g, either_spec)
    
    def __call__(self, *args: Any, **kwargs: Any) -> "Operation[T, S, C]":
        tpl = Template.from_call(args, kwargs)

        if self._s.bound_template or self._s.bound_args or self._s.bound_kwargs:
            return self._invoke_later(tpl)

        if any(a is _ for a in tpl.args) or any(v is _ for v in tpl.kwargs.values()):
            return self._bind_template(tpl)

        sig_params = [
            p for p in self._s.signature.parameters
            if p not in ("self", "context")
        ]
        arg_count = len(tpl.args) + len(tpl.kwargs)

        can_be_binding = (
            arg_count > 0
            and arg_count <= len(sig_params)
            and all(k in sig_params for k in tpl.kwargs)
        )

        if can_be_binding:
            return self._bind_constants(tpl)

        return self._invoke_later(tpl)

    def __await__(self) -> Any:
        return self.execute().__await__()

    def validate(self) -> None:
        """
        Walk the upstream DAG and raise if any *required* parameter is
        neither:

        • satisfied by a constant in `bound_args` / `bound_kwargs`
        • supplied by a **Template** entry that is a constant (≠ Placeholder)
        • wired from an upstream edge
        • given a default value in the function signature
        """
        from fp_ops.placeholder import Placeholder

        topo = self._g._upstream(self._s)

        for spec in topo:
            sig = spec.signature
            params = [
                name
                for name, p in sig.parameters.items()
                if p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]

            if not any(
                self._g._incoming_edges(HandleId(spec.id, p)) for p in params
            ):
                continue

            tpl = spec.bound_template

            for idx, pname in enumerate(params):
                if pname in spec.bound_kwargs or idx < len(spec.bound_args):
                    continue

                if tpl is not None:
                    if pname in tpl.kwargs:
                        if not isinstance(tpl.kwargs[pname], Placeholder):
                            continue
                    elif idx < len(tpl.args):
                        if not isinstance(tpl.args[idx], Placeholder):
                            continue

                if sig.parameters[pname].default is not inspect.Parameter.empty:
                    continue

                target = HandleId(spec.id, pname)
                if any(e.target == target for edges in self._g._edges_from.values() for e in edges):
                    continue

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

    # TODO: Why do we call .execute() here?
    def bind(self, binder: Callable[[S], "Operation | Result | Awaitable | R"]) -> "Operation":
        """
        `binder` may return:
            • Operation        → chained in the DAG
            • Result / Awaitable / plain value → lifted automatically
        """
        async def _bind(x):
            res = binder(x)
            if isinstance(res, Operation):
                # Execute immediately so we still end up with a single value
                return await res.execute()
            if isinstance(res, Result):
                return res
            if inspect.isawaitable(res):
                return await res
            return res

        return self >> _lift_function(_bind)
    
    def map(self, fn: Callable[[S], R]) -> "Operation":
        """Pure transformation of the previous value."""
        return self >> _lift_function(fn)
    
    def filter(self, pred: Callable[[S], bool],
                     msg : str = "Value did not satisfy predicate") -> "Operation":
        async def _flt(x):
            ok = await pred(x) if inspect.iscoroutinefunction(pred) else await asyncio.to_thread(pred, x)
            return x if ok else Result.Error(ValueError(msg))
        return self >> _lift_function(_flt)
    
    def catch(self, handler: Callable[[Exception], S]) -> "Operation":
        async def _handle(res: Result):
            if res.is_ok():
                return res             # already good
            exc = res.error
            try:
                val = handler(exc)
                if inspect.isawaitable(val):
                    val = await val
                return val
            except Exception as e:
                return Result.Error(e)
        # `self` yields a Result object → map /unwrap/ with _handle
        return self.map(lambda r: r).bind(_handle)
    
    def retry(self, attempts: int = 3, delay: float = 0.1) -> "Operation":
        async def _retry(x):
            last: Exception | None = None
            for _ in range(attempts):
                res = await self.execute(x)   # run the inner op
                if res.is_ok():
                    return res
                last = res.error
                await asyncio.sleep(delay)
            return Result.Error(last or Exception("retry exhausted"))
        return _lift_function(_retry)
    
    def tap(self, side: Callable[[S], Any]) -> "Operation":
        async def _tap(x):
            try:
                await safe_await(side(x))
            finally:
                return x
        return self >> _lift_function(_tap)
    
    @staticmethod
    def unit(value: T) -> "Operation[Any, T, None]":
        async def _const(*_a, **_kw):        # noqa: D401
            return value
        return operation(_const)
    
    def default_value(self, default: S) -> "Operation":
        return self.catch(lambda _e: default)
    
    @classmethod
    def sequence(cls, ops: Sequence["Operation"]) -> "Operation[Any, List[Any], None]":
        async def _seq(*args, **kw):
            out = []
            for op in ops:
                out.append(await op.execute(*args, **kw))
            return out
        return operation(_seq)


    @classmethod
    def combine(cls, **named: "Operation") -> "Operation[Any, Dict[str, Any], None]":
        async def _cmb(*args, **kw):
            res = {}
            for k, op in named.items():
                res[k] = await op.execute(*args, **kw)
            return res
        return operation(_cmb)

    

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


async def safe_await(value: Any) -> Any:
    """Safely await a value, handling both awaitable and non-awaitable values."""
    if inspect.isawaitable(value):
        return await value
    return value
