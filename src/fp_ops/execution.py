import collections
import inspect
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Tuple, Any, Optional, TypeVar, cast

from .graph import OpGraph, OpSpec
from fp_ops.context import BaseContext
from fp_ops.primitives import Template, Placeholder
from expression import Result, Ok, Error

T = TypeVar('T')

@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """
    Immutable, topologically-sorted description of the graph ready for the executor.

    Attributes:
        order (Tuple[OpSpec, ...]): OpSpecs in evaluation order.
        arg_render (Mapping[str, Callable[[object, object | None], Tuple[Tuple, Dict]]]):
            Mapping from node_id to a callable (prev_value, ctx) -> (args, kwargs).
            The callable embodies the Template logic so the executor does not need to know about placeholders.
        successors (Mapping[str, Tuple[str, ...]]):
            Mapping from node_id to a tuple of node_ids that depend on it.
            Useful when introducing parallel execution.
    """
    order: Tuple[OpSpec, ...]
    arg_render: Mapping[str, Callable[[object, object | None], Tuple[Tuple, Dict]]]
    successors: Mapping[str, Tuple[str, ...]] = field(repr=False)

    @classmethod
    def from_graph(cls, graph: OpGraph) -> "ExecutionPlan":
        order = graph.topological_order()

        renderers: Dict[str, Callable[[object, Optional[object]], Tuple[Tuple, Dict]]] = {}

        for spec in order:
            tpl = spec.template

            if tpl.has_placeholders() and graph.incoming(spec.id):
                # ── internal nodes get the placeholder renderer ──────────
                def make_renderer(template: Template) -> Callable[[object, Optional[object]], Tuple[Tuple, Dict]]:
                    def renderer(val: object, _ctx: Optional[object]) -> Tuple[Tuple, Dict]:
                        return cast(Tuple[Tuple, Dict], template.render(val))
                    return renderer
                renderers[spec.id] = make_renderer(tpl)
            elif tpl.has_placeholders():
                # ── HEAD nodes keep raw template so placeholders survive ─
                # But we need to handle placeholder rendering in the executor
                def make_head_renderer(template: Template) -> Callable[[object, Optional[object]], Tuple[Tuple, Dict]]:
                    def renderer(_v: object, _c: Optional[object]) -> Tuple[Tuple, Dict]:
                        return (tuple(template.args), dict(template.kwargs))
                    return renderer
                renderers[spec.id] = make_head_renderer(tpl)

            elif not tpl.args and not tpl.kwargs:
                params = [
                    p
                    for p in spec.signature.parameters.values()
                    if p.name not in ("self", "context")
                ]

                # plain unary func (first param is regular) 
                if params and params[0].kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    def make_unary_renderer() -> Callable[[object, Optional[object]], Tuple[Tuple, Dict]]:
                        def renderer(v: object, _c: Optional[object]) -> Tuple[Tuple, Dict]:
                            return ((v,), {}) if v is not None else ((), {})
                        return renderer
                    renderers[spec.id] = make_unary_renderer()
                # leading *args and no regular params (def f(*args, **kw))
                elif params and params[0].kind is inspect.Parameter.VAR_POSITIONAL:
                    def make_var_positional_renderer() -> Callable[[object, Optional[object]], Tuple[Tuple, Dict]]:
                        def renderer(v: object, _c: Optional[object]) -> Tuple[Tuple, Dict]:
                            return ((v,) if v is not None else (), {})
                        return renderer
                    renderers[spec.id] = make_var_positional_renderer()

                #  fallback: inject into the first named parameter 
                else:
                    first_name = params[0].name if params else None
                    def make_named_param_renderer(fn: Optional[str]) -> Callable[[object, Optional[object]], Tuple[Tuple, Dict]]:
                        def renderer(v: object, _c: Optional[object]) -> Tuple[Tuple, Dict]:
                            return ((), {fn: v} if fn else {})
                        return renderer
                    renderers[spec.id] = make_named_param_renderer(first_name)

            else:
                const_args = tuple(tpl.args)
                const_kwargs = dict(tpl.kwargs)
                def make_const_renderer(ca: Tuple, ck: Dict) -> Callable[[object, Optional[object]], Tuple[Tuple, Dict]]:
                    def renderer(_v: object, _c: Optional[object]) -> Tuple[Tuple, Dict]:
                        return (ca, ck)
                    return renderer
                renderers[spec.id] = make_const_renderer(const_args, const_kwargs)

        scc: Dict[str, List[str]] = collections.defaultdict(list)
        for node_id, edges in graph._out_edges.items():
            scc[node_id] = [e.target.node_id for e in edges]

        return cls(order=order, arg_render=renderers, successors={k: tuple(v) for k, v in scc.items()})
    

def _merge_first_call(
    signature: inspect.Signature,
    base_args: Tuple,
    base_kwargs: Dict[str, Any],
    rt_args: Tuple,
    rt_kwargs: Dict[str, Any],
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Merge the arguments that are already baked into the node
    (*base_args/base_kwargs*) with the runtime arguments supplied by the
    caller (*rt_args/rt_kwargs*).

    If the wrapped function accepts a var-positional parameter (``*args``)
    we **must not** try to map the caller's positional arguments onto a
    named parameter — we simply forward them as real positionals.

    Precedence for *named* parameters (no ``*args``):
        1. runtime keyword
        2. runtime positional (in order)
        3. pre-bound constant
    """
    
    # Fast path ── the callee exposes *args → keep all positional args
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL
           for p in signature.parameters.values()):
        merged_args = (*base_args, *rt_args)
        merged_kwargs = {**base_kwargs, **rt_kwargs}  # runtime kw override
        return merged_args, merged_kwargs

    # fast path – nothing supplied at call-time
    if not rt_args and not rt_kwargs:
        return base_args, dict(base_kwargs)

    param_names = [n for n in signature.parameters if n not in ("self",)]

    # always return **kwargs only – that guarantees we never send the same
    # value twice (once positionally *and* once by name)
    final: Dict[str, Any] = {}

    base_pos = iter(base_args)
    rt_pos = iter(rt_args)

    for name in param_names:
        if name in rt_kwargs:          # runtime kwarg top priority
            final[name] = rt_kwargs[name]
            continue

        try:                           # then runtime positional …
            final[name] = next(rt_pos)
            continue
        except StopIteration:
            pass

        if name in base_kwargs:        # then template kwarg …
            final[name] = base_kwargs[name]
            continue

        try:                           # finally template positional
            final[name] = next(base_pos)
        except StopIteration:
            # let Python supply its own default (if any)
            pass

    # any left-over runtime positionals → too many arguments
    remaining_rt_pos = list(rt_pos)
    if remaining_rt_pos:
        # Get the actual number of positional args expected
        pos_params = [p for p in signature.parameters.values() 
                      if p.kind in (inspect.Parameter.POSITIONAL_ONLY, 
                                    inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        raise TypeError(
            f"Too many positional arguments: got {len(rt_args)} (rt_args={rt_args!r}), "
            f"expected at most {len(pos_params)} for parameters {[p.name for p in pos_params]!r}. "
            f"Signature: {signature!r}, "
            f"base_args={base_args!r}, base_kwargs={base_kwargs!r}, "
            f"rt_args={rt_args!r}, rt_kwargs={rt_kwargs!r}"
        )
        
    for k, v in rt_kwargs.items():
        if k not in final:
            final[k] = v

    return (), final


def _has_nested_placeholder(obj: Any) -> bool:
    """Check if an object contains any Placeholder instances (including nested)."""
    from fp_ops.primitives import Placeholder
    
    if isinstance(obj, Placeholder):
        return True
    if isinstance(obj, dict):
        return any(_has_nested_placeholder(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_nested_placeholder(v) for v in obj)
    return False


class Executor:
    """
    Runs a pre-compiled `ExecutionPlan`.  It executes nodes strictly
    in topo-order and propagates only the *result* value downstream (matching
    the single-running-value assumption).
    """

    def __init__(self, plan: ExecutionPlan):
        self._plan = plan

    async def run(
        self,
        *first_args: Any,
        _context: BaseContext | None = None,
        **first_kwargs: Any,
    ) -> Result[Any, Exception]:
        """
        *first_args / first_kwargs* feed the *first* node.
        For every subsequent node we use the renderer stored in the plan.
        """
        id2value: Dict[str, Any] = {}
        ctx = _context
        last_result: Any = None

        for idx, spec in enumerate(self._plan.order):
            # Build call-args
            if idx == 0:
                # Start from template (pre-bound constants) then
                # overlay runtime arguments, avoiding duplicates
                args, kwargs = self._plan.arg_render[spec.id](None, None)
                
                # Special handling for single-step operations with placeholders
                # If template has placeholders and runtime args are provided,
                # check if we should use runtime args to render placeholders
                if (spec.template.has_placeholders() and first_args and 
                    len(self._plan.order) == 1):  # Single-step operation
                    # Check if any kwargs contain nested placeholders
                    has_nested = any(_has_nested_placeholder(v) for v in kwargs.values())
                    if has_nested:
                        # Use the first runtime arg to render placeholders
                        rendered_args, rendered_kwargs = spec.template.render(first_args[0])
                        # Update kwargs with rendered values
                        for k, v in rendered_kwargs.items():
                            if k in kwargs and _has_nested_placeholder(kwargs[k]):
                                kwargs[k] = v
                
                args, kwargs = _merge_first_call(
                    spec.signature, args, kwargs, first_args, first_kwargs
                )
            else:
                args, kwargs = self._plan.arg_render[spec.id](last_result, None)

            if spec.require_ctx:
                # pick whichever source (caller kwarg *or* propagated) is present
                cur_ctx = kwargs.get("context", ctx)

                # presence check
                if cur_ctx is None:
                    return Error(RuntimeError(f"{spec.func.__name__} requires a context"))

                # accept dict / other BaseContext and try to build the right class
                if spec.ctx_type and not isinstance(cur_ctx, spec.ctx_type):
                    if isinstance(cur_ctx, dict):
                        try:
                            cur_ctx = spec.ctx_type(**cur_ctx)
                        except Exception as exc:
                            return Error(RuntimeError(f"Invalid context: {exc}"))
                    else:
                        return Error(
                            RuntimeError(
                                f"Invalid context: Could not convert "
                                f"{type(cur_ctx).__name__} to {spec.ctx_type.__name__}"
                            )
                        )

                kwargs["context"] = cur_ctx

                # if the operation only *needs* the context, drop the
                # pipeline's running value so we don't send a spurious arg
                pos_ok = [
                    p for p in spec.signature.parameters.values()
                    if p.name not in ("self", "context")
                       and p.kind in (
                           inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.VAR_POSITIONAL,
                       )
                ]
                if not pos_ok:
                    args = ()
            # ------------------------------------------------------------------

            # Call the function (sync or async transparently)
            try:
                raw = await spec.func(*args, **kwargs)
            except Exception as exc:
                return Error(exc)

            result = raw if isinstance(raw, Result) else Ok(raw)
            if result.is_error():
                return result

            value = result.default_value(None)
            
            # ---------- propagate updated context -----------------------------
            if spec.require_ctx and isinstance(value, BaseContext):
                ctx = value
            # ------------------------------------------------------------------
            
            id2value[spec.id] = value
            last_result = value

        return result