import itertools
import inspect
from typing import Any, Dict, List, Tuple, MutableMapping, Callable

from fp_ops.primitives import Edge, HandleId, OpSpec
from fp_ops.execution import Step, ExecutionPlan
from fp_ops.utils import _contains_ph, _fill
from expression import Result

class OpGraph:
    """DAG of OpSpecs wired by Edges."""

    _id_counter = itertools.count().__next__

    def __init__(self):
        self._nodes: Dict[str, OpSpec] = {}
        self._edges_from: Dict[HandleId, List[Edge]] = {}

    # ---------- mutation helpers -------------------------------------
    def add_spec(self, spec: OpSpec) -> None:
        self._nodes[spec.id] = spec

    def connect(self, src: HandleId, tgt: HandleId) -> None:
        self._edges_from.setdefault(src, []).append(Edge(src, tgt))

    def new_id(self, base: str) -> str:
        return f"{base}_{OpGraph._id_counter()}"

    # ---------- graph traversal --------------------------------------
    def _upstream(self, start: OpSpec) -> List[OpSpec]:
        visited: set[str] = set()
        order: List[OpSpec] = []

        def dfs(spec: OpSpec):
            if spec.id in visited:
                return
            visited.add(spec.id)
            for handle_name in spec.params:
                src_handle = HandleId(spec.id, handle_name)
                # incoming edge? -> follow upstream
                for e in self._incoming_edges(src_handle):
                    dfs(self._nodes[e.source.node_id])
            order.append(spec)

        dfs(start)
        return order  # already in topological order (upstream first)

    def _incoming_edges(self, target: HandleId) -> List[Edge]:
        res: List[Edge] = []
        for edges in self._edges_from.values():
            for e in edges:
                if e.target == target:
                    res.append(e)
        return res
    # ---------- compiler ---------------------------------------------
    def compile(self, head_id: str, ext_args: Tuple[Any, ...], ext_kwargs: Dict[str, Any]) -> "ExecutionPlan":
        head = self._nodes[head_id]

        # Build topo-order upstream ------------------------------------------------
        visited: set[str] = set()
        order: List[OpSpec] = []

        def dfs(spec: OpSpec):
            if spec.id in visited:
                return
            visited.add(spec.id)
            for pname in spec.params:
                for e in self._incoming_edges(HandleId(spec.id, pname)):
                    dfs(self._nodes[e.source.node_id])
            order.append(spec)

        dfs(head)

        # Key helper --------------------------------------------------------------
        def handle(spec: OpSpec, name: str = "result") -> HandleId:
            return HandleId(spec.id, name)

        arg_cursor = list(ext_args)
        steps: List[Step] = []

        for spec in order:
            sig = spec.signature
            param_names = [p for p in sig.parameters if sig.parameters[p].kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )]
            arg_getters: List[Callable[[MutableMapping[HandleId, Result]], Any]] = []

            for idx, pname in enumerate(param_names):
                tgt_h = HandleId(spec.id, pname)

                # bound value (possibly with placeholders) -----------------
                template_val: Any | None = None
                if pname in spec.bound_kwargs:
                    template_val = spec.bound_kwargs[pname]
                elif idx < len(spec.bound_args):
                    template_val = spec.bound_args[idx]

                has_ph = template_val is not None and _contains_ph(template_val)
                inc_edges = self._incoming_edges(tgt_h)

                # ---- 1. plain constant bound (highest priority) ----------
                if template_val is not None and not has_ph:
                    arg_getters.append(lambda _st, v=template_val: v)
                    continue

                # ---- 2. incoming edge ------------------------------------
                if inc_edges:
                    src_h = inc_edges[-1].source
                    if has_ph:
                        arg_getters.append(
                            lambda st, sh=src_h, tpl=template_val: _fill(tpl, st[sh].default_value(None))
                        )
                    else:
                        arg_getters.append(lambda st, sh=src_h: st[sh].default_value(None))
                    continue

                # ---- 3. external args / kwargs ---------------------------
                if pname in ext_kwargs:
                    ext_v = ext_kwargs[pname]
                    if has_ph:
                        arg_getters.append(lambda _st, v=ext_v, tpl=template_val: _fill(tpl, v))
                    else:
                        arg_getters.append(lambda _st, v=ext_v: v)
                    continue
                if arg_cursor:
                    ext_v = arg_cursor.pop(0)
                    if has_ph:
                        arg_getters.append(lambda _st, v=ext_v, tpl=template_val: _fill(tpl, v))
                    else:
                        arg_getters.append(lambda _st, v=ext_v: v)
                    continue

                # ---- 4. default value ------------------------------------
                if sig.parameters[pname].default is not inspect.Parameter.empty:
                    default_v = sig.parameters[pname].default
                    arg_getters.append(lambda _st, v=default_v: v)
                    continue

                raise TypeError(f"{spec.id}: missing argument '{pname}'")


            steps.append(Step(func=spec.func, arg_getters=tuple(arg_getters), result_handle=handle(spec)))

        return ExecutionPlan(steps=steps, final_handle=HandleId(head.id, "result"))
