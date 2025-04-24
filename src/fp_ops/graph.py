import itertools
import inspect
from typing import Any, Dict, List, Tuple, MutableMapping, Callable

from fp_ops.primitives import Edge, HandleId, OpSpec
from fp_ops.execution import Step, ExecutionPlan
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
    def compile(
        self,
        head_id: str,
        ext_args: Tuple[Any, ...],
        ext_kwargs: Dict[str, Any],
    ) -> "ExecutionPlan":
        head = self._nodes[head_id]

        # ---- build topo-order upstream DAG ------------------------ #
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

        # ---- plan-building helpers -------------------------------- #
        def handle(spec: OpSpec, name: str = "result") -> HandleId:
            return HandleId(spec.id, name)

        arg_cursor = list(ext_args)
        steps: List[Step] = []

        # ========================================================== #
        # main loop over the topo-ordered specs                      #
        # ========================================================== #
        for spec in order:
            sig = spec.signature
            param_names = [
                p
                for p in sig.parameters
                if sig.parameters[p].kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]

            arg_getters: List[Callable[[MutableMapping[HandleId, Result]], Any]] = []

            for idx, pname in enumerate(param_names):
                tgt_h = HandleId(spec.id, pname)

                # ------------------------------------------------------------------
                # 0. bound *template*  (highest precedence)
                # ------------------------------------------------------------------
                if spec.bound_template is not None:
                    inc = self._incoming_edges(tgt_h)
                    if not inc:
                        raise TypeError(
                            f"{spec.id}: parameter '{pname}' is bound via template "
                            "but has no upstream value"
                        )
                    src_h = inc[-1].source  # last edge wins
                    tpl = spec.bound_template
                    if tpl.is_identity():
                        arg_getters.append(
                            lambda st, sh=src_h: st[sh].default_value(None)
                        )
                    else:
                        arg_getters.append(
                            lambda st, sh=src_h, t=tpl: t.render(
                                st[sh].default_value(None)
                            )[0][0]
                        )
                    continue

                # ------------------------------------------------------------------
                # 1. plain constant bind  (former bound_args / bound_kwargs)
                # ------------------------------------------------------------------
                if pname in spec.bound_kwargs:
                    value = spec.bound_kwargs[pname]
                    arg_getters.append(lambda _st, v=value: v)
                    continue
                if idx < len(spec.bound_args):
                    value = spec.bound_args[idx]
                    arg_getters.append(lambda _st, v=value: v)
                    continue

                # ------------------------------------------------------------------
                # 2. incoming edge
                # ------------------------------------------------------------------
                inc = self._incoming_edges(tgt_h)
                if inc:
                    src_h = inc[-1].source
                    arg_getters.append(lambda st, sh=src_h: st[sh].default_value(None))
                    continue

                # ------------------------------------------------------------------
                # 3. external args / kwargs
                # ------------------------------------------------------------------
                if pname in ext_kwargs:
                    v = ext_kwargs[pname]
                    arg_getters.append(lambda _st, v=v: v)
                    continue
                if arg_cursor:
                    v = arg_cursor.pop(0)
                    arg_getters.append(lambda _st, v=v: v)
                    continue

                # ------------------------------------------------------------------
                # 4. default value in signature
                # ------------------------------------------------------------------
                if sig.parameters[pname].default is not inspect.Parameter.empty:
                    v = sig.parameters[pname].default
                    arg_getters.append(lambda _st, v=v: v)
                    continue

                # ------------------------------------------------------------------
                # 5. nothing satisfied!
                # ------------------------------------------------------------------
                raise TypeError(f"{spec.id}: missing argument '{pname}'")

            # ---- record the step ----------------------------------- #
            steps.append(
                Step(
                    func=spec.func,
                    arg_getters=tuple(arg_getters),
                    result_handle=handle(spec),
                )
            )

        return ExecutionPlan(
            steps=steps,
            final_handle=HandleId(head.id, "result"),
        )
