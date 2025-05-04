import itertools
import inspect
import uuid
import collections
from typing import Any, Dict, List, Tuple, MutableMapping, Callable

from fp_ops.primitives import Edge, Port, OpSpec
from fp_ops.execution import Step, ExecutionPlan
from fp_ops.placeholder import _
from expression import Result


class OpGraph:
    """DAG of OpSpecs wired by Edges."""

    _id_counter = itertools.count().__next__

    def __init__(self):
        self._nodes: Dict[str, OpSpec] = {}
        self._graph_id = uuid.uuid4().hex[:8]
        self._out_edges: Dict[str, List[Edge]] = collections.defaultdict(list)
        self._in_edges:  Dict[str, List[Edge]] = collections.defaultdict(list)

    def add_spec(self, spec: OpSpec) -> None:
        self._nodes[spec.id] = spec

    def connect(self, src: Port, tgt: Port) -> None:
        e = Edge(src, tgt)
        self._out_edges[src.node_id].append(e)
        self._in_edges[tgt.node_id].append(e)

    def new_id(self, base: str) -> str:
        return f"{base}:{self._graph_id}:{OpGraph._id_counter()}"

    def _upstream(self, start: OpSpec) -> List[OpSpec]:
        visited: set[str] = set()
        order: List[OpSpec] = []

        def dfs(spec: OpSpec):
            if spec.id in visited:
                return
            visited.add(spec.id)
            for handle_name in spec.params:
                src_handle = Port(spec.id, handle_name)
                for e in self._incoming_edges(src_handle):
                    dfs(self._nodes[e.source.node_id])
            order.append(spec)

        dfs(start)
        return order

    def _incoming_edges(self, target: Port) -> List[Edge]:
        res: List[Edge] = []
        return [e for e in self._in_edges.get(target.node_id, ())
                 if e.target == target]
    def merge(self, other: "OpGraph") -> None:
        """
        Re-attach every node & edge from *other* into *this* graph.
        """
        if self is other:
            return

        for sp in other._nodes.values():
            self.add_spec(sp)
        for edges in other._out_edges.values():
            for e in edges:
                self.connect(e.source, e.target)

    def compile(
        self,
        head_id: str,
        ext_args: Tuple[Any, ...],
        ext_kwargs: Dict[str, Any],
    ) -> "ExecutionPlan":
        head = self._nodes[head_id]

        order: List[OpSpec] = []
        visited: set[str] = set()

        

        def dfs(spec: OpSpec) -> None:
            if spec.id in visited:
                return
            visited.add(spec.id)
            for pname in spec.params:
                for e in self._incoming_edges(Port(spec.id, pname)):
                    dfs(self._nodes[e.source.node_id])
            order.append(spec)

        dfs(head)

        def handle(spec: OpSpec, name: str = "result") -> Port:
            return Port(spec.id, name)

        arg_cursor = list(ext_args)
        steps: List[Step] = []

        for spec in order:
            sig = spec.signature
            param_list = [
                p for p in sig.parameters
                if sig.parameters[p].kind
                not in (inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD)
            ]

            tpl = spec.bound_template
            arg_getters: List[Callable[[MutableMapping[Port, Result]], Any]] = []

            for idx, pname in enumerate(param_list):
                target_h = Port(spec.id, pname)

                if tpl is not None:
                    if pname in tpl.kwargs:
                        tpl_val = tpl.kwargs[pname]
                    elif idx < len(tpl.args):
                        tpl_val = tpl.args[idx]
                    else:
                        tpl_val = None

                    if tpl_val is not None:
                        if tpl_val is not _:
                            arg_getters.append(lambda _st, v=tpl_val: v)
                            continue

                        incoming = self._incoming_edges(target_h)
                        if incoming:
                            src_h = incoming[-1].source
                            arg_getters.append(
                                lambda st, sh=src_h: st[sh].default_value(None)
                            )
                            continue

                        elif pname in ext_kwargs:
                            v = ext_kwargs[pname]
                            arg_getters.append(lambda _st, v=v: v)
                            continue

                        elif arg_cursor:
                            v = arg_cursor.pop(0)
                            arg_getters.append(lambda _st, v=v: v)
                            continue

                        elif sig.parameters[pname].default is not inspect.Parameter.empty:
                            v = sig.parameters[pname].default
                            arg_getters.append(lambda _st, v=v: v)
                            continue

                        raise TypeError(
                            f"{spec.id}: placeholder for '{pname}' has no value"
                        )

                if pname in spec.bound_kwargs:
                    v = spec.bound_kwargs[pname]
                    arg_getters.append(lambda _st, v=v: v)
                    continue

                if idx < len(spec.bound_args):
                    v = spec.bound_args[idx]
                    arg_getters.append(lambda _st, v=v: v)
                    continue

                incoming = self._incoming_edges(target_h)
                if incoming:
                    src_h = incoming[-1].source
                    arg_getters.append(lambda st, sh=src_h: st[sh].default_value(None))
                    continue

                if pname in ext_kwargs:
                    v = ext_kwargs[pname]
                    arg_getters.append(lambda _st, v=v: v)
                    continue

                if arg_cursor:
                    v = arg_cursor.pop(0)
                    arg_getters.append(lambda _st, v=v: v)
                    continue

                if sig.parameters[pname].default is not inspect.Parameter.empty:
                    v = sig.parameters[pname].default
                    arg_getters.append(lambda _st, v=v: v)
                    continue

                raise TypeError(f"{spec.id}: missing argument '{pname}'")

            steps.append(
                Step(
                    func=spec.func,
                    arg_getters=tuple(arg_getters),
                    result_handle=handle(spec),
                )
            )

        return ExecutionPlan(
            steps=steps,
            final_handle=Port(head.id, "result"),
        )
