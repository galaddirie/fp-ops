from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from .primitives import Edge, OpSpec, Port, PortType

class OpGraph:
    """
    A directed acyclic multi-graph whose vertices are `OpSpec`s and whose
    edges are the data paths (result / error / ctx) between them.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, OpSpec] = {}
        self._out_edges: Dict[str, List[Edge]] = collections.defaultdict(list)
        self._in_edges: Dict[str, List[Edge]] = collections.defaultdict(list)

    def add_node(self, spec: OpSpec) -> None:
        if spec.id in self._nodes:
            raise ValueError(f"Node {spec.id} already exists in graph.")
        self._nodes[spec.id] = spec

    def add_edge(self, edge: Edge) -> None:
        if edge.source.node_id not in self._nodes or edge.target.node_id not in self._nodes:
            raise ValueError("Both edge endpoints must be added to the graph first.")

        self._out_edges[edge.source.node_id].append(edge)
        self._in_edges[edge.target.node_id].append(edge)

    @property
    def nodes(self) -> Tuple[OpSpec, ...]:
        return tuple(self._nodes.values())

    def outgoing(self, node_id: str) -> Tuple[Edge, ...]:
        return tuple(self._out_edges.get(node_id, ()))

    def incoming(self, node_id: str) -> Tuple[Edge, ...]:
        return tuple(self._in_edges.get(node_id, ()))

    def topological_order(self) -> Tuple[OpSpec, ...]:
        in_degree = {nid: len(edges) for nid, edges in self._in_edges.items()}
        for nid in self._nodes:
            in_degree.setdefault(nid, 0)

        queue = collections.deque([nid for nid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)

            for e in self._out_edges.get(nid, ()):
                tgt = e.target.node_id
                in_degree[tgt] -= 1
                if in_degree[tgt] == 0:
                    queue.append(tgt)

        if len(order) != len(self._nodes):
            raise ValueError("Cycle detected in operation graph.")

        return tuple(self._nodes[nid] for nid in order)

    def head_ids(self) -> Tuple[str, ...]:
        return tuple(n.id for n in self.nodes if not self._in_edges.get(n.id))

    def tail_ids(self) -> Tuple[str, ...]:
        return tuple(n.id for n in self.nodes if not self._out_edges.get(n.id))

    def merged_with(self, other: "OpGraph") -> "OpGraph":
        g = OpGraph()
        for spec in (*self.nodes, *other.nodes):
            g.add_node(spec)
        for e in (*self._all_edges(), *other._all_edges()):
            g.add_edge(e)
        for tail in self.tail_ids():
            for head in other.head_ids():
                g.add_edge(
                    Edge(
                        source=Port(node_id=tail, port_type=PortType.SOURCE, name="result"),
                        target=Port(node_id=head, port_type=PortType.TARGET, name=None),
                    )
                )
        return g

    def _all_edges(self) -> Iterable[Edge]:
        for lst in self._out_edges.values():
            yield from lst