"""Closed immutable AST interpreted only by :mod:`fp_ops._runner`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from fp_ops._invoke import Invocation


class Node:
    """Marker base class for interpreter nodes."""


@dataclass(frozen=True, slots=True)
class IdentityNode(Node):
    pass


@dataclass(frozen=True, slots=True)
class ConstantNode(Node):
    value: Any


@dataclass(frozen=True, slots=True)
class FailNode(Node):
    error: Exception


@dataclass(frozen=True, slots=True)
class CallNode(Node):
    invocation: Invocation


@dataclass(frozen=True, slots=True)
class SequenceNode(Node):
    stages: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class FlatMapNode(Node):
    source: Node
    callback: Callable[..., Any]
    declared: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class RecoverNode(Node):
    source: Node
    handler: Callable[..., Any]
    with_operation: bool
    matches: Callable[[Exception], bool] | None = None
    declared: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class AlternativeNode(Node):
    alternatives: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class TapNode(Node):
    source: Node
    effect: Callable[..., Any]
    best_effort: bool


@dataclass(frozen=True, slots=True)
class RetryNode(Node):
    source: Node
    attempts: int
    backoff: float | Callable[..., Any]
    matches: Callable[[Exception], bool] | None = None


@dataclass(frozen=True, slots=True)
class BranchNode(Node):
    condition: Node
    then: Node
    otherwise: Node


@dataclass(frozen=True, slots=True)
class PassthroughNode(Node):
    effect: Node


@dataclass(frozen=True, slots=True)
class LoopNode(Node):
    step: Node
    max_steps: int | None


@dataclass(frozen=True, slots=True)
class BracketNode(Node):
    acquire: Node
    use: Node
    release: Node


@dataclass(frozen=True, slots=True)
class ProvideNode(Node):
    key: Any
    value: Node
    body: Node


@dataclass(frozen=True, slots=True)
class ApplyNode(Node):
    """Evaluate derived operands on the input, then call with ``(input, values)``."""

    operands: tuple[Node, ...]
    invocation: Invocation


@dataclass(frozen=True, slots=True)
class FanoutNode(Node):
    branches: tuple[tuple[str | None, Node], ...]
    mode: Literal["fail_fast", "all_settled"]
    named: bool
    limit: int | None


@dataclass(frozen=True, slots=True)
class MapEachNode(Node):
    callback: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class TraverseNode(Node):
    operation: Node
    mode: Literal["fail_fast", "all_settled", "lossy"]
    limit: int | None
    on_error: Callable[..., Any] | None = None


@dataclass(frozen=True, slots=True)
class FilterEachNode(Node):
    predicate: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class FilterOperationNode(Node):
    predicate: Node
    best_effort: bool


@dataclass(frozen=True, slots=True)
class FoldNode(Node):
    reducer: Callable[..., Any] | Node
    initial: Any
    operation_reducer: bool


@dataclass(frozen=True, slots=True)
class FanoutEachNode(Node):
    fanout: FanoutNode


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    kind: Literal["operation", "callable", "nested", "constant"]
    value: Any


@dataclass(frozen=True, slots=True)
class BuildNode(Node):
    fields: tuple[FieldSpec, ...]
    mode: Literal["fail_fast", "all_settled", "lossy"]
    model: type[Any] | None


@dataclass(frozen=True, slots=True)
class MergeSource:
    kind: Literal["operation", "callable", "mapping"]
    value: Any


@dataclass(frozen=True, slots=True)
class MergeNode(Node):
    sources: tuple[MergeSource, ...]


@dataclass(frozen=True, slots=True)
class AssignFieldsNode(Node):
    fields: tuple[FieldSpec, ...]
