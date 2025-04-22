from __future__ import annotations
import asyncio
import inspect
import uuid
from enum import Enum
from typing import (
    TypeVar,
    Callable,
    Any,
    Optional,
    List,
    Dict,
    Type,
    Awaitable,
    ParamSpec,
    TYPE_CHECKING,
)
from expression import Result

from .placeholder import Placeholder
from .context import BaseContext

if TYPE_CHECKING:
    from .operator import Operation


T = TypeVar("T")  # type of the input
S = TypeVar("S")  # type of the output
R = TypeVar("R")  # type of the result
E = TypeVar("E", bound=Exception)  # type of the error
C = TypeVar("C", bound=Optional[BaseContext])  # type of the context
P = ParamSpec("P")  # Captures all parameter types


class EdgeType(Enum):
    RESULT = "result"
    ERROR = "error"
    CONTEXT = "context"


class HandleType(Enum):
    TARGET = "target"
    SOURCE = "source"


class Handle:
    def __init__(
        self,
        node: Operation[Any, Any, Any],
        handle_type: HandleType,
        name: str,
        optional: bool = False,
        default_value: Any = None,
    ):
        self.node = node
        self.handle_type = handle_type
        self.name = name
        self.optional = optional
        self.default_value = default_value
        self.edges: List[Edge] = []

    def connect(
        self,
        target: Handle,
        edge_type: EdgeType = EdgeType.RESULT,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> Edge:
        if self.handle_type != HandleType.SOURCE:
            raise ValueError("Can only connect from SOURCE handles")
        if target.handle_type != HandleType.TARGET:
            raise ValueError("Can only connect to TARGET handles")
        edge = Edge(self, target, edge_type, transform)
        self.edges.append(edge)
        target.edges.append(edge)
        return edge


class Edge:
    def __init__(
        self,
        source_handle: Handle,
        target_handle: Handle,
        edge_type: EdgeType = EdgeType.RESULT,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.source_handle = source_handle
        self.target_handle = target_handle
        self.edge_type = edge_type
        self.transform = transform

    async def pipe(self, value: Any, context: Any = None) -> Result[Any, Exception]:
        res = value if isinstance(value, Result) else Result.Ok(value)
        if res.is_error() and self.edge_type != EdgeType.ERROR:
            return res
        if self.transform and res.is_ok():
            try:
                raw = res.default_value(None)
                if inspect.iscoroutinefunction(self.transform):
                    out = await self.transform(raw)
                else:
                    out = await asyncio.to_thread(self.transform, raw)
                return Result.Ok(out)
            except Exception as e:
                return Result.Error(e)
        return res

    def __str__(self) -> str:
        return (
            f"Edge({self.source_handle.node.name}.{self.source_handle.name} -> "
            f"{self.target_handle.node.name}.{self.target_handle.name}, type={self.edge_type.value})"
        )
