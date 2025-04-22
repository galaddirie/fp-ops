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
    TYPE_CHECKING
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
    """
    The type of an edge in the operation DAG.
    Represents the type of data that can be passed through the edge.

    RESULT: Represents a successful result connection. This is the standard edge type
    when connecting operations, e.g., a >> b

    ERROR: Represents an error connection for propagating errors.

    CONTEXT: Represents a context connection for propagating context objects.
    """

    RESULT = "result"
    ERROR = "error"
    CONTEXT = "context"


class HandleType(Enum):
    """Type of a port (input or output)."""

    TARGET = "target"
    SOURCE = "source"


class Handle:
    def __init__(self, node: 'Operation', handle_type: HandleType, name: str, optional: bool = False, default_value: Any = None):
        self.node = node
        self.handle_type = handle_type
        self.name = name
        self.optional = optional
        self.default_value = default_value
        self.edges = []  # type: list[Edge]

    def connect(self, target_handle: 'Handle', edge_type: EdgeType = EdgeType.RESULT, transform: Optional[Callable[[Any], Any]] = None):
        if self.handle_type != HandleType.SOURCE:
            raise ValueError(f"Can only connect from SOURCE ports, not {self.handle_type}")
        if target_handle.handle_type != HandleType.TARGET:
            raise ValueError(f"Can only connect to TARGET ports, not {target_handle.handle_type}")
        edge = Edge(self, target_handle, edge_type=edge_type, transform=transform)
        self.edges.append(edge)
        target_handle.edges.append(edge)
        return edge

class Edge:
    def __init__(self, source_handle: Handle, target_handle: Handle,
                 edge_type: EdgeType = EdgeType.RESULT,
                 transform: Optional[Callable[[Any], Any]] = None):
        self.source_handle = source_handle
        self.target_handle = target_handle
        self.edge_type = edge_type
        self.transform = transform

    async def pipe(self, value: Any, context: Any = None) -> Result:
        # Handle incoming Result or raw value
        if isinstance(value, Result):
            res = value
        else:
            res = Result.Ok(value)
        # If error and not an ERROR edge, propagate unchanged
        if res.is_error() and self.edge_type != EdgeType.ERROR:
            return res
        # Apply transform if present
        if self.transform and res.is_ok():
            try:
                out_val = res.default_value(None)
                if inspect.iscoroutinefunction(self.transform):
                    mapped = await self.transform(out_val)
                else:
                    mapped = await asyncio.to_thread(self.transform, out_val)
                return Result.Ok(mapped)
            except Exception as e:
                return Result.Error(e)
        return res

    def __str__(self):
        return (f"Edge({self.source_handle.node.name}.{self.source_handle.name} -> "
                f"{self.target_handle.node.name}.{self.target_handle.name}, "
                f"type={self.edge_type.value})")

