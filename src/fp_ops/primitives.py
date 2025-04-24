from __future__ import annotations
import asyncio
import inspect
import uuid
import itertools

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
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    Generic,
)
from types import MappingProxyType


from dataclasses import dataclass, field

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


@dataclass(slots=True, frozen=True)
class OpSpec(Generic[S, C]):
    """Pure data - no edges, no mutable state."""

    id: str
    func: Callable[..., Awaitable[S]]
    signature: inspect.Signature
    requires_ctx: bool
    ctx_type: Type[C] | None
    bound_args: Tuple[Any, ...] = ()
    bound_kwargs: Mapping[str, Any] = MappingProxyType({})

    @property
    def params(self) -> Sequence[str]:
        return [p for p in self.signature.parameters if p not in ("self",)]

class EdgeType(Enum):
    RESULT = "result"
    ERROR = "error"
    CONTEXT = "context"

class HandleType(Enum):
    TARGET = "target"
    SOURCE = "source"

@dataclass(slots=True, frozen=True)
class HandleId:
    node_id: str
    handle_type: HandleType
    name: Optional[str] = None                     # param or special (result/error/context)
    optional: bool = False
    default: Any = None




@dataclass(slots=True, frozen=True)
class Edge:
    source: HandleId
    target: HandleId
    type: EdgeType = EdgeType.RESULT
    transform: Callable[[Any], Awaitable[Any]] | None = None



