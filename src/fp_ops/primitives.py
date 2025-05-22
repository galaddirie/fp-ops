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

from .context import BaseContext
if TYPE_CHECKING:
    from .operator import Operation

# More descriptive type variables
T = TypeVar("T")
ResultType = TypeVar("ResultType")
ReturnType = TypeVar("ReturnType")
ExceptionType = TypeVar("ExceptionType", bound=Exception)
ContextType = TypeVar("ContextType", bound=Optional[BaseContext])
Params = ParamSpec("Params")

class Placeholder:
    """
    Singleton marker that will be replaced when a ``Template`` is rendered.
    """
    _instance = None

    def __new__(cls) -> "Placeholder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "_"


_ = Placeholder()


@dataclass(slots=True, frozen=True)
class Template(Generic[ResultType, ContextType]):
    """
    Template for operation arguments that can contain placeholders.
    
    This class represents a complete argument specification for an operation,
    storing both positional and keyword arguments that may include placeholders (`_`).
    When an operation is called, these placeholders can be replaced with actual values.
    
    Examples:
        # Identity template that passes through a single value
        Template(args=(_,))
        
        # Template with a placeholder as the first arg and a constant second arg
        Template(args=(_, "constant"))
        
        # Template with keyword arguments including placeholders
        Template(kwargs={"param1": _, "param2": "constant"})
    
    The Template maintains efficient caches of placeholder positions for quick rendering
    when the operation is executed.
    
    Notes:
        - "context" is a reserved keyword argument name and will be handled specially during
          execution if the operation requires a context.
    """
    args: Sequence[Any] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    
    # Cache positions of placeholders for efficient rendering
    _pos_indices: Tuple[int, ...] = field(init=False, repr=False)
    _kw_keys: Tuple[str, ...] = field(init=False, repr=False)
    _deep: bool = field(init=False, repr=False) #  any nested "_"?
    
    def __post_init__(self) -> None:
        object.__setattr__(self, "_pos_indices",
                           tuple(i for i, v in enumerate(self.args)
                                 if isinstance(v, Placeholder)))
        object.__setattr__(self, "_kw_keys",
                           tuple(k for k, v in self.kwargs.items()
                                 if isinstance(v, Placeholder)))

        def _contains(obj: Any) -> bool:
            if isinstance(obj, Placeholder):
                return True
            if isinstance(obj, (list, tuple, set)):
                return any(_contains(x) for x in obj)
            if isinstance(obj, Mapping):
                return any(_contains(x) for x in obj.values())
            return False

        object.__setattr__(self, "_deep",
                           _contains(self.args) or _contains(tuple(self.kwargs.values())))
    
    def has_placeholders(self) -> bool:
        """True if *any* placeholder (even nested) is present."""
        return bool(self._pos_indices or self._kw_keys or self._deep)
    
    def is_identity(self) -> bool:
        """Fast-path: the template is exactly one bare "_"."""
        return len(self.args) == 1 and isinstance(self.args[0], Placeholder) and not self.kwargs
    
    def render(self, value: Any) -> tuple[Tuple[Any, ...], dict[str, Any]]:
        """Recursively replace "_" with *value*."""
        if not self.has_placeholders():
            return tuple(self.args), dict(self.kwargs)
            
        def _replace(obj: Any) -> Any:
            if isinstance(obj, Placeholder):
                return value
            if isinstance(obj, list):
                return [_replace(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_replace(x) for x in obj)
            if isinstance(obj, dict):
                return {k: _replace(v) for k, v in obj.items()}
            return obj

        new_args = tuple(_replace(a) for a in self.args)
        new_kwargs = {k: _replace(v) for k, v in self.kwargs.items()}
        return new_args, new_kwargs
    
@dataclass(slots=True, frozen=True)
class OpSpec(Generic[ResultType, ContextType]):
    """
    Immutable specification for a single operation node in the computation graph.

    This class encapsulates all static information about an operation, including:
      - a unique identifier (`id`)
      - the underlying async function to execute (`func`)
      - its Python signature (`signature`)
      - the expected context type (`ctx_type`)
      - whether the operation requires a context (`require_ctx`)
      - a unified template for arguments (`template`)

    OpSpec contains no edge or graph connectivity information and holds no mutable state.
    It is used to describe the behavior and invocation details of an operation node,
    independent of how it is wired into a graph.
    
    Notes:
      - "context" is a reserved keyword argument name. If present in the function
        signature and require_ctx is True, the context will be automatically injected
        during execution.
    """

    id: str
    func: Callable[..., Awaitable[ResultType]]
    signature: inspect.Signature
    ctx_type: Type[ContextType] | None
    require_ctx: bool = False
    template: Template[ResultType, ContextType] = field(default_factory=Template)

    @property
    def params(self) -> Sequence[str]:
        """
        Get the parameter names expected by this operation.
        
        Returns:
            Sequence[str]: List of parameter names, excluding 'self'.
                           Note that 'context' may be in this list if the operation
                           accepts a context parameter.
        """
        return [p for p in self.signature.parameters if p not in ("self",)]
    
    @property
    def non_context_params(self) -> Sequence[str]:
        """
        Get the parameter names expected by this operation, excluding 'context'.
        
        Returns:
            Sequence[str]: List of parameter names, excluding 'self' and 'context'.
        """
        return [p for p in self.signature.parameters if p not in ("self", "context")]

class EdgeType(Enum):
    RESULT = "result"
    ERROR = "error"
    CONTEXT = "context"

class PortType(Enum):
    TARGET = "target"
    SOURCE = "source"

@dataclass(slots=True, frozen=True)
class Port:
    node_id: str
    port_type: PortType
    name: Optional[str] = None
    optional: bool = False
    default: Any = None

@dataclass(slots=True, frozen=True)
class Edge:
    source: Port
    target: Port
    type: EdgeType = EdgeType.RESULT
    transform: Callable[[Any], Awaitable[Any]] | None = None
