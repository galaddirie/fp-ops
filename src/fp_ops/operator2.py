from __future__ import annotations
from typing import (
    Callable,
    Awaitable,
    Any,
    Optional,
    Dict,
    List,
    TypeVar,
    Generic,
    overload,
    ParamSpec,
    Type,
    Union,
    Tuple,
)
from functools import wraps
import inspect
import copy

from fp_ops.node import Edge, Handle, HandleType
from fp_ops.placeholder import Placeholder, _
from expression import Result

T = TypeVar("T")
S = TypeVar("S")
C = TypeVar("C")
P = ParamSpec("P")
R = TypeVar("R")


class Operation(Generic[T, S, C]):
    """
    A DAG node wrapping a function into composable ports and execution.
    """

    # core metadata
    func: Callable[..., Awaitable[Any]]
    name: str
    requires_context: bool
    context_type: Optional[Type[C]]

    # binding state
    bound_args: Optional[Tuple[Any, ...]]
    bound_kwargs: Optional[Dict[str, Any]]
    is_bound: bool

    # I/O handles for DAG wiring
    target_handles: Dict[str, Handle]
    source_handles: Dict[str, Handle]

    # function introspection
    __name__: str
    __doc__: str
    __signature__: Optional[inspect.Signature]
    __annotations__: Dict[str, Any]
    __module__: str

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        name: Optional[str] = None,
        context: bool = False,
        context_type: Optional[Type[C]] = None,
        bound_args: Optional[Tuple[Any, ...]] = None,
        bound_kwargs: Optional[Dict[str, Any]] = None,
    ):
        wraps(func)(self)
        self.func = func
        self.name = name or func.__name__
        self.requires_context = (
            "context" in inspect.signature(func).parameters or context
        )
        self.context_type = context_type

        # binding initialization
        self.bound_args = bound_args
        self.bound_kwargs = bound_kwargs
        self.is_bound = bool(bound_args or bound_kwargs)

        # build wiring handles
        self.target_handles = {}
        self.source_handles = {}
        self._build_handles()

        # preserve introspection
        self.__name__ = getattr(func, "__name__", "unknown")
        self.__doc__ = getattr(func, "__doc__", "")
        self.__signature__ = inspect.signature(func)
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__module__ = getattr(func, "__module__", "unknown")

    def _build_handles(self) -> None:
        sig = inspect.signature(self.func)
        for pname, param in sig.parameters.items():
            optional = param.default is not inspect.Parameter.empty
            default = param.default if optional else None
            self.target_handles[pname] = Handle(
                self, HandleType.TARGET, pname, optional, default
            )
        for out in ("result", "error", "context"):
            self.source_handles[out] = Handle(self, HandleType.SOURCE, out)

    def __str__(self) -> str:
        return f"{self.name}: {self.__doc__}"

    def __repr__(self) -> str:
        return f"{self.name}: {self.__doc__}"

    def __rshift__(self, other: Operation) -> Operation:
        src = self.source_handles["result"]
        # build map of bound inputs from args and kwargs
        bound_map: Dict[str, Any] = {}
        if other.bound_kwargs:
            bound_map.update(other.bound_kwargs)
        if other.bound_args:
            params = list(other.target_handles.keys())
            for idx, val in enumerate(other.bound_args):
                if idx < len(params):
                    bound_map[params[idx]] = val

        # connect any placeholder bindings
        for name, val in bound_map.items():
            if isinstance(val, Placeholder):
                handle = other.target_handles[name]
                handle.optional = True
                src.connect(handle)

        # set defaults for constant binds
        for name, val in bound_map.items():
            if not isinstance(val, Placeholder):
                handle = other.target_handles[name]
                handle.optional = True
                handle.default_value = val

        # auto-wire first unbound input
        for name, handle in other.target_handles.items():
            if name == "context":
                continue
            if name not in bound_map:
                src.connect(handle)
                break
        return other

    def __and__(self, other: Operation) -> Operation:
        return self >> other

    def __or__(self, other: Operation) -> Operation:
        return self >> other

    def __call__(self, *args: Any, **kwargs: Any) -> Operation[T, S, C]:
        is_pipeline = len(self.crawl("upstream")) > 1
        sig = inspect.signature(self.func)

        # placeholder-binding
        if any(isinstance(a, Placeholder) for a in args) or any(
            isinstance(v, Placeholder) for v in kwargs.values()
        ):
            bound = sig.bind_partial(*args, **kwargs)

            @wraps(self.func)
            async def _wrapped(*in_args, **in_kwargs):
                final: Dict[str, Any] = {}
                iter_args = list(in_args)
                for name, val in bound.arguments.items():
                    if isinstance(val, Placeholder):
                        final[name] = iter_args.pop(0)
                    else:
                        final[name] = val
                for pname in sig.parameters:
                    if pname not in final:
                        final[pname] = iter_args.pop(0)
                return await self.func(**final)

            return Operation(
                _wrapped, name=self.name, bound_args=args, bound_kwargs=kwargs
            )

        # constant-binding
        try:
            bound = sig.bind_partial(*args, **kwargs)
            can_const = True
        except TypeError:
            can_const = False

        if can_const and (args or kwargs):

            @wraps(self.func)
            async def _wrapped_const(*in_args, **in_kwargs):
                final: Dict[str, Any] = {}
                iter_args = list(in_args)
                for name, val in bound.arguments.items():
                    final[name] = val
                for pname in sig.parameters:
                    if pname not in final and iter_args:
                        final[pname] = iter_args.pop(0)
                return await self.func(**final)

            return Operation(
                _wrapped_const, name=self.name, bound_args=args, bound_kwargs=kwargs
            )

        # pipeline-binding
        if is_pipeline:
            new = copy.copy(self)
            new.bound_args = args
            new.bound_kwargs = kwargs
            new.is_bound = True
            return new

        # fallback partial-binding
        bound = sig.bind_partial(*args, **kwargs)

        @wraps(self.func)
        async def _wrapped_partial(*in_args, **in_kwargs):
            final: Dict[str, Any] = {}
            iter_args = list(in_args)
            for name, val in bound.arguments.items():
                final[name] = val
            for pname in sig.parameters:
                if pname not in final and iter_args:
                    final[pname] = iter_args.pop(0)
            return await self.func(**final)

        return Operation(
            _wrapped_partial, name=self.name, bound_args=args, bound_kwargs=kwargs
        )

    def __await__(self) -> Any:
        return self.execute().__await__()

    async def execute(self, *args: Any, **kwargs: Any) -> Result[S, Exception]:
        if not args and not kwargs and self.bound_args:
            args = self.bound_args or ()
        if not kwargs and self.bound_kwargs:
            kwargs = self.bound_kwargs or {}

        nodes = list(reversed(self.crawl("upstream")))
        head = nodes[0]

        sig = inspect.signature(head.func)
        param_names = list(sig.parameters)
        inputs = dict(zip(param_names, args))
        inputs.update(kwargs)

        state: Dict[Tuple[Operation, str], Tuple[Result, Any]] = {}
        for node in nodes:
            for name, handle in node.target_handles.items():
                if not handle.edges:
                    if name in inputs:
                        state[(node, name)] = (Result.Ok(inputs[name]), None)
                    elif handle.optional:
                        state[(node, name)] = (Result.Ok(handle.default_value), None)
                    else:
                        raise ValueError(f"Missing input '{name}' for '{node.name}'")

        for node in nodes:
            ctx = None
            args_list: List[Any] = []
            for name in inspect.signature(node.func).parameters:
                handle = node.target_handles[name]
                if handle.edges:
                    edge = handle.edges[-1]
                    prev_res, prev_ctx = state[
                        (edge.source_handle.node, edge.source_handle.name)
                    ]
                    res = await edge.pipe(prev_res, prev_ctx)
                    state[(node, name)] = (res, prev_ctx)
                    args_list.append(res.default_value(None))
                else:
                    res, _ = state[(node, name)]
                    args_list.append(res.default_value(None))
            try:
                out_val = await node.func(*args_list)
                state[(node, "result")] = (Result.Ok(out_val), ctx)
            except Exception as e:
                state[(node, "error")] = (Result.Error(e), ctx)

        final_res, _ = state.get(
            (self, "result"), (Result.Error(ValueError("No result")), None)
        )
        return final_res

    def crawl(self, direction: str = "upstream") -> List[Operation]:
        visited: set[Operation] = set()
        order: List[Operation] = []

        def dfs(n: Operation):
            if n in visited:
                return
            visited.add(n)
            order.append(n)
            handles = (
                n.target_handles.values()
                if direction == "upstream"
                else n.source_handles.values()
            )
            for h in handles:
                for e in h.edges:
                    nxt = (
                        e.source_handle.node
                        if direction == "upstream"
                        else e.target_handle.node
                    )
                    dfs(nxt)

        dfs(self)
        return order

    def validate(self) -> None:
        for name, h in self.target_handles.items():
            if not h.optional and not h.edges and name != "context":
                raise ValueError(
                    f"Missing connection for required input '{name}' on '{self.name}'"
                )

    @staticmethod
    def unit(value: T) -> Operation[T, T, None]:
        return Operation(lambda x: x, name=f"unit({value})")

    def map(self, func: Callable[[Any], Any]) -> Operation:
        Edge(
            self.source_handles["result"], self.source_handles["result"], transform=func
        )
        return self

    def filter(self, cond: Callable[[Any], bool]) -> Operation:
        return self.map(lambda x: x if cond(x) else None)

    def sequence(self, *ops: Operation) -> Operation:
        cur = self
        for o in ops:
            cur = cur >> o
        return cur

    @classmethod
    def combine(cls, **ops: Operation) -> Operation:
        return cls.sequence(*ops)

    def dot_notation(self) -> str:
        """build graph viz using dot notation"""
        import graphviz

        dot = graphviz.Digraph(comment=self.name)
        for node in self.crawl("upstream"):
            dot.node(node.name, node.name)
        for node in self.crawl("downstream"):
            dot.node(node.name, node.name)
        return dot.source


@overload
def operation(
    func: Callable[P, R],
    *,
    context: bool = False,
    context_type: Optional[Type[C]] = None,
) -> Operation[Callable[P, R], S, C]: ...


@overload
def operation(
    func: None,
    *,
    context: bool = False,
    context_type: Optional[Type[C]] = None,
) -> Callable[[Callable[P, R]], Operation[Callable[P, R], S, C]]: ...


def operation(
    func: Optional[Callable[..., Any]] = None,
    *,
    context: bool = False,
    context_type: Optional[Type[C]] = None,
) -> Union[
    Operation[Callable[P, R], S, C],
    Callable[[Callable[P, R]], Operation[Callable[P, R], S, C]],
]:
    """Decorator to convert a function into an Operation node."""

    def decorator(f: Callable[P, R]) -> Operation[Callable[P, R], S, C]:
        return Operation(f, context=context, context_type=context_type)

    if func is None:
        return decorator
    return Operation(func, context=context, context_type=context_type)


@operation
async def identity(x: int) -> int:
    return x


@operation
async def constant(x: int) -> int:
    """Alias for identity"""
    return x
