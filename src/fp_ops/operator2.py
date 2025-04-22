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
import uuid

from fp_ops.node import Edge, Handle, HandleType, EdgeType
from fp_ops.placeholder import Placeholder, _

from expression import Result

T = TypeVar("T")
S = TypeVar("S")
C = TypeVar("C")
P = ParamSpec("P")
R = TypeVar("R")


class Operation(Generic[T, S, C]):
    '''A DAG node wrapping a function into composable ports and execution.'''
    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        name: Optional[str] = None,
        context: bool = False,
        context_type: Optional[Type[C]] = None
    ):
        wraps(func)(self)
        self.func = func
        self.name = name or func.__name__
        self.requires_context = 'context' in inspect.signature(func).parameters or context
        self._bound_args: Dict[str, Any] = {}
        self._placeholder_targets: List[str] = []
        self.target_handles: Dict[str, Handle] = {}
        self.source_handles: Dict[str, Handle] = {}
        self._build_handles()
        # for pipeline initial-binding
        self._initial_args: Tuple[Any, ...] = ()
        self._initial_kwargs: Dict[str, Any] = {}

    def _build_handles(self) -> None:
        sig = inspect.signature(self.func)
        for pname, param in sig.parameters.items():
            optional = param.default is not inspect.Parameter.empty
            default = param.default if optional else None
            self.target_handles[pname] = Handle(
                self, HandleType.TARGET, pname, optional, default
            )
        for out in ('result', 'error', 'context'):
            self.source_handles[out] = Handle(self, HandleType.SOURCE, out)

    def __rshift__(self, other: Operation) -> Operation:
        src = self.source_handles['result']
        # placeholder connections
        if other._placeholder_targets:
            for pname in other._placeholder_targets:
                src.connect(other.target_handles[pname])
            return other
        # auto-wire unbound inputs
        required = [h for h in other.target_handles.values() if h.name != 'context']
        bound = list(other._bound_args)
        if len(bound) < len(required):
            for h in required:
                if h.name not in bound:
                    src.connect(h)
                    break
        return other

    def __call__(self, *args: Any, **kwargs: Any) -> Operation[T, S, C]:
        # detect if this Operation is part of a composite pipeline
        is_pipeline = len(self.crawl('upstream')) > 1

        # 1) placeholder-binding (f(_), f(_,3), etc.)
        if any(isinstance(a, Placeholder) for a in args) or any(
            isinstance(v, Placeholder) for v in kwargs.values()
        ):
            sig = inspect.signature(self.func)
            bound = sig.bind_partial(*args, **kwargs)

            @wraps(self.func)
            async def _wrapped(*in_args, **in_kwargs):
                final: Dict[str, Any] = {}
                iter_args = list(in_args)
                # fill from bound args or placeholder pop
                for name, val in bound.arguments.items():
                    if isinstance(val, Placeholder):
                        final[name] = iter_args.pop(0)
                    else:
                        final[name] = val
                # fill remaining
                for pname in sig.parameters:
                    if pname not in final:
                        final[pname] = iter_args.pop(0)
                return await self.func(**final)

            new = Operation(_wrapped, name=self.name)
            new._bound_args = dict(bound.arguments)
            for name, val in bound.arguments.items():
                h = new.target_handles[name]
                h.optional = True
                if not isinstance(val, Placeholder):
                    h.default_value = val
                else:
                    new._placeholder_targets.append(name)
            return new

        # 2) constant-binding on standalone node (e.g., mul(1, 2) or add_one(a=2))
        sig = inspect.signature(self.func)
        try:
            bound = sig.bind_partial(*args, **kwargs)
            can_const_bind = True
        except TypeError:
            can_const_bind = False

        if can_const_bind and (args or kwargs):
            @wraps(self.func)
            async def _wrapped_const(*in_args, **in_kwargs):
                final: Dict[str, Any] = {}
                iter_args = list(in_args)
                # assign bound constants
                for name, val in bound.arguments.items():
                    final[name] = val
                # fill others from in_args
                for pname in sig.parameters:
                    if pname not in final and iter_args:
                        final[pname] = iter_args.pop(0)
                return await self.func(**final)

            new = Operation(_wrapped_const, name=self.name)
            new._bound_args = dict(bound.arguments)
            for name, val in bound.arguments.items():
                h = new.target_handles[name]
                h.optional = True
                h.default_value = val
            return new

        # 3) pipeline-binding on a composite (pipeline(…))
        if is_pipeline:
            new = copy.copy(self)
            new._initial_args = args
            new._initial_kwargs = kwargs
            return new

        # 4) fallback to creating a partially bound op
        # (covers cases like mul(1) binding first arg)
        bound = sig.bind_partial(*args, **kwargs)

        @wraps(self.func)
        async def _wrapped_partial(*in_args, **in_kwargs):
            final: Dict[str, Any] = {}
            iter_args = list(in_args)
            # assign bound constants
            for name, val in bound.arguments.items():
                final[name] = val
            # fill others from in_args
            for pname in sig.parameters:
                if pname not in final and iter_args:
                    final[pname] = iter_args.pop(0)
            return await self.func(**final)

        new = Operation(_wrapped_partial, name=self.name)
        new._bound_args = dict(bound.arguments)
        for name, val in bound.arguments.items():
            h = new.target_handles[name]
            h.optional = True
            h.default_value = val
        return new
        
    def __await__(self) -> Any:
        return self.execute().__await__()
    
    async def execute(self, *args: Any, **kwargs: Any) -> Result[S, Exception]:
        # use stored initial args if none passed
        if not args and not kwargs and self._initial_args:
            args = self._initial_args
            kwargs = self._initial_kwargs

        # build execution order, head first
        nodes = list(reversed(self.crawl('upstream')))
        head = nodes[0]

        # seed from head signature
        param_names = list(inspect.signature(head.func).parameters)
        inputs = dict(zip(param_names, args))
        inputs.update(kwargs)

        state: Dict[Tuple[Operation, str], Tuple[Result, Any]] = {}
        # seed defaults or inputs
        for node in nodes:
            for name, handle in node.target_handles.items():
                if not handle.edges:
                    if name in inputs:
                        state[(node, name)] = (Result.Ok(inputs[name]), None)
                    elif handle.optional:
                        state[(node, name)] = (Result.Ok(handle.default_value), None)
                    else:
                        raise ValueError(
                            f"Missing initial input '{name}' for node '{node.name}'"
                        )

        # execute each node
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
                state[(node, 'result')] = (Result.Ok(out_val), ctx)
            except Exception as e:
                state[(node, 'error')] = (Result.Error(e), ctx)

        final_res, _ = state.get(
            (self, 'result'), (Result.Error(ValueError('No result')), None)
        )
        return final_res

    def crawl(self, direction: str = 'upstream') -> List[Operation]:
        visited: set[Operation] = set()
        order: List[Operation] = []
        def dfs(n: Operation):
            if n in visited:
                return
            visited.add(n)
            order.append(n)
            handles = (
                n.target_handles.values()
                if direction == 'upstream'
                else n.source_handles.values()
            )
            for h in handles:
                for e in h.edges:
                    nxt = (
                        e.source_handle.node
                        if direction == 'upstream'
                        else e.target_handle.node
                    )
                    dfs(nxt)
        dfs(self)
        return order

    def validate(self) -> None:
        for name, h in self.target_handles.items():
            if not h.optional and not h.edges and name != 'context':
                raise ValueError(
                    f"Missing connection for required input '{name}' on node '{self.name}'"
                )

    def map(self, func: Callable[[Any], Any]) -> Operation:
        Edge(
            self.source_handles['result'],
            self.source_handles['result'],
            transform=func
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
        """ build graph viz using dot notation"""
        import graphviz
        dot = graphviz.Digraph(comment=self.name)
        for node in self.crawl('upstream'):
            dot.node(node.name, node.name)
        for node in self.crawl('downstream'):
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
def identity(x: int) -> int:
    return x


@operation
def constant(x: int) -> int:
    """Alias for identity operation"""
    return x
