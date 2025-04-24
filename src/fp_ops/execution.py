import itertools
from typing import Any, Awaitable, Callable, Dict, List, MutableMapping, Tuple
from dataclasses import dataclass

from fp_ops.primitives import HandleId

from expression import Result


@dataclass(slots=True)
class Step:
    func: Callable[..., Awaitable[Any]]
    arg_getters: Tuple[Callable[[MutableMapping[HandleId, Result]], Any], ...]
    result_handle: HandleId
    propagate_error: bool = True


@dataclass(slots=True)
class ExecutionPlan:
    steps: List[Step]
    final_handle: HandleId


class Executor:
    async def run(self, plan: ExecutionPlan) -> Result[Any, Exception]:
        state: Dict[HandleId, Result] = {}
        for step in plan.steps:
            try:
                args = [g(state) for g in step.arg_getters]
            except KeyError as ke:
                return Result.Error(KeyError(f"Uninitialized handle {ke}"))
            try:
                raw = await step.func(*args)
                res = raw if isinstance(raw, Result) else Result.Ok(raw)
            except Exception as exc:
                res = Result.Error(exc)
            state[step.result_handle] = res
            if res.is_error() and step.propagate_error:
                return res
        return state[plan.final_handle]

_default_executor = Executor()