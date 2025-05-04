import itertools
from typing import Any, Awaitable, Callable, Dict, List, MutableMapping, Tuple
from dataclasses import dataclass

from fp_ops.primitives import Port

from expression import Result


@dataclass(slots=True, frozen=True)
class Step:
    func: Callable[..., Awaitable[Any]]
    arg_getters: Tuple[Callable[[MutableMapping[Port, Result]], Any], ...]
    result_handle: Port
    propagate_error: bool = True


@dataclass(slots=True, frozen=True)
class ExecutionPlan:
    steps: List[Step]
    final_handle: Port

    def __hash__(self) -> int:
        return hash((tuple(self.steps), self.final_handle))


class Executor:
    async def run(self, plan: ExecutionPlan) -> Result[Any, Exception]:
        state: Dict[Port, Result] = {}
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