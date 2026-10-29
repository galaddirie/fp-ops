"""Control flow: branching, loops, resource scopes, and scoped environment values.

Each combinator here is a node in the immutable AST, so its environment
requirements are known when the operation is built and checked before it runs.
"""
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, cast

from fp_ops._ast import (
    BracketNode,
    BranchNode,
    LoopNode,
    PassthroughNode,
    ProvideNode,
)
from fp_ops.core import (
    Operation,
    _merge_requirements,
    environment_operation,
    identity,
    operation,
)
from fp_ops.environment import EnvKey

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
R = TypeVar("R")
S = TypeVar("S")
T = TypeVar("T")
S_co = TypeVar("S_co", covariant=True)


def branch(
    condition: Operation[A, bool],
    then: Operation[A, B],
    otherwise: Operation[A, C],
) -> Operation[A, B | C]:
    """Run ``then`` or ``otherwise`` on the original input, chosen by ``condition``.

    A condition that returns anything but a ``bool`` fails with ``TypeError``;
    only the chosen branch runs.
    """

    for item in (condition, then, otherwise):
        if not isinstance(item, Operation):
            raise TypeError("branch() requires three Operation instances")
    return cast(
        Operation[A, B | C],
        Operation._from_node(
            BranchNode(condition._node, then._node, otherwise._node),
            name=f"branch({condition.name})",
            requirements=_merge_requirements(
                condition._requirements, then._requirements, otherwise._requirements
            ),
        ),
    )


def passthrough(effect: Operation[A, object]) -> Operation[A, A]:
    """Run ``effect`` on the input for what it does, then continue with the input.

    Unlike :func:`~fp_ops.tap`, the effect is an operation: it can use the
    environment, and its failure fails the pipeline.
    """

    if not isinstance(effect, Operation):
        raise TypeError("passthrough() requires an Operation")
    return cast(
        Operation[A, A],
        Operation._from_node(
            PassthroughNode(effect._node),
            name=f"passthrough({effect.name})",
            requirements=effect._requirements,
        ),
    )


def sleep(seconds: float | Callable[[], float]) -> Operation[T, T]:
    """Wait, then pass the input through. A callable is sampled on each run,
    for jittered politeness delays: ``sleep(lambda: random.uniform(1, 3))``."""

    if not callable(seconds):
        if isinstance(seconds, bool) or seconds < 0:
            raise ValueError("sleep seconds must be a non-negative number")
        fixed = float(seconds)
        seconds = lambda: fixed  # noqa: E731

    sample = seconds

    async def wait(value: T) -> T:
        delay = sample()
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise TypeError("sleep callable must return a number")
        if delay < 0:
            raise ValueError("sleep callable returned a negative delay")
        await asyncio.sleep(float(delay))
        return value

    return operation(wait)


def jitter(low: float, high: float) -> Callable[[], float]:
    """A uniform random delay in ``[low, high]`` seconds, for :func:`sleep`."""

    if low < 0 or high < low:
        raise ValueError("jitter requires 0 <= low <= high")
    return lambda: random.uniform(low, high)  # noqa: S311


@dataclass(frozen=True, slots=True)
class Continue(Generic[S_co]):
    """Run another ``loop`` step with ``state``."""

    state: S_co


@dataclass(frozen=True, slots=True)
class Stop(Generic[S_co]):
    """End the ``loop`` with ``state`` as its result."""

    state: S_co


def loop(
    step: Operation[S, Continue[S] | Stop[S]],
    *,
    max_steps: int | None = None,
) -> Operation[S, S]:
    """Repeat ``step`` from the input state until it returns :class:`Stop`.

    Runs in constant stack space. ``Continue(s)`` feeds ``s`` to the next step;
    reaching ``max_steps`` returns the last state as a normal result. A failing
    step fails the loop with :class:`~fp_ops.LoopStepError`, which carries the
    step number and the state that step was given, so the work can resume.
    """

    if not isinstance(step, Operation):
        raise TypeError("loop() requires an Operation")
    if max_steps is not None and (isinstance(max_steps, bool) or max_steps < 1):
        raise ValueError("max_steps must be at least 1")
    return cast(
        Operation[S, S],
        Operation._from_node(
            LoopNode(step._node, max_steps),
            name=f"loop({step.name})",
            requirements=step._requirements,
        ),
    )


def continue_if(
    condition: Operation[S, bool],
    then: Operation[S, S],
) -> Operation[S, Continue[S] | Stop[S]]:
    """A ``loop`` step: when ``condition`` holds, run ``then`` and continue
    with its state; otherwise stop with the current state."""

    keep: Operation[S, S] = identity()
    advance: Operation[S, Continue[S]] = then.map(Continue)
    finish: Operation[S, Stop[S]] = keep.map(Stop)
    return branch(condition, advance, finish)


def bracket(
    acquire: Operation[A, R],
    use: Operation[R, B],
    release: Operation[R, object],
) -> Operation[A, B]:
    """Acquire a resource, use it, and always release it.

    * ``acquire`` failing returns its error; nothing is released.
    * ``use`` failing returns its error unchanged once the release succeeds.
    * A failed release returns :class:`~fp_ops.ReleaseError`, with the ``use``
      error (if any) as ``primary``.
    * When the run is cancelled, the release still runs to completion before
      the cancellation propagates, so concurrent branches never leak resources.
    """

    for item in (acquire, use, release):
        if not isinstance(item, Operation):
            raise TypeError("bracket() requires three Operation instances")
    return cast(
        Operation[A, B],
        Operation._from_node(
            BracketNode(acquire._node, use._node, release._node),
            name=f"bracket({use.name})",
            requirements=_merge_requirements(
                acquire._requirements, use._requirements, release._requirements
            ),
        ),
    )


def provide(
    key: EnvKey[T],
    value: Operation[A, T],
    body: Operation[A, B],
) -> Operation[A, B]:
    """Compute ``value`` from the input, then run ``body`` on the same input with
    ``key`` bound to it in the environment.

    The binding is lexical: it holds inside ``body`` only, shadows any outer
    value, and is invisible to sibling branches. Static requirements become
    ``value``'s plus ``body``'s minus ``key``, so preflight neither asks the
    caller for ``key`` nor misses anything ``body`` still needs.
    """

    if not isinstance(key, EnvKey):
        raise TypeError("provide() requires an EnvKey")
    if not isinstance(value, Operation) or not isinstance(body, Operation):
        raise TypeError("provide() requires Operation value and body")
    remaining = tuple(item for item in body._requirements if item is not key)
    return cast(
        Operation[A, B],
        Operation._from_node(
            ProvideNode(key, value._node, body._node),
            name=f"provide({key.name})",
            requirements=_merge_requirements(value._requirements, remaining),
        ),
    )


def from_environment(key: EnvKey[T]) -> Operation[object, T]:
    """Read ``key``'s value from the environment, ignoring the input."""

    def read(_value: object, dependency: T) -> T:
        return dependency

    read.__name__ = f"from_environment({key.name})"
    return environment_operation(key)(read)


__all__ = [
    "Continue",
    "Stop",
    "bracket",
    "branch",
    "continue_if",
    "from_environment",
    "jitter",
    "loop",
    "passthrough",
    "provide",
    "sleep",
]
