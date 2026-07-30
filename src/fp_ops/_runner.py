"""The sole interpreter for FP-Ops' closed immutable AST."""
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
import copy
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Callable, cast

from expression import Error, Ok, Result

from fp_ops._ast import (
    AlternativeNode,
    AssignFieldsNode,
    BuildNode,
    CallNode,
    ConstantNode,
    FailNode,
    FanoutEachNode,
    FanoutNode,
    FieldSpec,
    FilterEachNode,
    FilterOperationNode,
    FlatMapNode,
    FoldNode,
    IdentityNode,
    MapEachNode,
    MergeNode,
    MergeSource,
    Node,
    RecoverNode,
    RetryNode,
    SequenceNode,
    TapNode,
    TraverseNode,
)
from fp_ops._invoke import invoke
from fp_ops.environment import EnvKey, Environment
from fp_ops.result import BuildFieldError, TraversalItemError


async def run(
    node: Node,
    value: Any,
    environment: Environment,
    requirements: tuple[EnvKey[object], ...],
) -> Result[Any, Exception]:
    """Validate and execute an AST inside the public ``Result`` boundary."""

    try:
        if not isinstance(environment, Environment):
            raise TypeError("environment must be an Environment")
        _preflight(environment, requirements)
        return await _capture(node, value, environment)
    except Exception as error:
        return Error(error)


def _preflight(
    environment: Environment,
    requirements: tuple[EnvKey[object], ...],
) -> None:
    for key in requirements:
        environment.require(key)


async def _capture(
    node: Node,
    value: Any,
    environment: Environment,
) -> Result[Any, Exception]:
    try:
        return Ok(await _evaluate(node, value, environment))
    except Exception as error:
        return Error(error)


def _unwrap(result: Result[Any, Exception]) -> Any:
    if result.is_error():
        raise result.error
    return result.ok


async def _evaluate(node: Node, value: Any, environment: Environment) -> Any:
    if isinstance(node, IdentityNode):
        return value
    if isinstance(node, ConstantNode):
        return node.value
    if isinstance(node, FailNode):
        raise node.error
    if isinstance(node, CallNode):
        return await node.invocation.call(value, environment)
    if isinstance(node, SequenceNode):
        current = value
        for stage in node.stages:
            current = await _evaluate(stage, current, environment)
        return current
    if isinstance(node, FlatMapNode):
        current = await _evaluate(node.source, value, environment)
        returned = await invoke(node.callback, current)
        return await _run_returned_operation(returned, environment)
    if isinstance(node, RecoverNode):
        outcome = await _capture(node.source, value, environment)
        if outcome.is_ok():
            return outcome.ok
        recovered = await invoke(node.handler, outcome.error)
        if node.with_operation:
            return await _run_returned_operation(recovered, environment)
        return recovered
    if isinstance(node, AlternativeNode):
        last_error: Exception | None = None
        for alternative in node.alternatives:
            outcome = await _capture(alternative, value, environment)
            if outcome.is_ok():
                return outcome.ok
            last_error = outcome.error
        if last_error is None:
            raise ValueError("alternative node contains no operations")
        raise last_error
    if isinstance(node, TapNode):
        current = await _evaluate(node.source, value, environment)
        if node.best_effort:
            try:
                await invoke(node.effect, current)
            except Exception:
                pass
        else:
            await invoke(node.effect, current)
        return current
    if isinstance(node, RetryNode):
        return await _retry(node, value, environment)
    if isinstance(node, FanoutNode):
        return await _fanout(node, value, environment)
    if isinstance(node, MapEachNode):
        return await _map_each(node, value)
    if isinstance(node, TraverseNode):
        return await _traverse(node, value, environment)
    if isinstance(node, FilterEachNode):
        return await _filter_each(node, value)
    if isinstance(node, FilterOperationNode):
        return await _filter_operation(node, value, environment)
    if isinstance(node, FoldNode):
        return await _fold(node, value, environment)
    if isinstance(node, FanoutEachNode):
        return await _fanout_each(node, value, environment)
    if isinstance(node, BuildNode):
        return await _build(node, value, environment)
    if isinstance(node, MergeNode):
        return await _merge(node, value, environment)
    if isinstance(node, AssignFieldsNode):
        return await _assign_fields(node, value, environment)
    raise TypeError(f"unsupported operation node {type(node).__name__}")


async def _run_returned_operation(
    returned: Any,
    environment: Environment,
) -> Any:
    from fp_ops.core import UNIT, Operation

    if not isinstance(returned, Operation):
        raise TypeError(
            "operation-returning callback returned "
            f"{type(returned).__name__}, expected Operation"
        )
    _preflight(environment, returned._requirements)
    return await _evaluate(returned._node, UNIT, environment)


async def _retry(
    node: RetryNode,
    value: Any,
    environment: Environment,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, node.attempts + 1):
        outcome = await _capture(node.source, value, environment)
        if outcome.is_ok():
            return outcome.ok
        last_error = outcome.error
        if attempt == node.attempts:
            break

        if isinstance(node.backoff, float):
            delay = node.backoff
        else:
            delay = await invoke(
                cast(Callable[..., Any], node.backoff),
                attempt,
                last_error,
            )
        if isinstance(delay, bool) or not isinstance(delay, (int, float)):
            raise TypeError("retry backoff callable must return a number")
        if delay < 0:
            raise ValueError("retry backoff callable returned a negative delay")
        if delay:
            await asyncio.sleep(float(delay))

    if last_error is None:
        raise RuntimeError("retry reached an impossible empty state")
    raise last_error


async def _cancel_and_join(
    tasks: Iterable[asyncio.Task[Result[Any, Exception]]],
) -> None:
    snapshot = tuple(tasks)
    for task in snapshot:
        if not task.done():
            task.cancel()
    if snapshot:
        await asyncio.gather(*snapshot, return_exceptions=True)


async def _run_concurrent(
    calls: tuple[tuple[Node, Any], ...],
    environment: Environment,
    *,
    limit: int | None,
    fail_fast: bool,
) -> tuple[list[Result[Any, Exception]], tuple[int, Exception] | None]:
    if not calls:
        return [], None

    semaphore = asyncio.Semaphore(limit) if limit is not None else None

    async def call(node: Node, input_value: Any) -> Result[Any, Exception]:
        if semaphore is None:
            return await _capture(node, input_value, environment)
        async with semaphore:
            return await _capture(node, input_value, environment)

    tasks = [
        asyncio.create_task(call(call_node, input_value))
        for call_node, input_value in calls
    ]
    indices = {task: index for index, task in enumerate(tasks)}
    pending: set[asyncio.Task[Result[Any, Exception]]] = set(tasks)
    results: list[Result[Any, Exception] | None] = [None] * len(tasks)

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=(
                    asyncio.FIRST_COMPLETED
                    if fail_fast
                    else asyncio.ALL_COMPLETED
                ),
            )
            for task in sorted(done, key=indices.__getitem__):
                index = indices[task]
                outcome = task.result()
                results[index] = outcome
                if fail_fast and outcome.is_error():
                    await _cancel_and_join(pending)
                    return (
                        [
                            item if item is not None else Error(outcome.error)
                            for item in results
                        ],
                        (index, outcome.error),
                    )
    except BaseException:
        await _cancel_and_join(tasks)
        raise

    return (
        [cast(Result[Any, Exception], item) for item in results],
        None,
    )


async def _fanout(
    node: FanoutNode,
    value: Any,
    environment: Environment,
) -> Any:
    calls = tuple((branch, value) for _, branch in node.branches)
    results, failure = await _run_concurrent(
        calls,
        environment,
        limit=node.limit,
        fail_fast=node.mode == "fail_fast",
    )
    if failure is not None:
        raise failure[1]

    if node.mode == "all_settled":
        values: list[Any] = results
    else:
        values = [_unwrap(result) for result in results]

    if node.named:
        return {
            cast(str, name): branch_value
            for (name, _), branch_value in zip(node.branches, values)
        }
    return tuple(values)


def _collection_items(
    value: Any,
) -> tuple[bool, tuple[Hashable, ...], tuple[Any, ...]]:
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        return True, cast(tuple[Hashable, ...], keys), tuple(value.values())
    if not isinstance(value, Iterable):
        raise TypeError("collection operation input must be iterable")
    items = tuple(value)
    return False, tuple(range(len(items))), items


def _restore_collection(
    mapping: bool,
    keys: tuple[Hashable, ...],
    values: Iterable[Any],
) -> Any:
    snapshot = tuple(values)
    if mapping:
        return dict(zip(keys, snapshot))
    return list(snapshot)


async def _map_each(node: MapEachNode, value: Any) -> Any:
    mapping, keys, items = _collection_items(value)
    output: list[Any] = []
    for key, item in zip(keys, items):
        try:
            output.append(await invoke(node.callback, item))
        except Exception as error:
            raise TraversalItemError(key, error) from error
    return _restore_collection(mapping, keys, output)


async def _traverse(
    node: TraverseNode,
    value: Any,
    environment: Environment,
) -> Any:
    mapping, keys, items = _collection_items(value)
    calls = tuple((node.operation, item) for item in items)

    if node.limit is None:
        results: list[Result[Any, Exception]] = []
        for call_node, item in calls:
            outcome = await _capture(call_node, item, environment)
            results.append(outcome)
            if node.mode == "fail_fast" and outcome.is_error():
                key = keys[len(results) - 1]
                raise TraversalItemError(key, outcome.error) from outcome.error
    else:
        results, failure = await _run_concurrent(
            calls,
            environment,
            limit=node.limit,
            fail_fast=node.mode == "fail_fast",
        )
        if failure is not None:
            index, error = failure
            raise TraversalItemError(keys[index], error) from error

    if node.mode == "all_settled":
        return _restore_collection(mapping, keys, results)

    if node.mode == "lossy":
        kept_keys: list[Hashable] = []
        kept_values: list[Any] = []
        for key, item, outcome in zip(keys, items, results):
            if outcome.is_ok():
                kept_keys.append(key)
                kept_values.append(outcome.ok)
            elif node.on_error is not None:
                try:
                    await invoke(node.on_error, item, outcome.error)
                except Exception as error:
                    raise TraversalItemError(key, error) from error
        return _restore_collection(mapping, tuple(kept_keys), kept_values)

    return _restore_collection(mapping, keys, (_unwrap(item) for item in results))


async def _filter_each(node: FilterEachNode, value: Any) -> Any:
    mapping, keys, items = _collection_items(value)
    kept_keys: list[Hashable] = []
    kept_items: list[Any] = []
    for key, item in zip(keys, items):
        try:
            selected = await invoke(node.predicate, item)
        except Exception as error:
            raise TraversalItemError(key, error) from error
        if not isinstance(selected, bool):
            raise TraversalItemError(
                key,
                TypeError("filter predicate must return bool"),
            )
        if selected:
            kept_keys.append(key)
            kept_items.append(item)
    return _restore_collection(mapping, tuple(kept_keys), kept_items)


async def _filter_operation(
    node: FilterOperationNode,
    value: Any,
    environment: Environment,
) -> Any:
    mapping, keys, items = _collection_items(value)
    kept_keys: list[Hashable] = []
    kept_items: list[Any] = []
    for key, item in zip(keys, items):
        outcome = await _capture(node.predicate, item, environment)
        if outcome.is_error():
            if node.best_effort:
                continue
            raise TraversalItemError(key, outcome.error) from outcome.error
        selected = outcome.ok
        if not isinstance(selected, bool):
            error = TypeError("filter operation must return bool")
            if node.best_effort:
                continue
            raise TraversalItemError(key, error) from error
        if selected:
            kept_keys.append(key)
            kept_items.append(item)
    return _restore_collection(mapping, tuple(kept_keys), kept_items)


async def _fold(
    node: FoldNode,
    value: Any,
    environment: Environment,
) -> Any:
    if not isinstance(value, Iterable):
        raise TypeError("fold input must be iterable")
    accumulator = copy.deepcopy(node.initial)
    for index, item in enumerate(value):
        if node.operation_reducer:
            outcome = await _capture(
                cast(Node, node.reducer),
                (accumulator, item),
                environment,
            )
            if outcome.is_error():
                raise TraversalItemError(index, outcome.error) from outcome.error
            accumulator = outcome.ok
        else:
            try:
                accumulator = await invoke(
                    cast(Callable[..., Any], node.reducer),
                    accumulator,
                    item,
                )
            except Exception as error:
                raise TraversalItemError(index, error) from error
    return accumulator


async def _fanout_each(
    node: FanoutEachNode,
    value: Any,
    environment: Environment,
) -> list[tuple[Any, ...]]:
    if not isinstance(value, Iterable):
        raise TypeError("fanout_each input must be iterable")
    output: list[tuple[Any, ...]] = []
    for index, item in enumerate(value):
        try:
            output.append(await _fanout(node.fanout, item, environment))
        except Exception as error:
            raise TraversalItemError(index, error) from error
    return output


async def _evaluate_field(
    field: FieldSpec,
    value: Any,
    environment: Environment,
) -> Any:
    if field.kind == "operation":
        return await _evaluate(cast(Node, field.value), value, environment)
    if field.kind == "callable":
        return await invoke(cast(Callable[..., Any], field.value), value)
    if field.kind == "nested":
        return await _evaluate(cast(Node, field.value), value, environment)
    return copy.deepcopy(field.value)


async def _capture_field(
    field: FieldSpec,
    value: Any,
    environment: Environment,
) -> Result[Any, Exception]:
    try:
        return Ok(await _evaluate_field(field, value, environment))
    except Exception as error:
        return Error(error)


async def _build(
    node: BuildNode,
    value: Any,
    environment: Environment,
) -> Any:
    built: dict[str, Any] = {}
    settled: dict[str, Result[Any, Exception]] = {}
    for field in node.fields:
        outcome = await _capture_field(field, value, environment)
        if node.mode == "all_settled":
            settled[field.name] = outcome
        elif outcome.is_ok():
            built[field.name] = outcome.ok
        elif node.mode == "fail_fast":
            error = BuildFieldError(field.name, outcome.error)
            raise error from outcome.error

    if node.mode == "all_settled":
        return settled
    if node.model is None:
        return built
    if hasattr(node.model, "model_validate"):
        return node.model.model_validate(built)
    return node.model(**built)


async def _merge_source(
    source: MergeSource,
    value: Any,
    environment: Environment,
) -> Mapping[str, Any]:
    if source.kind == "operation":
        result = await _evaluate(cast(Node, source.value), value, environment)
    elif source.kind == "callable":
        result = await invoke(cast(Callable[..., Any], source.value), value)
    else:
        result = copy.deepcopy(source.value)
    if not isinstance(result, Mapping):
        raise TypeError(
            "shallow merge source returned "
            f"{type(result).__name__}, expected a mapping"
        )
    return cast(Mapping[str, Any], result)


async def _merge(
    node: MergeNode,
    value: Any,
    environment: Environment,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source in node.sources:
        output.update(await _merge_source(source, value, environment))
    return output


async def _assign_fields(
    node: AssignFieldsNode,
    value: Any,
    environment: Environment,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("assign_fields input must be a mapping")
    output = dict(value)
    for field in node.fields:
        outcome = await _capture_field(field, value, environment)
        if outcome.is_error():
            error = BuildFieldError(field.name, outcome.error)
            raise error from outcome.error
        output[field.name] = outcome.ok
    return output
