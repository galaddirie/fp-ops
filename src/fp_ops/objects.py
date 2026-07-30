"""Object paths, fail-fast/all-settled builds, and shallow assignment."""
# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar, cast, overload

from expression import Result

from fp_ops._ast import (
    AssignFieldsNode,
    BuildNode,
    FieldSpec,
    MergeNode,
    MergeSource,
)
from fp_ops.core import Operation, _merge_requirements
from fp_ops.environment import EnvKey
from fp_ops.result import PathNotFoundError

A = TypeVar("A")
T = TypeVar("T")


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __copy__(self) -> _Missing:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> _Missing:
        return self


MISSING = _Missing()


def _path_parts(path: str | Sequence[str | int]) -> tuple[str | int, ...]:
    if isinstance(path, str):
        if not path:
            return ()
        return tuple(
            int(match.group(1))
            if match.group(1) is not None
            else cast(str, match.group(2))
            for match in re.finditer(r"\[(\d+)\]|([^.\[\]]+)", path)
        )
    return tuple(path)


def get_path(
    path: str | Sequence[str | int],
    default: T | _Missing = MISSING,
) -> Operation[Any, Any | T]:
    """Read a nested mapping key, sequence index, or attribute.

    A present ``None`` is returned unchanged. Only an actually missing path uses
    ``default``; without a default the operation fails with
    :class:`PathNotFoundError`.
    """

    parts = _path_parts(path)
    path_text = path if isinstance(path, str) else ".".join(map(str, path))
    try:
        default_snapshot = copy.deepcopy(default)
    except Exception:
        default_snapshot = default

    def read(data: Any) -> Any | T:
        current = data
        for part in parts:
            if isinstance(current, Mapping):
                if part not in current:
                    break
                current = current[part]
            elif isinstance(part, int) and isinstance(current, Sequence):
                if isinstance(current, (str, bytes)) or not (
                    -len(current) <= part < len(current)
                ):
                    break
                current = current[part]
            elif isinstance(part, str) and hasattr(current, part):
                current = getattr(current, part)
            else:
                break
        else:
            return current

        if default_snapshot is MISSING:
            raise PathNotFoundError(str(path_text))
        return cast(T, default_snapshot)

    from fp_ops.core import operation

    return operation(read)


def _copy_config(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _snapshot_fields(
    schema: Mapping[str, Any],
) -> tuple[tuple[FieldSpec, ...], tuple[EnvKey[object], ...]]:
    if not isinstance(schema, Mapping):
        raise TypeError("schema must be a mapping")
    fields: list[FieldSpec] = []
    requirement_groups: list[tuple[EnvKey[object], ...]] = []
    for field, value in schema.items():
        if not isinstance(field, str):
            raise TypeError("schema field names must be strings")
        if isinstance(value, Operation):
            fields.append(FieldSpec(field, "operation", value._node))
            requirement_groups.append(value._requirements)
        elif isinstance(value, Mapping):
            nested_fields, nested_requirements = _snapshot_fields(value)
            fields.append(
                FieldSpec(
                    field,
                    "nested",
                    BuildNode(
                        nested_fields,
                        mode="fail_fast",
                        model=None,
                    ),
                )
            )
            requirement_groups.append(nested_requirements)
        elif callable(value) and not isinstance(value, type):
            fields.append(FieldSpec(field, "callable", value))
        else:
            fields.append(FieldSpec(field, "constant", _copy_config(value)))
    return tuple(fields), _merge_requirements(*requirement_groups)


@overload
def build(schema: Mapping[str, Any]) -> Operation[Any, dict[str, Any]]: ...


@overload
def build(
    schema: Mapping[str, Any],
    model: type[T],
) -> Operation[Any, T]: ...


def build(
    schema: Mapping[str, Any],
    model: type[T] | None = None,
) -> Operation[Any, dict[str, Any] | T]:
    """Build fields in insertion order and fail with ``BuildFieldError``."""

    if model is not None and not isinstance(model, type):
        raise TypeError("model must be a type")
    fields, requirements = _snapshot_fields(schema)
    return cast(
        Operation[Any, dict[str, Any] | T],
        Operation._from_node(
            BuildNode(fields, mode="fail_fast", model=model),
            name="build",
            requirements=requirements,
        ),
    )


def build_all_settled(
    schema: Mapping[str, Any],
) -> Operation[Any, dict[str, Result[Any, Exception]]]:
    """Build every field and retain a ``Result`` at each field."""

    fields, requirements = _snapshot_fields(schema)
    return cast(
        Operation[Any, dict[str, Result[Any, Exception]]],
        Operation._from_node(
            BuildNode(fields, mode="all_settled", model=None),
            name="build_all_settled",
            requirements=requirements,
        ),
    )


def build_lossy(
    schema: Mapping[str, Any],
) -> Operation[Any, dict[str, Any]]:
    """Build every field and omit fields that fail."""

    fields, requirements = _snapshot_fields(schema)
    return cast(
        Operation[Any, dict[str, Any]],
        Operation._from_node(
            BuildNode(fields, mode="lossy", model=None),
            name="build_lossy",
            requirements=requirements,
        ),
    )


def merge_shallow(
    *sources: (
        Mapping[str, Any]
        | Operation[A, Mapping[str, Any]]
        | Callable[[A], Mapping[str, Any]]
    ),
) -> Operation[A, dict[str, Any]]:
    """Evaluate same-input mapping sources in order and merge them shallowly."""

    snapshots: list[MergeSource] = []
    requirement_groups: list[tuple[EnvKey[object], ...]] = []
    for source in sources:
        if isinstance(source, Operation):
            snapshots.append(MergeSource("operation", source._node))
            requirement_groups.append(source._requirements)
        elif isinstance(source, Mapping):
            snapshots.append(MergeSource("mapping", _copy_config(dict(source))))
        elif callable(source):
            snapshots.append(MergeSource("callable", source))
        else:
            raise TypeError(
                "merge_shallow() sources must be mappings, callables, or operations"
            )
    return cast(
        Operation[A, dict[str, Any]],
        Operation._from_node(
            MergeNode(tuple(snapshots)),
            name="merge_shallow",
            requirements=_merge_requirements(*requirement_groups),
        ),
    )


def assign(
    *sources: (
        Mapping[str, Any]
        | Operation[A, Mapping[str, Any]]
        | Callable[[A], Mapping[str, Any]]
    ),
) -> Operation[A, dict[str, Any]]:
    """Alias for the explicit shallow same-input merge operation."""

    return merge_shallow(*sources)


def assign_fields(
    fields: Mapping[str, Any] | None = None,
    **named_fields: Any,
) -> Operation[Mapping[str, Any], dict[str, Any]]:
    """Copy an input mapping and assign snapshotted fields shallowly."""

    combined: dict[str, Any] = {}
    if fields is not None:
        if not isinstance(fields, Mapping):
            raise TypeError("assign_fields() fields must be a mapping")
        combined.update(fields)
    combined.update(named_fields)
    snapshots, requirements = _snapshot_fields(combined)
    return cast(
        Operation[Mapping[str, Any], dict[str, Any]],
        Operation._from_node(
            AssignFieldsNode(snapshots),
            name="assign_fields",
            requirements=requirements,
        ),
    )


__all__ = [
    "MISSING",
    "assign",
    "assign_fields",
    "build",
    "build_all_settled",
    "build_lossy",
    "get_path",
    "merge_shallow",
]
