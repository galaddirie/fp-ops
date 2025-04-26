from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple


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
class Template:
    """
    Immutable tree of positional / keyword arguments that may contain
    ``Placeholder`` objects.
    """

    args:  Sequence[Any] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict) 

    _pos_indices: Tuple[int, ...] = field(init=False, repr=False)
    _kw_keys: Tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_pos_indices",
            tuple(i for i, v in enumerate(self.args) if v is _),
        )
        object.__setattr__(
            self,
            "_kw_keys",
            tuple(k for k, v in self.kwargs.items() if v is _),
        )

    def is_identity(self) -> bool:
        """Fast-path: the template is exactly one bare “_”."""
        return len(self.args) == 1 and self.args[0] is _ and not self.kwargs

    def render(self, value: Any) -> tuple[Tuple[Any, ...], dict[str, Any]]:
        """Replace every placeholder with *value* and return *(args, kwargs)*."""
        args = list(self.args)
        for i in self._pos_indices:
            args[i] = value

        kwargs = dict(self.kwargs)
        for k in self._kw_keys:
            kwargs[k] = value

        return tuple(args), kwargs

    @staticmethod
    def from_call(
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> "Template":
        """Factory used by ``Operation.__call__``."""
        return Template(tuple(args), dict(kwargs))
