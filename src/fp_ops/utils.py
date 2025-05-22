from typing import Any


from fp_ops.placeholder import Placeholder


def _contains_ph(x: Any) -> bool:
    if isinstance(x, Placeholder):
        return True
    if isinstance(x, (list, tuple)):
        return any(_contains_ph(v) for v in x)
    if isinstance(x, dict):
        return any(_contains_ph(v) for v in x.values())
    return False


def _fill(template: Any, repl: Any) -> Any:
    """Replace placeholders inside *template* with *repl* (one-level recurse)."""
    if isinstance(template, Placeholder):
        return repl
    if isinstance(template, list):
        return [_fill(v, repl) for v in template]
    if isinstance(template, tuple):
        return tuple(_fill(v, repl) for v in template)
    if isinstance(template, dict):
        return {k: _fill(v, repl) for k, v in template.items()}
    return template

