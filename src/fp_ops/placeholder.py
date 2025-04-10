from typing import Any, Tuple


class Placeholder:
    """A placeholder object used in operations to represent where the previous result should be inserted."""

    def __repr__(self):
        return "_"


# Create a singleton placeholder instance
_ = Placeholder()


# Add these methods to the Operation class
def _has_placeholders(self) -> bool:
    """
    Check if this operation has placeholders in its bound arguments.

    This checks recursively through nested data structures.
    """
    return self._contains_placeholder(self.bound_args) or self._contains_placeholder(
        self.bound_kwargs
    )


def _contains_placeholder(self, obj: Any) -> bool:
    """
    Check if an object contains any Placeholder instances.

    This recursively checks lists, tuples, and dictionaries.

    Args:
        obj: The object to check.

    Returns:
        True if obj contains a Placeholder, False otherwise.
    """
    if isinstance(obj, Placeholder):
        return True

    if isinstance(obj, (list, tuple)):
        return any(self._contains_placeholder(item) for item in obj)

    if isinstance(obj, dict):
        return any(self._contains_placeholder(key) for key in obj) or any(
            self._contains_placeholder(value) for value in obj.values()
        )

    return False


def _substitute_placeholders(self, value: Any) -> Tuple[tuple, dict]:
    """
    Return new bound_args and bound_kwargs with placeholders substituted.

    This recursively substitutes placeholders in nested data structures.

    Args:
        value: The value to substitute for placeholders.

    Returns:
        A tuple of (new_args, new_kwargs) with placeholders substituted.
    """
    new_args = tuple(
        self._substitute_placeholder(arg, value) for arg in self.bound_args or ()
    )

    new_kwargs = {}
    if self.bound_kwargs:
        new_kwargs = {
            self._substitute_placeholder(key, value): self._substitute_placeholder(
                val, value
            )
            for key, val in self.bound_kwargs.items()
        }

    return new_args, new_kwargs


def _substitute_placeholder(self, obj: Any, value: Any) -> Any:
    """
    Substitute all Placeholder instances with the given value.

    This recursively processes lists, tuples, and dictionaries.

    Args:
        obj: The object to process.
        value: The value to substitute for placeholders.

    Returns:
        A new object with all placeholders replaced by the value.
    """
    if isinstance(obj, Placeholder):
        return value

    if isinstance(obj, list):
        return [self._substitute_placeholder(item, value) for item in obj]

    if isinstance(obj, tuple):
        return tuple(self._substitute_placeholder(item, value) for item in obj)

    if isinstance(obj, dict):
        return {
            self._substitute_placeholder(key, value): self._substitute_placeholder(
                val, value
            )
            for key, val in obj.items()
        }

    return obj
