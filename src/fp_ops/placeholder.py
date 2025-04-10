from typing import Any, Tuple


class Placeholder:
    """A placeholder object used in operations to represent where the previous result should be inserted."""

    def __repr__(self):
        return "_"


# singleton placeholder instance
_ = Placeholder()