class Placeholder:
    """A placeholder object used in operations to represent where the previous result should be inserted."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self) -> str:
        return "_"


# singleton placeholder instance
_ = Placeholder()