__all__ = [
    'ShapeError',
]


class ShapeError(RuntimeError):
    """Raised when the shape of a tensor is invalid for some functions."""
    pass
