__all__ = [
    'PhysicsError',
    'ShapeError',
]


class ShapeError(RuntimeError):
    """Raised when the shape of a tensor is invalid for some functions."""
    pass


class PhysicsError(RuntimeError):
    """Raised when some physical assumptions are not met."""
    pass
