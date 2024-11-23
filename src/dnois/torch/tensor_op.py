import torch

from ..base.typing import Ts, Sequence, overload

__all__ = [
    'as1d',
    'broadcastable',
]


def as1d(x: Ts, ndim: int = 1, dim: int = -1) -> Ts:
    """
    Transforms ``x`` into a ``ndim``-D tensor but arranges its all elements into dimension ``dim``.

    .. testsetup::

        import torch
        from dnois.torch import as1d

    >>> x = torch.arange(4)
    >>> as1d(x)
    tensor([0, 1, 2, 3])

    :param Tensor x: A tensor with any shape.
    :param int ndim: Number of dimensions of resulted tensor.
    :param dim: The dimension to place elements in.
    :return: The transformed tensor.
    :rtype: Tensor
    """
    shape = [1 for _ in range(ndim)]
    shape[dim] = -1
    return x.reshape(shape)


@overload
def broadcastable(*tensors: Ts) -> bool:
    pass


@overload
def broadcastable(*shapes: Sequence[int]) -> bool:
    pass


def broadcastable(*tensors_or_shapes: Sequence[int] | Ts) -> bool:
    r"""
    Check whether some tensors or some tensor shapes are broadcastable.

    :param tensors_or_shapes: Some tensors or some tensor shapes.
    :return: Whether they are broadcastable.
    :rtype: bool
    """
    is_tensor = all(torch.is_tensor(x) for x in tensors_or_shapes)
    is_shape = all(isinstance(x, Sequence) for x in tensors_or_shapes)
    if not is_tensor and not is_shape:
        raise TypeError(f'Arguments of {broadcastable.__name__} must be all tensors or all shapes, '
                        f'but got types {[type(x) for x in tensors_or_shapes]}')
    if is_tensor:
        shapes = [t.shape for t in tensors_or_shapes]
    else:
        shapes = tensors_or_shapes
    try:
        torch.broadcast_shapes(*shapes)
    except RuntimeError:
        return False
    else:
        return True
