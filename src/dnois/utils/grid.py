import torch

from ..base.typing import Ts, Sequence, is_scalar

__all__ = [
    'grid',
    'interval',
]


def _reshape(x: Ts, n: int, idx: int) -> Ts:
    x = x.reshape(*x.shape, *[1 for _ in range(n - 1)])
    x = x.transpose(idx - n, -n)
    return x


def interval(
    n: int, spacing: float | Ts = None, center: float | Ts = None, symmetric: bool = False, **kwargs
) -> Ts:
    """
    Create a 1D evenly spaced grid.

    .. testsetup::

        from dnois.utils import *
        import torch
        torch.set_printoptions(precision=2)

    .. doctest::
        :options: +NORMALIZE_WHITESPACE

        >>> interval(3, 0.1)
        tensor([-0.10, 0.00, 0.10])
        >>> interval(3, torch.tensor([0.1, 0.2]))
        tensor([[-0.10, 0.00, 0.10],
                [-0.20, 0.00, 0.20]])
        >>> interval(3, torch.tensor([0.1, 0.2]), torch.tensor([-1, 1]))
        tensor([[-1.10, -1.00, -0.90],
                [ 0.80,  1.00,  1.20]])
        >>> interval(4)
        tensor([-2., -1., 0., 1.])
        >>> interval(4, symmetric=True)
        tensor([-1.50, -0.50, 0.50, 1.50])

    :param int n: Number of grid points.
    :param spacing: Spacing between grid points. If a tensor with shape ``(...)``,
        the returned tensor will have shape ``(..., n)``. Default: 1.
    :type spacing: float or Tensor.
    :param center: Center of resulted grid points. If a tensor with shape ``(...)``,
        the returned tensor will have shape ``(..., n)``. Default: 0.
    :type center: float or Tensor.
    :param bool symmetric: If ``True``, grid points are symmetric w.r.t. ``center``.
        Otherwise, ``n // 2`` points are smaller, ``n // 2 - 1`` points are larger
        and one point is ``center`` value. Only matters when ``n`` is even. Default: ``False``.
    :keyword kwargs: Tensor creation arguments passes to :py:func:`torch.linspace` like ``device``.
    :return: A tensor of shape ``(n,)`` if ``spacing`` and ``center`` are both scalars,
        otherwise of shape ``(*<broadcast shape of spacing and center>, n)``.
    :rtype: Tensor
    """
    if n % 2 == 1 or symmetric:
        x = torch.linspace(-(n - 1) / 2, (n - 1) / 2, n, **kwargs)
    else:
        x = torch.linspace(-n / 2, n / 2 - 1, n, **kwargs)
    if spacing is not None:
        if is_scalar(spacing):
            x = x * spacing
        else:  # >1D tensor
            x = x * spacing.unsqueeze(-1)
    if center is not None:
        if is_scalar(center):
            x = x + center
        else:
            x = x + center.unsqueeze(-1)
    return x


def grid(
    n: Sequence[int],
    spacing: float | Ts | Sequence[float | Ts] = None,
    center: float | Ts | Sequence[float | Ts] = None,
    symmetric: bool = False,
    broadcast: bool = True,
    **kwargs
) -> list[Ts]:
    """
    Create a ``len(n)``-D evenly spaced grid.

    .. testsetup::

        from dnois.utils import *
        import torch
        torch.set_printoptions(precision=2)

    .. doctest::
        :options: +NORMALIZE_WHITESPACE

        >>> grid((2, 3), 0.1)
        [tensor([[-0.10, -0.10, -0.10],
                 [ 0.00,  0.00,  0.00]]),
         tensor([[-0.10,  0.00,  0.10],
                 [-0.10,  0.00,  0.10]])]
        >>> grid((2, 3), 0.1, 1.)
        [tensor([[0.90, 0.90, 0.90],
                 [1.00, 1.00, 1.00]]),
         tensor([[0.90, 1.00, 1.10],
                 [0.90, 1.00, 1.10]])]
        >>> grid((2, 3), (0.1, 0.2), (-1., 1.))
        [tensor([[-1.10, -1.10, -1.10],
                 [-1.00, -1.00, -1.00]]),
         tensor([[0.80, 1.00, 1.20],
                 [0.80, 1.00, 1.20]])]
        >>> grid((2, 3), torch.tensor([0.1, 0.2]), torch.tensor([1., 2.]))
        [tensor([[[0.90, 0.90, 0.90],
                  [1.00, 1.00, 1.00]],
        <BLANKLINE>
                 [[1.80, 1.80, 1.80],
                  [2.00, 2.00, 2.00]]]),
         tensor([[[0.90, 1.00, 1.10],
                  [0.90, 1.00, 1.10]],
        <BLANKLINE>
                 [[1.80, 2.00, 2.20],
                  [1.80, 2.00, 2.20]]])]
        >>> grid((2, 3), symmetric=True)
        [tensor([[-0.50, -0.50, -0.50],
                 [ 0.50,  0.50,  0.50]]),
         tensor([[-1.,  0.,  1.],
                 [-1.,  0.,  1.]])]
        >>> grid((2, 3), broadcast=False)
        [tensor([[-1.],
                 [ 0.]]),
         tensor([[-1.,  0.,  1.]])]

    :param Sequence[int] n: Number of grid points in each dimension.
    :param spacing: Spacing between grid points in each dimension.
        A single ``float`` or 0D tensor indicates the spacing for all dimensions. Default: 1.
    :type spacing: float or Tensor or Sequence[float | Tensor]
    :param center: Center of resulted grid points in each dimension.
        A single ``float`` or 0D tensor indicates the center for all dimensions. Default: 0.
    :type center: float or Tensor or Sequence[float | Tensor]
    :param bool symmetric: See :py:func:`interval`. Default: ``False``.
    :param bool broadcast: Whether to broadcast resulted tensors. Default: ``True``.
    :param kwargs: Tensor creation arguments passes to :py:func:`torch.linspace`.
    :return: A list of ``len(n)``-D tensors if ``spacing`` and ``center`` are both scalars,
        otherwise of shape ``(*<broadcast shape of spacing and center>, *n)``.
        See above examples.
    :rtype: list[Tensor]
    """
    dims = len(n)
    if not isinstance(spacing, Sequence):
        spacing = [spacing for _ in range(dims)]
    elif len(spacing) != dims:
        raise ValueError(f'Given dims={dims} but number of grid spacings is {len(spacing)}')
    if not isinstance(center, Sequence):
        center = [center for _ in range(dims)]
    elif len(center) != dims:
        raise ValueError(f'Given dims={dims} but number of offsets is {len(center)}')

    g = [
        _reshape(interval(n[i], spacing[i], center[i], symmetric, **kwargs), dims, i)
        for i in range(dims)
    ]
    if broadcast:
        g = list(torch.broadcast_tensors(*g))
    return g
