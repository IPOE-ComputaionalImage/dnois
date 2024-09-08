import torch

from .typing import Ts, Sequence, cast, is_scalar

__all__ = [
    'sym_interval',
    'sym_grid',
]


def _reshape(x: Ts, n: int, idx: int) -> Ts:
    x = x.reshape(*x.shape, *[1 for _ in range(n - 1)])
    x = x.transpose(idx - n, -n)
    return x


def sym_interval(
    n: int,
    spacing: float | Ts = None,
    symmetric_even_grid: bool = False,
    **kwargs
) -> Ts:
    """
    Create a 1D evenly spaced grid centered at the origin.

    .. testsetup::

        from dnois.utils import *
        import torch
        torch.set_printoptions(precision=2)

    .. doctest::
        :options: +NORMALIZE_WHITESPACE

        >>> sym_interval(3, 0.1)
        tensor([-0.10, 0.00, 0.10])
        >>> sym_interval(3, torch.tensor([0.1, 0.2]))
        tensor([[-0.10, 0.00, 0.10],
                [-0.20, 0.00, 0.20]])
        >>> sym_interval(4)
        tensor([-2., -1., 0., 1.])
        >>> sym_interval(4, symmetric_even_grid=True)
        tensor([-1.50, -0.50, 0.50, 1.50])

    :param int n: Number of grid points.
    :param spacing: Spacing between grid points. Default: 1.
    :type spacing: float or Tensor.
    :param bool symmetric_even_grid: If ``True``, grid points are symmetric w.r.t. 0.
        Otherwise, ``n // 2`` points are negative, ``n // 2 - 1`` points are positive
        and one point is zero. Only valid when ``n`` is even. Default: ``False``.
    :param kwargs: Tensor creation arguments passes to :py:func:`torch.linspace`.
    :return: A tensor of shape ``(n,)`` if ``spacing`` is a scalar,
        otherwise of shape ``(*spacing.shape, n)``.
    :rtype: Tensor
    """
    if n % 2 == 1 or symmetric_even_grid:
        x = torch.linspace(-(n - 1) / 2, (n - 1) / 2, n, **kwargs)
    else:
        x = torch.linspace(-n / 2, n / 2 - 1, n, **kwargs)
    if spacing is None:
        return x
    elif is_scalar(spacing):
        return x * spacing
    else:  # >1D tensor
        return x * spacing.unsqueeze(-1)


def sym_grid(
    dims: int,
    n: int | Sequence[int],
    spacing: float | Ts | Sequence[float | Ts] = None,
    symmetric_even_grid: bool = False,
    **kwargs
) -> list[Ts]:
    """
    Create a ``dims``-D evenly spaced grid centered at the origin.

    .. testsetup::

        from dnois.utils import *
        import torch
        torch.set_printoptions(precision=2)

    .. doctest::
        :options: +NORMALIZE_WHITESPACE

        >>> sym_grid(2, (2, 3), 0.1)
        [tensor([[-0.10],
                 [ 0.00]]),
         tensor([[-0.10, 0.00, 0.10]])]
        >>> sym_grid(2, (2, 3), 0.1, True)
        [tensor([[-0.05],
                 [ 0.05]]),
         tensor([[-0.10, 0.00, 0.10]])]
        >>> sym_grid(2, (2, 3), (0.1, 0.2))
        [tensor([[-0.10],
                 [ 0.00]]),
         tensor([[-0.20, 0.00, 0.20]])]

    :param int dims: Number of dimensions.
    :param n: Number of grid points in each dimension.
    :type n: int or Sequence[int]
    :param spacing: Spacing between grid points in each dimension. Default: 1.
    :type spacing: float or Tensor or Sequence[float | Tensor]
    :param bool symmetric_even_grid: See :py:func:`sym_interval`. Default: ``False``.
    :param kwargs: Tensor creation arguments passes to :py:func:`torch.linspace`.
    :return: A list of ``dims``-D tensors if ``spacing`` s are all scalar (see example
        above), with only one dimension is not trivial so that they can be broadcast
        together. Otherwise, their shapes are determined by their multiplication with
        corresponding ``spacing``.
    :rtype: list[Tensor]
    """
    if not isinstance(n, Sequence):
        n = [n] * dims
    elif len(n) != dims:
        raise ValueError(f'Given dims={dims} but length of grid size is {len(n)}')
    if not isinstance(spacing, Sequence):
        spacing = [spacing for _ in range(dims)]
    elif len(spacing) != dims:
        raise ValueError(f'Given dims={dims} but length of grid spacing is {len(spacing)}')
    return [
        _reshape(sym_interval(n[i], spacing[i], symmetric_even_grid, **kwargs), dims, i)
        for i in range(dims)
    ]
