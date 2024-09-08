r"""
.. testsetup::

    import torch
    from dnois.fourier import ft1

These functions compute fourier transform or its inverse for signals
defined on an interval or region which is symmetric w.r.t. the origin.
They differ from vanilla Discrete Fourier Transform (DFT) provided by
:py:mod:`torch.fft` in that correct scale and shift are considered and
thus serve as numerical approximation to continuous fourier transform.

For example, the function :py:func:`ft1` computes the 1D Fourier transform of signal
:math:`f`. Given signal length :math:`N`, assuming :math:`f[k]` represents sampled values on
points :math:`\{k\delta_f\}_{k=-\lfloor N/2\rfloor}^{\lfloor(N-1)/2\rfloor}`
where :math:`\delta_f` is sampling interval, then  :py:func:`ft1`  computes

.. math::
    \ft\{f\}[l]=h[l]=\sum_{k=-\lfloor N/2\rfloor}^{\lfloor(N-1)/2\rfloor}
    f[k]\e^{-\i 2\pi k\delta_f\times l\delta_h}\delta_f,
    l=-\lfloor\frac{N}{2}\rfloor,\cdots,\lfloor\frac{N-1}{2}\rfloor

where :math:`\delta_h=1/(N\delta_f)` is sampling interval in frequency domain.
Indices :math:`-\lfloor\frac{N}{2}\rfloor, \cdots,\lfloor\frac{N-1}{2}\rfloor`
for :math:`k` and :math:`l` correspond to ``0``, ..., ``N-1`` in the given array.
In other words, this function works like

>>> from torch.fft import fft, fftshift, ifftshift
>>>
>>> f = torch.rand(9)
>>> h1 = ft1(f, 0.1)
>>> h2 = fftshift(fft(ifftshift(f))) * 0.1
>>> torch.allclose(h1, h2)
True

.. note::

    The sampling interval (like ``delta`` argument for :py:func:`ft1`) will be
    multiplied to transformed signal if given, so it can be a :py:class:`float`
    or a tensor with shape broadcastable to original signal. But if it is not a
    0D tensor, its size on the dimensions to be transformed must be 1.
"""
# TODO: spectrum-shift FT
import torch
import torch.fft as _fft

from dnois.utils.typing import Ts, Spacing

from ._utils import _check_dim, _div, _mul

__all__ = [
    'ft1',
    'ift1',
    'ft2',
    'ift2',
]


def ft1(f: Ts, delta: Spacing = None, dim: int = -1) -> Ts:
    """
    Computes the 1D Fourier transform of signal ``f``.

    :param Tensor f: The function to be transformed.
    :param delta: Sampling spacing, either a float, a 0D tensor, or a tensor broadcastable
        with ``f``. Omitted if ``None`` (default).
    :type: float or Tensor
    :param int dim: The dimension to be transformed. Default: -1.
    :return: Fourier transform of ``f``.
    :rtype: Tensor
    """
    _check_dim('f', f.shape, (dim,), delta=delta)
    h = _fft.fftshift(_fft.fft(_fft.ifftshift(f, dim), dim=dim), dim)
    return _mul(h, delta)


def ift1(h: Ts, delta: Spacing = None, dim: int = -1) -> Ts:
    """
    Computes the 1D inverse Fourier transform of spectrum ``h``.

    :param Tensor h: The spectrum to be inverse transformed.
    :param delta: Sampling spacing in original domain, either a float, a 0D tensor,
        or a tensor broadcastable with ``h``. Omitted if ``None`` (default).
    :type: float or Tensor
    :param int dim: The dimension to be transformed. Default: -1.
    :return: Inverse Fourier transform of ``h``.
    :rtype: Tensor
    """
    _check_dim('h', h.shape, (dim,), delta=delta)
    f = _fft.fftshift(_fft.ifft(_fft.ifftshift(h, dim), dim=dim), dim)
    return _div(f, delta)


def ft2(f: Ts, dx: Spacing = None, dy: Spacing = None, dims: tuple[int, int] = (-2, -1)) -> Ts:
    """
    Computes the 2D Fourier transform of signal ``f``.

    :param Tensor f: The function to be transformed.
    :param dx: Sampling spacing in x direction i.e. the second dimension, either a float,
        a 0D tensor, or a tensor broadcastable with ``f``.
        Default: omitted.
    :type: float or Tensor
    :param dy: Sampling spacing in y direction i.e. the first dimension, similar to ``dx``.
        Default: identical to ``dx``.
    :type: float or Tensor
    :param dims: The dimensions to be transformed. Default: (-2, -1).
    :type: tuple[int, int]
    :return: Fourier transform of ``f``.
    :rtype: Tensor
    """
    if dy is None:
        dy = dx
    _check_dim('f', f.shape, dims, dx=dx, dy=dy)
    h = _fft.fftshift(_fft.fft2(_fft.ifftshift(f, dims), dim=dims), dims)
    return _mul(h, dx, dy)


def ift2(h: Ts, dx: Spacing = None, dy: Spacing = None, dims: tuple[int, int] = (-2, -1)) -> Ts:
    """
    Computes the 2D inverse Fourier transform of signal ``h``.

    :param Tensor h: The function to be inverse transformed.
    :param dx: Sampling spacing in x direction i.e. the second dimension of original domain,
        either a float, a 0D tensor, or a tensor broadcastable with ``h``.
        Default: omitted.
    :type: float or Tensor
    :param dy: Sampling spacing in y direction i.e. the first dimension, similar to ``dx``.
        Default: identical to ``dx``.
    :type: float or Tensor
    :param dims: The dimensions to be transformed. Default: (-2, -1).
    :type: tuple[int, int]
    :return: Inverse Fourier transform of ``h``.
    :rtype: Tensor
    """
    if dy is None:
        dy = dx
    _check_dim('h', h.shape, dims, dx=dx, dy=dy)
    f = _fft.fftshift(_fft.ifft2(_fft.ifftshift(h, dims), dim=dims), dims)
    return _div(f, dx, dy)
