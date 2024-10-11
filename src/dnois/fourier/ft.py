# TODO: spectrum-shift FT
import torch.fft as _fft

from dnois.base.typing import Ts, Spacing

from ._utils import _check_dim, _mul

__all__ = [
    'ft1',
    'ift1',
    'ft2',
    'ift2',
]


def _div(x: Ts, *fs: Spacing) -> Ts:
    for f in fs:
        if f is not None:
            x = x / f
    return x


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
