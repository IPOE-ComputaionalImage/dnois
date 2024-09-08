"""
These functions compute convolution for signals defined on an interval or
region symmetric w.r.t. the origin. Correct scale and shift are considered and
thus serve as numerical approximation to continuous convolution.
For example (see :py:func:`dnois.fourier.ft` for more details),

>>> import torch
>>> from dnois.fourier import ft1, ift1, conv1
>>>
>>> f, g = torch.rand(5), torch.rand(5)
>>> c1 = ift1(ft1(f, 0.1) * ft1(g, 0.1), 0.1)
>>> c2 = conv1(f, g, 0.1, real=False)
>>> torch.allclose(c1, c2)
True

.. note::
    If all the signals involved is real-valued, set ``real`` to ``True`` to improve
    the computation and return a real-valued tensor.

.. note::
    By default, these functions compute circular convolution
    because they are implemented by FFT. Set ``pad`` to ``'full'`` to compute
    linear convolution, at the expense of more computation load. See
    `here <https://en.wikipedia.org/wiki/Circular_convolution>`_ for their distinctions.
    Set ``pad`` to ``'full'`` is equivalent to pad the signal to double its
    original size. It also accepts an (or more) :py:class:`int` as padding amount.
"""
import warnings

import torch
import torch.fft as _fft

from dnois.utils.typing import Size2d, Ts, Spacing, size2d

from ._utils import _check_dim, _mul, _pad, _pad_in_dims

__all__ = [
    'conv1',
    'conv2',
    'conv2_mult',
    'conv2_partial',
    'init_conv2',
]


def conv1(
    f: Ts,
    g: Ts,
    dx: Spacing = None,
    dim: int = -1,
    pad: int | str = 0,
    real: bool = None
) -> Ts:
    """
    Computes the 1D convolution between ``f`` and ``g``.

    :param Tensor f: One of the functions to be convolved.
    :param Tensor g: Another function to be convolved.
    :param dx: Sampling spacing, either a float, a 0D tensor, or a tensor broadcastable
        with ``f`` and ``g``. Omitted if ``None`` (default).
    :type: float or Tensor
    :param int dim: The dimension to be transformed. Default: -1.
    :param int or str pad: Padding width at both ends of ``f`` and ``g``.
        ``f`` and ``g`` are periodically
        replicated during DFT-based convolution without padding, which is eliminated if they
        are padded to double of the original lengths. Default: 0.
    :param bool real: If both ``f`` and ``g`` are real-valued and ``real`` is ``True``,
        :py:func:`torch.fft.rfft` can be used to improve computation and a real-valued
        tensor will be returned. If either ``f`` or ``g`` is complex or `real=False`,
        a complex tensor will be returned.
        Default: depending on the dtype of ``f`` and ``g``.
    :return: Convolution between ``f`` and ``g``. Complex if either ``f`` or ``g`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    _check_dim('f', f.shape, (dim,), delta=dx)
    sl = f.size(dim)
    if pad == 'full':
        pad = sl
    if pad != 0:  # pad
        f, g = _pad_in_dims((dim,), (pad, pad), f, g)

    rfft_applicable = not (f.is_complex() or g.is_complex())
    if real is None:
        real = rfft_applicable
    elif real and not rfft_applicable:
        warnings.warn(f'Either f or g is complex but real=True given')
        real = False

    f, g = _fft.ifftshift(f, dim), _fft.ifftshift(g, dim)
    if real:
        c = _fft.irfft(_fft.rfft(f, dim=dim) * _fft.rfft(g, dim=dim), sl, dim)
    else:
        c = _fft.ifft(_fft.fft(f, dim=dim) * _fft.fft(g, dim=dim), dim=dim)
    c = _fft.fftshift(c, dim)

    if pad != 0:
        c = torch.narrow(c, dim, pad, sl)  # crop to original size
    return _mul(c, dx)


def init_conv2(g: Ts, dims: tuple[int, int] = (-2, -1), pad: Size2d = 0, real: bool = None) -> Ts:
    pad = size2d(pad)
    if pad != (0, 0):
        g = _pad(g, dims, (pad[1], pad[1], pad[0], pad[0]))

    rfft_applicable = g.is_complex()
    if real is None:
        real = rfft_applicable
    elif real and not rfft_applicable:
        warnings.warn(f'g is complex but real=True given')
        real = False
    g = _fft.ifftshift(g, dims)
    if real:
        return _fft.rfft(g, dim=dims)
    else:
        return _fft.fft(g, dim=dims)


def conv2(
    f: Ts,
    g: Ts,
    dx: Spacing = None,
    dy: Spacing = None,
    dims: tuple[int, int] = (-2, -1),
    pad: Size2d | str = 0,
    real: bool = None
) -> Ts:
    """
    Computes the 2D convolution between ``f`` and ``g``.

    :param Tensor f: One of the functions to be convolved.
    :param Tensor g: Another function to be convolved.
    :param dx: Sampling spacing in x direction i.e. the second dimension, either a float,
        a 0D tensor, or a tensor broadcastable with ``f`` and ``g``.
        Default: omitted.
    :type: float or Tensor
    :param dy: Sampling spacing in y direction i.e. the first dimension, similar to ``dx``.
        Default: identical to ``dx``.
    :type: float or Tensor
    :param dims: The dimensions to be transformed. Default: (-2, -1).
    :type: tuple[int, int] or str
    :param int pad: Padding width at all four edges of ``f`` and ``g``. ``f`` and ``g`` are periodically
        replicated during DFT-based convolution without padding, which is eliminated if they
        are padded to double of the original widths. Default: 0.
    :param bool real: If both ``f`` and ``g`` are real-valued and ``real`` is ``True``,
        :py:func:`torch.fft.rfft2` can be used to improve computation and a real-valued
        tensor will be returned. If either ``f`` or ``g`` is complex or `real=False`,
        a complex tensor will be returned.
        Default: depending on the dtype of ``f`` and ``g``.
    :return: Convolution between ``f`` and ``g``. Complex if either ``f`` or ``g`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    _check_dim('f', f.shape, dims, dx=dx, dy=dy)
    sl = f.size(dims[0]), f.size(dims[1])
    if pad == 'full':
        pad = sl
    pad = size2d(pad)
    if pad != (0, 0):
        f = _pad(f, dims, (pad[1], pad[1], pad[0], pad[0]))
        g = _pad(g, dims, (pad[1], pad[1], pad[0], pad[0]))

    rfft_applicable = not (f.is_complex() or g.is_complex())
    if real is None:
        real = rfft_applicable
    elif real and not rfft_applicable:
        warnings.warn(f'Either f or g is complex but real=True given')
        real = False

    f, g = _fft.ifftshift(f, dims), _fft.ifftshift(g, dims)
    if real:
        c = _fft.irfft2(_fft.rfft2(f, dim=dims) * _fft.rfft2(g, dim=dims), sl, dims)
    else:
        c = _fft.ifft2(_fft.fft2(f, dim=dims) * _fft.fft2(g, dim=dims), dim=dims)
    c = _fft.fftshift(c, dims)

    if pad != 0:
        for dim, pad_, sl_ in zip(dims, pad, sl):
            c = torch.narrow(c, dim, pad_, sl_)
    return _mul(c, dx, dy)


def conv2_mult(
    f: Ts,
    g: list[Ts],
    dx: Spacing = None,
    dy: Spacing = None,
    dims: tuple[int, int] = (-2, -1),
    pad: Size2d = 0,
    real: bool = None
) -> list[Ts]:
    f_ft = init_conv2(f, dims, pad, real)
    return [conv2_partial(g_, f_ft, dx, dy, dims, pad) for g_ in g]


def conv2_partial(
    f: Ts,
    g_ft: Ts,
    dx: Spacing = None,
    dy: Spacing = None,
    dims: tuple[int, int] = (-2, -1),
    pad: Size2d = 0,
    real_fft: bool = False,
    centered: bool = False,
) -> Ts:
    _check_dim('f', f.shape, dims, dx=dx, dy=dy)
    sl = f.size(dims[0]), f.size(dims[1])
    pad = size2d(pad)
    if pad != (0, 0):
        f = _pad(f, dims, (pad[1], pad[1], pad[0], pad[0]))

    f = _fft.ifftshift(f, dims)
    if real_fft:
        if f.is_complex():
            raise ValueError(f'f is complex but real_fft=True given')
        c = _fft.irfft2(_fft.rfft2(f, dim=dims) * g_ft, sl, dims)
    else:
        if centered:
            g_ft = _fft.ifftshift(g_ft, dims)
        c = _fft.ifft2(_fft.fft2(f, dim=dims) * g_ft, dim=dims)
    c = _fft.fftshift(c, dims)

    if pad != 0:
        for dim, pad_, sl_ in zip(dims, pad, sl):
            c = torch.narrow(c, dim, pad_, sl_)
    return _mul(c, dx, dy)
