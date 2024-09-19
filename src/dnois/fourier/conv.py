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

from dnois.base.typing import Size2d, Ts, Spacing, ConvOut, size2d

from ._utils import _check_dim, _mul, _pad, _pad_in_dims

__all__ = [
    'conv1',
    'conv2',
    'conv2_mult',
    'conv2_partial',
    'init_conv2',
    'lconv1',
    'lconv2',
]


def _dtm_real(f1: Ts, f2: Ts, real: bool) -> bool:
    rfft_applicable = not (f1.is_complex() or f2.is_complex())
    if real is None:
        return rfft_applicable
    elif real and not rfft_applicable:
        warnings.warn(f'Either f1 or f2 is complex but real=True given')
        return False
    else:
        return False


def _tail_pad(f: Ts, dims: tuple[int, ...], paddings: tuple[int, ...]) -> Ts:
    for dim, padding in zip(dims, paddings):
        shape = list(f.shape)
        shape[dim] = padding
        f = torch.cat((f, f.new_zeros(shape)), dim=dim)
    return f


def lconv1(
    f1: Ts,
    f2: Ts,
    dim: int = -1,
    out: ConvOut = 'full',
    real: bool = None,
) -> Ts:
    r"""
    Computes 1D linear convolution for two sequences :math:`f_1` and :math:`f_2`,
    implemented by FFT and appropriate padding is applied automatically
    to avoid circular convolution.

    This function is similar to |scipy_signal_fftconvolve|_ in 1D case.

    :param Tensor f1: The first sequence :math:`f_1` with length ``N`` in dimension ``dim``.
    :param Tensor f2: The second sequence :math:`f_2` with length ``M`` in dimension ``dim``.
    :param int dim: The dimension to be transformed. Default: -1.
    :param str out: One of the following options. Default: ``full``.

        ``full``
            Return complete result so the size of dimension ``dim`` is :math:`N+M-1`.

        ``same``
            Return the middle segment of result. In other words, drop the first
            ``min(N, M) // 2`` elements and the last ``(min(N, M) - 1) // 2`` elements
            so the size of dimension ``dim`` is :math:`\max(N, M)`.

        ``valid``
            Return only the segments where ``f1`` and ``f2`` overlap completely so
            the size of dimension ``dim`` is :math:`\max(N,M)-\min(N,M)+1`.
    :param bool real: If both ``f1`` and ``f2`` are real-valued and ``real`` is ``True``,
        :py:func:`torch.fft.rfft` can be used to improve computation and a real-valued
        tensor will be returned. If either ``f1`` or ``f2`` is complex or `real=False`,
        a complex tensor will be returned.
        Default: depending on the dtype of ``f1`` and ``f2``.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor

    .. |scipy_signal_fftconvolve| replace:: scipy.signal.fftconvolve
    .. _scipy_signal_fftconvolve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html#scipy.signal.fftconvolve
    """
    real = _dtm_real(f1, f2, real)
    n, m = f1.size(dim), f2.size(dim)
    min_l, max_l = min(n, m), max(n, m)
    sl = m + n - 1

    if real:
        g: Ts = _fft.irfft(_fft.rfft(f1, sl, dim) * _fft.rfft(f2, sl, dim), sl, dim)
    else:
        g: Ts = _fft.ifft(_fft.fft(f1, sl, dim) * _fft.fft(f2, sl, dim), sl, dim)

    if out == 'full':
        return g
    elif out == 'same':
        return g.narrow(dim, min_l // 2, max_l)
    elif out == 'valid':
        return g.narrow(dim, min_l - 1, max_l - min_l + 1)
    else:
        raise ValueError(f'Unknown output type: {out}')


def lconv2(
    f1: Ts,
    f2: Ts,
    dims: tuple[int, int] = (-2, -1),
    out: ConvOut = 'full',
    real: bool = None,
) -> Ts:
    r"""
    Computes 2D linear convolution for two 2D arrays :math:`f_1` and :math:`f_2`,
    implemented by FFT and appropriate padding is applied automatically
    to avoid circular convolution.

    This function is similar to |scipy_signal_fftconvolve|_ in 2D case.

    :param Tensor f1: The first array :math:`f_1` with size ``N1, N2`` in dimension ``dims``.
    :param Tensor f2: The second array :math:`f_2` with size ``M1, M2`` in dimension ``dims``.
    :param dims: The dimensions to be transformed. Default: ``(-2,-1)``.
    :type dims: tuple[int, int]
    :param str out: One of the following options. Default: ``full``. In each dimension:

        ``full``
            Return complete result so the size of each ``dims`` is :math:`N+M-1`.

        ``same``
            Return the middle segment of result. In other words, drop the first
            ``min(N, M) // 2`` elements and the last ``(min(N, M) - 1) // 2`` elements
            so the size of each ``dims`` is :math:`\max(N, M)`.

        ``valid``
            Return only the segments where ``f1`` and ``f2`` overlap completely so
            the size of each ``dims`` is :math:`\max(N,M)-\min(N,M)+1`.
    :param bool real: If both ``f1`` and ``f2`` are real-valued and ``real`` is ``True``,
        :py:func:`torch.fft.rfft` can be used to improve computation and a real-valued
        tensor will be returned. If either ``f1`` or ``f2`` is complex or `real=False`,
        a complex tensor will be returned.
        Default: depending on the dtype of ``f1`` and ``f2``.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor

    .. |scipy_signal_fftconvolve| replace:: scipy.signal.fftconvolve
    .. _scipy_signal_fftconvolve: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html#scipy.signal.fftconvolve

    """
    real = _dtm_real(f1, f2, real)
    (n1, n2), (m1, m2) = (f1.size(dims[0]), f1.size(dims[1])), (f2.size(dims[0]), f2.size(dims[1]))
    min_l1, min_l2, max_l1, max_l2 = min(n1, m1), min(n2, m2), max(n1, m2), max(n1, m2)
    sl = (n1 + m1 - 1, n2 + m2 - 1)

    if real:
        g: Ts = _fft.irfft2(_fft.rfft2(f1, sl, dims) * _fft.rfft2(f2, sl, dims), sl, dims)
    else:
        g: Ts = _fft.ifft2(_fft.fft2(f1, sl, dims) * _fft.fft2(f2, sl, dims), sl, dims)

    if out == 'full':
        return g
    elif out == 'same':
        return g.narrow(dims[0], min_l1 // 2, max_l1).narrow(dims[1], min_l2 // 2, max_l2)
    elif out == 'valid':
        g = g.narrow(dims[0], min_l1 - 1, max_l1 - min_l1 + 1)
        g = g.narrow(dims[1], min_l2 - 1, max_l2 - min_l2 + 1)
        return g
    else:
        raise ValueError(f'Unknown output type: {out}')


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
    :param pad: Padding width at both ends of ``f`` and ``g``. ``pad`` can be set to
        an integer ranging from 0 to the size of dimension
        (or equivalently, a str ``full``) ``dim`` to mitigate
        aliasing artifact caused by FFT. It computes circular convolution when
        ``pad`` is 0 and linear convolution when ``pad`` is ``full``.
    :type pad: int or str
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
    real = _dtm_real(f, g, real)
    if pad == 'full':
        pad = sl // 2
    if pad != 0:  # pad
        f, g = _pad_in_dims((dim,), (pad, pad), f, g)

    f, g = _fft.ifftshift(f, dim), _fft.ifftshift(g, dim)
    if real:
        c = _fft.irfft(_fft.rfft(f, dim=dim) * _fft.rfft(g, dim=dim), sl, dim)
    else:
        c = _fft.ifft(_fft.fft(f, dim=dim) * _fft.fft(g, dim=dim), dim=dim)
    c = _fft.fftshift(c, dim)

    if pad != 0:
        c = torch.narrow(c, dim, pad, sl)  # crop to original size
    return _mul(c, dx)


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
    :param pad: Padding width at both ends of ``f`` and ``g``. ``pad`` can be set to
        an integer ranging from 0 to the size of dimension
        (or equivalently, a str ``full``) ``dim`` to mitigate
        aliasing artifact caused by FFT. It computes circular convolution when
        ``pad`` is 0 and linear convolution when ``pad`` is ``full``.
    :type pad: int or str
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
    real = _dtm_real(f, g, real)
    if pad == 'full':
        pad = (sl[0] // 2, sl[1] // 2)
    pad = size2d(pad)
    if pad != (0, 0):
        f = _pad(f, dims, (pad[1], pad[1], pad[0], pad[0]))
        g = _pad(g, dims, (pad[1], pad[1], pad[0], pad[0]))

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
