"""
These functions compute convolution between two signals ``f1`` and ``f2``.
:py:func:`dconv1` and :py:func:`dconv2` computes discrete convolution.
Numerical issues and correct scale are considered 
in :py:func:`conv1` and :py:func:`conv2` so they
serve as numerical approximation to continuous convolution.

.. note::
    If all the signals involved is real-valued, set ``real`` to ``True`` to improve
    the computation and return a real-valued tensor.
"""
# TODO: doctest
import warnings

import torch
import torch.fft as _fft

from dnois.base.typing import Size2d, Ts, Spacing, ConvOut, size2d

from ._utils import _check_dim, _mul, _pad

__all__ = [
    'conv1',
    'conv2',
    'dconv1',
    'dconv2',
    # 'conv2_mult',
    # 'conv2_partial',
    # 'init_conv2',
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


def _dtm_padding(padding: int | str, linear_value: int) -> int:
    if padding == 'none':
        return 0
    elif padding == 'linear':
        return linear_value
    elif padding < 0:
        warnings.warn('Got negative padding, setting padding to zero')
        return 0
    elif padding > linear_value:
        warnings.warn(f'Got padding too large, setting padding to {linear_value}')
        return linear_value
    else:
        return padding


def _narrow_periodic(ts: Ts, dim: int, offset: int, target_len: int) -> Ts:
    sl = ts.size(dim)
    if sl < target_len + offset:
        return torch.cat((
            ts.narrow(dim, offset, sl - offset), ts.narrow(dim, 0, target_len + offset - sl)
        ), dim=dim)
    else:
        return ts.narrow(dim, offset, target_len)


def _simpson_weights(n: int, device, dtype) -> Ts:
    # odd: (1, 4, 2, 4, 2, ..., 4, 1) / 3
    # even: (1, 4, 2, 4, 2, ..., 4, 1, 3) / 3
    if n < 3:
        raise ValueError(f'n={n} is too small')
    s = torch.ones(n, dtype=dtype, device=device)
    s[1:-1:2] = 4
    s[2:-2:2] = 2
    if n % 2 == 0:
        s[-1] = 3
    s /= 3
    return s


def _simpson(f1: Ts, f2: Ts, dim: int) -> tuple[Ts, Ts]:
    if f1.size(dim) >= f2.size(dim):
        shape = [1 for _ in f1.shape]
        shape[dim] = -1
        f1 = f1 * _simpson_weights(f1.size(dim), f1.device, f1.dtype).reshape(shape)
    else:
        shape = [1 for _ in f2.shape]
        shape[dim] = -1
        f2 = f2 * _simpson_weights(f2.size(dim), f2.device, f2.dtype).reshape(shape)
    return f1, f2


def dconv1(
    f1: Ts,
    f2: Ts,
    dim: int = -1,
    out: ConvOut = 'full',
    padding: int | str = 'linear',
    real: bool = None,
) -> Ts:
    r"""
    Computes 1D discrete convolution for two sequences :math:`f_1` and :math:`f_2` utilizing FFT.

    The lengths of :math:`f_1`, :math:`N` and :math:`f_2`, :math:`M` can be different.
    In that case, the shorter sequence will be padded first to match length of the longer one.
    Notably, it is circular convolution that is computed without additional padding
    (if ``padding`` is ``0`` or ``none``). Specify ``padding`` as ``linear``
    or :math:`\min(N,M)-1` to compute linear convolution, with extra computational overhead.
    ``padding`` can also be set as a non-negative integer within this range for a balance.

    This function is similar to |scipy_signal_fftconvolve|_ in 1D case
    when ``padding`` is ``linear``.

    :param Tensor f1: The first sequence :math:`f_1` with length :math:`N` in dimension ``dim``.
    :param Tensor f2: The second sequence :math:`f_2` with length :math:`M` in dimension ``dim``.
    :param int dim: The dimension to be convolved. Default: -1.
    :param str out: One of the following options. Default: ``full``.

        ``full``
            Return complete result so the size of dimension ``dim`` is ``max(N, M) + padding - 1``.

        ``same``
            Return the middle segment of result. In other words, drop the first
            ``min(N, M) // 2`` elements and return subsequent ``max(N, M)`` ones.
            If the length is less than ``max(N, M)`` after dropping,
            the elements from head will be appended to reach this length.

        ``valid``
            Return only the segments where ``f1`` and ``f2`` overlap completely so
            the size of dimension ``dim`` is :math:`\max(N,M)-\min(N,M)+1`.
    :param padding: Padding amount. See description above. Default: ``linear``.
    :type padding: int or str
    :param bool real: If both ``f1`` and ``f2`` are real-valued and ``real`` is ``True``,
        :py:func:`torch.fft.rfft` can be used to improve computation and a real-valued
        tensor will be returned. If either ``f1`` or ``f2`` is complex or ``real=False``,
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
    padding = _dtm_padding(padding, min_l - 1)
    sl = max_l + padding

    if real:
        g: Ts = _fft.irfft(_fft.rfft(f1, sl, dim) * _fft.rfft(f2, sl, dim), sl, dim)
    else:
        g: Ts = _fft.ifft(_fft.fft(f1, sl, dim) * _fft.fft(f2, sl, dim), sl, dim)

    if out == 'full':
        return g
    elif out == 'same':
        return _narrow_periodic(g, dim, min_l // 2, max_l)
    elif out == 'valid':
        return g.narrow(dim, min_l - 1, max_l - min_l + 1)
    else:
        raise ValueError(f'Unknown output type: {out}')


def dconv2(
    f1: Ts,
    f2: Ts,
    dims: tuple[int, int] = (-2, -1),
    out: ConvOut = 'full',
    padding: Size2d | str = 'linear',
    real: bool = None,
) -> Ts:
    r"""
    Computes 2D discrete convolution for two arrays :math:`f_1` and :math:`f_2` utilizing FFT.

    In each involved dimension, the lengths of :math:`f_1`, :math:`N`
    and :math:`f_2`, :math:`M` can be different. In that case,
    the shorter sequence will be padded first to match length of the longer one.
    Notably, it is circular convolution that is computed without additional padding
    (if ``padding`` is ``0`` or ``none``). Specify ``padding`` as ``linear``
    or :math:`\min(N,M)-1` to compute linear convolution, with extra computational overhead.
    ``padding`` can also be set as a non-negative integer within this range for a balance.

    This function is similar to |scipy_signal_fftconvolve|_ in 2D case
    when ``padding`` is ``linear``.

    :param Tensor f1: The first array :math:`f_1` with size ``N1, N2`` in dimension ``dims``.
    :param Tensor f2: The second array :math:`f_2` with size ``M1, M2`` in dimension ``dims``.
    :param dims: The dimensions to be transformed. Default: ``(-2,-1)``.
    :type dims: tuple[int, int]
    :param str out: One of the following options. Default: ``full``. In each dimension:

        ``full``
            Return complete result so the size of dimension ``dim`` is ``max(N, M) + padding - 1``.

        ``same``
            Return the middle segment of result. In other words, drop the first
            ``min(N, M) // 2`` elements and return subsequent ``max(N, M)`` ones.
            If the length is less than ``max(N, M)`` after dropping,
            the elements from head will be appended to reach this length.

        ``valid``
            Return only the segments where ``f1`` and ``f2`` overlap completely so
            the size of dimension ``dim`` is :math:`\max(N,M)-\min(N,M)+1`.
    :param padding: Padding amount in two directions. See description above. Default: ``linear``.
    :type padding: int, tuple[int, int] or str
    :param bool real: If both ``f1`` and ``f2`` are real-valued and ``real`` is ``True``,
        :py:func:`torch.fft.rfft` can be used to improve computation and a real-valued
        tensor will be returned. If either ``f1`` or ``f2`` is complex or ``real=False``,
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
    if isinstance(padding, str):
        padding = (_dtm_padding(padding, min_l1 - 1), _dtm_padding(padding, min_l2 - 1))
    else:
        padding = size2d(padding)
        padding = (_dtm_padding(padding[0], min_l1 - 1), _dtm_padding(padding[1], min_l2 - 1))
    sl = (max_l1 + padding[0], max_l2 + padding[1])

    if real:
        g: Ts = _fft.irfft2(_fft.rfft2(f1, sl, dims) * _fft.rfft2(f2, sl, dims), sl, dims)
    else:
        g: Ts = _fft.ifft2(_fft.fft2(f1, sl, dims) * _fft.fft2(f2, sl, dims), sl, dims)

    if out == 'full':
        return g
    elif out == 'same':
        g = _narrow_periodic(g, dims[0], min_l1 // 2, max_l1)
        g = _narrow_periodic(g, dims[1], min_l2 // 2, max_l2)
        return g
    elif out == 'valid':
        g = g.narrow(dims[0], min_l1 - 1, max_l1 - min_l1 + 1)
        g = g.narrow(dims[1], min_l2 - 1, max_l2 - min_l2 + 1)
        return g
    else:
        raise ValueError(f'Unknown output type: {out}')


def conv1(
    f1: Ts,
    f2: Ts,
    dx: Spacing = None,
    dim: int = -1,
    out: ConvOut = 'full',
    padding: int | str = 'linear',
    simpson: bool = True,
    real: bool = None,
) -> Ts:
    r"""
    Computes 1D continuous convolution for two sequences :math:`f_1` and :math:`f_2` utilizing FFT.
    See :py:func:`dconv1` for more details.

    This function uses Simpson's rule to improve accuracy. Specifically, the weights
    of elements are :math:`1/3, 4/3, 2/3, 4/3, 2/3, \cdots, 4/3, 1/3` if the longer
    length is odd. If it is even, an additional 1 is appended.

    :param Tensor f1: The first sequence :math:`f_1` with length :math:`N` in dimension ``dim``.
    :param Tensor f2: The second sequence :math:`f_2` with length :math:`M` in dimension ``dim``.
    :param dx: Sampling spacing, either a float, a 0D tensor, or a tensor broadcastable
        with ``f1`` and ``f2``. Omitted if ``None`` (default).
    :type: float or Tensor
    :param int dim: The dimension to be convolved. Default: -1.
    :param str out: See :py:func:`dconv1`.
    :param padding: See :py:func:`dconv1`.
    :type padding: int or str
    :param bool simpson: Whether to apply Simpson's rule. Default: ``True``.
    :param bool real: See :py:func:`dconv1`.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    _check_dim('f1', f1.shape, (dim,), delta=dx)
    if simpson:
        f1, f2 = _simpson(f1, f2, dim)
    g = dconv1(f1, f2, dim, out, padding, real)
    return _mul(g, dx)


def conv2(
    f1: Ts,
    f2: Ts,
    dx: Spacing = None,
    dy: Spacing = None,
    dims: tuple[int, int] = (-2, -1),
    out: ConvOut = 'full',
    padding: Size2d | str = 'linear',
    simpson: bool = True,
    real: bool = None,
) -> Ts:
    r"""
    Computes 2D continuous convolution for two arrays :math:`f_1` and :math:`f_2` utilizing FFT.
    See :py:func:`dconv2` for more details.

    This function uses Simpson's rule to improve accuracy. Specifically, the weights
    of elements are :math:`1/3, 4/3, 2/3, 4/3, 2/3, \cdots, 4/3, 1/3` if the longer
    length is odd. If it is even, an additional 1 is appended.
    It is applied in all dimensions.

    :param Tensor f1: The first array :math:`f_1` with size ``N1, N2`` in dimension ``dims``.
    :param Tensor f2: The second array :math:`f_2` with size ``M1, M2`` in dimension ``dims``.
    :param dx: Sampling spacing in x direction i.e. the second dimension, either a float,
        a 0D tensor, or a tensor broadcastable with ``f1`` and ``f2``.
        Default: omitted.
    :type: float or Tensor
    :param dy: Sampling spacing in y direction i.e. the first dimension, similar to ``dx``.
        Default: identical to ``dx``.
    :type: float or Tensor
    :param dims: The dimensions to be transformed. Default: ``(-2,-1)``.
    :type dims: tuple[int, int]
    :param str out: See :py:func:`dconv2`.
    :param padding: See :py:func:`dconv2`.
    :type padding: int, tuple[int, int] or str
    :param bool simpson: Whether to apply Simpson's rule. Default: ``True``.
    :param bool real: See :py:func:`dconv2`.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    _check_dim('f1', f1.shape, dims, dx=dx, dy=dy)
    if simpson:
        f1, f2 = _simpson(f1, f2, dims[0])
        f1, f2 = _simpson(f1, f2, dims[1])
    g = dconv2(f1, f2, dims, out, padding, real)
    return _mul(g, dx, dy)


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
