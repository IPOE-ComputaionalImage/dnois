import warnings

import torch
import torch.fft as _fft

from dnois.base.typing import Size2d, Ts, Spacing, ConvOut, Sequence, Sizend, sizend

from ._utils import _check_dim, _mul

__all__ = [
    'conv',
    'conv1',
    'conv2',
    'conv_mult',
    'conv_partial',
    'dconv',
    'dconv1',
    'dconv2',
    'dconv_mult',
    'dconv_partial',
    'ft4dconv',
    'ft4conv',
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
    elif isinstance(padding, str):
        raise ValueError(f'Unknown padding type: {padding}')
    elif padding < 0:
        warnings.warn('Got negative padding, setting padding to zero')
        return 0
    elif padding > linear_value:
        warnings.warn(f'Got padding too large, setting padding to {linear_value}')
        return linear_value
    else:
        return padding


def _dtm_size(shape1, shape2, dims, padding):
    n_list = [shape1[dim] for dim in dims]
    m_list = [shape2[dim] for dim in dims]
    min_list = [min(n, m) for n, m in zip(n_list, m_list)]
    max_list = [max(n, m) for n, m in zip(n_list, m_list)]
    if isinstance(padding, str):
        padding = [_dtm_padding(padding, min_l - 1) for min_l in min_list]
    else:
        padding = sizend(padding, len(dims))
        padding = [_dtm_padding(p, min_l - 1) for p, min_l in zip(padding, min_list)]
    sl = [max_l + p for max_l, p in zip(max_list, padding)]
    return sl, min_list, max_list


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


def _simpson(f: Ts, dim: int) -> Ts:
    shape = [1 for _ in f.shape]
    shape[dim] = -1
    return f * _simpson_weights(f.size(dim), f.device, f.dtype).reshape(shape)


def _crop(g, out, dims, min_list, max_list) -> Ts:
    if out == 'full':
        return g
    elif out == 'same':
        for dim, min_l, max_l in zip(dims, min_list, max_list):
            g = _narrow_periodic(g, dim, min_l // 2, max_l)
        return g
    elif out == 'valid':
        for dim, min_l, max_l in zip(dims, min_list, max_list):
            g = g.narrow(dim, min_l - 1, max_l - min_l + 1)
        return g
    else:
        raise ValueError(f'Unknown output type: {out}')


def dconv1(
    f1: Ts,
    f2: Ts,
    dim: int = -1,
    out: ConvOut = 'full',
    padding: int | str = 'linear',
    real: bool = None,
) -> Ts:
    r"""
    1D version of :py:func:`dconv`.

    :param Tensor f1: The first sequence :math:`f_1`.
    :param Tensor f2: The second sequence :math:`f_2`.
    :param int dim: The dimension to be convolved. Default: -1.
    :param str out: See :py:func:`dconv`.
    :param padding: See :py:func:`dconv`.
    :type padding: int or str
    :param bool real: See :py:func:`dconv`.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    return dconv(f1, f2, (dim,), out, padding, real)


def dconv2(
    f1: Ts,
    f2: Ts,
    dims: tuple[int, int] = (-2, -1),
    out: ConvOut = 'full',
    padding: Size2d | str = 'linear',
    real: bool = None,
) -> Ts:
    r"""
    2D version of :py:func:`dconv`.

    :param Tensor f1: The first array :math:`f_1`.
    :param Tensor f2: The second array :math:`f_2`.
    :param dims: The dimensions to be convolved. Default: ``(-2,-1)``.
    :type dims: tuple[int, int]
    :param str out: See :py:func:`dconv`.
    :param padding: See :py:func:`dconv`.
    :type padding: int, tuple[int, int] or str
    :param bool real: See :py:func:`dconv`.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    return dconv(f1, f2, dims, out, padding, real)


def dconv(
    f1: Ts,
    f2: Ts,
    dims: Sequence[int] = None,
    out: ConvOut = 'full',
    padding: Sizend | str = 'linear',
    real: bool = None,
) -> Ts:
    r"""
    Computes n-D discrete convolution for two tensors :math:`f_1` and :math:`f_2` utilizing FFT.

    In each involved dimension, the lengths of :math:`f_1`, :math:`N`
    and :math:`f_2`, :math:`M` can be different. In that case,
    the shorter sequence will be padded first to match length of the longer one.
    Notably, it is circular convolution that is computed without additional padding
    (if ``padding`` is ``0`` or ``none``). Specify ``padding`` as ``linear``
    or :math:`\min(N,M)-1` to compute linear convolution, with extra computational overhead.
    ``padding`` can also be set as a non-negative integer within this range for a balance.
    Note that the lengths of each dimension can be different.

    See :ref:`ref_fourier_convolution` for examples.

    .. seealso::
        This function is similar to |scipy_signal_fftconvolve|_ when ``padding`` is ``linear``.
        :py:func:`dconv1` and :py:func:`dconv2` are 1D and 2D variants, respectively,
        with slightly different signature. :py:func:`conv` provides more functionalities.

    :param Tensor f1: The first tensor :math:`f_1`.
    :param Tensor f2: The second tensor :math:`f_2`.
    :param dims: The dimensions to be convolved. Default: the last ``ndim`` dimensions
        where ``ndim`` is the fewer number of dimensions of ``f1`` and ``f2``
    :type dims: Sequence[int]
    :param str out: One of the following options. Default: ``full``. In each dimension:

        ``full``
            Return complete result so the size of each dimension is ``max(N, M) + padding - 1``.

        ``same``
            Return the middle segment of result. In other words, drop the first
            ``min(N, M) // 2`` elements and return subsequent ``max(N, M)`` ones.
            If the length is less than ``max(N, M)`` after dropping,
            the elements from head will be appended to reach this length.

        ``valid``
            Return only the segments where ``f1`` and ``f2`` overlap completely so
            the size of each dimension is :math:`\max(N,M)-\min(N,M)+1`.
    :param padding: Padding amount in all dimensions. See description above. Default: ``linear``.
    :type padding: int, Sequence[int] or str
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
    if dims is None:
        dims = list(range(-min(f1.ndim, f2.ndim), 0))
    real = _dtm_real(f1, f2, real)
    sl, min_list, max_list = _dtm_size(f1.shape, f2.shape, dims, padding)

    if real:
        g: Ts = _fft.irfftn(_fft.rfftn(f1, sl, dims) * _fft.rfftn(f2, sl, dims), sl, dims)
    else:
        g: Ts = _fft.ifftn(_fft.fftn(f1, sl, dims) * _fft.fftn(f2, sl, dims), sl, dims)

    return _crop(g, out, dims, min_list, max_list)


def ft4dconv(
    f1_shape: Sequence[int],
    f2: Ts,
    dims: Sequence[int],
    real: bool,
    padding: Sizend | str = 'linear',
) -> Ts:
    """Discrete version of :py:func:`ft4conv`."""
    if any(s < 0 for s in f1_shape):
        raise ValueError(f'Got shape with negative element: {f1_shape}')
    sl, min_list, max_list = _dtm_size(f1_shape, f2.shape, dims, padding)

    if real:
        return _fft.rfftn(f2, sl, dims)
    else:
        return _fft.fftn(f2, sl, dims)


def dconv_partial(
    f1: Ts,
    f2_ft: Ts,
    f2_shape: Sequence[int],
    dims: Sequence[int],
    real: bool,
    out: ConvOut = 'full',
    padding: Sizend | str = 'linear',
) -> Ts:
    """Discrete version of :py:func:`conv_partial`."""
    if any(s < 0 for s in f2_shape):
        raise ValueError(f'Got shape with negative element: {f2_shape}')
    sl, min_list, max_list = _dtm_size(f1.shape, f2_shape, dims, padding)

    if real:
        g: Ts = _fft.irfftn(_fft.rfftn(f1, sl, dims) * f2_ft, sl, dims)
    else:
        g: Ts = _fft.ifftn(_fft.fftn(f1, sl, dims) * f2_ft, sl, dims)

    return _crop(g, out, dims, min_list, max_list)


def dconv_mult(
    f1s: Sequence[Ts],
    f2: Ts,
    dims: Sequence[int],
    out: ConvOut = 'full',
    padding: Sizend | str = 'linear',
    real: bool = None,
) -> list[Ts]:
    """Discrete version of :py:func:`conv_mult`."""
    real = all(_dtm_real(f1, f2, real) for f1 in f1s)
    f2_ft = ft4dconv(f1s[0].shape, f2, dims, real, padding)
    return [dconv_partial(f1, f2_ft, f2.shape, dims, real, out, padding) for f1 in f1s]


def conv1(
    f1: Ts,
    f2: Ts,
    dx: Spacing = None,
    dim: int = -1,
    out: ConvOut = 'full',
    padding: int | str = 'linear',
    simpson: bool = False,
    real: bool = None,
) -> Ts:
    r"""
    1D version of :py:func:`conv`.

    :param Tensor f1: The first sequence :math:`f_1`.
    :param Tensor f2: The second sequence :math:`f_2`.
    :param dx: Sampling spacing, either a float, a 0D tensor, or a tensor broadcastable
        with ``f1`` and ``f2``. Omitted if ``None`` (default).
    :type: float or Tensor
    :param int dim: The dimension to be convolved. Default: -1.
    :param str out: See :py:func:`conv`.
    :param padding: See :py:func:`conv`.
    :type padding: int or str
    :param bool simpson: Whether to apply Simpson's rule. Default: ``True``.
    :param bool real: See :py:func:`conv`.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    _check_dim('f1', f1.shape, (dim,), delta=dx)
    if simpson:
        f1 = _simpson(f1, dim)
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
    simpson: bool = False,
    real: bool = None,
) -> Ts:
    r"""
    2D version of :py:func:`conv`.

    .. seealso:: :py:func:`conv` and :py:func:`dconv2`.

    :param Tensor f1: The first array :math:`f_1`.
    :param Tensor f2: The second array :math:`f_2`.
    :param dx: Sampling spacing in x direction i.e. the second dimension, either a float,
        a 0D tensor, or a tensor broadcastable with ``f1`` and ``f2``.
        Default: omitted.
    :type: float or Tensor
    :param dy: Sampling spacing in y direction i.e. the first dimension, similar to ``dx``.
        Default: identical to ``dx``.
    :type: float or Tensor
    :param dims: The dimensions to be convolved. Default: ``(-2,-1)``.
    :type dims: tuple[int, int]
    :param str out: See :py:func:`conv`.
    :param padding: See :py:func:`conv`.
    :type padding: int, tuple[int, int] or str
    :param bool simpson: Whether to apply Simpson's rule. Default: ``True``.
    :param bool real: See :py:func:`conv`.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    _check_dim('f1', f1.shape, dims, dx=dx, dy=dy)
    if simpson:
        f1 = _simpson(f1, dims[0])
        f1 = _simpson(f1, dims[1])
    g = dconv2(f1, f2, dims, out, padding, real)
    return _mul(g, dx, dy)


def conv(
    f1: Ts,
    f2: Ts,
    dims: Sequence[int] = None,
    spacing: Spacing | Sequence[Spacing] = None,
    out: ConvOut = 'full',
    padding: Sizend | str = 'linear',
    simpson: bool = False,
    real: bool = None,
) -> Ts:
    r"""
    Computes continuous convolution for two tensors :math:`f_1` and :math:`f_2` utilizing FFT.
    See :py:func:`dconv` for more details.

    This function may use Simpson's rule to improve accuracy. Specifically, the weights
    of elements are :math:`1/3, 4/3, 2/3, 4/3, 2/3, \cdots, 4/3, 1/3` if the
    length is odd. If it is even, an additional 1 is appended.
    It is applied in all involved dimensions of ``f1``.

    See :ref:`ref_fourier_convolution` for examples.

    .. seealso::
        This function is equivalent to :py:func:`dconv`
        when ``spacing`` is ``None`` and ``simpson`` is ``False``.

    :param Tensor f1: The first tensor :math:`f_1`.
    :param Tensor f2: The second tensor :math:`f_2`.
    :param Sequence[int] dims: The dimensions to be convolved.Default: the last ``ndim`` dimensions
        where ``ndim`` is the fewer number of dimensions of ``f1`` and ``f2``
    :param spacing: Grid spacings in each dimension. Each element of ``spacing``
        can be a float, a 0D tensor or a tensor broadcastable with ``f1`` and ``f2``.
        If not a :py:class:`Sequence`, it serves as the spacings for all dimensions.
        Default: omitted.
    :type spacing: float, Tensor or Sequence[float, Tensor]
    :param str out: See :py:func:`dconv`.
    :param padding: See :py:func:`dconv`.
    :type padding: int, Sequence[int] or str
    :param bool simpson: Whether to apply Simpson's rule. Default: ``True``.
    :param bool real: See :py:func:`dconv`.
    :return: Convolution between ``f1`` and ``f2``. Complex if either ``f1`` or ``f2`` is
        complex or ``real=False``, real-valued otherwise.
    :rtype: Tensor
    """
    if dims is None:
        dims = list(range(-min(f1.ndim, f2.ndim), 0))
    if not isinstance(spacing, Sequence):  # including None
        spacing = [spacing] * len(dims)
    if len(spacing) != len(dims):
        raise ValueError(f'Number of spacings ({len(spacing)}) and dimensions ({len(dims)}) are unequal')
    _check_dim('f1', f1.shape, tuple(dims), **{f'd{i}': d for i, d in enumerate(spacing)})
    if simpson:
        for dim in dims:
            f1 = _simpson(f1, dim)
    g = dconv(f1, f2, dims, out, padding, real)
    return _mul(g, *spacing)


def ft4conv(
    f1_shape: Sequence[int],
    f2: Ts,
    dims: Sequence[int],
    real: bool,
    spacing: Spacing | Sequence[Spacing] = None,
    padding: Sizend | str = 'linear',
    simpson: bool = False,
) -> Ts:
    """
    See :py:func:`conv_mult`.

    :param Sequence[int] f1_shape: Shape of the original first tensor for convolution.
        This is needed to ensure correct padding.
    :param Tensor f2: The second tensor for convolution.
    :param Sequence[int] dims: Dimensions for convolution.
    :param bool real: Whether to use FFT for real tensors to ``f2``.
    :param spacing: See :py:func:`conv`.
    :param padding: See :py:func:`dconv`.
    :type padding: int, Sequence[int] or str
    :param bool simpson: Whether to apply Simpson's rule to ``f2`` before FFT. Default: ``True``.
    :return:
    """
    if not isinstance(spacing, Sequence):  # including None
        spacing = [spacing] * len(dims)
    if len(spacing) != len(dims):
        raise ValueError(f'Number of spacings ({len(spacing)}) and dimensions ({len(dims)}) are unequal')
    _check_dim('f1', f1_shape, tuple(dims), **{f'd{i}': d for i, d in enumerate(spacing)})
    if simpson:
        for dim in dims:
            f2 = _simpson(f2, dim)
    f2_ft = ft4dconv(f1_shape, f2, dims, real, padding)
    return _mul(f2_ft, *spacing)


def conv_partial(
    f1: Ts,
    f2_ft: Ts,
    f2_shape: Sequence[int],
    dims: Sequence[int],
    real: bool,
    out: ConvOut = 'full',
    padding: Sizend | str = 'linear',
    simpson: bool = True,
) -> Ts:
    """
    See :py:func:`conv_mult`. DFT-scale is not applied here.

    :param Tensor f1: The first tensor for convolution.
    :param Tensor f2_ft: The FFT of the second tensor for convolution.
        Typically produced by :py:func:`ft4conv`.
    :param Sequence[int] f2_shape: Shape of original second tensor.
        This is needed to ensure correct padding.
    :param Sequence[int] dims: Dimensions to be convolved.
    :param bool real: Whether to use FFT for real tensors to ``f1``.
        This should be consistent with that for ``f2_ft``.
    :param str out: See :py:func:`dconv`.
    :param padding: See :py:func:`dconv`.
    :type padding: int, Sequence[int] or str
    :param bool simpson: Whether to apply Simpson's rule to ``f1`` before FFT. Default: ``True``.
    :return: Convolution between ``f1`` and ``f2`` (whose FFT is ``f2_ft``).
        See :py:func:`dconv` for descriptions about its dtype and shape.
    """
    if simpson:
        for dim in dims:
            f1 = _simpson(f1, dim)
    return dconv_partial(f1, f2_ft, f2_shape, dims, real, out, padding)


def conv_mult(
    f1s: Sequence[Ts],
    f2: Ts,
    dims: Sequence[int],
    spacing: Spacing | Sequence[Spacing] = None,
    out: ConvOut = 'full',
    padding: Sizend | str = 'linear',
    simpson: bool = True,
    real: bool = None,
) -> list[Ts]:
    """
    Compute the convolutions between each signal in ``f1s`` and ``f2``.
    The FFT of ``f2`` is computed only once by calling :py:func:`ft4conv`,
    then passed to :py:func:`conv_partial` to compute convolutions.
    This is more efficient than calling :py:func:`conv` multiple times.
    By default, Simpson's rule and DFT-scale are only applied to ``f2``.

    Shapes of tensors in ``f1s`` on dimensions ``dims`` should be identical.
    Additionally, FFT for real signals is applicable only if ``real`` is ``True``
    and all the tensors in ``f1s`` are real tensors.

    .. seealso::
        :py:func:`dconv_mult` is the discrete counterpart of this function
        which drops ``spacing`` and ``simpson``.

    :param Sequence[Tensor] f1s: A sequence of tensors as one convolution operand.
    :param Tensor f2: The common tensor as another convolution operand.
    :param Sequence[int] dims: Dimensions to be convolved.
    :param spacing: See :py:func:`conv`.
    :type spacing: float, Tensor or Sequence[float, Tensor]
    :param str out: See :py:func:`dconv`.
    :param padding: See :py:func:`dconv`.
    :type padding: int, Sequence[int] or str
    :param bool simpson: See :py:func:`conv`.
    :param bool real: See :py:func:`dconv`.
    :return: A list of tensors representing convolutions between signals in ``f1s`` and ``f2``.
    :rtype: list[Tensor]
    """
    real = all(_dtm_real(f1, f2, real) for f1 in f1s)
    f2_ft = ft4conv(f1s[0].shape, f2, dims, False, spacing, padding, simpson)
    # apply Simpson's rule to f2 only
    return [conv_partial(f1, f2_ft, f2.shape, dims, real, out, padding, False) for f1 in f1s]
