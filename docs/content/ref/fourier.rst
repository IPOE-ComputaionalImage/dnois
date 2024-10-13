##################################################
Fourier Transform and Convolution
##################################################

.. automodule:: dnois.fourier

.. _ref_fourier_fourier_transform:

********************************
Fourier transform
********************************
.. testsetup::

    import torch
    import dnois
    from dnois.fourier import *

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

Multiplication with sampling interval is necessary if consistent scale
with continuous Fourier transform is desired, which is called *DFT-scale*.
Otherwise, it can be set to ``None``.

The following example illustrates its approximation to continuous Fourier transform,
as well as its precision in a simple occasion for 32-bit float.

>>> x = dnois.utils.sym_interval(1000, 4e-3)  # [-2, 2]
>>> y = torch.exp(-torch.pi * x.square())  # Gaussian function
>>> fx = dnois.utils.sym_interval(1000, 0.25)  # frequency
>>> g = torch.exp(-torch.pi * fx.square())  # ground truth spectrum
>>> torch.allclose(ft1(y, 4e-3).real, g, atol=1e-6)
True

.. note::

    The sampling interval (like ``delta`` argument for :py:func:`ft1`) will be
    multiplied to transformed signal if given, so it can be a :py:class:`float`
    or a tensor with shape broadcastable to original signal. But if it is not a
    0D tensor, its size on the dimensions to be transformed must be 1.

.. autosummary::
    :toctree: ../generated/ft/ft

    ft1
    ift1
    ft2
    ift2

.. _ref_fourier_convolution:

********************************
Convolution
********************************
These functions compute convolution between two signals ``f1`` and ``f2``.
:py:func:`dconv` computes discrete convolution.
Numerical optimization and correct scale are considered
in :py:func:`conv` so it serve as a numerical approximation to continuous convolution.

>>> f1, f2 = torch.tensor([1., 2., 3., 4.]), torch.tensor([1., 2., 3.])
>>> g_full_linear = torch.tensor([1., 4., 10., 16., 17., 12.])
>>> torch.allclose(dconv(f1, f2), g_full_linear)
True
>>> g_same_linear = torch.tensor([4., 10., 16., 17.])
>>> torch.allclose(dconv(f1, f2, out='same'), g_same_linear)
True
>>> g_full_circular = torch.tensor([18., 16., 10., 16.])
>>> torch.allclose(dconv(f1, f2, padding='none'), g_full_circular)
True

>>> f1, f2 = torch.rand(5), torch.rand(6)
>>> torch.allclose(conv(f1, f2, spacing=0.1), dconv(f1, f2) * 0.1)
True

.. note::
    If all the signals involved is real-valued, set ``real`` to ``True`` to improve
    the computation and return a real-valued tensor.

.. autosummary::
    :toctree: ../generated/ft/conv

    conv
    conv1
    conv2
    dconv
    dconv1
    dconv2

Multiple convolutions
=======================================

Sometimes one may need to compute convolutions between a group of tensors
and a single common tensor. In that case, it is sensible to compute the
Fourier transform of the common tensor only once. The following functions
serve for this purpose.

.. autosummary::
    :toctree: ../generated/ft/conv

    dconv_mult
    dconv_partial
    ft4dconv
    conv_mult
    conv_partial
    ft4conv
