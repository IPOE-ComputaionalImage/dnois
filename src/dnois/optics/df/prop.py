# TODO: detection
import abc
import re
import warnings

import torch
from torch import nn

from dnois import utils, fourier
from dnois.base.typing import (
    Ts, Spacing, Vector, Literal, Sequence, Size2d, Callable,
    cast, is_scalar, scalar, vector, size2d, pair,
)

__all__ = [
    'angular_spectrum',
    'delta_convert',
    'fraunhofer',
    'fresnel_2ft',
    'fresnel_conv',
    'fresnel_ft',
    'init_angular_spectrum',
    'init_fraunhofer',
    'init_fresnel_conv',
    'init_fresnel_ft',

    'AngularSpectrum',
    'DMode',
    'Fraunhofer',
    'FresnelConv',
    'FresnelFT',
]

_Cache = dict[str, Ts]
DMode = Literal['forward', 'backward']
DTy_ = Ts  # TODO


def _delta_convert(delta: Spacing, n: int, prod: Ts) -> Spacing:
    if is_scalar(delta):  # return scalar
        return prod / (delta * n)
    else:  # return tensor: ...,N_d,N_wl
        if is_scalar(prod):  # N_d=N_wl=1
            prod = prod.reshape(1, 1)
        if delta.shape[-2:] != prod.shape:
            raise ValueError(f'delta shape must be (...,N_d,N_wl). Got {delta.shape} '
                             f'but N_d={prod.size(-2)}, N_wl={prod.size(-1)}')
        return prod / (delta * n)


def _determine_cache(callback: Callable, cache: _Cache, *args) -> _Cache:
    cache_available = cache is not None
    if all(arg is not None for arg in args):
        if cache_available:
            warnings.warn(f'All required parameters are provided so cache is ignored')
        return callback()
    elif all(arg is None for arg in args):
        if not cache_available:
            raise ValueError(f'Either all of wl, distance and grid spacings or cache should be given')
        return cache
    else:
        raise ValueError(f'wl, distance and grid spacings must be given or not simultaneously')


def _init_ft_common(
    far_field: bool,  # Fraunhofer if True, Fresnel otherwise
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,  # Scalar or (...,N_d,N_wl)
    dy: Spacing = None,  # default to dx
    delta_mode: DMode = 'backward',
    pupil_exp_form: bool = True,
    post_phase_factor: bool = True,
    scale_factor: bool = True,
) -> _Cache:
    grid_size = size2d(grid_size)
    if dy is None:
        dy = dx
    if not torch.is_tensor(dx):
        dx = torch.tensor(dx)
    if not torch.is_tensor(dy):
        dy = torch.tensor(dy)
    if delta_mode == 'backward':
        dv, du = delta_convert((dy, dx), grid_size, wl, distance)
    elif delta_mode == 'forward':
        dv, du = dy, dx
        dy, dx = delta_convert((dy, dx), grid_size, wl, distance)
    else:
        raise ValueError(f'Unknown delta_mode: {delta_mode}')
    # shape of deltas at present: (,) or (...,N_d,N_wl)
    cache = {'du': du[..., None, None], 'dv': dv[..., None, None]}  # add two more dim for ft
    v, u = utils.sym_grid(2, grid_size, (dv, du))

    wl = vector(wl)
    distance = vector(distance)
    prod = wl * distance.unsqueeze(-1)  # N_d x N_wl
    phase_scale = torch.pi / prod  # k/(2d)
    quadratic_phase = phase_scale[..., None, None] * (u.square() + v.square())
    if not far_field:
        if pupil_exp_form:
            cache['quadratic_phase'] = quadratic_phase
        else:
            cache['quadratic_phase_factor'] = utils.expi(quadratic_phase)

    if post_phase_factor:
        y, x = utils.sym_grid(2, grid_size, (dy, dx))
        _post_phase = phase_scale[..., None, None] * (x.square() + y.square())
        _post_phase += 2 * torch.pi * distance.unsqueeze(-4) / wl - torch.pi / 2
        cache['post_phase_factor'] = utils.expi(_post_phase)

    if scale_factor:
        cache['scale_factor'] = 1 / prod[..., None, None]  # N_d x N_wl x 1 x 1

    return cache


def _init_as_common(
    paraxial: bool,
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    pad: Size2d = 0,
) -> _Cache:
    if dy is None:
        dy = dx
    grid_size = size2d(grid_size)
    pad = size2d(pad)
    grid_size = (grid_size[0] + 2 * pad[0], grid_size[1] + 2 * pad[1])
    dfy = 1 / (grid_size[0] * dy)
    dfx = 1 / (grid_size[1] * dx)
    fy, fx = utils.sym_grid(2, grid_size, (dfy, dfx))
    rou2 = fx.square() + fy.square()

    wl = vector(wl)
    distance = vector(distance)
    kd = 2 * torch.pi / wl * distance.unsqueeze(-1)
    if paraxial:
        phase_scale = - torch.pi * wl * distance.unsqueeze(-1)  # -pi lambda d
        phase_scale = phase_scale[..., None, None]
        argument = phase_scale * rou2
        argument += kd[..., None, None]
        transfer = utils.expi(argument)
    else:
        max_delta = max(dx.max().item(), dy.max().item())
        if max_delta > wl.max().item() / 2:
            warnings.warn(
                f'Grid spacing should be less than a half of the wavelength to '
                f'cover the pass band of angular spectrum transfer function. '
                f'Got max spacing={max_delta}, wavelength={wl}'
            )

        under_sqrt = 1 - wl.square().reshape(-1, 1, 1) * rou2
        mask = under_sqrt > 0
        factor = torch.where(mask, torch.sqrt(under_sqrt), torch.zeros_like(under_sqrt))
        argument = kd[..., None, None] * factor
        transfer = utils.expi(argument)
        transfer = torch.where(mask, transfer, torch.zeros_like(transfer))
    return {'transfer': transfer}


def _ft_common(
    far_field: bool,  # Fraunhofer if True, Fresnel otherwise
    pupil: Ts | tuple[Ts, Ts],
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    post_phase_factor: bool = False,
    scale_factor: bool = True,
    dft_scale: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    argument_available = isinstance(pupil, tuple)
    grid_size = pupil[0].shape[-2:] if argument_available else pupil.shape[-2:]
    intermediate = _determine_cache(lambda: _init_ft_common(
        far_field,
        grid_size, wl, distance, dx, dy, delta_mode,
        argument_available, post_phase_factor, scale_factor
    ), intermediate, wl, distance, dx)

    if far_field:
        if argument_available:
            pupil = torch.polar(*pupil)
    else:  # Fresnel diffraction needs quadratic phase
        if argument_available:
            phase = pupil[1] + intermediate['quadratic_phase']
            if pupil[0].dtype == torch.bool:
                phase_factor = utils.expi(phase)
                pupil = torch.where(pupil[0], phase_factor, torch.zeros_like(phase_factor))
            else:
                pupil = torch.polar(pupil[0], phase)
        else:
            pupil = pupil * intermediate['quadratic_phase_factor']

    if dft_scale:
        field = fourier.ft2(pupil, intermediate['du'], intermediate['dv'])
    else:
        field = fourier.ft2(pupil)
    if post_phase_factor:
        field = field * intermediate['post_phase_factor']
    if scale_factor:
        field = field * intermediate['scale_factor']
    return field


def _as_common(
    paraxial: bool,
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    pad: Size2d = 0,
    intermediate: _Cache = None,
) -> Ts:
    grid_size = (pupil.size(-2), pupil.size(-1))
    intermediate = _determine_cache(
        lambda: _init_as_common(paraxial, grid_size, wl, distance, dx, dy, pad),
        intermediate,
        wl, distance, dx
    )

    # grid spacings are not needed to ensure correct DFT-scale
    # because transfer function itself is already correctly DFT-scaled
    field = fourier.conv2_partial(
        pupil, intermediate['transfer'], pad=pad, centered=True
    )
    return field


def delta_convert(
    delta: Spacing | Sequence[Spacing],
    n: int | Sequence[int],
    wl: Vector,
    distance: Vector,
) -> Spacing | list[Spacing]:
    r"""
    Convert grid spacing between source plane (:math:`\delta_\text{s}`) and
    target plane (:math:`\delta_\text{t}`) in Fourier-transform-based
    Fresnel diffraction integral, using the following formula:

    .. math::
        \delta_\text{s}\delta_\text{t}=\lambda d/N

    :param delta: Grid spacing on source plane or target plane.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`, or a sequence of them.
    :param n: Number of samples :math:`N`. An integer or a sequence of integers with same
        length as ``delta``.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :return: Converted grid spacing. A scalar tensor if ``delta``, ``wl`` and ``distance``
        are all scalars, otherwise a tensor of shape :math:`(\cdots,N_d,N_\lambda)`.
        If ``delta`` and ``n`` are sequences, a list with same length will be returned.
    """
    is_seq = isinstance(delta, Sequence)
    if is_seq ^ isinstance(n, Sequence):
        msg = f'delta and n must be or not be sequences simultaneously. Got {type(delta)}, {type(n)}'
        raise ValueError(msg)
    if is_seq:
        len1, len2 = len(delta), len(cast(Sequence[int], n))
        if len1 != len2:
            raise ValueError(f'delta and n must have same length. Got {len1}, {len2}')
    if is_scalar(wl) and is_scalar(distance):
        prod = scalar(wl) * scalar(distance)  # 0d tensor
    else:
        prod = vector(wl) * vector(distance).unsqueeze(-1)  # N_d x N_wl

    if is_seq:
        delta: Sequence[Spacing]
        n: Sequence[int]
        return [_delta_convert(_delta, _n, prod) for _delta, _n in zip(delta, n)]
    else:
        return _delta_convert(delta, n, prod)


def init_fresnel_ft(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    pupil_exp_form: bool = True,
    post_phase_factor: bool = True,
    scale_factor: bool = True,
) -> _Cache:
    r"""
    Pre-compute the intermediate results in Fresnel diffraction integral and return them
    to avoid repeated computation in :py:func:`fresnel_ft`.

    :param grid_size: Size of spatial dimensions :math:`(H,W)`.
    :param wl: See :py:func:`fresnel_ft`.
    :param distance: See :py:func:`fresnel_ft`.
    :param dx: See :py:func:`fresnel_ft`.
    :param dy: See :py:func:`fresnel_ft`.
    :param delta_mode: See :py:func:`fresnel_ft`.
    :param pupil_exp_form: Whether the pupil function passed to :py:func:`fresnel_ft` is a
        a pair of tensors representing its amplitude and phase, or a single complex tensor
        if ``False``. See :py:func:`fresnel_ft`.
    :param post_phase_factor: See :py:func:`fresnel_ft`.
    :param scale_factor: See :py:func:`fresnel_ft`.
    :return: Intermediate results that can be passed to :py:func:`fresnel_ft` as
        ``intermediate`` argument.
    """
    return _init_ft_common(
        False,
        grid_size, wl, distance, dx, dy,
        delta_mode, pupil_exp_form, post_phase_factor, scale_factor
    )


def init_fresnel_conv(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    pad: Size2d = 0,
) -> _Cache:
    """
    Pre-compute the intermediate results in Fresnel diffraction integral and return them
    to avoid repeated computation in :py:func:`fresnel_conv`.

    :param grid_size: Grid size before padding. A 2-tuple of integers representing the
        height and width or a single integer if they are equal.
    :param wl: See :py:func:`fresnel_conv`.
    :param distance: See :py:func:`fresnel_conv`.
    :param dx: See :py:func:`fresnel_conv`.
    :param dy: See :py:func:`fresnel_conv`.
    :param pad: See :py:func:`fresnel_conv`.
    :return: Intermediate results that can be passed to :py:func:`fresnel_conv` as
        ``intermediate`` argument.
    """
    return _init_as_common(True, grid_size, wl, distance, dx, dy, pad)


def init_fraunhofer(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    post_phase_factor: bool = True,
    scale_factor: bool = True,
) -> _Cache:
    r"""
    Pre-compute the intermediate results in Fraunhofer diffraction integral and return them
    to avoid repeated computation in :py:func:`fraunhofer`.

    :param grid_size: Size of spatial dimensions :math:`(H,W)`.
    :param wl: See :py:func:`fraunhofer`.
    :param distance: See :py:func:`fraunhofer`.
    :param dx: See :py:func:`fraunhofer`.
    :param dy: See :py:func:`fraunhofer`.
    :param delta_mode: See :py:func:`fraunhofer`.
    :param post_phase_factor: See :py:func:`fraunhofer`.
    :param scale_factor: See :py:func:`fraunhofer`.
    :return: Intermediate results that can be passed to :py:func:`fraunhofer` as
        ``intermediate`` argument.
    """
    return _init_ft_common(
        False,
        grid_size, wl, distance, dx, dy,
        delta_mode, False, post_phase_factor, scale_factor
    )


def init_angular_spectrum(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    pad: Size2d = 0,
) -> _Cache:
    """
    Pre-compute the intermediate results in Fresnel diffraction integral and return them
    to avoid repeated computation in :py:func:`angular_spectrum`.

    :param grid_size: Grid size before padding. A 2-tuple of integers representing the
        height and width or a single integer if they are equal.
    :param wl: See :py:func:`angular_spectrum`.
    :param distance: See :py:func:`angular_spectrum`.
    :param dx: See :py:func:`angular_spectrum`.
    :param dy: See :py:func:`angular_spectrum`.
    :param pad: See :py:func:`angular_spectrum`.
    :return: Intermediate results that can be passed to :py:func:`angular_spectrum` as
        ``intermediate`` argument.
    """
    return _init_as_common(False, grid_size, wl, distance, dx, dy, pad)


def fresnel_ft(
    pupil: Ts | tuple[Ts, Ts],
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    post_phase_factor: bool = False,
    scale_factor: bool = True,
    dft_scale: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    r"""
    Computes Fresnel diffraction integral using one-step Fourier transform method:

    .. math::
        U'(x,y)&=\frac{\e^{\i kd}}{\i\lambda d}\iint U(u,v)
        \e^{\i\frac{k}{2d}\left((x-u)^2+(y-v)^2\right)}\d u\d v \\
        &=\frac{\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}}{\i\lambda d}
        \ft\left\{U(u,v)\e^{\i\frac{k}{2d}(u^2+v^2)}\right\}_
        {f_u=\frac{x}{\lambda d},f_v=\frac{y}{\lambda d}}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ft` represents Fourier transform and :math:`k=2\pi/\lambda`.

    This is typically used to compute the pulse response of an optical system given
    the pupil function :math:`U`. It is assumed to have non-zero value only on a finite
    region around the origin, covered by the region from which :math:`U` is sampled.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_fresnel_ft` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        If a single tensor, it serves as complex :math:`U`. If a tuple of tensors, they will be
        interpreted as the amplitude and phase of :math:`U` and must be real. The latter is
        more efficient if amplitude and phase are available (i.e. need not be computed from
        real and imaginary part).
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even thought ``dy`` is given.
    :param delta_mode: On which plane ``dx`` and ``dy`` are defined, source plane
        (``forward``) or target plane (``backward``). Default: target plane.
    :param post_phase_factor: Whether to multiply :math:`\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}/\i`.
        It can be omitted to reduce computation if only amplitude matters.
    :param scale_factor: Whether to multiply the factor :math:`1/(\lambda d)`.
        It can be omitted to reduce computation if relative scales between results across
        different :math:`d` and :math:`\lambda` do not matter.
    :param dft_scale: Whether to apply scale correction for DFT. Default: ``True``.
    :param intermediate: Cached intermediate results returned by :py:func:`init_fresnel_ft`.
        This can be used to speed up computation.
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    """
    return _ft_common(
        False,
        pupil, wl, distance, dx, dy,
        delta_mode, post_phase_factor, scale_factor, dft_scale, intermediate,
    )


def fresnel_2ft(

) -> Ts:  # TODO
    pass


def fresnel_conv(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    pad: Size2d = 0,
    intermediate: _Cache = None,
) -> Ts:
    r"""
    Computes Fresnel diffraction integral using convolution method:

    .. math::
        U'(x,y)&=\frac{\e^{\i kd}}{\i\lambda d}\iint U(u,v)
        \e^{\i\frac{k}{2d}\left((x-u)^2+(y-v)^2\right)}\d u\d v \\
        &=U(x,y)\ast\frac{\e^{\i kd}}{\i\lambda d}\e^{\i\frac{k}{2d}(x^2+y^2)}\\
        &=\ft^{-1}\left\{\ft\{U(x,y)\}\e^{\i kd}\e^{-\i\pi\lambda d(f_X^2+f_Y^2)}\right\}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ast` represents convolution and :math:`k=2\pi/\lambda`.

    This is typically used to compute the pulse response of an optical system given
    the pupil function :math:`U`. It is assumed to have non-zero value only on a finite
    region around the origin, covered by the region from which :math:`U` is sampled.

    Compared to :py:func:`fresnel_ft`, this function always computes the diffraction integral
    completely and exactly, including phase factor, scale factor and DFT scale.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_fresnel_conv` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even thought ``dy`` is given.
    :param pad: Padding width used for DFT. A 2-tuple of integers representing paddings
        for vertical and horizontal directions, or a single integer if they are equal.
        Default: no padding.
    :param intermediate: Cached intermediate results returned by :py:func:`init_fresnel_conv`.
        This can be used to speed up computation.
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    """
    return _as_common(True, pupil, wl, distance, dx, dy, pad, intermediate)


def fraunhofer(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    post_phase_factor: bool = False,
    scale_factor: bool = True,
    dft_scale: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    r"""
    Computes Fraunhofer diffraction integral using one-step Fourier transform method:

    .. math::
        U'(x,y)=\frac{\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}}{\i\lambda d}
        \ft\{U(u,v)\}_{f_u=\frac{x}{\lambda d},f_v=\frac{y}{\lambda d}}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ft` represents Fourier transform and :math:`k=2\pi/\lambda`.

    This is typically used to compute the pulse response of an optical system given
    the pupil function :math:`U`. It is assumed to have non-zero value only on a finite
    region around the origin, covered by the region from which :math:`U` is sampled.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_fraunhofer` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even thought ``dy`` is given.
    :param delta_mode: On which plane ``dx`` and ``dy`` are defined, source plane
        (``forward``) or target plane (``backward``). Default: target plane.
    :param post_phase_factor: Whether to multiply :math:`\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}/\i`.
        It can be omitted to reduce computation if only amplitude matters.
    :param scale_factor: Whether to multiply the factor :math:`1/(\lambda d)`.
        It can be omitted to reduce computation if relative scales between results across
        different :math:`d` and :math:`\lambda` do not matter.
    :param dft_scale: Whether to apply scale correction for DFT. Default: ``True``.
    :param intermediate: Cached intermediate results returned by :py:func:`init_fraunhofer`.
        This can be used to speed up computation.
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    """
    return _ft_common(
        True,
        pupil, wl, distance, dx, dy, delta_mode,
        post_phase_factor, scale_factor, dft_scale, intermediate
    )


def angular_spectrum(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    pad: Size2d = 0,
    intermediate: _Cache = None,
) -> Ts:
    r"""
    Computes angular spectrum diffraction integral using convolution    :

    .. math::
        U'(x,y)&=-\frac{1}{2\pi}\iint U(u,v)\pfrac{}{d}\frac{\e^{\i kr_d}}{r_d}\d u\d v\\
        &=-\frac{1}{2\pi}U(x,y)\ast\pfrac{}{d}\frac{\e^{\i k\sqrt{x^2+y^2+d^2}}}{\sqrt{x^2+y^2+d^2}}\\
        &=\ft^{-1}\left\{\ft\{U(x,y)\}\circfunc\left((\lambda f_X)^2+(\lambda f_Y)^2\right)
        \e^{\i kd\sqrt{1-(\lambda f_X)^2-(\lambda f_Y)^2}}\right\}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ast` represents convolution, :math:`k=2\pi/\lambda` and
    :math:`r_d=\sqrt{(x-u)^2+(y-v)^2+d^2}`. Angular spectrum diffraction is equivalent to
    the first Rayleigh-Sommerfeld solution.

    This is typically used to compute the pulse response of an optical system given
    the pupil function :math:`U`. It is assumed to have non-zero value only on a finite
    region around the origin, covered by the region from which :math:`U` is sampled.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_angular_spectrum` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even thought ``dy`` is given.
    :param pad: Padding width used for DFT. A 2-tuple of integers representing paddings
        for vertical and horizontal directions, or a single integer if they are equal.
        Default: no padding.
    :param intermediate: Cached intermediate results returned by :py:func:`init_angular_spectrum`.
        This can be used to speed up computation.
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    """
    return _as_common(False, pupil, wl, distance, dx, dy, pad, intermediate)


class Diffraction(nn.Module, metaclass=abc.ABCMeta):
    u: Ts
    v: Ts
    x: Ts
    y: Ts
    u_range: tuple[DTy_, DTy_]
    v_range: tuple[DTy_, DTy_]
    x_range: tuple[DTy_, DTy_]
    y_range: tuple[DTy_, DTy_]
    u_size: DTy_
    v_size: DTy_
    x_size: DTy_
    y_size: DTy_

    _wl: Ts
    _d: Ts
    _du: Ts
    _dv: Ts

    def __init__(
        self,
        grid_size: Size2d,
        wl: Vector = None,
        d: Vector = None,
    ):
        super().__init__()
        self._grid_size = size2d(grid_size)

        self._ptn = re.compile('^[uvxy]((_range)|(_size))?$')
        self._ready = False  # diffraction and grid
        self._cache = {}

        self.wl = wl
        self.d = d

    def __getattr__(self, item: str):
        if self._ptn.match(item):
            spacing = getattr(self, f'd{item[0]}')  # du/dv/dx/dy
            dim = 0 if item[0] == 'v' or item[0] == 'y' else 1
            size = self._grid_size[dim]
            if len(item) == 1:  # u/v/x/y
                axis = utils.sym_interval(size, spacing.unsqueeze(-1))
                return axis.unsqueeze(-1 - dim)  # 1 x W for u/x or H x 1 for v/y
            elif item[2] == 'r':  # range
                return -spacing * (size // 2), spacing * ((size - 1) // 2)
            else:  # size
                return spacing * size
        else:
            return super().__getattr__(item)

    @abc.abstractmethod
    def _init_cache(self):
        pass

    @property
    def wl(self) -> Ts:
        return self.__dict__['_buffers']['_wl']

    @wl.setter
    def wl(self, value: Vector):
        value = scalar(value) if is_scalar(value) else vector(value)
        self.register_buffer('_wl', value, False)
        self._ready = False

    @property
    def d(self) -> Ts:
        return self.__dict__['_buffers']['_d']

    @d.setter
    def d(self, value: Vector):
        value = scalar(value) if is_scalar(value) else vector(value)
        self.register_buffer('_d', value, False)
        self._ready = False

    @property
    def du(self) -> Ts:
        return self.__dict__['_buffers']['_du']

    @du.setter
    def du(self, value: Spacing):
        self._set_spacing('_du', value)

    @property
    def dv(self) -> Ts:
        return self.__dict__['_buffers']['_dv']

    @dv.setter
    def dv(self, value: Spacing):
        self._set_spacing('_dv', value)

    @property
    def dx(self) -> Ts:
        return self.du

    @dx.setter
    def dx(self, value: Spacing):
        self.du = value

    @property
    def dy(self) -> Ts:
        return self.dv

    @dy.setter
    def dy(self, value: Spacing):
        self.dv = value

    def _build(self):
        if self._ready:
            return
        if self.wl is None or self.d is None or self.du is None or self.dv is None:
            raise RuntimeError(f'Wavelength, distance or spacing has not been set')

        cache = self._init_cache()

        for k, v in cache.items():
            self.register_buffer(f'_cache_{k}', v, False)
        for k, v in cache.items():
            self._cache[k] = getattr(self, f'_cache_{k}')

        self._ready = True

    def _set_spacing(self, key, value):
        if is_scalar(value):
            value = scalar(value)
        self.register_buffer(key, value, False)
        self._ready = False


class FourierTransformBased(Diffraction):
    _far_field: bool
    _dx: Ts
    _dy: Ts

    def __init__(
        self,
        grid_size: Size2d,
        spacing: Spacing | tuple[Spacing, Spacing] = None,
        wl: Vector = None,
        d: Vector = None,
        ampl_phase_form: bool = True,
        delta_mode: DMode = 'backward',
        post_phase_factor: bool = False,
        scale_factor: bool = True,
        dft_scale: bool = True,
    ):
        super().__init__(grid_size, wl, d)
        self._apf = ampl_phase_form
        self._delta_mode = delta_mode
        self._ppf = post_phase_factor
        self._sf = scale_factor
        self._dft_scale = dft_scale

        if self._delta_mode == 'backward':
            self.dy, self.dx = pair(spacing)
        elif self._delta_mode == 'forward':
            self.dv, self.du = pair(spacing)
        else:
            raise ValueError(f'Unknown delta_mode: {delta_mode}')

    def _init_cache(self):
        d1, d2 = (self.du, self.dv) if self._delta_mode == 'forward' else (self.dx, self.dy)
        return _init_ft_common(
            self._far_field,  # Fresnel or Fraunhofer
            self._grid_size,
            self.wl, self.d, d1, d2, self._delta_mode,  # required parameters
            self._apf, self._ppf, self._sf,  # options
        )

    @property
    def du(self):
        return self.__dict__['_buffers']['_du']

    @du.setter
    def du(self, value):
        self._set_spacing('_du', value)
        self._set_spacing('_dx', delta_convert(self.du, self._grid_size[1], self.wl, self.d))

    @property
    def dv(self):
        return self.__dict__['_buffers']['_dv']

    @dv.setter
    def dv(self, value):
        self._set_spacing('_dv', value)
        self._set_spacing('_dy', delta_convert(self.dv, self._grid_size[0], self.wl, self.d))

    @property
    def dx(self):
        return self.__dict__['_buffers']['_dx']

    @dx.setter
    def dx(self, value):
        self._set_spacing('_dx', value)
        self._set_spacing('_du', delta_convert(self.dx, self._grid_size[1], self.wl, self.d))

    @property
    def dy(self):
        return self.__dict__['_buffers']['_dy']

    @dy.setter
    def dy(self, value):
        self._set_spacing('_dy', value)
        self._set_spacing('_dv', delta_convert(self.dy, self._grid_size[0], self.wl, self.d))

    def forward(self, pupil: Ts | tuple[Ts]) -> Ts:
        is_tuple = isinstance(pupil, tuple)
        if self._apf:
            if not is_tuple:
                raise RuntimeError(f'A pair of Tensors expected, got {type(pupil)}')
        elif is_tuple:
            raise RuntimeError(f'A single Tensors expected, got {type(pupil)}')

        self._build()
        return _ft_common(
            self._far_field,
            pupil,
            post_phase_factor=self._ppf,
            scale_factor=self._sf,
            dft_scale=self._dft_scale,
            intermediate=self._cache,
        )


class AngularSpectrumBased(Diffraction):
    _paraxial: bool

    def __init__(
        self,
        grid_size: Size2d,
        spacing: Spacing | tuple[Spacing, Spacing] = None,
        wl: Vector = None,
        d: Vector = None,
        pad: Size2d = 0,
    ):
        super().__init__(grid_size, wl, d)
        self._pad = size2d(pad)

        self.dv, self.du = pair(spacing)

    def _init_cache(self):
        return _init_as_common(
            self._paraxial,
            self._grid_size,
            self.wl, self.d, self.du, self.dv,
            self._pad
        )

    def forward(self, pupil: Ts) -> Ts:
        self._build()
        return _as_common(self._paraxial, pupil, pad=self._pad, intermediate=self._cache)


class FresnelFT(FourierTransformBased):
    _far_field: bool = False


class Fraunhofer(FourierTransformBased):
    _far_field: bool = True


class FresnelConv(AngularSpectrumBased):
    _paraxial: bool = True


class AngularSpectrum(AngularSpectrumBased):
    _paraxial: bool = False
