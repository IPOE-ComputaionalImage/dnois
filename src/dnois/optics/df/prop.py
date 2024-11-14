# TODO: detection; kwargs tables left aligning
import abc
import re
import warnings

import torch
from torch import nn

from dnois import base, utils, fourier
from dnois.base.typing import (
    Ts, Spacing, Vector, Literal, Sequence, Size2d, Callable,
    cast, is_scalar, scalar, vector, size2d, pair, overload
)

__all__ = [
    'delta_convert',
    'fraunhofer',
    'fresnel_as',
    'fresnel_conv',
    'fresnel_ft',
    'init_fraunhofer',
    'init_fresnel_as',
    'init_fresnel_conv',
    'init_fresnel_ft',
    'init_rayleigh_sommerfeld_as',
    'init_rayleigh_sommerfeld_conv',
    'rayleigh_sommerfeld_as',
    'rayleigh_sommerfeld_conv',

    'Diffraction',
    'DMode',
    'Fraunhofer',
    'FresnelAS',
    'FresnelConv',
    'FresnelFT',
    'RayleighSommerfeldAS',
    'RayleighSommerfeldConv',
]

_Cache = dict[str, Ts]
DMode = Literal['forward', 'backward']


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
    ampl_phase_form: bool = True,
    phase_factor: bool = True,
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
        if ampl_phase_form:
            cache['quadratic_phase'] = quadratic_phase
        else:
            cache['quadratic_phase_factor'] = base.expi(quadratic_phase)

    if phase_factor:
        y, x = utils.sym_grid(2, grid_size, (dy, dx))
        _post_phase = phase_scale[..., None, None] * (x.square() + y.square())
        _post_phase += 2 * torch.pi * distance.unsqueeze(-4) / wl - torch.pi / 2
        cache['phase_factor'] = base.expi(_post_phase)

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
) -> _Cache:
    if dy is None:
        dy = dx
    grid_size = size2d(grid_size)
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
        transfer = base.expi(argument)
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
        transfer = base.expi(argument)
        transfer = torch.where(mask, transfer, torch.zeros_like(transfer))
    return {'transfer': transfer}


def _init_conv_common(
    paraxial: bool,
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    ksize: Size2d = None,
    padding: int | str = 'linear',
    dft_scale: bool = True,
    short_wl: bool = True,
    phase_factor: bool = True,
    scale_factor: bool = True,
) -> _Cache:
    if dy is None:
        dy = dx
    grid_size = size2d(grid_size)
    if ksize is None:
        ksize = grid_size
    ksize = size2d(ksize)
    wl = vector(wl).reshape(-1, 1, 1)
    k = 2 * torch.pi / wl
    distance = vector(distance).reshape(-1, 1, 1, 1)

    v, u = utils.sym_grid(2, ksize, (dy, dx))
    lateral_r2 = v.square() + u.square()
    wl_d_prod = wl * distance
    if paraxial:  # fresnel
        if short_wl is not ...:
            warnings.warn(f'{short_wl=} ignored when {paraxial=}')
        phase_scale = torch.pi / wl_d_prod  # k/(2d)
        argument = phase_scale * lateral_r2
        if phase_factor:
            argument = argument + (k * distance - torch.pi / 2)
        kernel = base.expi(argument)
        if scale_factor:
            kernel = kernel / wl_d_prod
    else:
        if phase_factor is not ...:
            warnings.warn(f'{phase_factor=} ignored when {paraxial=}')
        if scale_factor is not ...:
            warnings.warn(f'{scale_factor=} ignored when {paraxial=}')
        r2 = lateral_r2 + distance.square()
        r = r2.sqrt()
        argument = k * r
        ampl = distance / r2
        if short_wl:
            argument = argument - torch.pi / 2
            ampl = ampl / wl
        kernel = torch.polar(ampl, argument)
        if not short_wl:
            kernel = kernel * torch.complex(1 / r, -k) / (2 * torch.pi)

    spacing = (dy, dx) if dft_scale else (None, None)
    transfer = fourier.ft4conv(grid_size, kernel, (-2, -1), False, spacing, padding, False)
    return {'transfer': transfer}


def _ft_common(
    far_field: bool,  # Fraunhofer if True, Fresnel otherwise
    pupil: Ts | tuple[Ts, Ts],
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    phase_factor: bool = True,
    scale_factor: bool = True,
    dft_scale: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    argument_available = isinstance(pupil, tuple)
    grid_size = pupil[0].shape[-2:] if argument_available else pupil.shape[-2:]
    intermediate = _determine_cache(lambda: _init_ft_common(
        far_field,
        grid_size, wl, distance, dx, dy, delta_mode,
        argument_available, phase_factor, scale_factor
    ), intermediate, wl, distance, dx)

    if far_field:
        if argument_available:
            pupil = torch.polar(*pupil)
    else:  # Fresnel diffraction needs quadratic phase
        if argument_available:
            phase = pupil[1] + intermediate['quadratic_phase']
            if pupil[0].dtype == torch.bool:
                _phase_factor = base.expi(phase)
                pupil = torch.where(pupil[0], _phase_factor, torch.zeros_like(_phase_factor))
            else:
                pupil = torch.polar(pupil[0], phase)
        else:
            pupil = pupil * intermediate['quadratic_phase_factor']

    if dft_scale:
        field = fourier.ft2(pupil, intermediate['du'], intermediate['dv'])
    else:
        field = fourier.ft2(pupil)
    if phase_factor:
        field = field * intermediate['phase_factor']
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
    intermediate: _Cache = None,
) -> Ts:
    grid_size = (pupil.size(-2), pupil.size(-1))
    intermediate = _determine_cache(
        lambda: _init_as_common(paraxial, grid_size, wl, distance, dx, dy),
        intermediate, wl, distance, dx
    )

    # grid spacings are not needed to ensure correct DFT-scale
    # because transfer function itself is already correctly DFT-scaled
    field = fourier.ift2(fourier.ft2(pupil) * intermediate['transfer'])
    return field


def _conv_common(
    paraxial: bool,
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    ksize: Size2d = None,
    padding: int | str = 'linear',
    dft_scale: bool = True,
    short_wl: bool = True,
    phase_factor: bool = True,
    scale_factor: bool = True,
    simpson: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    grid_size = (pupil.size(-2), pupil.size(-1))
    ksize = size2d(ksize)
    intermediate = _determine_cache(
        lambda: _init_conv_common(
            paraxial, grid_size, wl, distance, dx, dy, ksize, padding,
            dft_scale, short_wl, phase_factor, scale_factor
        ), intermediate, wl, distance, dx
    )
    return fourier.conv_partial(
        pupil, intermediate['transfer'], ksize, (-2, -1), False, 'same', padding, simpson
    )


def _ft_args(kwargs: dict) -> tuple:
    delta_mode = cast(DMode, kwargs.pop('delta_mode', 'backward'))
    phase_factor = kwargs.pop('phase_factor', True)
    scale_factor = kwargs.pop('scale_factor', True)
    dft_scale = kwargs.pop('dft_scale', True)
    intermediate = kwargs.pop('intermediate', None)
    if len(kwargs) > 0:
        raise TypeError(f'Unexpected keyword arguments: {kwargs}')
    return delta_mode, phase_factor, scale_factor, dft_scale, intermediate


def _conv_args(kwargs: dict, paraxial: bool, additional: bool = False) -> tuple:
    if paraxial:
        kwargs['short_wl'] = ...
    else:
        kwargs['phase_factor'] = ...
        kwargs['scale_factor'] = ...
    dft_scale = kwargs.pop('dft_scale', True)
    ksize = cast(Size2d, kwargs.pop('ksize', None))
    padding = kwargs.pop('padding', 'linear')
    short_wl = kwargs.pop('short_wl', True)
    phase_factor = kwargs.pop('phase_factor', True)
    scale_factor = kwargs.pop('scale_factor', True)
    args = (ksize, padding, dft_scale, short_wl, phase_factor, scale_factor)
    if additional:
        simpson = kwargs.pop('simpson', True)
        intermediate = kwargs.pop('intermediate', None)
        args += (simpson, intermediate)
    if len(kwargs) > 0:
        raise TypeError(f'Unexpected keyword arguments: {kwargs}')
    return args


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


@overload
def init_fresnel_ft(  # hint for complete argument list
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    ampl_phase_form: bool = True,
    phase_factor: bool = True,
    scale_factor: bool = True,
) -> _Cache:
    pass


def init_fresnel_ft(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    **kwargs
) -> _Cache:
    r"""
    Pre-compute the intermediate results for :py:func:`fresnel_ft` and return them
    to avoid repeated computation.

    :param grid_size: Size of spatial dimensions :math:`(H, W)`.
    :type grid_size: int or tuple[int, int]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: distance: Propagation distance :math:`d`.
        A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: See :py:func:`fresnel_ft`.
    :type dx: float or Tensor
    :param dy: See :py:func:`fresnel_ft`.
    :type dy: float or Tensor
    :param kwargs: Optional keyword arguments:

        ===================  ========  ================
        Parameter            Type      Description
        ===================  ========  ================
        **delta_mode**       *str*     See :py:func:`fresnel_ft`.
        **ampl_phase_form**  *bool*    ``True`` if the pupil function passed to :py:func:`fresnel_ft`
                                       is a pair of tensors representing its amplitude and phase,
                                       otherwise a single complex tensor. See :py:func:`fresnel_ft`.
        **phase_factor**     *bool*    See :py:func:`fresnel_ft`.
        **scale_factor**     *bool*    See :py:func:`fresnel_ft`.
        ===================  ========  ================

    :return: Intermediate results that can be passed to :py:func:`fresnel_ft` as
        ``intermediate`` argument.
    :rtype: dict[str, Tensor]
    """
    delta_mode = cast(DMode, kwargs.get('delta_mode', 'backward'))
    phase_factor = kwargs.get('phase_factor', True)
    ampl_phase_form = kwargs.get('ampl_phase_form', True)
    scale_factor = kwargs.get('scale_factor', True)
    return _init_ft_common(
        False,
        grid_size, wl, distance, dx, dy,
        delta_mode, ampl_phase_form, phase_factor, scale_factor
    )


def init_fresnel_as(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
) -> _Cache:
    r"""
    Pre-compute the intermediate results for :py:func:`fresnel_as` and return them
    to avoid repeated computation.

    :param grid_size: Size of spatial dimensions :math:`(H, W)`.
    :type grid_size: int or tuple[int, int]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: distance: Propagation distance :math:`d`.
        A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: See :py:func:`fresnel_as`.
    :type dx: float or Tensor
    :param dy: See :py:func:`fresnel_as`.
    :type dy: float or Tensor
    :return: Intermediate results that can be passed to :py:func:`fresnel_as` as
        ``intermediate`` argument.
    :rtype: dict[str, Tensor]
    """
    return _init_as_common(True, grid_size, wl, distance, dx, dy)


@overload
def init_fraunhofer(  # hint for complete argument list
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    phase_factor: bool = True,
    scale_factor: bool = True,
) -> _Cache:
    pass


def init_fraunhofer(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    **kwargs
) -> _Cache:
    r"""
    Pre-compute the intermediate results for :py:func:`fraunhofer` and return them
    to avoid repeated computation.

    :param grid_size: Size of spatial dimensions :math:`(H, W)`.
    :type grid_size: int or tuple[int, int]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: distance: Propagation distance :math:`d`.
        A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: See :py:func:`fraunhofer`.
    :type dx: float or Tensor
    :param dy: See :py:func:`fraunhofer`.
    :type dy: float or Tensor
    :param kwargs: Additional keyword arguments, allowing all the additional keyword arguments
        of :py:func:`fraunhofer` other than ``dft_scale`` and ``intermediate``.
    :return: Intermediate results that can be passed to :py:func:`fraunhofer` as
        ``intermediate`` argument.
    :rtype: dict[str, Tensor]
    """
    delta_mode = cast(DMode, kwargs.get('delta_mode', 'backward'))
    phase_factor = kwargs.get('phase_factor', True)
    scale_factor = kwargs.get('scale_factor', True)
    return _init_ft_common(
        False,
        grid_size, wl, distance, dx, dy,
        delta_mode, False, phase_factor, scale_factor
    )


def init_rayleigh_sommerfeld_as(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
) -> _Cache:
    r"""
    Pre-compute the intermediate results for :py:func:`rayleigh_sommerfeld_as` and return them
    to avoid repeated computation.

    :param grid_size: Size of spatial dimensions :math:`(H, W)`.
    :type grid_size: int or tuple[int, int]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: distance: Propagation distance :math:`d`.
        A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: See :py:func:`rayleigh_sommerfeld_as`.
    :type dx: float or Tensor
    :param dy: See :py:func:`rayleigh_sommerfeld_as`.
    :type dy: float or Tensor
    :return: Intermediate results that can be passed to :py:func:`rayleigh_sommerfeld_as` as
        ``intermediate`` argument.
    :rtype: dict[str, Tensor]
    """
    return _init_as_common(False, grid_size, wl, distance, dx, dy)


@overload
def init_fresnel_conv(  # hint for complete argument list
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    ksize: Size2d = None,
    padding: int | str = 'linear',
    dft_scale: bool = True,
    phase_factor: bool = True,
    scale_factor: bool = True,
) -> _Cache:
    pass


def init_fresnel_conv(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    **kwargs
) -> _Cache:
    r"""
    Pre-compute the intermediate results for :py:func:`fresnel_conv` and return them
    to avoid repeated computation.

    :param grid_size: Size of spatial dimensions :math:`(H, W)`.
    :type grid_size: int or tuple[int, int]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: distance: Propagation distance :math:`d`.
        A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: See :py:func:`fresnel_conv`.
    :type dx: float or Tensor
    :param dy: See :py:func:`fresnel_conv`.
    :type dy: float or Tensor
    :param kwargs: Additional keyword arguments, allowing all the additional keyword arguments
        of :py:func:`fresnel_conv` other than ``simpson`` and ``intermediate``.
    :return: Intermediate results that can be passed to :py:func:`fresnel_conv` as
        ``intermediate`` argument.
    :rtype: dict[str, Tensor]
    """
    return _init_conv_common(True, grid_size, wl, distance, dx, dy, *_conv_args(kwargs, True))


@overload
def init_rayleigh_sommerfeld_conv(  # hint for complete argument list
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    ksize: Size2d = None,
    padding: int | str = 'linear',
    dft_scale: bool = True,
    short_wl: bool = True,
) -> _Cache:
    pass


def init_rayleigh_sommerfeld_conv(
    grid_size: Size2d,
    wl: Vector,
    distance: Vector,
    dx: Spacing,
    dy: Spacing = None,
    **kwargs
) -> _Cache:
    r"""
    Pre-compute the intermediate results for :py:func:`rayleigh_sommerfeld_conv` and return them
    to avoid repeated computation.

    :param grid_size: Size of spatial dimensions :math:`(H, W)`.
    :type grid_size: int or tuple[int, int]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: distance: Propagation distance :math:`d`.
        A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: See :py:func:`rayleigh_sommerfeld_conv`.
    :type dx: float or Tensor
    :param dy: See :py:func:`rayleigh_sommerfeld_conv`.
    :type dy: float or Tensor
    :param kwargs: Additional keyword arguments, allowing all the additional keyword arguments
        of :py:func:`rayleigh_sommerfeld_conv` other than ``simpson`` and ``intermediate``.
    :return: Intermediate results that can be passed to :py:func:`rayleigh_sommerfeld_conv` as
        ``intermediate`` argument.
    :rtype: dict[str, Tensor]
    """
    return _init_conv_common(False, grid_size, wl, distance, dx, dy, *_conv_args(kwargs, False))


@overload
def fresnel_ft(
    pupil: Ts | tuple[Ts, Ts],
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    phase_factor: bool = True,
    scale_factor: bool = True,
    dft_scale: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    pass


def fresnel_ft(
    pupil: Ts | tuple[Ts, Ts],
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    **kwargs
) -> Ts:
    r"""
    Computes Fresnel diffraction integral using Fourier transform method:

    .. math::
        U'(x,y)&=\frac{\e^{\i kd}}{\i\lambda d}\iint U(u,v)
        \e^{\i\frac{k}{2d}\left((x-u)^2+(y-v)^2\right)}\d u\d v \\
        &=\frac{\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}}{\i\lambda d}
        \ft\left\{U(u,v)\e^{\i\frac{k}{2d}(u^2+v^2)}\right\}_
        {f_u=\frac{x}{\lambda d},f_v=\frac{y}{\lambda d}}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ft` represents Fourier transform and :math:`k=2\pi/\lambda`.

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
    :type pupil: Tensor or tuple[Tensor, Tensor]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :type dx: float or Tensor
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even though ``dy`` is given.
    :type dy: float or Tensor
    :param kwargs: Optional keyword arguments:

        ================  ======  ================
        Parameter         Type    Description
        ================  ======  ================
        **delta_mode**    *str*   On which plane ``dx`` and ``dy`` are defined, source plane
                                  (``forward``) or target plane (``backward``).
                                  Default: ``backward``.
        **phase_factor**  *bool*  Whether to multiply :math:`\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}/\i`.
                                  It can be omitted to reduce computation if only amplitude matters.
                                  Default: ``True``.
        **scale_factor**  *bool*  Whether to multiply the factor :math:`1/(\lambda d)`.
                                  It can be omitted to reduce computation if relative scales
                                  across different :math:`d` and :math:`\lambda` do not matter.
                                  Default: ``True``.
        **dft_scale**     *bool*  Whether to apply DFT-scale.
                                  See :ref:`ref_fourier_fourier_transform`. Default: ``True``.
        **intermediate**  *dict*  Cached intermediate results returned by :py:func:`init_fresnel_ft`.
                                  This can be used to speed up computation. Default: omitted.
        ================  ======  ================
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :rtype: Tensor
    """
    return _ft_common(False, pupil, wl, distance, dx, dy, *_ft_args(kwargs))


def fresnel_2ft(

) -> Ts:  # TODO
    pass


def fresnel_as(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    intermediate: _Cache = None,
) -> Ts:
    r"""
    Computes Fresnel diffraction integral using angular spectrum method:

    .. math::
        U'(x,y)&=\frac{\e^{\i kd}}{\i\lambda d}\iint U(u,v)
        \e^{\i\frac{k}{2d}\left((x-u)^2+(y-v)^2\right)}\d u\d v \\
        &=U(x,y)\ast\frac{\e^{\i kd}}{\i\lambda d}\e^{\i\frac{k}{2d}(x^2+y^2)}\\
        &=\ft^{-1}\left\{\ft\{U(x,y)\}\e^{\i kd}\e^{-\i\pi\lambda d(f_X^2+f_Y^2)}\right\}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ast` represents convolution and :math:`k=2\pi/\lambda`.

    Compared to :py:func:`fresnel_ft`, this function always computes the diffraction integral
    completely and exactly, including phase factor, scale factor and DFT scale.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_fresnel_as` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param Tensor pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :type dx: float or Tensor
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even though ``dy`` is given.
    :type dy: float or Tensor
    :param dict intermediate: Cached intermediate results returned by :py:func:`init_fresnel_as`.
        This can be used to speed up computation.
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :rtype: Tensor
    """
    return _as_common(True, pupil, wl, distance, dx, dy, intermediate)


@overload
def fraunhofer(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    delta_mode: DMode = 'backward',
    phase_factor: bool = True,
    scale_factor: bool = True,
    dft_scale: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    pass


def fraunhofer(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    **kwargs
) -> Ts:
    r"""
    Computes Fraunhofer diffraction integral using Fourier transform method:

    .. math::
        U'(x,y)&=\frac{\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}}{\i\lambda d}
        \iint U(u,v)\e^{-\i\frac{k}{d}(xu+yv)}\d u\d v \\
        &=\frac{\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}}{\i\lambda d}
        \ft\{U(u,v)\}_{f_u=\frac{x}{\lambda d},f_v=\frac{y}{\lambda d}}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ft` represents Fourier transform and :math:`k=2\pi/\lambda`.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_fraunhofer` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param Tensor pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :type dx: float or Tensor
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even though ``dy`` is given.
    :type dy: float or Tensor
    :param kwargs: Optional keyword arguments:

        ================  ======  ================
        Parameter         Type    Description
        ================  ======  ================
        **delta_mode**    *str*   On which plane ``dx`` and ``dy`` are defined, source plane
                                  (``forward``) or target plane (``backward``). Default: ``backward``.
        **phase_factor**  *bool*  Whether to multiply :math:`\e^{\i kd}\e^{\i\frac{k}{2d}(x^2+y^2)}/\i`.
                                  It can be omitted to reduce computation if only amplitude matters.
                                  Default: ``True``.
        **scale_factor**  *bool*  Whether to multiply the factor :math:`1/(\lambda d)`.
                                  It can be omitted to reduce computation if relative scales
                                  across different :math:`d` and :math:`\lambda` do not matter.
                                  Default: ``True``.
        **dft_scale**     *bool*  Whether to apply DFT-scale.
                                  See :ref:`ref_fourier_fourier_transform`. Default: ``True``.
        **intermediate**  *dict*  Cached intermediate results returned by :py:func:`init_fraunhofer`.
                                  This can be used to speed up computation. Default: omitted.
        ================  ======  ================
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :rtype: Tensor
    """
    return _ft_common(True, pupil, wl, distance, dx, dy, *_ft_args(kwargs))


def rayleigh_sommerfeld_as(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    intermediate: _Cache = None,
) -> Ts:
    r"""
    Computes the first Rayleigh-Sommerfeld diffraction integral using convolution    :

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

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_rayleigh_sommerfeld_as` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param Tensor pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :type dx: float or Tensor
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even though ``dy`` is given.
    :type dy: float or Tensor
    :param dict intermediate: Cached intermediate results returned by :py:func:`init_rayleigh_sommerfeld_as`.
        This can be used to speed up computation.
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :rtype: Tensor
    """
    return _as_common(False, pupil, wl, distance, dx, dy, intermediate)


@overload
def fresnel_conv(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    ksize: Size2d = None,
    padding: int | str = 'linear',
    dft_scale: bool = True,
    phase_factor: bool = True,
    scale_factor: bool = True,
    simpson: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    pass


def fresnel_conv(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    **kwargs
) -> Ts:
    r"""
    Computes Fresnel diffraction integral using convolution method:

    .. math::
        U'(x,y)&=\frac{\e^{\i kd}}{\i\lambda d}\iint U(u,v)
        \e^{\i\frac{k}{2d}\left((x-u)^2+(y-v)^2\right)}\d u\d v \\
        &=U(x,y)\ast\frac{\e^{\i kd}}{\i\lambda d}\e^{\i\frac{k}{2d}(x^2+y^2)}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes, :math:`\lambda`
    is wavelength, :math:`\ast` represents convolution and :math:`k=2\pi/\lambda`.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_fresnel_conv` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param Tensor pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :type dx: float or Tensor
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even though ``dy`` is given.
    :type dy: float or Tensor
    :param kwargs: Optional keyword arguments:

        ================  ===========  ================
        Parameter         Type         Description
        ================  ===========  ================
        **ksize**         *int |       Spatial dimension of the kernel function
                          tuple[int,   in vertical and horizontal directions.
                          int]*        Default: identical to that of ``pupil``.
        **padding**       *int |       See :py:func:`~dnois.fourier.dconv`.
                          str*         Default: ``linear``.
        **dft_scale**     *bool*       Whether to apply DFT-scale.
                                       See :ref:`ref_fourier_fourier_transform`. Default: ``True``.
        **phase_factor**  *bool*       Whether to multiply :math:`\e^{\i kd}/\i` in kernel function.
                                       It can be omitted to reduce computation if only amplitude matters.
                                       Default: ``True``.
        **scale_factor**  *bool*       Whether to multiply :math:`1/(\lambda d)` in kernel function.
                                       It can be omitted to reduce computation if relative scales
                                       across different :math:`d` and :math:`\lambda` do not matter.
                                       Default: ``True``.
        **simpson**       *bool*       Whether to apply Simpson's rule. Default: ``True``.
        **intermediate**  *dict*       Cached intermediate results returned by :py:func:`init_fresnel_conv`.
                                       This can be used to speed up computation. Default: omitted.
        ================  ===========  ================
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :rtype: Tensor
    """
    return _conv_common(True, pupil, wl, distance, dx, dy, *_conv_args(kwargs, True, True))


@overload
def rayleigh_sommerfeld_conv(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    ksize: Size2d = None,
    padding: int | str = 'linear',
    dft_scale: bool = True,
    short_wl: bool = True,
    simpson: bool = True,
    intermediate: _Cache = None,
) -> Ts:
    pass


def rayleigh_sommerfeld_conv(
    pupil: Ts,
    wl: Vector = None,
    distance: Vector = None,
    dx: Spacing = None,
    dy: Spacing = None,
    **kwargs,
) -> Ts:
    r"""
    Computes Fresnel diffraction integral using convolution method:

    .. math::
        U'(x,y)&=-\frac{1}{2\pi}\iint U(u,v)\pfrac{}{d}\frac{\e^{\i kr_d}}{r_d}\d u\d v\\
        &=-\frac{1}{2\pi}U(x,y)\ast\pfrac{}{d}\frac{\e^{\i kr}}{r}\\
        &=U(x,y)\ast\frac{z}{2\pi r^2}(\frac{1}{r}-\i k)\e^{\i kr}

    where :math:`U` and :math:`U'` are the complex amplitude on source and target
    plane, respectively, :math:`d` is the distance between two planes,
    :math:`r=\sqrt{x^2+y^2+d^2}`, :math:`\lambda` is wavelength, :math:`k=2\pi/\lambda`
    and :math:`\ast` represents convolution.

    Note that some intermediate computational steps don't depend on ``pupil`` and pre-computing
    their results can avoid repeated computation if this function is expected to be called
    multiple times, as long as the arguments other than ``pupil`` don't change. This can be
    done by calling :py:func:`init_rayleigh_sommerfeld_conv` and passing its return value as
    ``intermediate`` argument. In this case, ``wl``, ``distance``, ``dx`` and ``dy`` cannot
    be passed. If all of them are given, ``intermediate`` will be ignored anyway.

    :param Tensor pupil: Pupil function :math:`U`. A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
    :type wl: float, Sequence[float] or Tensor
    :param distance: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
    :type distance: float, Sequence[float] or Tensor
    :param dx: Grid spacing in horizontal direction to ensure correct scale for DFT.
        A scalar or a tensor of shape :math:`(\cdots,N_d,N_\lambda)`. Default: ignored.
    :type dx: float or Tensor
    :param dy: Grid spacing in vertical direction, similar to ``dx``. Default: same as ``dx``.
        Note that ``dy`` will be also ignored if ``dx`` is ``None`` even though ``dy`` is given.
    :type dy: float or Tensor
    :param kwargs: Optional keyword arguments:

        ================  ===========  ================
        Parameter         Type         Description
        ================  ===========  ================
        **ksize**         *int |       Spatial dimension of the kernel function
                          tuple[int,   in vertical and horizontal directions.
                          int]*        Default: identical to that of ``pupil``.
        **padding**       *int | str*  See :py:func:`~dnois.fourier.dconv`. Default: ``linear``.
        **dft_scale**     *bool*       Whether to apply DFT-scale.
                                       See :ref:`ref_fourier_fourier_transform`. Default: ``True``.
        **short_wl**      *bool*       Whether to ignore :math:`\frac{1}{r}` term in kernel function.
                                       Default: ``True``.
        **intermediate**  *dict*       Cached intermediate results returned by
                                       :py:func:`init_rayleigh_sommerfeld_conv`. This can be used
                                       to speed up computation. Default: omitted.
        ================  ===========  ================
    :return: Diffracted complex amplitude :math:`U'`.
        A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
    :rtype: Tensor
    """
    return _conv_common(False, pupil, wl, distance, dx, dy, *_conv_args(kwargs, False, True))


class Diffraction(base.TensorContainerMixIn, nn.Module, metaclass=abc.ABCMeta):
    r"""
    Base class for all diffraction modules. Its subclasses implement the computation
    of various diffraction integral. One advantage of using module API rather than
    functional API is that intermediate results are handled automatically.

    :param grid_size: Size of spatial dimensions :math:`(H, W)`.
    :type grid_size: int or tuple[int, int]
    :param wl: Wavelengths :math:`\lambda`. A scalar or a tensor of shape :math:`(N_\lambda,)`.
        Default: ignored temporarily.
    :type wl: float, Sequence[float] or Tensor
    :param d: Propagation distance :math:`d`. A scalar or a tensor of shape :math:`(N_d,)`.
        Default: ignored temporarily.
    :type d: float, Sequence[float] or Tensor
    """
    u: Ts  #: One of coordinates on source plane.
    v: Ts  #: One of coordinates on source plane.
    x: Ts  #: One of coordinates on target plane.
    y: Ts  #: One of coordinates on target plane.
    u_range: tuple[Ts, Ts]  #: Range of :attr:`u`. A pair of 0D tensor.
    v_range: tuple[Ts, Ts]  #: Range of :attr:`v`. A pair of 0D tensor.
    x_range: tuple[Ts, Ts]  #: Range of :attr:`x`. A pair of 0D tensor.
    y_range: tuple[Ts, Ts]  #: Range of :attr:`y`. A pair of 0D tensor.
    u_size: Ts  #: Span of :attr:`u`, i.e. ``u_range[1] - u_range[0]``.
    v_size: Ts  #: Span of :attr:`v`, i.e. ``v_range[1] - v_range[0]``.
    x_size: Ts  #: Span of :attr:`x`, i.e. ``x_range[1] - x_range[0]``.
    y_size: Ts  #: Span of :attr:`y`, i.e. ``y_range[1] - y_range[0]``.

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
            # device and dtype of spacing are already consistent with self
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
        """
        Wavelengths :math:`\\lambda`. It can be assigned with a float, sequence of float,
        0D or 1D tensor. This property returns a 0D tensor if assigned with a float or 0D
        tensor, or a 1D tensor otherwise.
        """
        return self.__dict__['_buffers']['_wl']

    @wl.setter
    def wl(self, value: Vector):
        value = scalar(value) if is_scalar(value) else vector(value)
        self.register_buffer('_wl', value, False)
        self._ready = False

    @property
    def d(self) -> Ts:
        """
        Diffraction distances :math:`z`. It can be assigned with a float, sequence of float,
        0D or 1D tensor. This property returns a 0D tensor if assigned with a float or 0D
        tensor, or a 1D tensor otherwise.
        """
        return self.__dict__['_buffers']['_d']

    @d.setter
    def d(self, value: Vector):
        value = scalar(value) if is_scalar(value) else vector(value)
        self.register_buffer('_d', value, False)
        self._ready = False

    @property
    def du(self) -> Ts:
        """
        Grid spacing for :math:`u` coordinate. See corresponding functional API reference
        for allowed values. Note that this property depends on :attr:`dx`, vice versa.
        """
        return self.__dict__['_buffers']['_du']

    @du.setter
    def du(self, value: Spacing):
        self._set_spacing('_du', value)

    @property
    def dv(self) -> Ts:
        """
        Grid spacing for :math:`v` coordinate. See corresponding functional API reference
        for allowed values. Note that this property depends on :attr:`dy`, vice versa.
        """
        return self.__dict__['_buffers']['_dv']

    @dv.setter
    def dv(self, value: Spacing):
        self._set_spacing('_dv', value)

    @property
    def dx(self) -> Ts:
        """
        Grid spacing for :math:`x` coordinate. See corresponding functional API reference
        for allowed values. Note that this property depends on :attr:`du`, vice versa.
        """
        return self.du

    @dx.setter
    def dx(self, value: Spacing):
        self.du = value

    @property
    def dy(self) -> Ts:
        """
        Grid spacing for :math:`y` coordinate. See corresponding functional API reference
        for allowed values. Note that this property depends on :attr:`dv`, vice versa.
        """
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

    def _delegate(self) -> Ts:
        return self.wl

    def _set_spacing(self, key, value):
        if is_scalar(value):
            value = scalar(value)
        self.register_buffer(key, value, False)
        self._ready = False


class FourierTransformBased(Diffraction):
    _far_field: bool
    # either dx/dy or du/dv are stored, another pair is computed dynamically
    _dx: Ts
    _dy: Ts

    def __init__(
        self,
        grid_size: Size2d,
        spacing: Spacing | tuple[Spacing, Spacing] = None,  # not dx/dy or du/dv explicitly
        wl: Vector = None,
        d: Vector = None,
        delta_mode: DMode = 'backward',
        phase_factor: bool = True,
        scale_factor: bool = True,
        dft_scale: bool = True,
    ):
        super().__init__(grid_size, wl, d)
        self._delta_mode = delta_mode
        self._ppf = phase_factor
        self._sf = scale_factor
        self._dft_scale = dft_scale
        self._apf = False

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
        r"""
        Compute diffraction integral for given pupil function ``pupil``

        :param pupil: See :func:`fresnel_ft`.
        :return: Diffracted complex amplitude :math:`U'`.
            A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        :rtype: Tensor
        """
        is_tuple = isinstance(pupil, tuple)
        if self._apf:
            if not is_tuple:
                raise RuntimeError(f'A pair of Tensors expected, got {type(pupil)}')
        elif is_tuple:
            raise RuntimeError(f'A single Tensors expected, got {type(pupil)}')

        self._build()
        return _ft_common(
            self._far_field,
            pupil,  # leave other parameters not provided, they are not used with cache present
            phase_factor=self._ppf,
            scale_factor=self._sf,
            dft_scale=self._dft_scale,
            intermediate=self._cache,
        )


class AngularSpectrumBased(Diffraction):
    _paraxial: bool

    def __init__(
        self,
        grid_size: Size2d,
        dx: Spacing = None,
        dy: Spacing = None,
        wl: Vector = None,
        d: Vector = None,
    ):
        super().__init__(grid_size, wl, d)
        if dy is None:
            dy = dx
        self.dv, self.du = dy, dx  # dx/dy and du/dv are shared

    def _init_cache(self):
        return _init_as_common(
            self._paraxial,
            self._grid_size,
            self.wl, self.d, self.du, self.dv,
        )

    def forward(self, pupil: Ts) -> Ts:
        r"""
        Compute diffraction integral for given pupil function ``pupil``

        :param Tensor pupil: Pupil function :math:`U`.
            A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        :return: Diffracted complex amplitude :math:`U'`.
            A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        :rtype: Tensor
        """
        self._build()
        return _as_common(self._paraxial, pupil, intermediate=self._cache)


class ConvolutionBased(Diffraction):
    _paraxial: bool

    def __init__(
        self,
        grid_size: Size2d,
        wl: Vector = None,
        d: Vector = None,
        ksize: Size2d = None,
        padding: int | str = 'linear',
        dft_scale: bool = True,
        simpson: bool = True,
    ):
        super().__init__(grid_size, wl, d)
        self._ksize = self._grid_size if ksize is None else size2d(ksize)
        self._padding = padding
        self._dft_scale = dft_scale
        self._simpson = simpson

    def forward(self, pupil: Ts) -> Ts:
        r"""
        Compute diffraction integral for given pupil function ``pupil``

        :param Tensor pupil: Pupil function :math:`U`.
            A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        :return: Diffracted complex amplitude :math:`U'`.
            A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        :rtype: Tensor
        """
        self._build()
        return _conv_common(
            self._paraxial, pupil,
            ksize=self._ksize,
            padding=self._padding,
            simpson=self._simpson,
            intermediate=self._cache,
        )

    def _init_cache(self):
        return _init_conv_common(
            self._paraxial, self._grid_size,
            self.wl, self.d, self.dx, self.dy,
            self._ksize, self._padding, self._dft_scale, **self._additional_args(),
        )

    def _additional_args(self) -> dict:
        raise NotImplementedError()  # not as abstract for conciseness. Equivalent.


class FresnelFT(FourierTransformBased):
    """
    Module for computing Fresnel diffraction using Fourier transform method.
    See :func:`fresnel_ft` and :func:`init_fresnel_ft` for more details and
    description of parameters.
    """
    _far_field: bool = False

    def __init__(
        self,
        grid_size: Size2d,
        spacing: Spacing | tuple[Spacing, Spacing] = None,  # not dx/dy or du/dv explicitly
        wl: Vector = None,
        d: Vector = None,
        delta_mode: DMode = 'backward',
        phase_factor: bool = True,
        scale_factor: bool = True,
        dft_scale: bool = True,
        ampl_phase_form: bool = True,
    ):
        super().__init__(grid_size, spacing, wl, d, delta_mode, phase_factor, scale_factor, dft_scale)
        self._apf = ampl_phase_form


class Fraunhofer(FourierTransformBased):
    """
    Module for computing Fraunhofer diffraction using Fourier transform method.
    See :func:`fraunhofer` and :func:`fraunhofer` for more details and
    description of parameters.
    """
    _far_field: bool = True

    def forward(self, pupil: Ts) -> Ts:  # override signature and docstring
        r"""
        Compute diffraction integral for given pupil function ``pupil``

        :param Tensor pupil: Pupil function :math:`U`.
            A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        :return: Diffracted complex amplitude :math:`U'`.
            A tensor of shape :math:`(\cdots,N_d,N_\lambda,H,W)`.
        :rtype: Tensor
        """
        return super().forward(pupil)


class FresnelAS(AngularSpectrumBased):
    """
    Module for computing Fresnel diffraction using angular spectrum method.
    See :func:`fresnel_as` and :func:`init_fresnel_as` for more details and
    description of parameters.
    """
    _paraxial: bool = True


class RayleighSommerfeldAS(AngularSpectrumBased):
    """
    Module for computing Rayleigh-Sommerfeld diffraction using angular spectrum method.
    See :func:`rayleigh_sommerfeld_as` and :func:`init_rayleigh_sommerfeld_as`
    for more details and description of parameters.
    """
    _paraxial: bool = False


class FresnelConv(ConvolutionBased):
    """
    Module for computing Fresnel diffraction using convolution method.
    See :func:`fresnel_conv` and :func:`init_fresnel_conv`
    for more details and description of parameters.
    """
    _paraxial = True

    def __init__(
        self,
        grid_size: Size2d,
        wl: Vector = None,
        d: Vector = None,
        ksize: Size2d = None,
        padding: int | str = 'linear',
        phase_factor: bool = True,
        scale_factor: bool = True,
        dft_scale: bool = True,
        simpson: bool = True
    ):
        super().__init__(grid_size, wl, d, ksize, padding, dft_scale, simpson)
        self._pf = phase_factor
        self._sf = scale_factor

    def _additional_args(self) -> dict:
        return {'phase_factor': self._pf, 'scale_factor': self._sf}


class RayleighSommerfeldConv(ConvolutionBased):
    """
    Module for computing Rayleigh-Sommerfeld diffraction using convolution method.
    See :func:`rayleigh_sommerfeld_conv` and :func:`init_rayleigh_sommerfeld_conv`
    for more details and description of parameters.
    """
    _paraxial = False

    def __init__(
        self,
        grid_size: Size2d,
        wl: Vector = None,
        d: Vector = None,
        ksize: Size2d = None,
        padding: int | str = 'linear',
        short_wl: bool = True,
        dft_scale: bool = True,
        simpson: bool = True
    ):
        super().__init__(grid_size, wl, d, ksize, padding, dft_scale, simpson)
        self._short = short_wl

    def _additional_args(self) -> dict:
        return {'short_wl': self._short}
