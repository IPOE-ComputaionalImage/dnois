import abc
import math

import torch
from torch import nn

from . import _surf
from ._surf import *
from ... import mt, torch as _t
from ...base.typing import Any, Ts, Scalar, Sequence, scalar, cast

__all__ = [
    'Conic',
    'EvenAspherical',
    'Fresnel',
    'PolynomialPhase',
    'Spherical',
    'Standard',
]
__all__ += _surf.__all__


def _spherical(r2: Ts, c: Ts, k: Ts = None) -> Ts:  # TODO: optimize
    a = c.square() if k is None else c.square() * (1 + k)
    return c * r2 / (1 + torch.sqrt(1 - r2 * a))


def _spherical_der_wrt_r2(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    _1 = c.square() if k is None else c.square() * (1 + k)
    _2 = r2 * _1
    _3 = torch.sqrt(1 - _2)
    _4 = _3 + 1
    return c / _4 * (1 + _2 / (2 * _3 * _4))


class _SphericalBase(CircularSurface, metaclass=abc.ABCMeta):  # docstring for Spherical
    r"""
    Spherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-c^2r^2}}

    where :math:`c` is radius of curvature.

    See :py:class:`CircularSurface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    """

    def __init__(
        self, roc: Scalar,
        material: mt.Material | str,
        aperture: Aperture | float = None,
        reflective: bool = False,
        newton_config: dict[str, Any] = None,
        *,
        d: Scalar = None
    ):
        super().__init__(material, aperture, reflective, newton_config, d=d)
        self.roc: nn.Parameter = nn.Parameter(scalar(roc))  #: Radius of curvature.

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['roc'] = self._attr2dictitem('roc', keep_tensor)
        return d

    @property
    def geo_radius(self) -> Ts:
        return self.roc


class Spherical(_SphericalBase):
    __doc__ = _SphericalBase.__doc__

    def h_r2(self, r2: Ts) -> Ts:
        return _spherical(r2, 1 / self.roc)

    def h_derivative_r2(self, r2: Ts) -> Ts:
        return _spherical_der_wrt_r2(r2, 1 / self.roc)

    def _solve_t(self, ray: BatchedRay) -> Ts:
        if self.roc.isinf().all():
            return (self.context.z - ray.z) / ray.d_z

        o_hat = torch.cat([ray.o[..., :2], ray.z.unsqueeze(-1) - self.context.z], -1) / self.roc
        qc_b = torch.sum(o_hat * ray.d, -1) - ray.d_z  # quadratic coefficient: b
        qc_c = o_hat.square().sum(-1) - 2 * o_hat[..., 2]  # quadratic coefficient: c
        q_sqrt_delta = torch.sqrt(qc_b.square() - qc_c)  # sqrt of delta in quadratic equation
        q_sqrt_delta = torch.copysign(q_sqrt_delta, ray.d_z)
        t_hat = -qc_b - q_sqrt_delta
        t = t_hat * self.roc

        nan_mask = t.isnan()
        if nan_mask.any():
            h_ext_value = self.context.z + self.roc
            t = torch.where(nan_mask, (h_ext_value - ray.z) / ray.d_z, t)
        return t


class _ConicBase(_SphericalBase, metaclass=abc.ABCMeta):  # docstring for Conic
    r"""
    Conic surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-(1+k)c^2r^2}}

    where :math:`c` is radius of curvature and :math:`k` is conic coefficient.

    See :py:class:`CircularSurface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    :param conic: Conic coefficient.
    :type conic: float or Tensor
    """

    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        material: mt.Material | str,
        aperture: Aperture | float = None,
        reflective: bool = False,
        newton_config: dict[str, Any] = None,
        *,
        d: Scalar = None
    ):
        super().__init__(roc, material, aperture, reflective, newton_config, d=d)
        self.conic: nn.Parameter = nn.Parameter(scalar(conic))  #: Conic coefficient.

    def h_derivative_r2(self, r2: Ts) -> Ts:
        return _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic)

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['conic'] = self._attr2dictitem('conic', keep_tensor)
        return d

    @property
    def geo_radius(self) -> Ts:
        if self.conic.item() <= -1:
            return self.conic.new_tensor(float('inf'))
        else:
            return self.roc / torch.sqrt(1 + self.conic)


class Conic(_ConicBase):
    __doc__ = _ConicBase.__doc__

    def h_r2(self, r2: Ts) -> Ts:
        return _spherical(r2, 1 / self.roc, self.conic)

    def _solve_t(self, ray: BatchedRay) -> Ts:
        if self.roc.isinf().all():
            return (self.context.z - ray.z) / ray.d_z

        o_hat = torch.cat([ray.o[..., :2], ray.z.unsqueeze(-1) - self.context.z], -1) / self.roc
        qc_a = 1 + self.conic * ray.d_z.square()  # quadratic coefficient: a
        _1 = o_hat * ray.d
        _1[..., 2] = _1[..., 2] * (self.conic + 1)
        qc_b = _1.sum(-1) - ray.d_z  # quadratic coefficient: b
        _2 = o_hat.square()
        _2[..., 2] = _2[..., 2] * (self.conic + 1)
        qc_c = _2.sum(-1) - 2 * o_hat[..., 2]  # quadratic coefficient: c
        q_sqrt_delta = torch.sqrt(qc_b.square() - qc_a * qc_c)
        q_sqrt_delta = torch.copysign(q_sqrt_delta, ray.d_z)
        t_hat = -(qc_b + q_sqrt_delta) / qc_a
        t = t_hat * self.roc

        nan_mask = t.isnan()
        if nan_mask.any():
            h_ext_value = self.context.z + self.roc
            t = torch.where(nan_mask, (h_ext_value - ray.z) / ray.d_z, t)
        return t


class Standard(Conic):
    """Alias for :py:class:`~Conic`. This name complies with the convention of Zemax."""


class EvenAspherical(_ConicBase):
    r"""
    Even aspherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-(1+k)c^2r^2}}+\sum_{i=1}^N a_i r^{2i}

    where :math:`c` is radius of curvature, :math:`k` is conic coefficient
    and :math:`\{a_i\}_{i=1}^N` are even aspherical coefficients.

    See :py:class:`CircularSurface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    :param conic: Conic coefficient.
    :type conic: float or Tensor
    :param coefficients: Even aspherical coefficients.
    :type coefficients: Sequence[float | Tensor]
    """

    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        coefficients: Sequence[Scalar],
        material: mt.Material | str,
        aperture: Aperture | float = None,
        reflective: bool = False,
        newton_config: dict[str, Any] = None,
        *,
        d: Scalar = None
    ):
        super().__init__(roc, conic, material, aperture, reflective, newton_config, d=d)
        for i, a in enumerate(coefficients):
            self.register_parameter(f'a{i + 1}', nn.Parameter(scalar(a)))
        self._n = len(coefficients)

    def h_r2(self, r2: Ts) -> Ts:
        a = 0
        for c in reversed(self.coefficients):
            a = (a + c) * r2
        return _spherical(r2, 1 / self.roc, self.conic) + a

    def h_derivative_r2(self, r2: Ts) -> Ts:
        s_der = _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic)
        a_der = 0
        for i in range(len(self.coefficients), 0, -1):
            a_der = a_der * r2 + self.coefficients[i - 1] * i
        return s_der + a_der

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['coefficients'] = self.coefficients if keep_tensor else [c.item() for c in self.coefficients]
        return d

    @property
    def coefficients(self) -> list[Ts]:
        r"""
        Aspherical coefficients. Note that the element with index ``i``
        represents coefficient :math:`a_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'a{i + 1}') for i in range(self._n)]

    @property
    def even_aspherical_items(self) -> int:
        """
        Number of even aspherical coefficients.

        :type: int
        """
        return self._n


def _diff(x: Ts, d: float, dim: int) -> Ts:
    s = x.size(dim)
    diff_central = (x.narrow(dim, 2, s - 2) - x.narrow(dim, 0, s - 2)) / (2 * d)  # central difference
    diff_lower = (x.narrow(dim, 1, 1) - x.narrow(dim, 0, 1)) / d  # forward difference
    diff_upper = (x.narrow(dim, -1, 1) - x.narrow(dim, -2, 1)) / d  # backward difference
    diff = torch.cat([diff_lower, diff_central, diff_upper], dim)
    return diff


def _coordinate2index(x: Ts, n: int) -> tuple[Ts, Ts]:
    if n % 2 == 0:  # even
        x_cell = x.floor() + 0.5  # round to nearest half integers
    else:  # odd
        x_cell = x.round()  # round to nearest integers
    idx = x_cell + (n - 1) / 2
    return idx.int().clamp(0, n - 1), x - x_cell


class PlanarPhase(Planar, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        r"""
        Computes gradient of imparted phase shift :math:`(\pfrac{\phi}{x},\pfrac{\phi}{y})`.

        :param Tensor x: x coordinate. A tensor of shape `(...)`.
        :param Tensor y: y coordinate. A tensor of shape `(...)`.
        :return: Gradient. A tensor of shape `(..., 2)`.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        r"""
        Refracts rays with direction :math:`\mathbf{d}=(d_x,d_y,d_z)`
        according to generalized Snell's law [#gsl]_:

        .. math::
            \left\{\begin{array}{l}
            n_2d_x'=n_1d_x+\frac{\lambda}{2\pi}\pfrac{\phi}{x}(x_0,y_0)\\
            n_2d_y'=n_1d_y+\frac{\lambda}{2\pi}\pfrac{\phi}{y}(x_0,y_0)\\
            d_z'=\sqrt{1-d_x'^2-d_y'^2}
            \end{array}\right.

        where :math:`\mathbf{d'}=(d_x',d_y',d_z')` is direction of refractive ray,
        :math:`n_1` and :math:`n_2` are refractive indices before and behind this surface,
        :math:`\lambda` is the wavelength in vacuum, :math:`\phi` is imparted phase
        and :math:`(x_0,y_0)` is the ray-surface intersection. Both direction vectors
        have length 1.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay

        .. [#gsl] Yu, N., Genevet, P., Kats, M. A., Aieta, F., Tetienne, J. P.,
            Capasso, F., & Gaburro, Z. (2011). Light propagation with phase discontinuities:
            generalized laws of reflection and refraction. science, 334(6054), 333-337.
        """
        n1 = self.context.material_before.n(ray.wl, 'm')
        n2 = self.material.n(ray.wl, 'm')
        inv_k = ray.wl / (2 * torch.pi)
        phase_x, phase_y = self.phase_grad(ray.x, ray.y)

        ndx = (n1 * ray.d_x + inv_k * phase_x) / n2
        ndy = (n1 * ray.d_y + inv_k * phase_y) / n2
        ndz2 = 1 - ndx.square() - ndy.square()
        new_d = torch.stack([ndx, ndy, ndz2.relu().sqrt()], dim=-1)

        ray.d = new_d
        ray._d_normed = True
        ray.update_valid_(ndz2 >= 0)
        return ray


def _term_grad(x_exp: int, y_exp: int, x: Ts, y: Ts) -> Ts:
    # This function is intended to compute partial derivative correctly
    # when exp is 0 or 1 and there is 0 in x or y
    if x_exp == 0:  # y ** y_exp
        return torch.zeros_like(y)
    elif x_exp == 1:  # x * y ** y_exp
        return torch.ones_like(x) if y_exp == 0 else y.pow(y_exp)
    elif y_exp == 0:  # x ** x_exp
        return x_exp * x.pow(x_exp - 1)
    else:  # x ** x_exp * y ** y_exp
        return x_exp * x.pow(x_exp - 1) * y.pow(y_exp)


def _rect_grad(i: int, x: Ts, y: Ts) -> tuple[Ts, Ts]:
    # If k is an integer, its ceiling may be k+1 rather than k due to floating point error
    # so decrease i a little to avoid this problem
    fi = i - 1e-5
    k = (math.sqrt(9 + 8 * fi) - 3) / 2
    k = int(math.ceil(k))
    lb = (k - 1) * (k + 2) // 2
    y_exp = i - lb - 1
    x_exp = k - y_exp
    return _term_grad(x_exp, y_exp, x, y), _term_grad(y_exp, x_exp, y, x)


class PolynomialPhase(PlanarPhase, CircularSurface):
    r"""
    A planar surface imparting a phase shift to incident rays, parameterized as follows:

    .. math::

        \phi(r)=\sum_{i=1}^n a_i r^{2i}+\sum_{i=1}^m b_i p_i(x,y)

    where :math:`r=\sqrt{x^2+y^2}`. :math:`p_i` is the :math:`i`-th polynomial
    of :math:`(x,y)`, i.e. :math:`x`, :math:`y`, :math:`x^2`, :math:`xy`,
    :math:`y^2`, :math:`x^3` and so on.

    See :py:class:`CircularSurface` for descriptions of more parameters.

    :param a: Radial coefficients :math:`a_1,\ldots,a_n`.
    :type a: Sequence[float | Tensor]
    :param b: Rectangular coefficients :math:`b_1,\ldots,b_m`.
    :type b: Sequence[float | Tensor]
    """

    def __init__(
        self,
        a: Sequence[Scalar],
        b: Sequence[Scalar],
        material: mt.Material | str,
        aperture: Aperture | float = None,
        reflective: bool = False,
        *,
        d: Scalar = None
    ):
        CircularSurface.__init__(self, material, aperture, reflective, d=d)

        for i, _a in enumerate(a):
            self.register_parameter(f'a{i + 1}', nn.Parameter(scalar(_a)))
        self.n: int = len(a)  #: Number of radial coefficients :math:`n`.
        for i, _b in enumerate(b):
            self.register_parameter(f'b{i + 1}', nn.Parameter(scalar(_b)))
        self.m: int = len(b)  #: Number of rectangular coefficients :math:`m`.

    def h_r2(self, r2: Ts) -> Ts:
        return torch.zeros_like(r2)

    def h_derivative_r2(self, r2: Ts) -> Ts:
        return torch.zeros_like(r2)

    def phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        double_phase_grad_r2 = self._radial_phase_grad_r2(x.square() + y.square()) * 2
        rect_grad_x, rect_grad_y = self._rect_phase_grad(x, y)
        return double_phase_grad_r2 * x + rect_grad_x, double_phase_grad_r2 * y + rect_grad_y

    @property
    def a(self) -> list[nn.Parameter]:
        r"""
        Radial coefficients :math:`a_1,\ldots,a_n`. Note that the element
        with index ``i`` represents coefficient :math:`a_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'a{i + 1}') for i in range(self.n)]

    @property
    def b(self) -> list[nn.Parameter]:
        r"""
        Rectangular coefficients :math:`b_1,\ldots,b_m`. Note that the element
        with index ``i`` represents coefficient :math:`b_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'b{i + 1}') for i in range(self.m)]

    def _radial_phase_grad_r2(self, r2: Ts) -> Ts:
        c = self.coefficients
        c = [(i + 1) * _c for i, _c in enumerate(c)]
        return _t.polynomial(r2, c)

    def _rect_phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        term_grads = [_rect_grad(i, x, y) for i in range(1, self.m + 1)]
        x_grads, y_grads = zip(*term_grads)
        x_grad = sum(x_grad_i * b_i for x_grad_i, b_i in zip(x_grads, self.b))
        y_grad = sum(y_grad_i * b_i for y_grad_i, b_i in zip(y_grads, self.b))
        return cast(Ts, x_grad), cast(Ts, y_grad)


class Fresnel(Planar, EvenAspherical):
    """
    Fresnel lens surface. It is regarded as a planar surface generally,
    but refracts rays like a :class:`EvenAspherical` surface. Specifically,
    when calculating the direction of refractive rays, its normal vector at
    :math:`(x,y)` is that of a :class:`EvenAspherical` at the same point.

    See :class:`EvenAspherical` for description of parameters.
    """

    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        coefficients: Sequence[Scalar],
        material: mt.Material | str,
        aperture: Aperture | float = None,
        reflective: bool = False,
        *,
        d: Scalar = None
    ):
        EvenAspherical.__init__(self, roc, conic, coefficients, material, aperture, reflective, d=d)

    def h_r2(self, r2: Ts) -> Ts:
        return torch.zeros_like(r2)

    def h_derivative_r2(self, r2: Ts) -> Ts:
        return torch.zeros_like(r2)

    def _optical_normal(self, x: Ts, y: Ts) -> Ts:
        r2 = x.square() + y.square()
        lim2 = self.geo_radius.square()

        _der = EvenAspherical.h_derivative_r2(self, r2) * 2
        phpx, phpy = _der * x, _der * y
        if not lim2.isinf().all():
            mask = r2 <= lim2
            phpx, phpy = torch.where(mask, phpx, 0), torch.where(mask, phpy, 0)
        f_grad = torch.stack((-phpx, -phpy, torch.ones_like(phpx)), dim=-1)
        return f_grad / f_grad.norm(2, -1, True)
