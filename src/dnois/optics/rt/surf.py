import abc

import torch
from torch import nn

from . import _surf
from ._surf import *
from ... import mt, base
from ...base.typing import Any, Ts, Scalar, Vector, scalar, pair

__all__ = [
    'build_surface',

    'Conic',
    'EvenAspherical',
    'Fresnel',
    'PlanarPhase',
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
        distance: Scalar,
        aperture: Aperture | float = None,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(material, distance, aperture, newton_config)
        self.roc: nn.Parameter = nn.Parameter(scalar(roc))  #: Radius of curvature.

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
        ray = ray.norm_d()
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
        distance: Scalar,
        aperture: Aperture | float = None,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, material, distance, aperture, newton_config)
        self.conic: nn.Parameter = nn.Parameter(scalar(conic))  #: Conic coefficient.

    def h_derivative_r2(self, r2: Ts) -> Ts:
        return _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic)

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
    :type coefficients: float, Sequence[float] or 1D Tensor
    """

    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        coefficients: Vector,
        material: mt.Material | str,
        distance: Scalar,
        aperture: Aperture | float = None,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, conic, material, distance, aperture, newton_config)
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

    @property
    def coefficients(self):
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


class PlanarPhase(Planar):
    """
    This class is subject to change.
    """

    def __init__(
        self,
        size: float | tuple[float, float],
        phase: Ts,  # small row index means small y coordinate
        material: mt.Material | str,
        distance: Scalar,
        aperture: Aperture = None,
        optimize_phase: bool = True,
    ):
        size = pair(size)
        if aperture is None:
            aperture = CircularAperture(min(size) / 2)
        super().__init__(material, distance, aperture)

        if phase.ndim != 2:
            raise base.ShapeError(f'Shape of phase must be 2D, but got {phase.shape}')
        #: Physical size of effective region of the surface, in vertical and horizontal directions.
        self.size: tuple[float, float] = size
        # no constraint on value of phase currently
        self.phase = nn.Parameter(phase, requires_grad=optimize_phase)  #: Phase shift.

    def phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        p = self.phase  # N_y x N_x
        dy, dx = self.cell_size
        grad_y = _diff(self.phase, dy, 0)  # N_y x N_x
        grad_x = _diff(self.phase, dx, 1)  # N_y x N_x

        y, x = y.clamp(-self.size[0], self.size[0]), x.clamp(-self.size[1], self.size[1])
        y, x = y / dy, x / dx
        y_idx, y_bias = _coordinate2index(y, p.size(0) - 1)  # row index for upper left
        x_idx, x_bias = _coordinate2index(x, p.size(1) - 1)  # col index for upper left
        y_idx2, x_idx2 = y_idx + 1, x_idx + 1  # indices for lower right

        # bilinear interpolation
        y_w = torch.stack([0.5 - y_bias, 0.5 + y_bias], -1).unsqueeze(-2)  # ... x 1 x 2
        x_w = torch.stack([0.5 - x_bias, 0.5 + x_bias], -1).unsqueeze(-1)  # ... x 2 x 1
        grad_y_points = torch.stack([
            torch.stack([grad_y[y_idx, x_idx], grad_y[y_idx, x_idx2]], -1),
            torch.stack([grad_y[y_idx2, x_idx], grad_y[y_idx2, x_idx2]], -1),
        ], -1)  # ... x 2 x 2
        grad_x_points = torch.stack([
            torch.stack([grad_x[y_idx, x_idx], grad_x[y_idx, x_idx2]], -1),
            torch.stack([grad_x[y_idx2, x_idx], grad_x[y_idx2, x_idx2]], -1),
        ], -1)  # ... x 2 x 2
        return (y_w @ grad_x_points @ x_w).squeeze(-2, -1), (y_w @ grad_y_points @ x_w).squeeze(-2, -1)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        ray = ray.clone(False).norm_d_()
        n1 = self.context.material_before.n(ray.wl, 'm')
        n2 = self.material.n(ray.wl, 'm')
        inv_k = ray.wl / (2 * torch.pi)
        phase_x, phase_y = self.phase_grad(ray.x, ray.y)

        ndx = (n1 * ray.d_x + inv_k * phase_x) / n2
        ndy = (n1 * ray.d_y + inv_k * phase_y) / n2
        ndz2 = 1 - ndx.square() - ndy.square()
        new_d = torch.stack([ndx, ndy, ndz2.clamp(min=0).sqrt()], dim=-1)

        ray.d = new_d
        ray._d_normed = True
        ray.update_valid_(ndz2 >= 0)
        return ray

    @property
    def cell_size(self) -> tuple[float, float]:
        return 2 * self.size[0] / (self.phase.size(0) - 1), 2 * self.size[1] / (self.phase.size(1) - 1)


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
        coefficients: Vector,
        material: mt.Material | str,
        distance: Scalar,
        aperture: Aperture | float = None,
    ):
        EvenAspherical.__init__(self, roc, conic, coefficients, material, distance, aperture)

    def normal(self, x: Ts, y: Ts, curved: bool = True) -> Ts:
        if not curved:
            return Planar.normal(self, x, y)
        else:
            return EvenAspherical.normal(self, x, y)


def build_surface(surface_config: dict[str, Any]) -> CircularSurface:
    raise NotImplementedError()
