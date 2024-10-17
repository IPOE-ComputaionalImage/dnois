import torch
from torch import nn

from . import _surf
from ._surf import *
from ... import mt, base
from ...base.typing import Any, Ts, Scalar, Vector, scalar, vector

__all__ = [
    'build_surface',

    'Conic',
    'EvenAspherical',
    'PlanarPhase',
    'Spherical',
    'Standard',
]
__all__ += _surf.__all__

EDGE_CUTTING: float = 1 - 1e-5


def _spherical(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    a = c.square() if k is None else c.square() * (1 + k)
    return c * r2 / (1 + torch.sqrt(1 - r2 * a))


def _spherical_der_wrt_r2(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    a = c.square() if k is None else c.square() * (1 + k)
    b = torch.sqrt(1 - r2 * a)
    return c / ((1 + b) * b)


class _SphericalBase(CircularSurface):
    def __init__(
        self, roc: Scalar,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(radius, material, distance, newton_config)
        self.roc: nn.Parameter = nn.Parameter(scalar(roc))  #: Radius of curvature.

    def h_r2(self, r2: Ts) -> Ts:
        return _spherical(r2, 1 / self.roc)

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        der = _spherical_der_wrt_r2(r2, 1 / self.roc) * 2
        return der * x, der * y

    @property
    def geo_radius(self) -> Ts:
        return self.roc * EDGE_CUTTING


class Spherical(_SphericalBase):
    r"""
    Spherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-c^2r^2}}

    where :math:`c` is radius of curvature.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    """

    # TODO: implement optimized intercept
    def __init__(
        self, roc: Scalar,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, radius, material, distance, newton_config)


class Conic(_SphericalBase):
    r"""
    Conic surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-(1+k)c^2r^2}}

    where :math:`c` is radius of curvature and :math:`k` is conic coefficient.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    :param conic: Conic coefficient.
    :type conic: float or Tensor
    """

    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, radius, material, distance, newton_config)
        self.conic: nn.Parameter = nn.Parameter(scalar(conic))  #: Conic coefficient.

    def h_r2(self, r2: Ts) -> Ts:
        return _spherical(r2, 1 / self.roc, self.conic)

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        der = _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic) * 2
        return der * x, der * y

    @property
    def geo_radius(self) -> Ts:
        if self.conic.item() <= -1:
            return self.conic.new_tensor(float('inf'))
        else:
            return self.roc / torch.sqrt(1 + self.conic) * EDGE_CUTTING


class Standard(Conic):
    """Alias for :py:class:`~Conic`. This name complies with the convention of Zemax."""


class EvenAspherical(Conic):
    r"""
    Even aspherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-c^2r^2}}+\sum_{i=1}^N a_i r^{2i}

    where :math:`c` is radius of curvature, :math:`k` is conic coefficient
    and :math:`\{a_i\}_{i=1}^N` are even aspherical coefficients.

    See :py:class:`Surface` for more description of arguments.

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
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, conic, radius, material, distance, newton_config)
        self.coefficients: nn.Parameter = nn.Parameter(vector(coefficients))  #: Aspherical coefficients.

    def h_r2(self, r2: Ts) -> Ts:
        a = 0
        for c in reversed(self.coefficients):
            a = (a + c) * r2
        return super().h_r2(r2) + a

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        s_der = _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic)
        a_der = 0
        for i in range(self.coefficients.numel(), 0, -1):
            a_der = a_der * r2 + self.coefficients[i - 1] * i
        der = (s_der + a_der) * 2
        return der * x, der * y

    @property
    def even_aspherical_items(self) -> int:
        """
        Number of even aspherical coefficients.

        :type: int
        """
        return self.coefficients.size(0)


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


class PlanarPhase(Surface):
    """
    This class is subject to change.
    """

    def __init__(
        self,
        radius: float,
        phase: Ts,  # small row index means small y coordinate
        material: mt.Material | str,
        distance: Scalar,
        optimize_phase: bool = True,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(radius * 2, material, distance, newton_config)
        self.radius: float = radius  #: Radius of circular aperture of the surface.
        if phase.ndim != 2:
            raise base.ShapeError(f'Shape of phase must be 2D, but got {phase.shape}')
        # no constraint on value of phase currently
        self.phase = nn.Parameter(phase, requires_grad=optimize_phase)  #: Phase shift.

    def h(self, x: Ts, y: Ts) -> Ts:
        return torch.zeros_like(torch.broadcast_tensors(x, y)[0])

    def h_extended(self, x: Ts, y: Ts) -> Ts:
        return torch.zeros_like(torch.broadcast_tensors(x, y)[0])

    def h_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        return torch.zeros_like(x), torch.zeros_like(y)

    def h_grad_extended(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        return torch.zeros_like(x), torch.zeros_like(y)

    def intercept(self, ray: BatchedRay) -> BatchedRay:
        new_ray = ray.march_to(self.context.z)
        new_ray.update_valid_(self._valid(ray), 'copy')
        return new_ray

    def phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        p = self.phase  # N_y x N_x
        dy, dx = self.cell_size
        grad_y = _diff(self.phase, dy, 0)  # N_y x N_x
        grad_x = _diff(self.phase, dx, 1)  # N_y x N_x

        y, x = y.clamp(-self.radius, self.radius), x.clamp(-self.radius, self.radius)
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
        ray.update_valid_(ndz2 >= 0, 'copy')
        return ray

    @property
    def cell_size(self) -> tuple[float, float]:
        return 2 * self.radius / (self.phase.size(0) - 2), 2 * self.radius / (self.phase.size(1) - 2)

    def _valid_coordinates(self, x: Ts, y: Ts) -> Ts:
        return x.square() + y.square() <= self.radius ** 2


def build_surface(surface_config: dict[str, Any]) -> CircularSurface:
    raise NotImplementedError()
