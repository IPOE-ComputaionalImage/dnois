import math
import warnings

import matplotlib.pyplot as plt
import torch

from . import surf, SurfaceList
from .ray import BatchedRay
from ..system import RenderingOptics
from ... import scene as _sc, base, utils, fourier, torch as _t
from ...base import typing
from ...base.typing import Ts, Any, Size2d, Vector, Scalar, Self, PsfCenter
from ...sensor import Sensor

__all__ = [
    'CoaxialRayTracing',
]

DEFAULT_FIND_CHIEF_SAMPLES: int = 101
DEFAULT_SAMPLES: int = 512
FOV_THRESHOLD4CHIEF_RAY = math.radians(0.01)

_PsfType = typing.Literal['inc_rect', 'inc_gaussian', 'coh_kirchoff', 'coh_huygens', 'coh_fraunhofer']


def _check_arg(arg: Any, n1: str, n2: str):
    if arg is not None:
        warnings.warn(f'{n1} and {n2} are given simultaneously, {n2} will be ignored')


def _plot_set_ax(ax: plt.Axes, x_range: float):
    ax.set_xlim(-x_range * 0.1, x_range * 1.01)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    ax.set_aspect('equal')


def _plot_fov_alphas(n: int) -> list[float]:
    return torch.linspace(1., 0.3, n).tolist()


def _plot_rays_3d(ax: plt.Axes, start: Ts, end: Ts, valid: Ts, colors: list[str], alphas: list[float]):
    # shape: N_fov x N_wl x N_spp x 3
    for alpha, fov_slc1, fov_slc2, v in zip(alphas, start, end, valid):  # N_wl x N_spp x 3
        for clr, wl_slc1, wl_slc2, vv in zip(colors, fov_slc1, fov_slc2, v):  # N_spp x 3
            ax.plot(
                (utils.t4plot(wl_slc1[:, 2][vv]), utils.t4plot(wl_slc2[:, 2][vv])),
                (utils.t4plot(wl_slc1[:, 1][vv]), utils.t4plot(wl_slc2[:, 1][vv])),
                color=clr, alpha=alpha, linewidth=0.5
            )


def _make_direction(sampled_point: Ts, origin: Ts, normalize: bool = False) -> tuple[Ts, Ts | None]:
    # Typically, to create rays, some origins in object space are selected
    # and some points are sampled on the first surface in a system. The directions
    # of rays are thus the vectors pointing to sampled points from origins.
    # However, the origins may be located at infinity. In that case, the x and
    # y coordinates of origins are assumed to be finite and serve as the tangents
    # of x and y FoV, respectively.
    is_inf = origin[..., 2].isinf()
    if is_inf.all():
        d = torch.cat([-origin[..., :2], torch.ones_like(origin[..., [2]])], -1)
    else:
        d = sampled_point - origin
        if is_inf.any():
            d = torch.where(
                is_inf.unsqueeze(-1),
                torch.cat([-origin[..., :2], torch.ones_like(origin[..., [2]])], -1),
                d
            )
    if normalize:
        length = d.norm(2, -1)
        return d / length.unsqueeze(-1), length
    else:
        return d, None


class CoaxialRayTracing(RenderingOptics):
    """
    A class of sequential and ray-tracing-based optical system model.

    See :class:`~dnois.optics.RenderingOptics` for descriptions of more parameters.

    :param CoaxialSurfaceList surfaces: Surface list object.
    :param str psf_type: The way to calculate PSF. Default: ``inc_rect``.

        ``inc_rect``
            TODO

        ``inc_gaussian``
            TODO

        ``coh_kirchoff``
            TODO

        ``coh_huygens``
            TODO

        ``coh_fraunhofer``
            TODO
    :param str psf_center: The way to determine centers of computed PSFs.

        ``'linear'``
            PSFs are centered around ideal image points thus realistic distortion is simulated.

        ``'mean'``
            PSFs are centered around their "center of gravity".

        ``'chief'``
            PSFs are centered around the intersections of corresponding chief rays and image plane.
    :param Callable sampler: A callable object whose signature is described by
        :meth:`dnois.optics.rt.Aperture.sampler`. This is typically created by this method as well.
    :param bool vignette: Whether to simulate vignette for ``'inc_rect'`` PSF. Default: ``False``.
    :param int coherent_tracing_samples: Number of samples in two directions
        for coherent tracing. Default: 512.
    :param str coherent_tracing_sampling_pattern: Sampling pattern for coherent tracing.
        Default: ``'quadrapolar'``.
    :param kwargs: Additional keyword arguments passed to :class:`RenderingOptics`.
    """
    _inherent = RenderingOptics._inherent + ['surfaces']
    _external = RenderingOptics._external + [
        'psf_type', 'psf_center',
        'vignette', 'coherent_tracing_samples', 'coherent_tracing_sampling_pattern'
    ]

    PsfType = _PsfType

    def __init__(
        self,
        surfaces: surf.CoaxialSurfaceList,
        image_distance: float,
        sensor: Sensor = None,
        psf_type: _PsfType = 'inc_rect',
        psf_center: PsfCenter = 'linear',
        sampler: surf.Sampler = None,
        vignette: bool = False,
        coherent_tracing_samples: int = 512,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
        **kwargs
    ):
        super().__init__(image_distance, sensor, **kwargs)
        self.surfaces: surf.CoaxialSurfaceList = surfaces  #: Surface list.
        self.psf_type: _PsfType = psf_type  #: See :class:`CoaxialRayTracing`.
        self.psf_center: PsfCenter = psf_center  #: See :class:`CoaxialRayTracing`.
        self.sampler: surf.Sampler = sampler  #: See :class:`CoaxialRayTracing`.
        self.vignette: bool = vignette  #: See :class:`CoaxialRayTracing`.
        #: See :class:`CoaxialRayTracing`.
        self.coherent_tracing_samples: int = coherent_tracing_samples
        #: See :class:`CoaxialRayTracing`.
        self.coherent_tracing_sampling_pattern: str = coherent_tracing_sampling_pattern

        if self.sampler is None:
            self.sampler = self.surfaces.first.aperture.sampler('random', 256)

    def pointwise_render(
        self,
        scene: _sc.Scene,
        direct: bool = True,
        wl: Vector = None,
        depth: Vector | tuple[Ts, Ts] = None,
        psf_size: Size2d = None,
        psf_type: _PsfType = None,
        psf_center: PsfCenter = None,
        x_symmetric: bool = None,
        y_symmetric: bool = None,
        **kwargs,
    ) -> Ts:
        """
        .. warning::

            This method is subject to change.
        """
        wl = self.pick('wl', wl)
        depth = self.pick('depth', depth)
        psf_type = self.pick('psf_type', psf_type)
        psf_size = self.pick('psf_size', psf_size)
        psf_center = self.pick('psf_center', psf_center)

        self._check_scene(scene)
        scene: _sc.ImageScene
        if direct:
            return self._direct_pw_render(scene, wl, depth, **kwargs)
        else:
            return super().pointwise_render(
                scene, wl, depth, psf_size, x_symmetric, y_symmetric,
                psf_center=psf_center, psf_type=psf_type, **kwargs
            )

    def cam2lens_z(self, depth: float | Ts) -> Ts:
        """
        Converts z-coordinates in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        (i.e. depth) to those in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.

        .. seealso::
            This is the inverse of :meth:`.len2cam_z`.

        :param depth: Depth.
        :type depth: float | Tensor
        :return: Z-coordinate in lens system. If ``depth`` is a float, returns a 0D tensor.
        :rtype: Tensor
        """
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        return self.principal1 - depth

    def len2cam_z(self, z: float | Ts) -> Ts:
        """
        Converts z-coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>`
        to those in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>` (i.e. depth).

        .. seealso::
            This is the inverse of :meth:`.cam2lens_z`.

        :param z: Z-coordinate in lens' coordinate system.
        :type z: float | Tensor
        :return: Depth. If ``z`` is a float, returns a 0D tensor.
        :rtype: Tensor
        """
        if not torch.is_tensor(z):
            z = self.new_tensor(z)
        return self.principal1 - z

    def lens2cam(self, point: Ts) -> Ts:
        """
        Converts coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>` to those in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.

        :param Tensor point: Coordinates in lens' coordinate system. A tensor with shape ``(..., 3)``.
        :return: Coordinates in camera's coordinate system. A tensor with shape ``(..., 3)``.
        :rtype: Tensor
        """
        _t.check_3d_vector(point, f'point in {self.lens2cam.__qualname__}')

        return torch.stack([-point[..., 0], point[..., 1], self.len2cam_z(point[..., 2])], -1)

    def cam2lens(self, point: Ts) -> Ts:
        """
        Converts coordinates in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        into coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.

        :param Tensor point: Coordinates in camera's coordinate system. A tensor with shape ``(..., 3)``.
        :return: Coordinates in lens' coordinate system. A tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        _t.check_3d_vector(point, f'point in {self.cam2lens.__qualname__}')

        return torch.stack([-point[..., 0], point[..., 1], self.cam2lens_z(point[..., 2])], -1)

    def obj_proj_lens(self, point: Ts) -> Ts:
        """
        Returns x and y coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>` of perspective projections
        of points in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        ``point``. They can be viewed as ideal image points of object points ``point``.

        :param Tensor point: Points in camera's coordinate system, a tensor with shape ``(..., 3)``.
            It complies with :ref:`guide_imodel_ccs_inf`.
        :return: x and y coordinate of projected points, a tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        xy_on_sensor = self.obj2tanfov(point) * self.reference.focal_length
        xy_on_sensor[..., 0] = -xy_on_sensor[..., 0]
        return xy_on_sensor

    def trace_ray(self, ray: BatchedRay) -> BatchedRay:
        """
        Trace a group of rays through surfaces until the image plane.
        If you want to trace rays until the last surface, call ``self.surfaces(ray)``.

        :param BatchedRay ray: Rays to trace.
        :return: Rays after tracing. Their origins are located at the image plane.
        :rtype: BatchedRay
        """
        out_ray: BatchedRay = self.surfaces(ray)
        ref_idx = self.surfaces.last.material.n(out_ray.wl, 'm')
        return out_ray.march_to(self.surfaces.total_length, ref_idx)

    def trace_point(self, point: Ts, wl: Vector, sampler: surf.Sampler = None) -> BatchedRay:
        """
        Trace a group of rays emitted from ``point`` through surfaces until the image plane.

        :param Tensor point: Coordinate of points in
            :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.
            A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths of rays. A float, a sequence of float or a tensor of shape ``(N_wl,)``.
        :param Callable sampler: A callable object whose signature is described by
            :meth:`dnois.optics.rt.Aperture.sampler`. This is typically created by this method as well.
        :return: Rays after tracing. Their origins are located at the image plane.
        :rtype: BatchedRay
        """
        sampler = self.pick('sampler', sampler)
        sampled = self.surfaces.first.sample(sampler)  # N_spp x 3
        d, _ = _make_direction(sampled, self.cam2lens(point).unsqueeze(-2))  # ... x N_spp|1 x 3
        ray = BatchedRay(sampled, d.unsqueeze(-3), wl.unsqueeze(-1))  # ... x N_wl x N_spp x 3
        out_ray = self.trace_ray(ray)  # ... x N_wl x N_spp
        return out_ray

    @torch.no_grad()
    def focus_to_(self, depth: Scalar) -> Self:
        """
        .. warning::

            This method is subject to change.
        """
        depth = typing.scalar(depth, dtype=self.dtype, device=self.device)
        z = self.cam2lens_z(depth)
        o = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z))  # 3
        points = self.surfaces.first.sample(self.sampler)  # N_spp x 3
        d, _ = _make_direction(points, o, True)
        wl = self.wl.reshape(-1, 1)
        ray = BatchedRay(points, d, wl)  # N_wl x N_spp

        out_ray = self.trace_ray(ray)

        # solve marching distance by least square
        t = -(out_ray.x * out_ray.d_x + out_ray.y * out_ray.d_y)
        t = t / (out_ray.d_x.square() + out_ray.d_y.square())
        new_z = out_ray.z + t * out_ray.d_z
        new_z = new_z[out_ray.valid & new_z.isnan().logical_not()].mean()
        move = new_z - self.surfaces.total_length
        self.surfaces.last.distance.data += move

        return self

    def psf(
        self,
        origins: Ts,
        size: Size2d = None,
        wl: Vector = None,
        psf_type: _PsfType = None,
        psf_center: PsfCenter = None,
        **kwargs
    ) -> Ts:
        size = self.pick('psf_size', size)
        wl = self.pick('wl', wl)
        psf_type = self.pick('psf_type', psf_type)
        psf_center = self.pick('psf_center', psf_center)

        _t.check_3d_vector(origins, f'origins in {self.psf.__qualname__}')

        if psf_type == 'inc_rect':
            return self._psf_inc_rect(origins, size, wl, psf_center, **kwargs)
        elif psf_type == 'inc_gaussian':
            return self._psf_inc_gaussian(origins, size, wl, psf_center, **kwargs)
        elif psf_type == 'coh_huygens':
            return self._psf_coherent(origins, size, wl, psf_center, False)
        elif psf_type == 'coh_kirchoff':
            return self._psf_coherent(origins, size, wl, psf_center, True)
        elif psf_type == 'coh_fraunhofer':
            return self._psf_from_wavefront(origins, size, wl, psf_center, **kwargs)
        else:
            raise ValueError(f'Unknown PSF type: {psf_type}')

    def entrance_pupil(self) -> tuple[Ts, Ts]:
        stop = self.surfaces.stop
        if stop is None:
            raise NotImplementedError()
        ap = stop.aperture
        if not isinstance(ap, surf.CircularAperture):
            raise NotImplementedError()
        idx = stop.ctx.index
        if idx != 0:
            raise NotImplementedError()
        return self.surfaces.ctxs[idx].baseline, ap.radius

    # def entrance_pupil_probe(self, origin: Ts, wl: typing.Vector = None) -> tuple[Ts, Ts]:
    #     """
    #     .. attention::
    #
    #         This method applies when the apertures of all surfaces are circular.
    #
    #     .. warning::
    #
    #         This method is subject to change.
    #
    #     Finds the diameter and z-value of the entrance pupil using *probe* method.
    #
    #     :param Tensor origin: Origin of probe rays, a tensor with shape ``(..., 3)``.
    #         Its coordinate is defined in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.
    #     :param wl: Wavelengths.
    #     :type wl: float | Sequence[float] | Tensor
    #     :return: Diameter and z-value of the entrance pupil, a pair of tensors of shape ``(..., N_wl)``.
    #     :rtype: tuple[Tensor, Tensor]
    #     """
    #     wl = self.pick('wl', wl)
    #
    #     self._check_circular()
    #     _t.check_3d_vector(origin, f'origin in {self.entrance_pupil_probe.__qualname__}')
    #     wl = wl.unsqueeze(-1)
    #
    #     origin = origin.unsqueeze(-2).unsqueeze(-3)  # ... x 1 x 1 x 3
    #     d_parallel, _ = _make_direction(
    #         torch.cat([self.new_tensor([0., 0.]), self.principal1.view(1)]), origin, True
    #     )  # ... x 1 x 1 x 3
    #
    #     r = self.surfaces.first.aperture.radius  # scalar
    #     edge_h = self.surfaces.first.h_extended(torch.zeros_like(r), r)  # scalar
    #     r = r + torch.sqrt(d_parallel[..., 2].reciprocal().square() - 1) * edge_h  # ... x 1 x 1
    #     axis_tmp = torch.linspace(-1, 1, find_chief_samples, device=self.device, dtype=self.dtype)
    #     x, y = torch.meshgrid(axis_tmp, axis_tmp, indexing='ij')
    #     x, y = x.flatten(), y.flatten()  # N_spp
    #     points_pre = torch.stack([x, y, torch.zeros_like(x)], dim=-1)  # N_spp x 3
    #     points_pre = points_pre * r.unsqueeze(-1)  # ... x 1 x N_spp x 3
    #     ray = BatchedRay(points_pre, d_parallel, wl)  # ... x N_wl x N_spp
    #     out_ray = self.surfaces(ray)
    #
    #     valid = out_ray.valid.broadcast_to(out_ray.shape)  # ... x N_wl x N_spp
    #     xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
    #     xy_mean = xy_valid.nanmean(-2, True)  # ... x N_wl x 1 x 2
    #     points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)  # ... x N_wl x 1 x 3
    #     d_chief, l0_chief = _make_direction(points_chief, origin, True)  # ... x 1 x 1( x 3)
    #     chief_ray = BatchedRay(points_chief, d_chief, wl, 0.)  # ... x N_wl x 1

    # @torch.no_grad()
    # def show_spot_diagram(self, fov: Vector = None, wl: Vector = None) -> plt.Figure:
    #     """
    #     .. warning::
    #
    #         This method is subject to change.
    #     """
    #     if fov is None:
    #         fov_half = self.reference.fov_half
    #         fov = [0., fov_half * 0.5 ** 0.5, fov_half]
    #     wl = self.pick('wl', wl)
    #     fov = typing.vector(fov, device=self.device, dtype=self.dtype)
    #     wl = typing.vector(wl, device=self.device, dtype=self.dtype)
    #
    #     n_fov = fov.numel()
    #     n_row = math.floor(math.sqrt(n_fov) + 1e-5)
    #     n_col = math.ceil(n_fov / n_row)
    #     fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 3))
    #
    #     for i, fov in enumerate(fov.tolist()):
    #         o = self.fovd2obj((0, fov), float('inf'), True)
    #         _, chief_ray = self._generate_rays(o, wl, 1)
    #         radial_offset = torch.sqrt(chief_ray.o[..., :2].square().sum(-1))
    #         d_proj = torch.sqrt(chief_ray.d[..., :2].square().sum(-1))
    #         rs_roc = radial_offset / d_proj  # N_wl x 1
    #         enter_pupil_z = torch.sqrt(rs_roc.square() - radial_offset.square()).squeeze(-1)  # N_wl
    #         enter_pupil_z += chief_ray.z
    #         pupil_ap=surf.CircularAperture()

    @torch.no_grad()
    def show_cross_section(
        self,
        figsize: float | tuple[float, float] = (12.8, 9.6),
        fov: Vector = None,
        wl: Vector = None,
        spp: int = 10,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        .. warning::

            This method is subject to change.
        """
        if fov is None:
            fov_half = self.reference.fov_half
            fov = [0., fov_half * 0.5 ** 0.5, fov_half]
        wl = self.pick('wl', wl)
        fov = typing.vector(fov, device=self.device, dtype=self.dtype)
        wl = typing.vector(wl, device=self.device, dtype=self.dtype)

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'frameon': False})

        self._plot_components(ax)
        # image_plane
        if self.sensor is not None:
            diag_length = (self.sensor.h ** 2 + self.sensor.w ** 2) ** 0.5
            y = torch.linspace(-diag_length / 2, diag_length / 2, 100, device=self.device)
            ax.plot(
                utils.t4plot(torch.full_like(y, self.surfaces.total_length.item())), utils.t4plot(y),
                color='black', linewidth=2
            )
        # rays
        o = self.surfaces[0].sample('diameter', spp, torch.pi / 2)  # N_spp x 3
        d = torch.stack([torch.zeros_like(fov), fov.tan(), torch.ones_like(fov)], dim=-1)
        d = d.unsqueeze(1).unsqueeze(1)  # N_fov x 1 x 1 x 3
        ray = BatchedRay(o, d, wl.reshape(1, -1, 1))  # N_fov x N_wl x N_spp
        self._plot_rays(ax, ray, fov, wl)

        _plot_set_ax(ax, self.surfaces.total_length.item())
        return fig, ax

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    def wavefront_map(
        self,
        origin: Ts,
        wl: Vector = None,
        coherent_tracing_samples: int = DEFAULT_SAMPLES,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
    ) -> tuple[BatchedRay, Ts]:
        wl = self.pick('wl', wl)
        samples = self.pick('coherent_tracing_samples', coherent_tracing_samples)
        sampling_pattern = self.pick('coherent_tracing_sampling_pattern', coherent_tracing_sampling_pattern)

        chief_ray, ray, rs_roc, exit_pupil_distance = self._trace_opl_with_chief(
            origin, wl, samples, sampling_pattern
        )
        ref_idx = self.surfaces.mt_tail.n(ray.wl, 'm')
        opd = chief_ray.march(-rs_roc, ref_idx).opl - ray.opl  # ... x N_wl x N_spp
        opd[~ray.valid] = float('nan')
        return ray, opd / wl.unsqueeze(-1)  # ... x N_wl x N_spp

    def chief_ray(self, origin: Ts, wl: Vector = None) -> BatchedRay:
        wl = typing.vector(wl, device=self.device, dtype=self.dtype)

        ap_z, _ = self.entrance_pupil()
        zero = torch.zeros_like(ap_z)
        chief_point = torch.stack([zero, zero, ap_z])  # 3
        d, _ = _make_direction(chief_point, origin, True)  # ... x 3
        chief = BatchedRay(chief_point, d.unsqueeze(-2), wl)  # ... x N_wl
        return chief

    # Optical parameters
    # =============================

    @property
    def fov_full(self) -> Ts:
        return self.fov_half * 2

    @property
    def fov_half(self) -> Ts:
        raise NotImplementedError()

    @property
    def focal1(self) -> Ts:
        raise NotImplementedError()

    @property
    def focal2(self) -> Ts:
        raise NotImplementedError()

    @property
    def principal1(self) -> Ts:
        # TODO: currently depth=0 plane is assumed to be z=0 plane, which is incorrect
        return self.new_tensor(0.)

    @property
    def principal2(self) -> Ts:
        raise NotImplementedError()

    def _check_circular(self):
        for s in self.surfaces:
            if not isinstance(s.aperture, surf.CircularAperture):
                raise NotImplementedError(f'A function called requires all the surfaces '
                                          f'have circular apertures, which is not satisfied')

    # Serialization
    # ===========================
    @classmethod
    def _pre_from_dict(cls, d: dict):
        d = super()._pre_from_dict(d)
        d['surfaces'] = SurfaceList.from_dict(d['surfaces'])
        return d

    # protected
    # ========================

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    @torch.no_grad()
    def _generate_rays(
        self,
        origin: Ts,
        wl: Ts,
        samples: int,
        sampling_pattern: str = 'quadrapolar',
        find_chief_samples: int = DEFAULT_FIND_CHIEF_SAMPLES,
    ) -> tuple[BatchedRay, BatchedRay]:
        """
        .. warning::

            This method is subject to change.
        """
        # origin is in lens' coordinate system
        if not isinstance(self.surfaces.first.aperture, surf.CircularAperture):
            raise NotImplementedError()
        _t.check_3d_vector(origin, f'origin in {self._generate_rays.__qualname__}')
        wl = wl.unsqueeze(-1)

        origin = origin.unsqueeze(-2).unsqueeze(-3)  # ... x 1 x 1 x 3
        d_parallel, _ = _make_direction(
            torch.cat([self.new_tensor([0., 0.]), self.principal1.view(1)]), origin, True
        )  # ... x 1 x 1 x 3

        r = self.surfaces.first.aperture.radius  # scalar
        edge_h = self.surfaces.first.h_extended(torch.zeros_like(r), r)  # scalar
        r = r + torch.sqrt(d_parallel[..., 2].reciprocal().square() - 1) * edge_h  # ... x 1 x 1
        axis_tmp = torch.linspace(-1, 1, find_chief_samples, device=self.device, dtype=self.dtype)
        x, y = torch.meshgrid(axis_tmp, axis_tmp, indexing='ij')
        x, y = x.flatten(), y.flatten()  # N_spp
        points_pre = torch.stack([x, y, torch.zeros_like(x)], dim=-1)  # N_spp x 3
        points_pre = points_pre * r.unsqueeze(-1)  # ... x 1 x N_spp x 3
        ray = BatchedRay(points_pre, d_parallel, wl)  # ... x N_wl x N_spp
        out_ray = self.surfaces(ray)

        valid = out_ray.valid.broadcast_to(out_ray.shape)  # ... x N_wl x N_spp
        xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
        xy_mean = xy_valid.nanmean(-2, True)  # ... x N_wl x 1 x 2
        points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)  # ... x N_wl x 1 x 3
        d_chief, l0_chief = _make_direction(points_chief, origin, True)  # ... x 1 x 1( x 3)
        chief_ray = BatchedRay(points_chief, d_chief, wl, 0.)  # ... x N_wl x 1

        # mimicking np.nanmax and np.nanmin
        xy_min = xy_valid.nan_to_num(nan=float('inf')).amin(-2, True)
        xy_max = xy_valid.nan_to_num(nan=-float('inf')).amax(-2, True)
        xy_shift_min = torch.abs(xy_min - xy_mean).unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2
        xy_shift_max = torch.abs(xy_max - xy_mean).unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2

        axis = torch.linspace(-1, 1, samples, dtype=self.dtype, device=self.device)
        if sampling_pattern == 'quadrapolar':
            h_p, w_p = torch.meshgrid(-axis, axis, indexing='ij')
            theta = torch.arctan2(h_p, w_p)
            o_p = torch.stack((theta.cos(), theta.sin()), dim=-1)
            o_p *= torch.max(h_p.abs(), w_p.abs()).unsqueeze(-1)  # samples x samples x 2
        elif sampling_pattern == 'rect':
            o_p = torch.stack(torch.meshgrid(axis, -axis, indexing='xy'), -1)
        else:
            raise ValueError(f'Unknown sampling pattern for {self._generate_rays.__qualname__}: {sampling_pattern}')

        o_shape = xy_valid.shape[:-2] + (samples, samples, 3)
        o = torch.zeros(o_shape, dtype=self.dtype, device=self.device)
        o[..., :samples // 2, :, 1] = o_p[:samples // 2, :, 1] * xy_shift_max[..., 1]
        o[..., samples // 2:, :, 1] = o_p[samples // 2:, :, 1] * xy_shift_min[..., 1]
        o[..., :, :samples // 2, 0] = o_p[:, :samples // 2, 0] * xy_shift_max[..., 0]
        o[..., :, samples // 2:, 0] = o_p[:, samples // 2:, 0] * xy_shift_min[..., 0]
        o[..., :2] += xy_mean.unsqueeze(-2)
        points_sample = o.flatten(-3, -2)
        d_sample, l0 = _make_direction(points_sample, origin, True)  # ... x 1 x N_spp'( x 3)
        # to reduce magnitude of opl and subsequently floating point error
        l0 = l0 - l0_chief  # ... x 1 x N_spp'
        if origin[..., 2].isinf().any():
            l0 = torch.where(
                origin[..., 2].isinf(),  # ... x 1 x 1
                torch.sum((points_sample - points_chief) * d_parallel, -1),
                l0
            )  # ... x N_wl x N_spp'
        ref_idx = self.surfaces.mt_head.n(wl, 'm')
        # ... x N_wl x N_spp'
        ray = BatchedRay(points_sample, d_sample, wl, l0 * ref_idx)
        return ray, chief_ray

    def _trace_opl_with_chief(
        self,
        origin: Ts,
        wl: Vector,
        samples: int = DEFAULT_SAMPLES,
        sampling_pattern: str = 'quadrapolar',
    ) -> tuple[BatchedRay, BatchedRay, Ts, Ts]:
        # ... x N_wl x N_spp(1)
        ray, chief_ray = self._generate_rays(self.cam2lens(origin), wl, samples, sampling_pattern)

        ray = self.surfaces(ray)
        chief_ray = self.trace_ray(chief_ray)
        radial_offset = torch.sqrt(chief_ray.o[..., :2].square().sum(-1))
        d_proj = torch.sqrt(chief_ray.d[..., :2].square().sum(-1))
        rs_roc = radial_offset / d_proj  # ... x N_wl x 1
        exit_pupil_distance = torch.sqrt(rs_roc.square() - radial_offset.square()).squeeze(-1)  # ... x N_wl

        shift = chief_ray.o - ray.o
        dp = torch.sum(shift * ray.d, dim=-1)  # dot product
        length2rs = dp - torch.sqrt(dp.square() - shift.square().sum(-1) + rs_roc.square())
        ref_idx = self.surfaces.mt_tail.n(ray.wl, 'm')
        ray.march_(length2rs, ref_idx)
        return chief_ray, ray, rs_roc, exit_pupil_distance  # ... x N_wl x N_spp

    def _plot_components(self, ax):
        uppers = []
        lowers = []
        for sf in self.surfaces:  # surfaces
            sf: surf.Surface
            y = torch.linspace(-sf.aperture.radius, sf.aperture.radius, 100, device=self.device)
            z = sf.h_extended(torch.zeros_like(y), y) + sf.context.baseline
            ax.plot(utils.t4plot(z), utils.t4plot(y), color='black', linewidth=1)
            uppers.append((z[-1].item(), y[-1].item()))
            lowers.append((z[0].item(), y[0].item()))
        for i in range(len(self.surfaces) - 1):  # edges
            if self.surfaces[i].material.name == 'vacuum':
                continue
            ax.plot(*list(zip(uppers[i], uppers[i + 1])), color='black', linewidth=1)
            ax.plot(*list(zip(lowers[i], lowers[i + 1])), color='black', linewidth=1)

    def _plot_rays(self, ax, ray: BatchedRay, fov: Ts, wl: Ts):  # ray: N_fov x N_wl x N_spp
        ray.broadcast_().march_to_(ray.new_tensor(0.))
        colors = [utils.wl2rgb(_wl, output_format='hex') for _wl in wl.tolist()]
        alphas = _plot_fov_alphas(fov.numel())
        rays_record = []
        for sf in self.surfaces:
            out_ray = sf(ray)
            rays_record.append(out_ray.broadcast_())
            ray = out_ray
        out_ray = ray.march_to(self.surfaces.total_length)
        rays_record.append(out_ray.broadcast_())
        for ray, next_ray in zip(rays_record[:-1], rays_record[1:]):
            _plot_rays_3d(ax, ray.o, next_ray.o, out_ray.valid, colors, alphas)

    def _direct_pw_render(
        self,
        scene: _sc.ImageScene,
        wl: Ts,
        depth: Ts | tuple[Ts, Ts],
        sampler: surf.Sampler,
        vignette: bool,
    ) -> Ts:
        self._check_scene(scene)
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {self.wl.numel()} wavelengths expected, got {scene.n_wl}')
        if len(self.surfaces) == 0:
            raise RuntimeError(f'No surface available')

        scene = scene.batch()
        n_b, n_wl, n_h, n_w = scene.image.shape

        depth_map = self._make_depth_map(scene, depth)  # B x H x W
        o = self.points_grid((n_h, n_w), depth_map, True)  # B x H x W x 3
        o = self.cam2lens(o)  # B x H x W x 3

        points = self.surfaces.first.sample(sampler)  # N_spp x 3
        spp = points.size(0)
        o = o.unsqueeze(-2).unsqueeze(-5)
        d, _ = _make_direction(points, o, True)  # B x 1 x H x W x N_spp x 3

        wl = wl.view(1, -1, 1, 1, 1)
        ray = BatchedRay(points, d, wl)  # B x N_wl x H x W x N_spp

        out_ray = self.trace_ray(ray)

        x, y = -out_ray.x / self.sensor.pixel_size[1], out_ray.y / self.sensor.pixel_size[0]
        x, y = x + self.sensor.pixel_num[1] / 2, y + self.sensor.pixel_num[0] / 2
        c_a, r_a = torch.floor(x.detach() + 0.5).long(), torch.floor(y.detach() + 0.5).long()
        in_region = (c_a >= 0) & (c_a <= n_w) & (r_a >= 0) & (r_a <= n_h)  # B x N_wl x H x W x N_spp
        c_a[~in_region] = 0
        r_a[~in_region] = 0
        c_as, r_as = c_a - 1, r_a - 1
        w_c, w_r = c_a - x + 0.5, r_a - y + 0.5
        iw_c, iw_r = 1 - w_c, 1 - w_r

        b_wl_idx = (
            torch.arange(n_b, device=self.device).view(-1, 1, 1, 1, 1),
            torch.arange(n_wl, device=self.device).view(1, -1, 1, 1, 1),
        )
        image = self.new_zeros((n_b, n_wl, n_h + 2, n_w + 2))
        mask = out_ray.valid & in_region
        gt_image = scene.image.unsqueeze(-1)
        if vignette:
            gt_image = gt_image * spp / (out_ray.valid.sum(dim=-1, keepdim=True) + 1e-5)
        _gt1 = gt_image * w_c
        _gt2 = gt_image * iw_c
        image.index_put_(b_wl_idx + (r_as, c_as), torch.where(mask, _gt1 * w_r, 0), True)  # top left
        image.index_put_(b_wl_idx + (r_a, c_as), torch.where(mask, _gt1 * iw_r, 0), True)  # bottom left
        image.index_put_(b_wl_idx + (r_as, c_a), torch.where(mask, _gt2 * w_r, 0), True)  # top right
        image.index_put_(b_wl_idx + (r_a, c_a), torch.where(mask, _gt2 * iw_r, 0), True)  # bottom right
        image = image[..., :-2, :-2] / spp
        return image

    def _find_xy_center(self, psf_center, origins, out_ray, wl):
        if psf_center == 'linear':
            xy_center = self.obj_proj_lens(origins)[..., None, None, :]  # ... x 1 x 1 x 2
        elif psf_center == 'mean':
            xy_center = out_ray.o[..., :2]  # ... x N_wl x N_spp x 2
            valid = out_ray.valid.unsqueeze(-1)  # ... x N_wl x N_spp x 1
            xy_center = torch.where(valid, xy_center, 0).sum(-2, True) / valid.sum(-2, True)  # ... x N_wl x 1 x 2
            xy_center = xy_center.mean(-3, True)  # ... x 1 x 1 x 2
        elif psf_center == 'chief':
            chief = self.chief_ray(origins, wl)  # ... x N_wl
            out_chief = self.trace_ray(chief)
            xy_center = out_chief.o[..., None, :2]  # ... x N_wl x 1 x 2
            xy_center = xy_center.mean(-3, True)  # ... x 1 x 1 x 2
        else:
            raise ValueError(f'Unsupported PSF center type for simple incoherent PSF: {psf_center}')
        return xy_center

    def _psf_inc_rect(
        self,
        origins: Ts,  # ... x 3
        size: tuple[int, int],
        wl: Ts,  # N_wl
        psf_center: PsfCenter,
        sampler: surf.Sampler = None,
        vignette: bool = None,
    ) -> Ts:
        sampler = self.pick('sampler', sampler)
        vignette = self.pick('vignette', vignette)

        out_ray = self.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp
        out_ray.update_valid_(~out_ray.x.isnan() & ~out_ray.y.isnan())  # TODO:?
        n_spp = out_ray.shape[-1]

        xy_center = self._find_xy_center(psf_center, self.cam2lens(origins), out_ray, wl)
        xy = out_ray.o[..., :2] - xy_center  # ... x N_wl x N_spp x 2

        x, y = xy[..., 0] / self.sensor.pixel_size[1], xy[..., 1] / self.sensor.pixel_size[0]  # ... x N_wl x N_spp
        # if PSF size is odd, the center is N/2, relative positions are -N//2, ..., N//2
        # if PSF size is even, the center is (N+1)/2, relative positions are -N//2, ..., N//2-1
        x, y = x + size[1] // 2 + 0.5, y + size[0] // 2 + 0.5
        c_a, r_a = torch.floor(x.detach() + 0.5).long(), torch.floor(y.detach() + 0.5).long()
        in_region = c_a.ge(0) & c_a.le(size[1]) & r_a.ge(0) & r_a.le(size[0])  # ... x N_wl x N_spp
        mask = out_ray.valid & in_region  # ... x N_wl x N_spp
        c_a[~mask] = 0
        r_a[~mask] = 0
        c_as, r_as = c_a - 1, r_a - 1
        w_c, w_r = c_a - x + 0.5, r_a - y + 0.5
        iw_c, iw_r = 1 - w_c, 1 - w_r

        psf = self.new_zeros(out_ray.shape[:-1] + (size[0] + 2, size[1] + 2))  # ... x N_wl x (H+2) x (W+2)
        pre_idx = [
            _t.as1d(torch.arange(dim_size, device=self.device), mask.ndim, i)
            for i, dim_size in enumerate(mask.shape[:-1])
        ]
        psf.index_put_(pre_idx + [r_as, c_as], torch.where(mask, w_c * w_r, 0), True)  # top left
        psf.index_put_(pre_idx + [r_a, c_as], torch.where(mask, w_c * iw_r, 0), True)  # bottom left
        psf.index_put_(pre_idx + [r_as, c_a], torch.where(mask, iw_c * w_r, 0), True)  # top right
        psf.index_put_(pre_idx + [r_a, c_a], torch.where(mask, iw_c * iw_r, 0), True)  # bottom right
        psf = psf[..., :-2, :-2]  # ... x N_wl x H x W

        if vignette:
            psf = psf / n_spp
        else:
            psf = psf / psf.sum((-2, -1), True)

        psf = psf.flip(-1)
        return psf

    def _psf_inc_gaussian(
        self,
        origins: Ts,  # ... x 3
        size: tuple[int, int],
        wl: Ts,  # N_wl
        psf_center: PsfCenter,
        sampler: surf.Sampler = None,
    ) -> Ts:
        sampler = self.pick('sampler', sampler)

        out_ray = self.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp
        out_ray.update_valid_(~out_ray.x.isnan() & ~out_ray.y.isnan())  # TODO:?

        xy_center = self._find_xy_center(psf_center, self.cam2lens(origins), out_ray, wl)
        xy = out_ray.o[..., :2] - xy_center  # ... x N_wl x N_spp x 2
        py, px = self.sensor.pixel_size
        valid = xy[..., 0].abs().le(px * (size[1] / 2 + 5)) & xy[..., 1].abs().le(py * (size[0] / 2 + 5))
        valid.logical_and_(out_ray.valid)  # ... x N_wl x N_spp

        psf = self.new_empty(out_ray.shape[:-1] + size)  # ... x N_wl x H x W
        ry, rx = utils.grid(size, self.sensor.pixel_size, dtype=self.dtype, device=self.device)
        rxy = torch.stack([rx, ry], -1)  # H x W x 2
        pixel_diag = math.sqrt(px ** 2 + py ** 2)
        sigma = pixel_diag / 3
        a, b = 1 / (math.sqrt(2 * math.pi) * sigma), -1 / (2 * sigma * sigma)
        for i in range(psf.size(-2)):
            for j in range(psf.size(-1)):
                r2 = torch.square(rxy[i, j] - xy).sum(-1)  # ... x N_wl x N_spp
                w = a * torch.exp(b * r2)  # ... x N_wl x N_spp
                psf[..., i, j] = torch.where(valid, w, 0).sum(-1)  # ... x N_wl

        psf = psf / psf.sum((-2, -1), True)
        psf = psf.flip(-1)
        return psf

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    def _psf_coherent(
        self,
        origins: Ts,
        size: tuple[int, int],
        wl: Ts,
        psf_center: PsfCenter,
        oblique: bool,
        samples: int = DEFAULT_SAMPLES,
    ) -> Ts:
        """This method is subject to change."""
        chief_ray, ray, rs_roc, _ = self._trace_opl_with_chief(origins, wl, samples)

        lim = ((size[0] - 1) // 2, (size[1] - 1) // 2)
        # x and y should decrease when index gets large since:
        # x coordinate of PSF should be in camera's coordinate system
        # but the computation is performed in lens' coordinate system
        # so a horizontal flipping is needed
        # and large index for y means lower position i.e. small y
        y, x = [torch.linspace(
            lim[i], -lim[i], size[i], device=self.device, dtype=self.dtype
        ) * self.sensor.pixel_size[i] for i in range(2)]
        x, y = torch.meshgrid(x, y, indexing='xy')  # H x W

        if psf_center == 'linear':
            center_xy = self.obj_proj_lens(origins)[..., None, None, None, :]  # ... x 1 x 1 x 1 x 2
        elif psf_center == 'chief':
            center_xy = chief_ray.o[..., :2].unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2
        else:
            raise ValueError(f'Unsupported PSF center type for coherent PSF: {psf_center}')

        sampling_grid = center_xy + torch.stack([x, y], -1)  # ... x N_wl x H x W x 2
        sampling_grid = torch.cat([
            sampling_grid, self.surfaces.total_length.broadcast_to(sampling_grid.shape[:-1]).unsqueeze(-1)
        ], -1)  # ... x N_wl x H x W x 3

        # ... x N_wl x H x W x N_spp x 3
        rs2grid_points = sampling_grid.unsqueeze(-2) - ray.o[..., None, None, :, :]
        r_proj = torch.sum(ray.d[..., None, None, :, :] * rs2grid_points, -1)
        wave_vec = base.wave_vec(wl.reshape(-1, 1, 1, 1))
        phase = (r_proj + ray.opl[..., None, None, :]) * wave_vec

        if oblique:
            r_unit = torch.linalg.vector_norm(rs2grid_points)  # ... x N_wl x H x W x N_spp x 3
            rs_normal = ray.o - chief_ray.o
            rs_normal = torch.linalg.vector_norm(rs_normal)  # ... x N_wl x N_spp x 3
            cosine_prop = torch.sum(rs_normal[..., None, None, :, :] * r_unit, -1)
            cosine_rs = torch.sum(rs_normal * ray.d, -1)  # N_fov x N_D x N_wl x N_spp
            obliquity = (cosine_rs[..., None, None, :] + cosine_prop) / 2
            field = torch.polar(obliquity, phase)
        else:
            field = _t.expi(phase)
        field = field.sum(-1)  # N_fov x N_D x N_wl x H x W
        psf = _t.abs2(field)
        return psf / psf.sum((-2, -1), True)  # N_fov x N_D x N_wl x H x W

    def _psf_from_wavefront(
        self,
        origins: Ts,
        size: tuple[int, int],
        wl: Ts,
        psf_center: PsfCenter,
        samples: int = DEFAULT_SAMPLES,
    ) -> Ts:
        """This method is subject to change."""
        if psf_center != 'chief':
            raise ValueError(f'Unsupported PSF center type for wavefront PSF: {psf_center}')

        chief_ray, ray, rs_roc, exit_pupil_distance = self._trace_opl_with_chief(origins, wl, samples, 'rect')
        exit_pupil_distance = exit_pupil_distance.detach()  # TODO: bug to fix

        ref_idx = self.surfaces.mt_tail.n(ray.wl, 'm')
        opd = chief_ray.march(-rs_roc, ref_idx).opl - ray.opl  # ... x N_wl x N_spp
        opd[~ray.valid] = float('nan')
        phase = opd * base.wave_vec(wl.unsqueeze(-1))
        if phase.requires_grad:  # TODO: bug to fix
            phase.register_hook(lambda g: torch.nan_to_num(g, nan=0.))

        spp = phase.size(-1)
        grid_size = int(math.sqrt(spp))
        # ... x N_wl x samples x samples, in lens' coordinate system
        phase = phase.reshape(phase.shape[:-1] + (grid_size, grid_size))
        phase = phase.flip(-2)  # phase on exit pupil in camera's coordinate system

        # TODO: wfe_u and wfe_v are assumed to be uniform in current code, enabling direct bilinear interpolation
        # ... x N_wl x samples x samples
        wfe_u = ray.x.reshape(ray.x.shape[:-1] + (grid_size, grid_size))
        wfe_v = ray.y.reshape(ray.y.shape[:-1] + (grid_size, grid_size))
        u_mean, v_mean = wfe_u.nanmean(-2), wfe_v.nanmean(-1)  # ... x N_wl x samples
        # ... x N_wl
        du, dv = (u_mean.amax(-1) - u_mean.amin(-1)) / grid_size, (v_mean.amax(-1) - v_mean.amin(-1)) / grid_size
        scale = exit_pupil_distance * wl  # ... x N_wl

        # all: ... x N_wl
        factor_x = grid_size * du * self.sensor.pixel_size[1] / scale
        factor_y = grid_size * dv * self.sensor.pixel_size[0] / scale
        factor_x, factor_y = factor_x.max().ceil().int().item(), factor_y.max().ceil().int().item()
        range_u, range_v = factor_x * scale / self.sensor.pixel_size[1], factor_y * scale / self.sensor.pixel_size[0]
        new_u_num, new_v_num = range_u / du, range_v / dv
        new_u_num, new_v_num = new_u_num.mean().round().int().item(), new_v_num.mean().round().int().item()
        du2, dv2 = range_u / new_u_num, range_v / new_v_num

        # bilinear interpolation
        new_v, new_u = utils.grid(
            (new_v_num, new_u_num), (dv2, du2), symmetric=True, dtype=self.dtype, device=self.device,
        )  # ... x N_wl x MH x MW
        new_r = new_v / dv[..., None, None] + (grid_size - 1) / 2
        new_c = new_u / du[..., None, None] + (grid_size - 1) / 2
        upper_r, left_c = new_r.floor().int(), new_c.floor().int()
        lower_r, right_c = upper_r + 1, left_c + 1
        valid_r1, valid_r2 = upper_r.clamp(0, grid_size - 1), lower_r.clamp(0, grid_size - 1)
        valid_c1, valid_c2 = left_c.clamp(0, grid_size - 1), right_c.clamp(0, grid_size - 1)
        _r_vec = torch.stack([lower_r - new_r, new_r - upper_r], -1).unsqueeze(-2)  # ... x N_wl x MH x MW x 1 x 2
        _c_vec = torch.stack([right_c - new_c, new_c - left_c], -1).unsqueeze(-1)  # ... x N_wl x MH x MW x 2 x 1
        pre_idx = [torch.arange(dim_size, device=self.device) for dim_size in phase.shape[:-2]]
        pre_idx = [_t.as1d(idx, len(pre_idx) + 2, i) for i, idx in enumerate(pre_idx)]
        _mat = torch.stack([
            torch.stack([phase[*pre_idx, valid_r1, valid_c1], phase[*pre_idx, valid_r1, valid_c2]], -1),
            torch.stack([phase[*pre_idx, valid_r2, valid_c1], phase[*pre_idx, valid_r2, valid_c2]], -1),
        ], -2)  # ... x N_wl x MH x MW x 2 x 2
        interp_phase = _r_vec @ _mat @ _c_vec
        interp_phase = interp_phase.squeeze(-1).squeeze(-1)  # ... x N_wl x MH x MW
        interp_phase[
            (upper_r != valid_r1) | (lower_r != valid_r2) | (left_c != valid_c1) | (right_c != valid_c2)
            ] = float('nan')  # ... x N_wl x MH x MW

        ep_field = _t.expi(interp_phase)
        ep_field[interp_phase.isnan()] = 0.

        psf = _t.abs2(fourier.ft2(ep_field))  # ... x N_wl x samples x samples

        if factor_x == 1 and factor_y == 1:
            psf = utils.resize(psf, size)
        else:
            psf = utils.resize(psf, (size[0] * factor_y, size[1] * factor_x))
            slices = [[
                psf[..., i::factor_y, j::factor_x] for j in range(factor_x)
            ] for i in range(factor_y)]
            psf = sum([sum(slc) for slc in slices]) / (factor_x * factor_y)

        psf = psf / psf.sum((-2, -1), True)
        psf = psf.flip(-2)
        return psf
