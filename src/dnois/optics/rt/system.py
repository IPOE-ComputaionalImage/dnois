import json
import math
import warnings

import matplotlib.pyplot as plt
import torch

from . import surf
from .ray import BatchedRay
from ..system import RenderingOptics, Pinhole
from ... import scene as _sc, base, utils
from ...base import typing
from ...base.typing import (
    Ts, Any, IO, Size2d, Vector, FovSeg, SclOrVec, SurfSample, Scalar, Self, Sequence,
    scalar, size2d
)

__all__ = [
    'SequentialRayTracing',
]

DEFAULT_PRE_SAMPLES: int = 101
DEFAULT_SAMPLES: int = 201
FOV_THRESHOLD4CHIEF_RAY = math.radians(0.01)


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


def _get_fov(fov: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(fov) <= 0:
        raise ValueError(f'No FoV value given')
    for i, fov_item in enumerate(fov):
        if len(fov_item) != 2:
            raise ValueError(f'Each element in fov must be a pair of floats, got {fov_item} as item {i}')
        if not all(-torch.pi / 2 < angle < torch.pi / 2 for angle in fov_item):
            raise ValueError(f'Invalid FoV (index {i}): {fov_item}')
        if fov_item[0] ** 2 + fov_item[1] ** 2 < FOV_THRESHOLD4CHIEF_RAY ** 2:
            raise NotImplementedError(f'FoV too small (index {i}): {fov_item}, intractable currently')
    return list(fov)


class SequentialRayTracing(RenderingOptics):
    """
    A class of sequential and ray-tracing-based optical system model.
    See :class:`~dnois.optics.RenderingOptics` for descriptions of more parameters.

    :param ``SurfaceList`` surfaces: Optical surfaces.
    :param sampling_arg: Number of sampling rays for each object point.
        See :meth:`CircularSurface.sample` for more details. Default: 256.
    :type sampling_arg: int or tuple[int, int]
    :param str sampling_mode: Mode for sampling rays.
        See :meth:`~dnois.optics.Aperture.sample` for more details. Default: ``random``.
    :param bool final_superposition_coherent: Whether to use coherent superposition
        to compute final PSF. If ``True``, use the algorithm in
        *"Chen, S., Feng, H., Pan, D., Xu, Z., Li, Q., & Chen, Y. (2021).
        Optical aberrations correction in postprocessing using imaging simulation.
        ACM Transactions on Graphics (TOG), 40(5), 1-15."*, otherwise that in
        *"Yang, X., Fu, Q., Elhoseiny, M., & Heidrich, W. (2023).
        Aberration-aware depth-from-focus.
        IEEE Transactions on Pattern Analysis and Machine Intelligence."*
        This must be ``True`` if ``coherent`` is ``True``. Default: same as ``coherent``.
    """

    def __init__(
        self,
        surfaces: surf.SurfaceList,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        nominal_focal_length: float = None,
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = 'paraxial',
        depth: SclOrVec | tuple[Ts, Ts] = None,
        depth_aware: bool = False,
        polarized: bool = False,
        coherent: bool = False,
        sampling_arg: int | tuple[int, int] = 256,
        sampling_mode: SurfSample = 'random',
        final_superposition_coherent: bool = None,
        psf_size: Size2d = None,
    ):
        super().__init__(
            pixel_num, pixel_size, nominal_focal_length, wavelength,
            fov_segments, depth, depth_aware, polarized, coherent,
        )
        if final_superposition_coherent is None:
            final_superposition_coherent = coherent
        if self.coherent and not final_superposition_coherent:
            raise ValueError(f'final_superposition_coherent must be True when coherent is True')
        if psf_size is None:
            raise NotImplementedError()
        self.surfaces: surf.SurfaceList = surfaces  #: Surface list.
        #: Number of sampling rays when computing a PSF.
        self.sampling_arg: int | tuple[int, int] = sampling_arg
        #: Mode for sampling on a surface.
        #: See :meth:`CircularSurface.sample` for more details.
        self.sampling_mode: SurfSample = sampling_mode
        #: Whether to use coherent superposition to compute final PSF. See above descriptions.
        self.final_superposition_coherent: bool = final_superposition_coherent
        self.psf_size: tuple[int, int] = size2d(psf_size)  #: Height and width of evaluated PSF in pixels.

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    def psf_on_grid(
        self,
        grid_size: Size2d,
        grid_spacing: float | tuple[float, float],
        depth: Vector,
        fov: Sequence[tuple[float, float]],
        wl: Vector = None,
    ) -> tuple[BatchedRay, Ts]:  # N_fov x N_D x N_wl x H x W
        grid_size = size2d(grid_size)
        grid_spacing = typing.pair(grid_spacing)
        wl = self._get_wl(wl)  # N_wl
        z = self.depth2z(typing.vector(depth))  # N_D
        fov = self.new_tensor(_get_fov(fov))  # N_fov x 2

        ray, chief_ray = self._generate_rays(fov, z, wl.unsqueeze(-1))  # N_fov x N_d x N_wl x N_spp(1)

        ray = self._obj2img(ray).norm_d_()
        chief_ray = self._obj2img(chief_ray).norm_d_()
        radial_offset = torch.sqrt(chief_ray.o[..., :2].square().sum(-1))
        d_proj = torch.sqrt(chief_ray.d[..., :2].square().sum(-1))
        rs_roc = radial_offset / d_proj  # N_fov x N_d x N_wl x 1

        shift = ray.o[..., :2] - chief_ray.o[..., :2]
        dp = torch.sum(shift * ray.d[..., :2], dim=-1)  # dot product
        # N_fov x N_D x N_wl x N_spp
        length2rs = dp + torch.sqrt(dp.square() - shift.square().sum(-1) + rs_roc.square())
        ref_idx = self.surfaces.mt_tail.n(ray.wl)
        ray = ray.march_(-length2rs, ref_idx)

        lim = ((grid_size[0] - 1) // 2, (grid_size[1] - 1) // 2)
        y, x = [torch.linspace(
            -lim[i], lim[i], grid_size[i], device=self.device, dtype=self.dtype
        ) * grid_spacing[i] for i in range(2)]
        x, y = torch.meshgrid(x, y, indexing='xy')  # H x W
        # N_fov x N_d x N_wl x H x W x 3
        relative_position = chief_ray.o.unsqueeze(-2) + torch.stack([x, y, torch.zeros_like(x)], -1)
        r = relative_position.unsqueeze(-2) - ray.o[..., None, None, :, :]
        r_unit = base.normalize(r)  # N_fov x N_D x N_wl x H x W x N_spp x 3
        rs_normal = ray.o - chief_ray.o
        rs_normal = base.normalize(rs_normal)  # N_fov x N_D x N_wl x N_spp x 3
        cosine_prop = torch.sum(rs_normal[..., None, None, :, :] * r_unit, -1)
        cosine_rs = torch.sum(rs_normal * ray.d, -1)  # N_fov x N_D x N_wl x N_spp
        obliquity = (cosine_rs[..., None, None, :] + cosine_prop) / 2
        r_proj = torch.sum(ray.d[..., None, None, :, :] * r, -1)
        wave_vec = base.wave_vec(wl.reshape(-1, 1, 1, 1))
        field = torch.polar(obliquity, (r_proj + ray.opl[..., None, None, :]) * wave_vec)
        field = field.sum(-1)  # N_fov x N_D x N_wl x H x W
        psf = base.abs2(field)
        return psf / psf.sum((-2, -1), True)  # N_fov x N_D x N_wl x H x W

    def psf_on_points(
        self,
        points: Ts,
        wl: Vector = None,
        polarized: bool = False,
        coherent_superposition: bool = False
    ) -> Ts:
        if polarized:
            raise ValueError(f'Polarization-aware PSF calculation '
                             f'is not available for {self.__class__.__name__}')
        if points.ndim < 1 or points.size(-1) != 3:
            raise base.ShapeError(f'Size of last dimension of points must be 3, '
                                  f'but its shape is {points.shape}')
        if points.isinf().any():
            raise ValueError(f'Coordinates of sources cannot be infinite')
        wl = self._get_wl(wl)  # N_wl

        o = points.unsqueeze(-2)  # ... 1 x 1 x 3
        points = self.surfaces.first.sample(self.sampling_mode, self.sampling_arg)  # N_spp x 3
        d = points.unsqueeze(-3) - o  # ... x 1 x N_spp x 3
        ray = BatchedRay(o, d, wl.unsqueeze(-1))  # ... x N_wl x N_spp
        ray.norm_d_()

        out_ray = self._obj2img(ray)
        if coherent_superposition:
            return self._final_ray2psf_coherent(out_ray)
        else:
            return self._final_ray2psf_incoherent(out_ray)

    def pointwise_render(
        self,
        scene: _sc.ImageScene,
        wavelength: Vector = None,
        sampling_arg: int | tuple[int, int] = None,  # TODO
        vignette: bool = True,
    ) -> Ts:
        """subject to change"""
        if sampling_arg is None:
            sampling_arg = self.sampling_arg
        wl = self._get_wl(wavelength)
        self._check_scene(scene)
        if self.polarized or scene.n_plr != 0:
            raise ValueError(f'Polarization-aware rendering is not available for {self.__class__.__name__}')
        if scene.intrinsic is not None:
            raise NotImplementedError()
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {self.wavelength.numel()} wavelengths expected, '
                             f'got {scene.n_wl}')
        if len(self.surfaces) == 0:
            raise RuntimeError(f'No surface available')

        scene = scene.batch()
        device = self.device
        rm = self.reference
        n_b, n_wl, n_h, n_w = scene.image.shape

        depth = None  # B x H x W
        if scene.depth_aware:
            if self.depth_aware:
                depth = scene.depth
            else:
                warnings.warn(f'Trying to render a depth-aware image by an instance of '
                              f'{self.__class__.__name__} that does not support it')
        if depth is None:
            depth = torch.stack([self.sample_depth() for _ in range(n_b)])
            depth = depth.reshape(-1, 1, 1).expand(-1, n_h, n_w)
        z = self._restrict_in_obj(self.depth2z(depth))
        depth = self.z2depth(z)
        x = torch.linspace(1, -1, n_w, device=device).unsqueeze(0)  # from right to left
        x = x * math.tan(rm.fov_half_x) * depth
        y = torch.linspace(1, -1, n_h, device=device).unsqueeze(-1)  # from top to bottom
        y = y * math.tan(rm.fov_half_y) * depth
        o = torch.stack([x, y, z], -1)  # B x H x W x 3

        points = self.surfaces.first.sample(self.sampling_mode, sampling_arg)  # N_spp x 3
        spp = points.size(0)
        o = o.unsqueeze(-2).unsqueeze(-5)
        d = points - o  # B x 1 x H x W x N_spp x 3

        wl = wl.view(1, -1, 1, 1, 1)
        ray = BatchedRay(o, d, wl)  # B x N_wl x H x W x N_spp
        ray.to_(device=device)
        ray.norm_d_()

        out_ray = self._obj2img(ray)

        x, y = out_ray.x / self.pixel_size[1], out_ray.y / self.pixel_size[0]
        x, y = x + self.pixel_num[1] / 2, y + self.pixel_num[0] / 2
        c_a, r_a = torch.floor(x.detach() + 0.5).long(), torch.floor(y.detach() + 0.5).long()
        in_region = (c_a >= 0) & (c_a <= n_w) & (r_a >= 0) & (r_a <= n_h)  # B x N_wl x H x W x N_spp
        c_a[~in_region] = 0
        r_a[~in_region] = 0
        c_as, r_as = c_a - 1, r_a - 1
        w_c, w_r = c_a - x + 0.5, r_a - y + 0.5
        iw_c, iw_r = 1 - w_c, 1 - w_r

        b_wl_idx = (
            torch.arange(n_b, device=device).view(-1, 1, 1, 1, 1),
            torch.arange(n_wl, device=device).view(1, -1, 1, 1, 1),
        )
        image = self.wavelength.new_zeros(n_b, n_wl, n_h + 2, n_w + 2)
        mask = out_ray.valid & in_region
        gt_image = scene.image.unsqueeze(-1)
        if vignette:
            gt_image = gt_image * spp / out_ray.valid.sum(dim=-1, keepdim=True)
        _gt1 = gt_image * w_c
        _gt2 = gt_image * iw_c
        image.index_put_(b_wl_idx + (r_as, c_as), torch.where(mask, _gt1 * w_r, 0), True)  # top left
        image.index_put_(b_wl_idx + (r_a, c_as), torch.where(mask, _gt1 * iw_r, 0), True)  # bottom left
        image.index_put_(b_wl_idx + (r_as, c_a), torch.where(mask, _gt2 * w_r, 0), True)  # top right
        image.index_put_(b_wl_idx + (r_a, c_a), torch.where(mask, _gt2 * iw_r, 0), True)  # bottom right
        image = image[..., :-2, :-2] / spp
        return image

    def depth2z(self, depth: float | Ts) -> Ts:
        """
        Converts depth to z-coordinate in lens system.

        .. seealso::
            This is the inverse of :meth:`.z2depth`.

        :param depth: Depth.
        :type depth: float | Tensor
        :return: Z-coordinate in lens system.
        :rtype: Tensor
        """
        # TODO: currently depth=0 plane is assumed to be z=0 plane, which is incorrect
        if not torch.is_tensor(depth):
            depth = torch.tensor(depth, device=self.device, dtype=self.dtype)
        return -depth

    @torch.no_grad()
    def focus_to_(self, depth: Scalar) -> Self:
        """This method is subject to change."""
        depth = scalar(depth).to(self.device)
        z = self._restrict_in_obj(self.depth2z(depth))
        o = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z))  # 3
        d = self.surfaces.first.sample(self.sampling_mode, self.sampling_arg) - o  # N_spp x 3
        wl = self.wavelength.reshape(-1, 1)
        ray = BatchedRay(o, d, wl)  # N_wl x N_spp
        ray.norm_d_()

        out_ray = self._obj2img(ray)

        # solve marching distance by least square
        t = -(out_ray.x * out_ray.d_x + out_ray.y * out_ray.d_y)
        t = t / (out_ray.d_x.square() + out_ray.d_y.square())
        new_z = out_ray.z + t * out_ray.d_z
        new_z = new_z[out_ray.valid & new_z.isnan().logical_not()].mean()
        move = new_z - self.surfaces.total_length
        self.surfaces.last.distance += move

        return self

    @torch.no_grad()
    def plot_cross_section(
        self,
        fov: Vector = None,
        wl: Vector = None,
        spp: int = 10,
    ) -> plt.Figure:
        """This method is subject to change."""
        if fov is None:
            fov_half = self.reference.fov_half
            fov = [0., fov_half * 0.5 ** 0.5, fov_half]
        if wl is None:
            wl = self.wavelength
        fov = typing.vector(fov, device=self.device, dtype=self.dtype)
        wl = typing.vector(wl, device=self.device, dtype=self.dtype)

        fig, ax = plt.subplots(figsize=(12.8, 9.6), subplot_kw={'frameon': False})

        self._plot_components(ax)
        # image_plane
        diag_length = (self.sensor_size[0] ** 2 + self.sensor_size[1] ** 2) ** 0.5
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
        return fig

    @torch.no_grad()
    def plot_spot_diagram(self) -> plt.Figure:
        raise NotImplementedError()

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    def wavefront_map(
        self,
        depth: Vector,
        fov: Sequence[tuple[float, float]],
        wl: Vector = None,
    ) -> tuple[BatchedRay, Ts]:  # N_fov x N_D x N_wl x H x W
        wl = self._get_wl(wl)  # N_wl
        z = self.depth2z(typing.vector(depth))  # N_D
        fov = self.new_tensor(_get_fov(fov))  # N_fov x 2

        ray, chief_ray = self._generate_rays(fov, z, wl.unsqueeze(-1))  # N_fov x N_d x N_wl x N_spp(1)

        ray = self._obj2img(ray)
        chief_ray = self._obj2img(chief_ray)
        radial_offset = torch.sqrt(chief_ray.o[..., :2].square().sum(-1))
        d_proj = torch.sqrt(chief_ray.d[..., :2].square().sum(-1))
        rs_roc = radial_offset / d_proj  # N_fov x N_d x N_wl x 1

        shift = ray.o[..., :2] - chief_ray.o[..., :2]
        dp = torch.sum(shift * ray.d[..., :2], dim=-1)  # dot product
        length2rs = dp + torch.sqrt(dp.square() - shift.square().sum(-1) + rs_roc.square())
        ref_idx = self.surfaces.mt_tail.n(ray.wl)
        opd = chief_ray.march_(-rs_roc, ref_idx).opl - ray.march_(-length2rs, ref_idx).opl
        return ray, opd / wl.unsqueeze(-1)  # N_fov x N_d x N_wl x N_spp

    def z2depth(self, z: float | Ts) -> Ts:
        """
        Converts z-coordinate in lens system to depth.

        .. seealso::
            This is the inverse of :meth:`.depth2z`.

        :param z: Z-coordinate in lens system.
        :type z: float | Tensor
        :return: Depth.
        :rtype: Tensor
        """
        # TODO: currently depth=0 plane is assumed to be z=0 plane, which is incorrect
        if not torch.is_tensor(z):
            z = torch.tensor(z, device=self.device, dtype=self.dtype)
        return -z

    @property
    def reference(self) -> Pinhole:
        if self.nominal_focal_length is None:
            raise NotImplementedError()
        return Pinhole(self.nominal_focal_length, self.pixel_num, self.pixel_size)

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
        raise NotImplementedError()

    @property
    def principal2(self) -> Ts:
        raise NotImplementedError()

    # Constructors
    # ===========================

    @classmethod
    def from_json(
        cls, *,
        path: Any = None,
        file: IO = None,
        data: str | bytes | bytearray = None,
        **kwargs
    ) -> 'SequentialRayTracing':
        if data is not None:
            _check_arg(file, 'data', 'file')
            _check_arg(path, 'data', 'path')
            file, path = None, None
        elif file is not None:
            _check_arg(path, 'file', 'path')
            path = None
        elif path is None:
            raise ValueError('Either path, file or data must be provided')

        info = ...
        if path is not None:
            with open(path, 'r', encoding='utf8') as f:
                info = json.load(f, **kwargs)
        if file is not None:
            info = json.load(file, **kwargs)
        if data is not None:
            info = json.loads(data, **kwargs)

        return cls.from_pyds(info)

    @classmethod
    def from_pyds(cls, info: dict[str, Any]) -> 'SequentialRayTracing':
        # surfaces
        sl = surf.SurfaceList([surf.build_surface(cfg) for cfg in info['surfaces']], info['environment_material'])

        # sensor
        sensor_cfg = info['sensor']
        pixel_num = (sensor_cfg['pixels_h'], sensor_cfg['pixels_w'])
        pixel_size = (sensor_cfg['pixel_size_h'], sensor_cfg['pixel_size_w'])

        # optional
        kwargs = {}
        if 'render' in info:
            render_cfg = info['render']

            def _set(k):
                if k in render_cfg:
                    kwargs[k] = render_cfg[k]

            _set('nominal_focal_length')
            _set('wavelength')
            _set('fov_segments')
            _set('depth')
            _set('depth_aware')
            _set('polarized')
            _set('coherent')
            _set('sampling_arg')
        return SequentialRayTracing(sl, pixel_num, pixel_size, **kwargs)

    # protected
    # ========================
    def _restrict_in_obj(self, z: Ts, warn: bool = False) -> Ts:
        if z.gt(0).any() or z.lt(-self.optical_infinity).any():
            if warn:
                warnings.warn(f'z-coordinate exceeding valid region, clamped to [-{self.optical_infinity}, 0]')
            z = z.clamp(-self.optical_infinity, 0)
        return z

    def _obj2img(self, ray: BatchedRay) -> BatchedRay:
        out_ray: BatchedRay = self.surfaces(ray)
        ref_idx = self.surfaces.last.material.n(out_ray.wl, 'm')
        return out_ray.march_to_(self.surfaces.total_length, ref_idx)

    def _final_ray2psf_coherent(self, ray: BatchedRay) -> Ts:
        raise NotImplementedError()

    def _final_ray2psf_incoherent(self, ray: BatchedRay) -> Ts:
        raise NotImplementedError()

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    def _generate_rays(
        self, fov: Ts, z: Ts, wl: Ts, pre_samples: int = DEFAULT_PRE_SAMPLES, samples: int = DEFAULT_SAMPLES
    ) -> tuple[BatchedRay, BatchedRay]:
        if not isinstance(self.surfaces.first.aperture, surf.CircularAperture):
            raise NotImplementedError()

        xy = fov.tan().unsqueeze(-2) * z.unsqueeze(-1)  # N_fov x N_D x 2
        origin = torch.cat([xy, z.reshape(1, -1, 1).expand(xy.size(0), -1, -1)], dim=-1)  # N_fov x N_D x 3
        origin = origin.unsqueeze(-2).unsqueeze(-3)  # N_fov x N_D x 1 x 1 x 3
        d_parallel = torch.stack([fov[:, 0].tan(), fov[:, 1].tan(), fov.new_ones(fov.size(0))], -1)
        d_parallel = d_parallel / d_parallel.norm(2, -1, True)
        d_parallel = d_parallel.view(-1, 1, 1, 1, 3)  # -1 <=> N_fov

        def _make_d(_points: Ts) -> tuple[Ts, Ts]:
            _d = _points - origin
            _length = _d.norm(2, -1)
            _d = _d / _length.unsqueeze(-1)
            if z.isinf().any():
                _d = torch.where(z.isinf().reshape(-1, 1, 1, 1), d_parallel, _d)
            return _d, _length

        r = self.surfaces.first.aperture.radius
        edge_h = self.surfaces.first.h_extended(torch.zeros_like(r), r)
        r = r + torch.sqrt(d_parallel[..., 2].reciprocal().square() - 1) * edge_h  # N_fov x 1 x 1 x 1
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, pre_samples, device=self.device, dtype=self.dtype),
            torch.linspace(-1, 1, pre_samples, device=self.device, dtype=self.dtype),
            indexing='ij'
        )
        x, y = x.flatten(), y.flatten()
        points_pre = torch.stack([x, y, torch.zeros_like(x)], dim=-1)  # N_spp x 3
        points_pre = points_pre * r.unsqueeze(-1)  # N_fov x 1 x 1 x N_spp x 3
        ray = BatchedRay(points_pre, d_parallel, wl, d_normalized=True)
        out_ray = self.surfaces(ray)

        valid = out_ray.valid.broadcast_to(out_ray.shape)  # N_fov x N_D x N_wl x N_spp
        xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
        xy_mean = xy_valid.nanmean(-2, True)  # N_fov x N_D x N_wl x 1 x 2
        points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)
        d_chief, l0_chief = _make_d(points_chief)  # N_fov x N_D x N_wl x 1( x 3)
        chief_ray = BatchedRay(points_chief, d_chief, wl, 0., d_normalized=True)  # N_fov x N_D x N_wl x 1

        # mimicking np.nanmax and np.nanmin
        xy_min = xy_valid.nan_to_num(nan=float('inf')).amin(-2, True)
        xy_max = xy_valid.nan_to_num(nan=-float('inf')).amax(-2, True)
        xy_shift_min = torch.abs(xy_min - xy_mean).unsqueeze(-2)  # N_fov x N_D x N_wl x 1 x 1 x 2
        xy_shift_max = torch.abs(xy_max - xy_mean).unsqueeze(-2)  # N_fov x N_D x N_wl x 1 x 1 x 2

        h_p = torch.linspace(-1, 1, samples, dtype=self.dtype, device=self.device)
        h_p, w_p = torch.meshgrid(-h_p, h_p.clone(), indexing='ij')
        theta = torch.arctan2(h_p, w_p)
        o_p = torch.stack((theta.cos(), theta.sin()), dim=-1)
        o_p *= torch.max(h_p.abs(), w_p.abs()).unsqueeze(-1)  # N_spp' x N_spp' x 2

        o_shape = xy_valid.shape[:3] + (samples, samples, 3)
        o = torch.zeros(o_shape, dtype=self.dtype, device=self.device)
        o[..., :samples // 2, :, 1] = o_p[:samples // 2, :, 1] * xy_shift_max[..., 1]
        o[..., samples // 2:, :, 1] = o_p[samples // 2:, :, 1] * xy_shift_min[..., 1]
        o[..., :, :samples // 2, 0] = o_p[:, :samples // 2, 0] * xy_shift_max[..., 0]
        o[..., :, samples // 2:, 0] = o_p[:, samples // 2:, 0] * xy_shift_min[..., 0]
        o[..., :2] += xy_mean.unsqueeze(-2)
        points_sample = o.flatten(-3, -2)
        d_sample, l0 = _make_d(points_sample)  # N_fov x N_D x N_wl x N_spp'( x 3)
        # to reduce magnitude of opl and subsequently floating point error
        l0 = l0 - l0_chief
        if z.isinf().any():
            l0 = torch.where(
                z.isinf().reshape(-1, 1, 1),
                torch.sum((points_sample - points_chief) * d_parallel, -1),
                l0
            )
        ref_idx = self.surfaces.env_material.n(wl, 'm')
        ray = BatchedRay(points_sample, d_sample, wl, l0 * ref_idx, d_normalized=True)  # ... x N_wl x N_spp
        return ray, chief_ray

    def _plot_components(self, ax):
        uppers = []
        lowers = []
        for sf in self.surfaces:  # surfaces
            sf: surf.Surface
            y = torch.linspace(-sf.aperture.radius, sf.aperture.radius, 100, device=self.device)
            z = sf.h_extended(torch.zeros_like(y), y) + sf.context.z
            ax.plot(utils.t4plot(z), utils.t4plot(y), color='black', linewidth=1)
            uppers.append((z[-1].item(), y[-1].item()))
            lowers.append((z[0].item(), y[0].item()))
        for i in range(len(self.surfaces) - 1):  # edges
            if self.surfaces[i].material.name == 'vacuum':
                continue
            ax.plot(*list(zip(uppers[i], uppers[i + 1])), color='black', linewidth=1)
            ax.plot(*list(zip(lowers[i], lowers[i + 1])), color='black', linewidth=1)

    def _plot_rays(self, ax, ray: BatchedRay, fov: Ts, wl: Ts):  # ray: N_fov x N_wl x N_spp
        ray.norm_d_().broadcast_().march_to_(ray.new_tensor(0.))
        colors = [utils.wl2rgb(_wl, output_format='hex') for _wl in wl.tolist()]
        alphas = _plot_fov_alphas(fov.numel())
        rays_record = []
        for sf in self.surfaces:
            out_ray = sf(ray)
            rays_record.append(out_ray)
            ray = out_ray
        out_ray = ray.march_to(self.surfaces.total_length)
        rays_record.append(out_ray)
        for ray, next_ray in zip(rays_record[:-1], rays_record[1:]):
            _plot_rays_3d(ax, ray.o, next_ray.o, out_ray.valid, colors, alphas)
