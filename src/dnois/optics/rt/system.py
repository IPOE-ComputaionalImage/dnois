import json
import math
import warnings

import matplotlib.pyplot as plt
import torch

from ... import scene as _sc, base
from ...base import typing
from ...base.typing import (
    Ts, Any, IO, Size2d, Vector, FovSeg, SclOrVec, SurfSample, Scalar, Self,
    scalar, size2d
)
from ...utils import wl2rgb, t4plot
from ..system import RenderingOptics, Pinhole
from .ray import BatchedRay
from .surf import build_surface, SurfaceList, Surface, CircularSurface

__all__ = [
    'SequentialRayTracing',
]

DEFAULT_SPP: int = 256


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
    # shape: N_wl x N_fov x N_spp x 3
    for clr, wl_slc1, wl_slc2, v in zip(colors, start, end, valid):  # N_fov x N_spp x 3
        for alpha, fov_slc1, fov_slc2, vv in zip(alphas, wl_slc1, wl_slc2, v):  # N_spp x 3
            ax.plot(
                (t4plot(fov_slc1[..., 2][vv]), t4plot(fov_slc2[..., 2][vv])),
                (t4plot(fov_slc1[..., 1][vv]), t4plot(fov_slc2[..., 1][vv])),
                color=clr, alpha=alpha, linewidth=0.5
            )


class SequentialRayTracing(RenderingOptics):
    """
    A class of sequential and ray-tracing-based optical system model.
    See :class:`~dnois.optics.RenderingOptics` for descriptions of more parameters.

    :param SurfaceList surfaces: Optical surfaces.
    :param samples_per_point: Number of sampling rays for each object point.
        See :meth:`CircularSurface.sample` for more details. Default: 256.
    :type samples_per_point: int or tuple[int, int]
    :param str sampling_mode: Mode for sampling rays.
        See :meth:`CircularSurface.sample` for more details. Default: ``random``.
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
        surfaces: SurfaceList,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        nominal_focal_length: float = None,
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = 'paraxial',
        depth: SclOrVec | tuple[Ts, Ts] = None,
        depth_aware: bool = False,
        polarized: bool = False,
        coherent: bool = False,
        samples_per_point: int | tuple[int, int] = 256,
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
        if isinstance(samples_per_point, int) and sampling_mode in ('rectangular', 'circular'):
            raise TypeError(f'A pair of int expected for samples_per_point when sampling_mode '
                            f'is {sampling_mode}, got {samples_per_point}')
        elif isinstance(samples_per_point, tuple) and sampling_mode == 'random':
            raise TypeError(f'A single int expected for samples_per_point when sampling_mode is '
                            f'random, got {samples_per_point}')
        if psf_size is None:
            raise NotImplementedError()
        self.surfaces: SurfaceList = surfaces  #: Surface list.
        #: Number of sampling rays when computing a PSF.
        #: See :meth:`CircularSurface.sample` for more details.
        self.samples_per_point: int | tuple[int, int] = samples_per_point
        #: Mode for sampling on a surface.
        #: See :meth:`CircularSurface.sample` for more details.
        self.sampling_mode: SurfSample = sampling_mode
        #: Whether to use coherent superposition to compute final PSF. See above descriptions.
        self.final_superposition_coherent: bool = final_superposition_coherent
        self.psf_size: tuple[int, int] = size2d(psf_size)  #: Height and width of evaluated PSF.

    def psf_on_grid(
        self,
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = None,
        depths: Vector | int = None,
        polarized: bool = False
    ) -> Ts:
        raise NotImplementedError()

    def psf_on_points(self, points: Ts, wl: Vector = None, polarized: bool = False) -> Ts:
        if polarized:
            raise NotImplementedError()
        if points.ndim < 1 or points.size(-1) != 3:
            raise base.ShapeError(f'Size of last dimension of points must be 3, '
                                  f'but its shape is {points.shape}')
        wl = self._get_wl(wl)

        o = points.unsqueeze(-2).unsqueeze(-2)  # ... 1 x 1 x 3
        x0, y0 = self.surfaces[0].sample(self.samples_per_point, self.sampling_mode)
        points = torch.stack([x0, y0, torch.zeros_like(x0)], dim=-1)  # N_spp x 3
        d = points.unsqueeze(-2) - o.unsqueeze(-2)  # ... x N_spp x 1 x 3
        ray = BatchedRay(o, d, wl.unsqueeze(-1))  # ... x N_spp x N_wl
        ray.norm_d_()

        out_ray = self.surfaces(ray)
        if self.final_superposition_coherent:
            return self._final_ray2psf_coherent(out_ray)
        else:
            return self._final_ray2psf_incoherent(out_ray)

    def pointwise_render(
        self,
        scene: _sc.ImageScene,
        vignette: bool = True,
    ) -> Ts:
        self._check_scene(scene)
        if self.polarized or scene.n_plr != 0:
            raise NotImplementedError()
        if scene.intrinsic is not None:
            raise NotImplementedError()
        if self.wavelength.numel() != scene.n_wl:
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
        depth = depth.clamp(0, self.optical_infinity)
        x = torch.linspace(1, -1, n_w, device=device).unsqueeze(0)  # from right to left
        x = x * math.tan(rm.fov_half_x) * depth
        y = torch.linspace(1, -1, n_h, device=device).unsqueeze(-1)  # from top to bottom
        y = y * math.tan(rm.fov_half_y) * depth
        o = torch.stack([x, y, self.depth2z(depth)], -1)  # B x H x W x 3

        x0, y0 = self.surfaces[0].sample(self.samples_per_point, self.sampling_mode)
        spp = x0.numel()
        points = torch.stack([x0, y0, torch.zeros_like(x0)], dim=-1)  # N_spp x 3
        o = o.unsqueeze(-2).unsqueeze(-5)
        d = points - o  # B x 1 x H x W x N_spp x 3

        wl = self.wavelength.view(1, -1, 1, 1, 1)
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
        z = self.depth2z(depth)
        z = z.clamp(-self.optical_infinity, 0)
        o = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z))  # 3

        x0, y0 = self.surfaces[0].sample(self.samples_per_point, self.sampling_mode)
        points = torch.stack([x0, y0, torch.zeros_like(x0)], dim=-1).to(o)  # N_spp x 3
        d = points - o  # N_spp x 3

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
        self.surfaces[-1].distance += move

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
        fov = typing.vector(fov, device=self.device)
        wl = typing.vector(wl, device=self.device)

        fig, ax = plt.subplots(figsize=(12.8, 9.6), subplot_kw={'frameon': False})

        uppers = []
        lowers = []
        for sf in self.surfaces:  # surfaces
            sf: Surface
            y = torch.linspace(-sf.radius, sf.radius, 100, device=self.device)
            z = sf.h_extended(torch.zeros_like(y), y) + sf.context.z
            ax.plot(t4plot(z), t4plot(y), color='black', linewidth=1)
            uppers.append((z[-1].item(), y[-1].item()))
            lowers.append((z[0].item(), y[0].item()))
        for i in range(len(self.surfaces) - 1):  # edges
            if self.surfaces[i].material.name == 'vacuum':
                continue
            ax.plot(*list(zip(uppers[i], uppers[i + 1])), color='black', linewidth=1)
            ax.plot(*list(zip(lowers[i], lowers[i + 1])), color='black', linewidth=1)
        # sensor
        diag_length = (self.sensor_size[0] ** 2 + self.sensor_size[1] ** 2) ** 0.5
        y = torch.linspace(-diag_length / 2, diag_length / 2, 100, device=self.device)
        ax.plot(
            t4plot(torch.full_like(y, self.surfaces.total_length)), t4plot(y),
            color='black', linewidth=2
        )
        # rays
        total_length = self.surfaces.total_length.item()
        sf1 = self.surfaces[0]
        o_y = torch.linspace(-sf1.radius * 0.98, sf1.radius * 0.98, spp, device=self.device)
        o = torch.stack([torch.zeros_like(o_y), o_y, torch.zeros_like(o_y)], -1)  # N_spp x 3
        d = torch.stack([torch.zeros_like(fov), fov.tan(), torch.ones_like(fov)], dim=-1)
        d = d.unsqueeze(1)  # N_fov x 1 x 3
        ray = BatchedRay(o, d, wl.reshape(-1, 1, 1))  # N_wl x N_fov x N_spp
        ray.norm_d_().broadcast_()
        colors = [wl2rgb(_wl, output_format='hex') for _wl in wl.tolist()]
        alphas = _plot_fov_alphas(fov.numel())
        for sf in self.surfaces:
            out_ray = sf(ray)
            _plot_rays_3d(ax, ray.o, out_ray.o, out_ray.valid, colors, alphas)
            ray = out_ray
        out_ray = ray.march_to(total_length)
        _plot_rays_3d(ax, ray.o, out_ray.o, out_ray.valid, colors, alphas)

        _plot_set_ax(ax, total_length)
        return fig

    @torch.no_grad()
    def plot_spot_diagram(self) -> plt.Figure:
        """This method is subject to change."""
        if self.sampling_mode != 'unipolar':
            raise NotImplementedError()

        fig, ax = plt.subplots(3, 1)

        sf1 = self.surfaces[0]
        if not isinstance(sf1, CircularSurface):
            raise NotImplementedError()
        x, y = sf1.sample(self.samples_per_point, self.sampling_mode)
        o = torch.stack([x, y, torch.zeros_like(y)], -1)  # N_spp x 3

        fov_half = self.reference.fov_half
        fov = torch.tensor([0., fov_half * 0.5 ** 0.5, fov_half], device=self.device)
        d = torch.stack([torch.zeros_like(fov), fov.tan(), torch.ones_like(fov)], dim=-1)
        d = d.unsqueeze(1)  # N_fov x 1 x 3
        ray = BatchedRay(o, d, ...)

        # TODO: complete

        return fig

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
        sl = SurfaceList([build_surface(cfg) for cfg in info['surfaces']], info['environment_material'])

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
            _set('samples_per_point')
            _set('sampling_mode')
        return SequentialRayTracing(sl, pixel_num, pixel_size, **kwargs)

    # protected
    # ========================

    def _obj2img(self, ray: BatchedRay) -> BatchedRay:
        out_ray: BatchedRay = self.surfaces(ray)
        return out_ray.march_to_(self.surfaces.total_length)

    def _final_ray2psf_coherent(self, ray: BatchedRay) -> Ts:
        raise NotImplementedError()

    def _final_ray2psf_incoherent(self, ray: BatchedRay) -> Ts:
        raise NotImplementedError()
