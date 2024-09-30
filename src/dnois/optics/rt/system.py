import json
import math
import warnings

import torch

import dnois.scene as _sc
from dnois.base.typing import (
    Ts, Any, IO, Size2d, Vector, FovSeg, SclOrVec, SurfSample, Scalar, Self,
    scalar
)

from ..system import RenderingOptics, Pinhole
from .ray import BatchedRay
from ._surf import SurfaceList
from .surf import build_surface

__all__ = [
    'SequentialRayTracing',
]

DEFAULT_SPP: int = 256


def _check_arg(arg: Any, n1: str, n2: str):
    if arg is not None:
        warnings.warn(f'{n1} and {n2} are given simultaneously, {n2} will be ignored')


class SequentialRayTracing(RenderingOptics):
    """
    A class of sequential and ray-tracing-based optical system model.

    :param SurfaceList surfaces: Optical surfaces.
    :param samples_per_point: Number of sampling rays for each object point.
        A single int for ``random`` mode or a 2-tuple of ints
        for ``rectangular`` and ``circular`` mode.
    :type samples_per_point: int or tuple[int, int]
    :param str sampling_mode: Mode for sampling rays. Choices (default: ``random``):

        ``random``
            Rays are sampled randomly (see :py:meth:`Surface.sample_random`).

        ``rectangular``
            Rays are sampled on evenly spaced rectangular grid.

        ``circular``
            Rays are sampled on evenly spaced circular grid.
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
    ):
        super().__init__(
            pixel_num, pixel_size, nominal_focal_length, wavelength,
            fov_segments, depth, depth_aware, polarized, coherent,
        )
        if isinstance(samples_per_point, int) and sampling_mode in ('rectangular', 'circular'):
            raise TypeError(f'A pair of int expected for samples_per_point when sampling_mode '
                            f'is {sampling_mode}, got {samples_per_point}')
        elif isinstance(samples_per_point, tuple) and sampling_mode == 'random':
            raise TypeError(f'A single int expected for samples_per_point when sampling_mode is '
                            f'random, got {samples_per_point}')
        self.surfaces: SurfaceList = surfaces  #: Surface list.
        #: Number of sampling rays when computing a PSF.
        #: A single int for random sample, or a pair of int representing
        #: the numbers along two directions.
        self.samples_per_point: int | tuple[int, int] = samples_per_point
        #: Mode for sampling on a surface, either ``rectangular``, ``unipolar`` or ``random``.
        self.sampling_mode: SurfSample = sampling_mode

    def psf(
        self,
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = None,
        depth_map: Ts = None,
        depth: Vector = None,
        polarized: bool = False
    ) -> Ts:
        raise NotImplementedError()

    def pointwise_render(
        self,
        scene: _sc.ImageScene,
        vignette: bool = True,
        samples_per_point: int | tuple[int, int] = None,
        sampling_mode: SurfSample = None,
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
        sampling_mode = sampling_mode or self.sampling_mode
        samples_per_point = samples_per_point or self.samples_per_point

        scene = scene.batch()
        device = self.device
        rm = self.reference
        n_b, n_wl, n_h, n_w = scene.image.shape

        warning_str = (f'Trying to render a {{0}} image by an instance of '
                       f'{self.__class__.__name__} that does not support it')
        depth = None  # B x H x W
        if scene.depth_aware:
            if self.depth_aware:
                depth = scene.depth
            else:
                warnings.warn(warning_str.format('depth aware'))
        if depth is None:
            depth = torch.stack([self.sample_depth() for _ in range(n_b)])
            depth = depth.reshape(-1, 1, 1).expand(-1, n_h, n_w)
        depth = depth.clamp(0, self.optical_infinity)
        x = torch.linspace(1, -1, n_w, device=device).unsqueeze(0)  # from right to left
        x = x * math.tan(rm.fov_half_x) * depth
        y = torch.linspace(1, -1, n_h, device=device).unsqueeze(-1)  # from top to bottom
        y = y * math.tan(rm.fov_half_y) * depth
        o = torch.stack([x, y, -depth], -1)  # B x H x W x 3
        # TODO: currently depth=0 plane is assumed to be z=0 plane, which is incorrect

        x0, y0 = self.surfaces[0].sample(samples_per_point, sampling_mode)
        spp = x0.numel()
        points = torch.stack([x0, y0, torch.zeros_like(x0)], dim=-1)  # N_spp x 3
        o = o.unsqueeze(-2).unsqueeze(-5)
        d = points - o  # B x 1 x H x W x N_spp x 3

        wl = self.wavelength.view(1, -1, 1, 1, 1)
        ray = BatchedRay(o, d, wl)  # B x N_wl x H x W x N_spp
        ray.to_(device=device)
        ray.norm_d_()

        out_ray: BatchedRay = self.surfaces(ray)
        out_ray.march_to_(self.surfaces.total_length)

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

    @torch.no_grad()
    def focus_to_(self, z: Scalar) -> Self:  # TODO: focus to depth rather than z
        z = scalar(z).to(self.device)
        z = z.clamp(-self.optical_infinity, 0)
        o = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z))  # 3

        x0, y0 = self.surfaces[0].sample(self.samples_per_point, self.sampling_mode)
        points = torch.stack([x0, y0, torch.zeros_like(x0)], dim=-1).to(o)  # N_spp x 3
        d = points - o  # N_spp x 3

        wl = self.wavelength.reshape(-1, 1)
        ray = BatchedRay(o, d, wl)  # N_wl x N_spp
        ray.norm_d_()

        out_ray: BatchedRay = self.surfaces(ray)
        out_ray.march_to_(self.surfaces.total_length)

        # solve marching distance by least square
        t = -(out_ray.x * out_ray.d_x + out_ray.y * out_ray.d_y)
        t = t / (out_ray.d_x.square() + out_ray.d_y.square())
        new_z = out_ray.z + t * out_ray.d_z
        new_z = new_z[out_ray.valid & new_z.isnan().logical_not()].mean()
        move = new_z - self.surfaces.total_length
        self.surfaces[-1].distance += move

        return self

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
        # lens group
        env_mt = info['environment_material']
        surfaces = [build_surface(cfg) for cfg in info['surfaces']]
        sl = SurfaceList(surfaces, env_mt)

        # sensor
        sensor_cfg = info['sensor']
        pixel_num = (sensor_cfg['pixels_h'], sensor_cfg['pixels_w'])
        pixel_size = (sensor_cfg['pixel_size_h'], sensor_cfg['pixel_size_w'])

        # optional
        kwargs = {}
        if 'render' in info:
            render_cfg = info['render']
            if 'fov_segments' in render_cfg:
                kwargs['fov_segments'] = render_cfg['fov_segments']
        return SequentialRayTracing(sl, pixel_num, pixel_size, **kwargs)
