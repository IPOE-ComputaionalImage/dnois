import json
import warnings

import dnois.scene as _sc
from dnois.base.typing import Ts, Any, IO, Size2d, Vector, FovSeg

from ..system import RenderingOptics, Pinhole
from .lg import LensGroup
from .surf import build_surface


def _check_arg(arg: Any, n1: str, n2: str):
    if arg is not None:
        warnings.warn(f'{n1} and {n2} are given simultaneously, {n2} will be ignored')


class SequentialRayTracing(RenderingOptics):
    def __init__(
        self,
        lens_group: LensGroup,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = 'paraxial',
        depth_aware: bool = False,
    ):
        super().__init__(pixel_num, pixel_size, wavelength, fov_segments, depth_aware)
        self.lens_group: LensGroup = lens_group  #: Lens group.

    def psf(
        self,
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = None,
        depth_map: Ts = None,
        depth: Vector = None,
        polarized: bool = False
    ) -> Ts:
        raise NotImplementedError()

    def pointwise_render(self, scene: _sc.ImageScene) -> Ts:
        if self.polarized or scene.n_plr != 0:
            raise NotImplementedError()

        warning_str = (f'Trying to render a {{0}} image by an instance of '
                       f'{self.__class__.__name__} that does not support it')
        if scene.depth_aware:
            if self.depth_aware:
                depth = scene.depth
            else:
                warnings.warn(warning_str.format('depth aware'))
                depth = self.sample_depth()
        else:
            depth = self.sample_depth()

        raise NotImplementedError()

    @property
    def reference(self) -> Pinhole:
        raise NotImplementedError()

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
        lg = LensGroup(surfaces, env_mt)

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
        return SequentialRayTracing(lg, pixel_num, pixel_size, **kwargs)
