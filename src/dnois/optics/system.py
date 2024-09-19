import abc
import warnings

import torch
from torch import nn

from dnois.base import FRAUNHOFER_LINES, ShapeError
import dnois.scene as _sc
from dnois.base.typing import (
    Ts, Size2d, FovSeg, Vector, SclOrVec, Callable,
    size2d, vector, cast, scl_or_vec
)


class Optics(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, scene: _sc.Scene) -> Ts:
        pass


class StandardOptics(Optics, metaclass=abc.ABCMeta):
    def __init__(
        self,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
    ):
        super().__init__()
        pixel_num = size2d(pixel_num)
        if not isinstance(pixel_size, tuple):
            pixel_size = (pixel_size, pixel_size)
        if any(n < 0 for n in pixel_num):
            raise ValueError(f'Number of pixels must be positive, got {pixel_num}')
        if any(s < 0 for s in pixel_size):
            raise ValueError(f'Pixel size must be positive, got {pixel_size}')
        #: Numbers of pixels in vertical and horizontal directions.
        self.pixel_num: tuple[int, int] = pixel_num
        #: Height and width of a pixel in meters.
        self.pixel_size: tuple[float, float] = pixel_size

    @abc.abstractmethod
    @property
    def reference(self) -> 'Pinhole':
        pass


class RenderingOptics(StandardOptics, metaclass=abc.ABCMeta):
    def __init__(
        self,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = 'paraxial',
        depth: SclOrVec | tuple[Ts, Ts] = None,
        depth_aware: bool = False,
        polarized: bool = False,
    ):
        super().__init__(pixel_num, pixel_size)
        if not isinstance(fov_segments, str):
            fov_segments = size2d(fov_segments)
        if wavelength is None:
            wavelength = FRAUNHOFER_LINES['d']
        wavelength = vector(wavelength)
        if isinstance(depth, tuple) and len(depth) == 2 and all(torch.is_tensor(t) for t in depth):
            if depth[0].ndim != 0 or depth[1].ndim != 0:
                raise ShapeError(f'If a pair of tensor, both of them should be 0D, got {depth}')
        else:
            depth = scl_or_vec(depth)

        #: Number of segments of FoV when rendering images.
        #: See :py:class:`~SequentialRayTracing` for options.
        self.fov_segments: FovSeg | tuple[int, int] = fov_segments
        self.depth_aware: bool = depth_aware  #: Whether to render images in a depth-aware manner.
        self.polarized: bool = polarized  #: Whether supports polarization-aware imaging.

        self.register_buffer('wavelength', wavelength, False)
        self.wavelength: Ts = wavelength  #: Wavelengths considered.
        if torch.is_tensor(depth) or depth is None:
            self.register_buffer('depth', depth, False)
        else:
            self.register_buffer('depth_min', depth[0], False)
            self.register_buffer('depth_max', depth[1], False)
        self.depth: Ts | tuple[Ts, Ts] = depth  #: Depth values if a scene has no depth information.

    @abc.abstractmethod
    def psf(
        self,
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = None,
        depth_map: Ts = None,
        depth: Vector = None,
        polarized: bool = False,
    ) -> Ts:  # X x Y x D x P x C x H x W
        pass

    def forward(self, scene: _sc.Scene) -> Ts:
        self._check_scene(scene)
        scene: _sc.ImageScene
        if self.fov_segments == 'paraxial':
            return self.conv_render(scene)
        elif self.fov_segments == 'pointwise':
            return self.point_render(scene)
        else:
            return self.patchwise_render(scene, cast(tuple[int, int], self.fov_segments))

    def pointwise_render(self, scene: _sc.ImageScene) -> Ts:
        self._check_scene(scene)
        raise NotImplementedError()

    def patchwise_render(self, scene: _sc.ImageScene, segments: tuple[int, int]) -> Ts:
        self._check_scene(scene)
        raise NotImplementedError()

    def conv_render(self, scene: _sc.ImageScene) -> Ts:
        self._check_scene(scene)
        kwargs = {}
        warning_str = (f'Trying to render a {{0}} image by an instance of '
                       f'{self.__class__.__name__} that does not support it')
        depth_aware, polarized = False, False
        if scene.depth_aware:
            if self.depth_aware:
                kwargs['depth_map'] = scene.depth
                depth_aware = True
            else:
                warnings.warn(warning_str.format('depth aware'))
        if scene.n_plr != 0:
            if self.polarized:
                kwargs['polarized'] = True
                polarized = True
            else:
                warnings.warn(warning_str.format('polarized'))

        # PSF shape: (D x )(P x )C x H x W
        psf = self.psf(fov_segments='paraxial', **kwargs)
        raise NotImplementedError()

    def sample_depth(self, sampling_curve: Callable[[Ts], Ts] = None, probabilities: Ts = None) -> Ts:
        depth = self.depth
        if isinstance(depth, tuple):  # randomly sampling from a depth range
            d1, d2 = self.depth_min, self.depth_max
            t = torch.rand_like(d2)
            if sampling_curve is not None:
                t = sampling_curve(t)
            return d1 + (d2 - d1) * t
        elif depth.ndim == 0:  # fixed depth
            return depth
        else:  # randomly sampling from a set of depth values
            if probabilities is None:
                idx = torch.randint(depth.numel(), ()).item()
            else:
                if probabilities.ndim != 1:
                    raise ShapeError(f'probabilities must be a 1D tensor')
                idx = torch.multinomial(probabilities, 1).squeeze().item()
            return depth[idx]

    def _check_scene(self, scene: _sc.Scene):
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'{self.__class__.__name__} only supports ImageScene at present')
        if scene.n_plr != 0:
            raise RuntimeError(f'{self.__class__.__name__} does not support polarization currently')


class Pinhole(StandardOptics):
    def __init__(
        self,
        focal_length: float,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
    ):
        super().__init__(pixel_num, pixel_size)
        if focal_length <= 0:
            raise ValueError(f'Focal length must be positive, got {focal_length}')
        self.focal_length: float = focal_length  #: Focal length in meters.

    @property
    def reference(self) -> 'Pinhole':
        return self

    def forward(self, scene: _sc.Scene) -> Ts:
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'{self.__class__.__name__} only supports ImageScene at present')
        if (si := scene.intrinsic) is not None:
            if not torch.allclose(si, self.intrinsic().broadcast_to(si.size(0), -1, -1)):
                raise NotImplementedError()
        if (scene.height, scene.width) != self.pixel_num:
            raise NotImplementedError()
        return scene.image

    def intrinsic(self, **kwargs) -> Ts:
        """
        Intrinsic matrix of the pinhole camera.

        :param kwargs: Keyword arguments passed to :py:func:`torch.zeros`
            to construct the matrix.
        :return: A tensor of shape (3, 3).
        :rtype: Tensor
        """
        kwargs.setdefault('dtype', torch.float32)
        i = torch.zeros(3, 3, **kwargs)
        i[0, 0] = self.focal_length / self.pixel_size[1]
        i[1, 1] = self.focal_length / self.pixel_size[0]
        i[2, 2] = 1
        i[0, 2] = self.pixel_num[1] / 2 * self.pixel_size[1]
        i[1, 2] = self.pixel_num[0] / 2 * self.pixel_size[0]
        return i

    @property
    def fov_full(self) -> Ts:
        return self.fov_half * 2

    @property
    def fov_half(self) -> Ts:
        half_h = torch.tensor(self.pixel_size[0] * self.pixel_num[0]) / 2
        half_w = torch.tensor(self.pixel_size[1] * self.pixel_num[1]) / 2
        tan = torch.sqrt(half_h.square() + half_w.square()) / self.focal_length
        return tan.atan().rad2deg()
