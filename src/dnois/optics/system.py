import abc
import math
import warnings

import torch
from torch import nn

from .. import scene as _sc
from ..base import FRAUNHOFER_LINES, ShapeError, TensorContainerMixIn
from ..base.typing import (
    Ts, Size2d, FovSeg, Vector, SclOrVec, Callable, Any,
    size2d, vector, cast, scl_or_vec
)

__all__ = [
    'Optics',
    'Pinhole',
    'RenderingOptics',
    'StandardOptics',
]

_ERMSG_POSITIVE = '{0} must be positive, got {1}'


class Optics(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, scene: _sc.Scene) -> Ts:
        pass


class StandardOptics(Optics, metaclass=abc.ABCMeta):
    """
    TODO
    """

    def __init__(
        self,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        nominal_focal_length: float = None,
    ):
        super().__init__()
        pixel_num = size2d(pixel_num)
        if not isinstance(pixel_size, tuple):
            pixel_size = (pixel_size, pixel_size)
        if any(n <= 0 for n in pixel_num):
            raise ValueError(_ERMSG_POSITIVE.format('Number of pixels', pixel_num))
        if any(s <= 0 for s in pixel_size):
            raise ValueError(_ERMSG_POSITIVE.format('Pixel size', pixel_size))
        if nominal_focal_length is not None and nominal_focal_length <= 0:
            raise ValueError(_ERMSG_POSITIVE.format('Nominal focal length', nominal_focal_length))
        #: Numbers of pixels in vertical and horizontal directions.
        self.pixel_num: tuple[int, int] = pixel_num
        #: Height and width of a pixel in meters.
        self.pixel_size: tuple[float, float] = pixel_size
        #: Focal length of the reference model.
        self.nominal_focal_length: float | None = nominal_focal_length

    @property
    @abc.abstractmethod
    def reference(self) -> 'Pinhole':
        """
        Returns the reference model of this object.

        :type: Pinhole
        """
        pass

    @property
    def sensor_size(self) -> tuple[float, float]:
        """
        Returns the physical size i.e. height and width of the sensor.

        :type: tuple[float, float]
        """
        return self.pixel_size[0] * self.pixel_num[0], self.pixel_size[1] * self.pixel_num[1]


class RenderingOptics(StandardOptics, TensorContainerMixIn, metaclass=abc.ABCMeta):
    """
    TODO

    :param pixel_num: Number of pixels in vertical and horizontal directions or the sensor.
    :type pixel_num: int or tuple[int, int]
    :param pixel_size: Physical size of a pixel in vertical and horizontal directions.
    :type pixel_size: float or tuple[float, float]
    :param float nominal_focal_length: Nominal focal length. Default: omitted.
    :param wavelength: Wavelengths considered. Default: omitted.
    :type wavelength: float, Sequence[float, float] or 1D Tensor
    :param fov_segments: Number of field-of-views when rendering images. Default: ``paraxial``.

        ``int`` or ``tuple[int, int]``
            The numbers in vertical and horizontal directions.

        ``paraxial``
            Only paraxial points are considered.

        ``pointwise``
            The optical responses of every individual object points will be computed.
    :type fov_segments: int, tuple[int, int] or str
    :param depth: Depth adopted for rendering images when the scene to be imaged
        carries no depth information or ``depth_aware`` is ``False``. Default: omitted.

        ``float`` or 0D tensor
            The value will always be used as depth.

        ``Sequence[float]`` or 1D tensor
            Randomly select a value from given value for each image.

        A 2-``tuple`` of 0D tensors
            They are interpreted as minimum and maximum values
            for random sampling (see :py:meth:`~sample_depth`).
    :type depth: float, Sequence[float], Tensor or tuple[Tensor, Tensor]
    :param bool depth_aware: Whether this model supports depth-aware rendering.
        If not, scenes to be images are assumed to lie within a plane
        with single depth even with depth information given. Default: ``False``.
    :param bool polarized: Whether this model supports polarized scenes.
        If not, all the polarized channels of a polarized scene will be treated identically.
        Default: ``False``.
    :param bool coherent: Whether this model renders images coherently. Default: ``False``.
    """
    optical_infinity: float = 1e3  #: Optical "infinite" depth.

    def __init__(
        self,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        nominal_focal_length: float = None,
        wavelength: Vector = None,
        fov_segments: FovSeg | Size2d = 'paraxial',
        depth: SclOrVec | tuple[Ts, Ts] = None,
        depth_aware: bool = False,
        polarized: bool = False,
        coherent: bool = False,
    ):
        super().__init__(pixel_num, pixel_size, nominal_focal_length)
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
        self.coherent: bool = coherent  #: Whether this system is coherent

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

    def forward(self, scene: _sc.Scene, **kwargs) -> Ts:
        self._check_scene(scene)
        scene: _sc.ImageScene
        if self.fov_segments == 'paraxial':
            return self.conv_render(scene, **kwargs)
        elif self.fov_segments == 'pointwise':
            return self.pointwise_render(scene, **kwargs)
        else:  # tuple[int, int]
            return self.patchwise_render(scene, cast(tuple[int, int], self.fov_segments), **kwargs)

    def pointwise_render(self, scene: _sc.ImageScene, **kwargs) -> Ts:
        self._check_scene(scene)
        raise NotImplementedError()

    def patchwise_render(self, scene: _sc.ImageScene, segments: tuple[int, int], **kwargs) -> Ts:
        self._check_scene(scene)
        raise NotImplementedError()

    def conv_render(self, scene: _sc.ImageScene, **kwargs) -> Ts:
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

    def _delegate(self) -> Ts:
        return self.wavelength

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
        super().__init__(pixel_num, pixel_size, focal_length)
        if focal_length <= 0:
            raise ValueError(_ERMSG_POSITIVE.format('Focal length', focal_length))

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
    def focal_length(self) -> float:
        return self.nominal_focal_length

    @property
    def fov_full(self) -> float:
        return self.fov_half * 2

    @property
    def fov_half(self) -> float:
        half_h = self.pixel_size[0] * (self.pixel_num[0] - 1) / 2
        half_w = self.pixel_size[1] * (self.pixel_num[1] - 1) / 2
        tan = math.sqrt(half_h * half_h + half_w * half_w) / self.focal_length
        return math.atan(tan)

    @property
    def fov_half_x(self) -> float:
        half_w = self.pixel_size[1] * (self.pixel_num[1] - 1) / 2
        return math.atan(half_w / self.focal_length)

    @property
    def fov_half_y(self) -> float:
        half_h = self.pixel_size[0] * (self.pixel_num[0] - 1) / 2
        return math.atan(half_h / self.focal_length)
