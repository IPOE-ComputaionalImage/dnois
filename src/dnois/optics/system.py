import abc
import math
import warnings

import torch
from torch import nn

from . import formation
from .. import scene as _sc, base, utils, torch as _t
from ..base import ShapeError, typing
from ..base.typing import (
    Ts, Size2d, FovSeg, Vector, Callable, Any,
    size2d, vector, cast
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
    """Base class of :ref:`standard optical system <guide_imodel_standard_optical_system>`."""

    def __init__(
        self,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        image_distance: float = None,
    ):
        super().__init__()
        pixel_num = size2d(pixel_num)
        if not isinstance(pixel_size, tuple):
            pixel_size = (pixel_size, pixel_size)
        if any(n <= 0 for n in pixel_num):
            raise ValueError(_ERMSG_POSITIVE.format('Number of pixels', pixel_num))
        if any(s <= 0 for s in pixel_size):
            raise ValueError(_ERMSG_POSITIVE.format('Pixel size', pixel_size))
        if image_distance is not None and image_distance <= 0:
            raise ValueError(_ERMSG_POSITIVE.format('Image distance', image_distance))
        #: Numbers of pixels in vertical and horizontal directions.
        self.pixel_num: tuple[int, int] = pixel_num
        #: Height and width of a pixel.
        self.pixel_size: tuple[float, float] = pixel_size
        #: Focal length of the reference model, i.e. the image distance.
        self.image_distance: float | None = image_distance

    def tanfovd2obj(self, tanfov: typing.Sequence[tuple[float, float]] | Ts, depth: float | Ts) -> Ts:
        r"""
        Computes 3D coordinates of points in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        given tangents of their FoV angles and depths:

        .. math::
            (x,y,z)=z(-\tan\varphi_x,-\tan\varphi_y,1)

        where :math:`z` indicates depth. Returned coordinates comply with
        :ref:`guide_imodel_ccs_inf`.

        :param tanfov: Tangents of FoV angles of points in radians. A tensor with shape ``(..., 2)``
            where the last dimension indicates x and y FoV angles.
        :type tanfov: Sequence[tuple[float, float]] or Tensor
        :param depth: Depths of points. A tensor with any shape that is
            broadcastable with ``tanfov`` other than its last dimension.
        :type depth: float | Tensor
        :return: 3D coordinates of points, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if not torch.is_tensor(tanfov):
            tanfov = self.new_tensor(tanfov)
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        typing.check_2d_vector(tanfov, f'tanfov in {self.fovd2obj.__qualname__}')

        tanx, tany, z = torch.broadcast_tensors(-tanfov[..., 0], -tanfov[..., 1], depth)  # ...
        is_inf = z.isinf()
        if is_inf.any():
            point_at_inf = torch.stack([tanx, tany, z], -1)  # ... x 3
            if is_inf.all():
                return point_at_inf
            else:
                return torch.where(is_inf, point_at_inf, torch.stack([tanx * z, tany * z, z], -1))
        else:
            return torch.stack([tanx * z, tany * z, z], -1)  # ... x 3

    def fovd2obj(
        self, fov: typing.Sequence[tuple[float, float]] | Ts, depth: float | Ts, in_degrees: bool = False
    ) -> Ts:
        r"""
        Similar to :meth:`.tanfovd2obj`, but computes coordinates
        from FoV angles rather than their tangents.

        :param fov: FoV angles of points in radians. A tensor with shape ``(..., 2)``
            where the last dimension indicates x and y FoV angles.
        :type fov: Sequence[tuple[float, float]] or Tensor
        :param depth: Depths of points. A tensor with any shape that is
            broadcastable with ``fov`` other than its last dimension.
        :type depth: float | Tensor
        :param bool in_degrees: Whether ``fov`` is in degrees. Default: ``False``.
        :return: 3D coordinates of points, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if not torch.is_tensor(fov):
            fov = self.new_tensor(fov)
        if in_degrees:
            fov = fov.deg2rad()
        if not torch.all(fov.gt(-torch.pi / 2) & fov.lt(torch.pi / 2)):
            raise ValueError(f'FoV angle must lie in the range (-pi/2, pi/2)')
        return self.tanfovd2obj(fov.tan(), depth)

    def obj2tanfov(self, point: Ts) -> Ts:
        r"""
        Converts coordinates of points in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        into tangent of corresponding FoV angles:

        .. math::
            \tan\varphi_x=-x/z\\
            \tan\varphi_y=-y/z

        ``point`` complies with :ref:`guide_imodel_ccs_inf`.

        :param Tensor point: Coordinates of points. A tensor with shape ``(..., 3)``
            where the last dimension indicates coordinates of points in camera's coordinate system.
        :return: Tangent of x and y FoV angles. A tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        typing.check_3d_vector(point, f'point in {self.obj2fov.__qualname__}')

        is_inf = point[..., [2]].isinf()  # ... x 1
        point2d = -point[..., :2]
        if is_inf.all():
            return point2d  # ... x 2
        else:
            if is_inf.any():
                return torch.where(is_inf, point2d, point2d / point[..., [2]])
            else:
                return point2d / point[..., [2]]

    def obj2fov(self, point: Ts, in_degrees: bool = False) -> Ts:
        """
        Similar to :meth:`.point2tanfov`, but returns FoV angles rather than tangents.

        :param Tensor point: Coordinates of points. A tensor with shape ``(..., 3)``
            where the last dimension indicates coordinates of points in camera's coordinate system.
        :param bool in_degrees: Whether to return FoV angles in degrees. Default: ``False``.
        :return: x and y FoV angles. A tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        fov = self.obj2tanfov(point).arctan()
        return fov.rad2deg() if in_degrees else fov

    def obj_proj(self, point: Ts, flip: bool = True) -> Ts:
        r"""
        Projects coordinates of points in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        to image plane in a perspective manner:

        .. math::
            \left\{\begin{array}{l}
            x'=-\frac{f}{z}x
            y'=-\frac{f}{z}y
            \end{array}\right.

        where :math:`f` is the focal length of :ref:`reference model <guide_imodel_ref_model>`.
        The negative sign is eliminated if ``flip`` is ``True``.

        :param Tensor point: Coordinates of points. A tensor with shape ``(..., 3)``
            where the last dimension indicates coordinates of points in camera's coordinate system.
        :param bool flip: If ``True``, returns coordinates projected on flipped (virtual) image plane.
            Otherwise, returns those projected on original image plane.
        :return: Projected x and y coordinates of points. A tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        xy = self.obj2tanfov(point) * self.reference.focal_length
        return -xy if flip else xy

    @property
    @abc.abstractmethod
    def reference(self) -> 'Pinhole':
        """
        Returns the :ref:`reference model <guide_imodel_ref_model>` of  this object.

        :type: :class:`Pinhole`
        """
        pass

    @property
    def sensor_size(self) -> tuple[float, float]:
        """
        Returns the physical size i.e. height and width of the sensor.

        :type: tuple[float, float]
        """
        return self.pixel_size[0] * self.pixel_num[0], self.pixel_size[1] * self.pixel_num[1]


class RenderingOptics(_t.TensorContainerMixIn, StandardOptics, metaclass=abc.ABCMeta):
    """
    Base class for optical systems with optical imaging behavior defined.
    See :doc:`/content/guide/optics/imodel` for details.

    See :class:`StandardOptics` for descriptions about more parameters.

    :param wl: Wavelengths for imaging. Default: Fraunhofer *d* line.
        See :func:`~dnois.fraunhofer_line` for details.
    :type wl: float, Sequence[float] or 1D Tensor
    :param fov_segments: Number of field-of-view segments when rendering images.
        Default: ``'paraxial'``.

        ``int`` or ``tuple[int, int]``
            The numbers of FoV segments in vertical and horizontal directions.
            PSFs in each segment are assumed to be FoV-invariant.

        ``'paraxial'``
            Only "paraxial" points are considered. In other words,
            PSF is assumed to not depend on FoV.

        ``'pointwise'``
            The optical responses of every individual object points will be computed.
    :type fov_segments: int, tuple[int, int] or str
    :param depth: Depth adopted for rendering images when the scene to be imaged
        carries no depth information or ``depth_aware`` is ``False``. Default: infinity.

        float or 0D tensor
            The value will always be used as depth.

        Sequence[float] or 1D tensor
            Randomly select a value from given value for each image.

        A pair of 0D tensors
            They are interpreted as minimum and maximum values
            for random sampling (see :py:meth:`~sample_depth`).
    :type depth: float, Sequence[float], Tensor or tuple[Tensor, Tensor]
    :param bool depth_aware: Whether this model supports depth-aware rendering.
        If not, scenes to be images are assumed to lie within a plane
        with single depth even with depth map given. Default: ``False``.
    :param psf_size: Height and width of PSF (i.e. convolution kernel) used to simulate imaging.
        Default: ``(64, 64)``.
    :type psf_size: int or tuple[int, int]
    :param patch_padding: Padding amount for each patch used for patch-wise (spatially variant) imaging.
        See :func:`~dnois.optics.space_variant` for more details. Default: ``(0, 0)``.
    :type patch_padding: int or tuple[int, int]
    :param bool linear_conv: Whether to compute linear convolution rather than
        circular convolution when computing blurred image. Default: ``False``.
    :param str patch_merging: Merging method to use for patch-wise (spatially variant) imaging.
        See :func:`~dnois.optics.space_variant` for more details. Default: ``'slope'``.
    """

    def __init__(
        self,
        pixel_num: Size2d,
        pixel_size: float | tuple[float, float],
        image_distance: float = None,
        *,
        wl: Vector = None,
        fov_segments: FovSeg | Size2d = 'paraxial',
        depth: Vector | tuple[Ts, Ts] = float('inf'),
        depth_aware: bool = False,
        psf_size: Size2d = 64,
        patch_padding: Size2d = 0,
        linear_conv: bool = False,
        patch_merging: utils.PatchMerging = 'slope',
    ):
        super().__init__(pixel_num, pixel_size, image_distance)
        if wl is None:
            wl = base.fraunhofer_line('d', 'He')  # self.wl is assumed to never be None

        self.register_buffer('_b_wl', None, False)
        self.register_buffer('_b_depth', None, False)
        self.register_buffer('_b_depth_min', None, False)
        self.register_buffer('_b_depth_max', None, False)

        self.wl = wl  # property setter
        self.depth = depth  # property setter
        #: See :class:`RenderingOptics`.
        self.fov_segments: FovSeg | tuple[int, int] = cast(FovSeg | tuple[int, int], fov_segments)
        self.depth_aware: bool = depth_aware  #: See :class:`RenderingOptics`.
        self.psf_size: tuple[int, int] = psf_size  #: See :class:`RenderingOptics`.
        self.patch_padding: tuple[int, int] = patch_padding  #: See :class:`RenderingOptics`.
        self.linear_conv: bool = linear_conv  #: See :class:`RenderingOptics`.
        self.patch_merging: utils.PatchMerging = patch_merging  #: See :class:`RenderingOptics`.

    def __setattr__(self, name, value):
        normalizer = getattr(self, '_normalize_' + name, None)
        if normalizer is not None:
            value = normalizer(value)
        return super().__setattr__(name, value)

    def pick(self, name: str, value=None) -> Any:
        """
        Determine the value of an :doc:`external parameter </content/guide/exparam>`
        ``name``. The return value is determined by the eponymous attribute of ``self``
        if ``value`` is ``None``, or ``value`` otherwise.

        :param str name: Name of the external parameter.
        :param Any value: Candidate of the external parameter.
            Default: the eponymous attribute of ``self``.
        :return: Value of the external parameter.
        :rtype: Any
        """
        picker = getattr(self, '_pick_' + name, None)
        if picker is not None:
            return picker(value)

        # This is needed even when value is not None to ensure name represents a valid config item
        attr = getattr(self, name, ...)
        if attr is ...:
            raise ValueError(f'Unknown external parameter for {self.__class__.__name__}: {name}')
        if value is None:
            return attr

        normalizer = getattr(self, '_normalize_' + name, None)
        return value if normalizer is None else normalizer(value)

    @abc.abstractmethod
    def psf(self, origins: Ts, size: Size2d = None, wl: Vector = None, **kwargs) -> Ts:
        r"""
        Returns PSF of points whose coordinates
        in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        are given by ``points``.

        The coordinate direction of returned PSF is defined as follows.
        Horizontal and vertical directions represent x- and y-axis, respectively.
        x is positive in left side and y is positive in upper side.
        In 3D space, the directions of x- and y-axis are identical to that of
        camera's coordinate system. In this way, returned PSF can be convolved with
        a clear image directly to produce a blurred image.

        :param Tensor origins: Source points of which to evaluate PSF. A tensor with shape
            ``(..., 3)`` where the last dimension indicates coordinates of points in camera's
            coordinate system. The coordinates comply with :ref:`guide_imodel_ccs_inf`.
        :param size: Numbers of pixels of PSF in vertical and horizontal directions.
        :type size: int or tuple[int, int]
        :param wl: Wavelengths to evaluate PSF on. Default: wavelength of ``self``.
        :type wl: float, Sequence[float] or Tensor
        :return: PSF conditioned on ``origins``. A tensor with shape :math:`(\cdots, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        pass

    def forward(self, scene: _sc.Scene, fov_segments: FovSeg | Size2d = None, **kwargs) -> Ts:
        r"""
        Implementation of :doc:`imaging simulation </content/guide/overview>`.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param fov_segments: See :class:`~.ConfigItems`.
        :param kwargs: Additional keyword arguments.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        self._check_scene(scene)
        scene: _sc.ImageScene
        fov_segments = self.pick('fov_segments', fov_segments)
        if fov_segments == 'paraxial':
            return self.conv_render(scene, **kwargs)
        elif fov_segments == 'pointwise':
            return self.pointwise_render(scene, **kwargs)
        else:  # tuple[int, int]
            return self.patchwise_render(scene, **kwargs)

    def pointwise_render(
        self,
        scene: _sc.ImageScene,
        wl: Vector = None,
        depth: Vector | tuple[Ts, Ts] = None,
        psf_size: Size2d = None,
        **kwargs,
    ) -> Ts:
        raise NotImplementedError()

    def patchwise_render(
        self,
        scene: _sc.ImageScene,
        segments: Size2d = None,
        wl: Vector = None,
        depth: Vector | tuple[Ts, Ts] = None,
        psf_size: Size2d = None,
        pad: Size2d = None,
        linear_conv: bool = None,
        merging: utils.PatchMerging = None,
        **kwargs
    ) -> Ts:
        segments = self.pick('fov_segments', segments)
        wl = self.pick('wl', wl)
        depth = self.pick('depth', depth)
        pad = self.pick('patch_padding', pad)
        linear_conv = self.pick('linear_conv', linear_conv)
        merging = self.pick('patch_merging', merging)
        psf_size = self.pick('psf_size', psf_size)

        self._check_scene(scene)
        if not isinstance(segments, tuple) or not len(segments) == 2:
            raise ValueError(f'segments must be a pair of ints, got {type(segments)}')
        if scene.intrinsic is not None:
            raise NotImplementedError()
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, '
                             f'got {scene.n_wl}')
        if scene.depth_aware:
            warnings.warn(f'Depth-aware rendering is not supported currently for {self.__class__.__name__}')

        scene = scene.batch()
        rm = self.reference
        n_b, n_wl, n_h, n_w = scene.image.shape

        if not (torch.is_tensor(depth) and depth.ndim == 0):
            depth = torch.stack([self.random_depth(depth) for _ in range(n_b)])  # B(1)
        tanfov_y, tanfov_x = utils.sym_grid(
            2, segments, (2 / segments[0], 2 / segments[1]), True, device=self.device, dtype=self.dtype
        )
        tanfov_x, tanfov_y = tanfov_x * math.tan(rm.fov_half_x), tanfov_y * math.tan(rm.fov_half_y)
        tanfov_x, tanfov_y = torch.broadcast_tensors(tanfov_x, tanfov_y)  # N_x x N_y
        # B(1) x N_x x N_y
        obj_points = self.fovd2obj(torch.stack([tanfov_x, tanfov_y], -1), depth.view(-1, 1, 1))

        psf = self.psf(obj_points, psf_size, wl, **kwargs)  # B(1) x N_x x N_y x N_wl x H x W

        psf = psf.permute(0, 3, 1, 2, 4, 5)  # B(1) x N_wl x N_x x N_y x H x W
        image_blur = formation.space_variant(scene.image, psf, pad, linear_conv, merging)  # B x N_wl x H x W
        return image_blur

    def conv_render(self, scene: _sc.ImageScene, **kwargs) -> Ts:
        raise NotImplementedError()

    def seq_depth(
        self,
        depth: Vector | tuple[Ts, Ts] = None,
        sampling_curve: Callable[[Ts], Ts] = None,
        n: int = None
    ) -> Ts:
        r"""
        Returns a 1D tensor representing a series of depths, inferred from ``depth``:

        - If ``depth`` is a pair of 0D tensor, i.e. lower and upper bound of depth,
          returns a tensor with length ``n`` whose values are

          .. math::
            \text{depth}=\text{depth}_\min+(\text{depth}_\max-\text{depth}_\min)\times \Gamma(t).

          where :math:`t` is drawn uniformly from :math:`[0,1]`. An optional ``sampling_curve``
          (denoted by :math:`\Gamma`) can be given to control its values.
          By default, :math:`\Gamma` is constructed so that the inverse of depth is evenly spaced.
        - If a 0D tensor, returns it but as a 1D tensor with length one.
        - If a 1D tensor, returns it as-is.

        :param depth: See the eponymous argument of :class:`RenderingOptics.ConfigItems` for details.
            Default: :attr:`.depth`.
        :type depth: float, Sequence[float], Tensor or tuple[Tensor, Tensor]
        :param sampling_curve: Sampling curve :math:`\Gamma`,
            only makes sense in the first case above. Default: omitted.
        :type sampling_curve: Callable[[Tensor], Tensor]
        :param int n: Number of depths, only makes sense in the first case above. Default: omitted.
        :return: 1D tensor of depths.
        :rtype: Tensor
        """
        depth = self.pick('depth', depth)
        if isinstance(depth, tuple):  # [min, max]
            if n is None:
                raise TypeError(f'Number of depths must be given because only the lower and '
                                f'upper bound is specified for depth')
            d1, d2 = depth
            t = torch.linspace(0, 1, n, device=d2.device, dtype=d2.dtype)
            if sampling_curve is not None:
                t = sampling_curve(t)
            else:
                t = d1 * t / (d2 - (d2 - d1) * t)  # default sampling curve
            return d1 + (d2 - d1) * t
        elif sampling_curve is not None or n is not None:
            warnings.warn(f'sampling_curve and n are ignored because depth is already specified')
        return depth

    def random_depth(
        self,
        depth: Vector | tuple[Ts, Ts] = None,
        sampling_curve: Callable[[Ts], Ts] = None,
        probabilities: Ts = None
    ) -> Ts:
        r"""
        Randomly sample a depth and returns it, inferred from ``depth``:

        - If ``depth`` is a pair of 0D tensor, i.e. lower and upper bound of depth, returns

          .. math::
            \text{depth}=\text{depth}_\min+(\text{depth}_\max-\text{depth}_\min)\times \Gamma(t).

          where :math:`t` is drawn uniformly from :math:`[0,1]`. An optional ``sampling_curve``
          (denoted by :math:`\Gamma`) can be given to control its distribution.
          By default, :math:`\Gamma` is constructed so that the inverse of depth is evenly spaced.
        - If a 1D tensor, randomly draws a value from it. Corresponding probability
          distribution can be given by ``probabilities``.

        :param depth: See the eponymous argument of :class:`ConfigItems` for details.
            Default: :attr:`.depth`.
        :type depth: float, Sequence[float], Tensor or tuple[Tensor, Tensor]
        :param sampling_curve: Sampling curve :math:`\Gamma`,
            only makes sense in the first case above. Default: omitted.
        :type sampling_curve: Callable[[Tensor], Tensor]
        :param Tensor probabilities: A 1D tensor with same length as :attr:`.depth`,
            only makes sense in the third case above. Default: omitted.
        :return: A 0D tensor of randomly sampled depth.
        :rtype: Tensor
        """
        depth = self.pick('depth', depth)
        if isinstance(depth, tuple):  # randomly sampling from a depth range
            d1, d2 = depth
            t = torch.rand_like(d2)
            if sampling_curve is not None:
                t = sampling_curve(t)
            else:
                t = d1 * t / (d2 - (d2 - d1) * t)  # default sampling curve
            return d1 + (d2 - d1) * t
        else:  # randomly sampling from a set of depth values
            if probabilities is None:
                idx = torch.randint(depth.numel(), ()).item()
            else:
                if probabilities.ndim != 1:
                    raise ShapeError(f'probabilities must be a 1D tensor')
                idx = torch.multinomial(probabilities, 1).squeeze().item()
            return depth[idx]

    @property
    def depth(self) -> Ts | tuple[Ts, Ts]:
        """
        Depth values used when a scene has no depth information.

        :type: 0D Tensor or 1D Tensor or a pair of 0D Tensor
        """
        d = self._b_depth
        return (self._b_depth_min, self._b_depth_max) if d is None else d

    @depth.setter
    def depth(self, value: Ts | tuple[Ts, Ts]):  # already normalized in __setattr__
        if torch.is_tensor(value):
            self._b_depth, self._b_depth_min, self._b_depth_max = value, None, None
        else:
            self._b_depth, self._b_depth_min, self._b_depth_max = None, value[0], value[1]

    @property
    def wl(self) -> Ts:
        """Wavelength for rendering.\n\n:type: 1D Tensor"""
        return self._b_wl

    @wl.setter
    def wl(self, value: Ts):  # already normalized in __setattr__
        self._b_wl = value

    def _delegate(self) -> Ts:
        return self._b_wl

    def _check_scene(self, scene: _sc.Scene):
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'{self.__class__.__name__} only supports ImageScene at present')
        if scene.n_plr != 0:
            raise RuntimeError(f'{self.__class__.__name__} does not support polarization currently')

    def _normalize_depth(self, depth: Vector | tuple[Ts, Ts]) -> Ts | tuple[Ts, Ts]:
        if isinstance(depth, tuple) and len(depth) == 2 and all(torch.is_tensor(t) for t in depth):
            if depth[0].ndim != 0 or depth[1].ndim != 0:
                raise ShapeError(f'If a pair of tensor, both of them should be 0D, got {depth}')
        else:
            depth = vector(depth, dtype=self.dtype, device=self.device)
        return depth

    def _normalize_wl(self, wl: Vector):
        return vector(wl, dtype=self.dtype, device=self.device)

    @staticmethod
    def _normalize_fov_segments(fov_segments: FovSeg | Size2d) -> FovSeg | tuple[int, int]:
        if not isinstance(fov_segments, str):
            fov_segments = size2d(fov_segments)
        return fov_segments

    _normalize_psf_size = staticmethod(size2d)
    _normalize_patch_padding = staticmethod(size2d)


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

    def forward(self, scene: _sc.Scene) -> Ts:
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'{self.__class__.__name__} only supports ImageScene at present')
        if (si := scene.intrinsic) is not None:
            if not torch.allclose(si, self.intrinsic().broadcast_to(si.size(0), -1, -1)):
                raise NotImplementedError()
        if (scene.height, scene.width) != self.pixel_num:
            raise RuntimeError(f'Got {_sc.ImageScene.__name__} with shape {scene.height}x{scene.width}, '
                               f'but size of sensor is {self.pixel_num[0]}x{self.pixel_num[1]}')
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
        return self.image_distance

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

    @property
    def reference(self) -> 'Pinhole':
        return self
