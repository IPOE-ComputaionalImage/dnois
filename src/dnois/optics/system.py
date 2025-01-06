import abc
import math
import warnings

import torch
from torch import nn

from . import formation, _func
from .. import base, utils, depth as _d, scene as _sc, torch as _t
from ..base import ShapeError, typing
from ..base.typing import (
    Ts, Size2d, FovSeg, Vector, Callable, Any,
    size2d, vector, cast
)
from ..sensor import Sensor

__all__ = [
    'IdealOptics',
    'Optics',
    'Pinhole',
    'RenderingOptics',
    'StandardOptics',
]


class Optics(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, scene: _sc.Scene) -> Ts:
        pass


class StandardOptics(Optics, metaclass=abc.ABCMeta):
    """
    Base class of :ref:`standard optical system <guide_imodel_standard_optical_system>`.

    :param float image_distance: Focal length of the reference model, i.e. the image distance.
    :param Sensor sensor: TODO
    """
    _inherent = ['image_distance', 'sensor']

    def __init__(self, image_distance: float, sensor: Sensor = None):
        super().__init__()
        if image_distance is not None:
            utils.check.positive(image_distance, 'image_distance')
        #: Attached :class:`~dnois.sensor.Sensor` object.
        self.sensor: Sensor | None = sensor
        #: Focal length of the reference model, i.e. the image distance.
        self.image_distance: float | None = image_distance

    def __getattribute__(self, item: str):
        attr = super().__getattribute__(item)
        if item == 'sensor' and attr is None:
            raise RuntimeError(f'Trying to get sensor from a {type(self).__name__} object without sensor')
        return attr

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
        _t.check_2d_vector(tanfov, f'tanfov in {self.fovd2obj.__qualname__}')

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
        _t.check_3d_vector(point, f'point in {self.obj2fov.__qualname__}')

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
    def reference(self) -> 'Pinhole':
        """
        Returns the :ref:`reference model <guide_imodel_ref_model>` of  this object.

        :type: :class:`Pinhole`
        """
        return Pinhole(self.image_distance, self.sensor)


def _symmetric_patch(obj_points: Ts, x_symmetric: bool, y_symmetric: bool) -> Ts:
    if x_symmetric:
        obj_points = obj_points[:, :(obj_points.size(1) + 1) // 2]
    if y_symmetric:
        obj_points = obj_points[:, :, :(obj_points.size(2) + 1) // 2]
    return obj_points


def _stitch_symmetric(psf: Ts, h: int, w: int, x_symmetric: bool, y_symmetric: bool) -> Ts:
    if x_symmetric:
        x_copy = psf[:, :h // 2].flip(1)
        psf = torch.cat([psf, x_copy], 1)
    if y_symmetric:
        y_copy = psf[:, :, :w // 2].flip(2)
        psf = torch.cat([psf, y_copy], 2)
    return psf


class RenderingOptics(_t.TensorContainerMixIn, StandardOptics, base.AsJsonMixIn, metaclass=abc.ABCMeta):
    """
    Base class for optical systems with optical imaging behavior defined.
    See :doc:`/content/guide/optics/imodel` for details.

    If two object points symmetric w.r.t. x-axis (i.e. whose x coordinates are equal
    and y coordinates are opposite) are expected to produce PSFs symmetric w.r.t. x-axis,
    one can set ``x_symmetric`` to ``True`` to compute PSFs in one side only,
    which is more efficient than computing two symmetric PSFs. It is similar
    for ``y_symmetric``. Specifically, an axisymmetric system allows both as ``True``.

    See :class:`StandardOptics` for descriptions about more parameters.

    .. warning::

        TODO

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
        carries no depth information. Default: infinity.

        float or 0D tensor
            The value will always be used as depth.

        Sequence[float] or 1D tensor
            Randomly select a value from given value for each image.

        A pair of 0D tensors
            They are interpreted as minimum and maximum values
            for random sampling (see :py:meth:`~sample_depth`).
    :type depth: float, Sequence[float], Tensor or tuple[Tensor, Tensor]
    :param psf_size: Height and width of PSF (i.e. convolution kernel) used to simulate imaging.
        Default: ``(64, 64)``.
    :type psf_size: int or tuple[int, int]
    :param cropping: Widths in pixels for cropping after rendering to alleviate
        aliasing (caused by circular convolution) or dimming (caused by linear convolution)
        in edges. Default: 0.
    :param bool x_symmetric: Whether this system is symmetric w.r.t. x-axis.
        See descriptions above. Default: ``False``.
    :param bool y_symmetric: Whether this system is symmetric w.r.t. y-axis.
        See descriptions above. Default: ``False``.
    """
    _external = [
        'wl', 'fov_segments', 'depth', 'psf_size', 'x_symmetric', 'y_symmetric'
    ]

    def __init__(
        self,
        image_distance: float,
        sensor: Sensor = None,
        *,
        wl: Vector = None,
        fov_segments: FovSeg | Size2d = 'paraxial',
        depth: Vector | tuple[Ts, Ts] = float('inf'),
        psf_size: Size2d = 64,
        cropping: Size2d = 0,
        x_symmetric: bool = False,
        y_symmetric: bool = False,
    ):
        super().__init__(image_distance, sensor)
        if wl is None:
            wl = base.fraunhofer_line('d', 'He')  # self.wl is assumed to never be None

        self.register_buffer('_b_wl', None)
        self.register_buffer('_b_depth', None)
        self.register_buffer('_b_depth_min', None)
        self.register_buffer('_b_depth_max', None)

        self.wl = wl  # property setter
        self.depth = depth  # property setter
        #: Number of field-of-view segments when rendering images. See :class:`RenderingOptics`.
        self.fov_segments: FovSeg | tuple[int, int] = cast(FovSeg | tuple[int, int], fov_segments)
        #: Height and width of PSF (i.e. convolution kernel) used to simulate imaging.
        #: See :class:`RenderingOptics`.
        self.psf_size: tuple[int, int] = size2d(psf_size)
        self.cropping: tuple[int, int] = size2d(cropping)  #: See :class:`RenderingOptics`.
        self.x_symmetric: bool = x_symmetric  #: See :class:`RenderingOptics`.
        self.y_symmetric: bool = y_symmetric  #: See :class:`RenderingOptics`.

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
            Default: :attr:`.psf_size`.
        :type size: int or tuple[int, int]
        :param wl: Wavelengths to evaluate PSF on. Default: :attr:`.wl`.
        :type wl: float, Sequence[float] or Tensor
        :return: PSF conditioned on ``origins``. A tensor with shape ``(..., N_wl, H, W)``.
        :rtype: Tensor
        """
        pass

    def forward(self, scene: _sc.Scene, fov_segments: FovSeg | Size2d = None, **kwargs) -> Ts:
        r"""
        Implementation of :doc:`imaging simulation </content/guide/overview>`.
        This method will call either of three imaging methods:

        - If ``fov_segments`` is ``'paraxial'``, call :meth:`conv_render`;
        - If ``'pointwise'``, call :meth:`pointwise_render`;
        - Otherwise, ``fov_segments`` is a pair of integers, call :meth:`patchwise_render`.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param fov_segments: See :class:`RenderingOptics`. Default: :attr:`.fov_segments`.
        :param kwargs: Additional keyword arguments passed to the underlying imaging methods.
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
            return self.patchwise_render(scene, fov_segments, **kwargs)

    def pointwise_render(
        self,
        scene: _sc.Scene,
        wl: Vector = None,
        depth: Vector | tuple[Ts, Ts] = None,
        psf_size: Size2d = None,
        x_symmetric: bool = None,
        y_symmetric: bool = None,
        **kwargs,
    ) -> Ts:
        r"""
        Renders :ref:`imaged radiance field <guide_overview_irf>` in a point-wise manner,
        i.e. PSFs of all the pixels are computed and superposed.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param wl: See :class:`RenderingOptics`. Default: :attr:`.wl`.
        :param depth: See :class:`RenderingOptics`. Default: :attr:`.depth`.
        :param psf_size: See :class:`RenderingOptics`. Default: :attr:`.psf_size`.
        :param bool x_symmetric: See :class:`RenderingOptics`. Default: :attr:`.x_symmetric`.
        :param bool y_symmetric: See :class:`RenderingOptics`. Default: :attr:`.y_symmetric`.
        :param kwargs: Additional keyword arguments passed to :meth:`.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        wl = self.pick('wl', wl)
        depth = self.pick('depth', depth)
        psf_size = self.pick('psf_size', psf_size)
        x_symmetric = self.pick('x_symmetric', x_symmetric)
        y_symmetric = self.pick('y_symmetric', y_symmetric)

        self._check_scene(scene)
        scene: _sc.ImageScene
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')

        scene = scene.batch()
        _, _, n_h, n_w = scene.image.shape
        depth_map = self._make_depth_map(scene, depth)  # B|1 x H x W
        obj_points = self.points_grid((n_h, n_w), depth_map, True)  # B|1 x H x W x 3
        if not scene.depth_aware:
            obj_points = _symmetric_patch(obj_points, x_symmetric, y_symmetric)

        psf = self.psf(obj_points, psf_size, wl, **kwargs)  # B|1 x H x W x N_wl x H_P x W_P
        if not scene.depth_aware:
            psf = _stitch_symmetric(psf, n_h, n_w, x_symmetric, y_symmetric)
        psf = psf.permute(0, 3, 1, 2, 4, 5)  # B|1 x N_wl x H x W x H_P x W_P

        image = formation.superpose(scene.image, psf)  # B x N_wl x H x W

        image = self.crop(image)
        return image

    def patchwise_render(
        self,
        scene: _sc.Scene,
        pad: Size2d = 0,
        linear_conv: bool = True,
        merging: utils.PatchMerging = 'slope',
        segments: Size2d = None,
        wl: Vector = None,
        depth: Vector | tuple[Ts, Ts] = None,
        psf_size: Size2d = None,
        x_symmetric: bool = None,
        y_symmetric: bool = None,
        **kwargs
    ) -> Ts:
        r"""
        Renders :ref:`imaged radiance field <guide_overview_irf>` in a patch-wise manner.
        In other words, the image plane is partitioned into non-overlapping
        patches and PSF is assumed to be space-invariant in each patch, but varies
        from patch to patch.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param pad: Padding amount for each patch. See :func:`~dnois.optics.space_variant`
            for more details. Default: ``(0, 0)``.
        :type pad: int or tuple[int, int]
        :param bool linear_conv: Whether to compute linear convolution rather than
            circular convolution when computing blurred image. Default: ``True``.
        :param str merging: Merging method to use for patch-wise (spatially variant) imaging.
            See :func:`~dnois.optics.space_variant` for more details. Default: ``'slope'``.
        :param segments: See :class:`RenderingOptics`. Default: :attr:`.fov_segments`.
        :param wl: See :class:`RenderingOptics`. Default: :attr:`.wl`.
        :param depth: See :class:`RenderingOptics`. Default: :attr:`.depth`.
        :param psf_size: See :class:`RenderingOptics`. Default: :attr:`.psf_size`.
        :param bool x_symmetric: See :class:`RenderingOptics`. Default: :attr:`.x_symmetric`.
        :param bool y_symmetric: See :class:`RenderingOptics`. Default: :attr:`.y_symmetric`.
        :param kwargs: Additional keyword arguments passed to :meth:`.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        segments = self.pick('fov_segments', segments)
        wl = self.pick('wl', wl)
        depth = self.pick('depth', depth)
        psf_size = self.pick('psf_size', psf_size)
        x_symmetric = self.pick('x_symmetric', x_symmetric)
        y_symmetric = self.pick('y_symmetric', y_symmetric)

        pad = size2d(pad)

        self._check_scene(scene)
        scene: _sc.ImageScene
        if not isinstance(segments, tuple) or not len(segments) == 2:
            raise ValueError(f'segments must be a pair of ints, got {type(segments)}')
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')
        if scene.depth_aware:
            warnings.warn(f'Depth-aware rendering is not supported currently '
                          f'for {self.patchwise_render.__qualname__}')

        scene = scene.batch()
        n_b, n_wl, n_h, n_w = scene.image.shape
        if not (torch.is_tensor(depth) and depth.numel() == 1):
            depth = torch.stack([self.random_depth(depth) for _ in range(n_b)])  # B(1)
        # B(1) x N_y x N_x x 3
        obj_points = self.points_grid(cast(tuple[int, int], segments), depth.flatten())
        obj_points = _symmetric_patch(obj_points, x_symmetric, y_symmetric)

        psf = self.psf(obj_points, psf_size, wl, **kwargs)  # B(1) x N_y x N_x x N_wl x H x W
        psf = _stitch_symmetric(psf, segments[0], segments[1], x_symmetric, y_symmetric)

        psf = psf.permute(0, 3, 1, 2, 4, 5)  # B(1) x N_wl x N_y x N_x x H x W
        image_blur = formation.space_variant(scene.image, psf, pad, linear_conv, merging)  # B x N_wl x H x W

        image_blur = self.crop(image_blur)
        return image_blur

    def conv_render(
        self,
        scene: _sc.ImageScene,
        wl: Vector = None,
        depth: Vector | tuple[Ts, Ts] = None,
        psf_size: Size2d = None,
        pad: Size2d | str = 'linear',
        occlusion_aware: bool = False,
        depth_quantization_level: int = 16,
        **kwargs
    ) -> Ts:
        r"""
        Renders :ref:`imaged radiance field <guide_overview_irf>` via vanilla convolution.
        It means that PSF is considered as space-invariant.

        :param scene: The scene to be imaged.
        :type scene: :class:`~dnois.scene.Scene`
        :param wl: See :class:`RenderingOptics`. Default: :attr:`.wl`.
        :param depth: See :class:`RenderingOptics`. Default: :attr:`.depth`.
        :param psf_size: See :class:`RenderingOptics`. Default: :attr:`.psf_size`.
        :param pad: Padding width used to mitigate aliasing. See :func:`dnois.fourier.dconv2`
            for more details. Default: ``'linear'``.
        :type pad: int, tuple[int, int] or str
        :param bool occlusion_aware: Whether to use occlusion-aware image formation algorithm.
            See :func:`dnois.optics.depth_aware` for more details.
            This matters only when ``scene`` carries depth map. Default: ``False``.
        :param int depth_quantization_level: Number of quantization levels for depth-aware imaging.
            This matters only when ``scene`` carries depth map. Default: ``16``.
        :param kwargs: Additional keyword arguments passed to :meth:`.psf`.
        :return: Computed :ref:`imaged radiance field <guide_overview_irf>`.
            A tensor of shape :math:`(B, N_\lambda, H, W)`.
        :rtype: Tensor
        """
        wl = self.pick('wl', wl)
        depth = self.pick('depth', depth)
        psf_size = self.pick('psf_size', psf_size)

        self._check_scene(scene)
        scene: _sc.ImageScene
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')

        scene = scene.batch()
        if scene.depth_aware:
            if not isinstance(depth, tuple):
                raise ValueError(f'depth must be a pair of 0D tensors for depth-aware imaging')
            min_d, max_d = depth
            q_ips = torch.arange(depth_quantization_level, device=self.device)
            q_ips = (q_ips + 0.5) / depth_quantization_level
            q_depth = _d.ips2depth(q_ips, min_d, max_d)  # D
            obj_points = self.fovd2obj([(0., 0.)], q_depth)  # D x 3
            psf = self.psf(obj_points, psf_size, wl, **kwargs)  # D x N_wl x H_P x W_P
            psf = psf.transpose(0, 1)  # N_wl x D x H_P x W_P

            masks = _d.quantize_depth_map(scene.depth, min_d, max_d, depth_quantization_level)
            masks = torch.stack(masks, 1).unsqueeze(1)  # B x 1 x D x H x W
            image = formation.depth_aware(scene.image, masks, psf, pad, occlusion_aware)  # B x N_wl x H x W
        else:
            if not (torch.is_tensor(depth) and depth.numel() == 1):
                depth = torch.stack([self.random_depth(depth) for _ in range(scene.batch_size)])  # B(1)
            obj_points = self.fovd2obj([(0., 0.)], depth)  # B(1) x 3

            psf = self.psf(obj_points, psf_size, wl, **kwargs)  # B(1) x N_wl x H_P x W_P

            image = formation.simple(scene.image, psf, pad)  # B x N_wl x H x W

        image = self.crop(image)
        return image

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

        :param depth: See the eponymous argument of :class:`RenderingOptics` for details.
            Default: :attr:`.depth`.
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

        :param depth: See the eponymous argument of :class:`RenderingOptics` for details.
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

    def points_grid(self, segments: Size2d, depth: float | Ts, depth_as_map: bool = False) -> Ts:
        """
        Creates some points in object space, each of which is mapped to the center of
        one of non-overlapping patches on the image plane by perspective projection.

        :param segments: Number of patches in vertical (``N_y``) and horizontal (``N_x``) directions.
        :type segments: int or tuple[int, int]
        :param depth: Depth of resulted points. A ``float`` or a tensor of any shape ``(...)``.
            if ``depth_as_map`` is ``False``. Otherwise, must be a tensor of shape ``(..., N_y, N_x)``.
        :type depth: float or Tensor
        :param bool depth_as_map: See description of ``depth``.
        :return: A tensor of shape ``(..., N_y, N_x, 3)`` representing the coordinates of points in
            :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.
        :rtype: Tensor
        """
        segments = size2d(segments)
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        if depth_as_map and depth.shape[-2:] != segments:
            raise ShapeError(f'The last two dimensions of depth map ({depth.shape[-2:]} '
                             f'must match the number of segments ({segments})')

        rm = self.reference
        tanfov_y, tanfov_x = utils.grid(
            segments, (2 / segments[0], 2 / segments[1]), symmetric=True, device=self.device, dtype=self.dtype
        )
        tanfov_x, tanfov_y = tanfov_x * math.tan(rm.fov_half_x), tanfov_y * math.tan(rm.fov_half_y)
        tanfov_x, tanfov_y = torch.broadcast_tensors(tanfov_x, tanfov_y)  # N_x x N_y
        if not depth_as_map:
            depth = depth[..., None, None]
        return self.tanfovd2obj(torch.stack([tanfov_x, tanfov_y], -1), depth)  # ... x N_x x N_y x 3

    def crop(self, image: Ts) -> Ts:
        return utils.crop(image, self.cropping)

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = {k: self._attr2dictitem(k, keep_tensor) for k in self._inherent}
        d.update({k: self._attr2dictitem(k, keep_tensor) for k in self._external})
        return d

    @property
    def depth(self) -> Ts | tuple[Ts, Ts]:
        """
        Depth values used when a scene has no depth information.
        A 0D Tensor, 1D Tensor or a pair of 0D Tensor. See :class:`RenderingOptics`.

        :type: Tensor or tuple[Tensor, Tensor]
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
        """Wavelength for rendering. A 1D tensor.\n\n:type: Tensor"""
        return self._b_wl

    @wl.setter
    def wl(self, value: Ts):  # already normalized in __setattr__
        self._b_wl = value

    def _check_scene(self, scene: _sc.Scene):
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'{self.__class__.__name__} only supports ImageScene at present')
        if scene.n_plr != 0:
            raise RuntimeError(f'{self.__class__.__name__} does not support polarization currently')
        if scene.intrinsic is not None:
            raise NotImplementedError()

    def _make_depth_map(self, scene: _sc.ImageScene, depth: Vector | tuple[Ts, Ts]) -> Ts:
        scene = scene.batch()
        n_b, _, n_h, n_w = scene.image.shape

        if scene.depth_aware:
            depth_map = scene.depth
        else:
            if not (torch.is_tensor(depth) and depth.numel() == 1):
                depth = torch.stack([self.random_depth(depth) for _ in range(n_b)])
            depth_map = depth.reshape(-1, 1, 1).expand(-1, n_h, n_w)
        return depth_map  # B|1 x H x W

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

    def _todict_depth(self, keep_tensor: bool = True):
        depth = self.depth
        if keep_tensor:
            return depth
        if torch.is_tensor(depth):
            return depth.tolist()
        return {'min': depth[0].tolist(), 'max': depth[1].tolist()}

    def _todict_sensor(self, _: bool = True):
        return None if self.sensor is None else {
            'pixel_size': self.sensor.pixel_size,
            'pixel_num': self.sensor.pixel_num,
        }

    @classmethod
    def _pre_from_dict(cls, d: dict):
        d = super()._pre_from_dict(d)
        depth = d['depth']
        if isinstance(depth, dict):
            d['depth'] = (torch.tensor(depth['min']), torch.tensor(depth['max']))
        if d['sensor'] is not None:
            d['sensor'] = Sensor(**d['sensor'])
        return d


class IdealOptics(RenderingOptics):
    """
    Ideal optics model.

    See :class:`RenderingOptics` for descriptions of more parameters.

    :param float pupil_diameter: Diameter of the light-passing pupil on principal planes.
    :param float fl1: Focal length in object space.
    :param float fl2: Focal length in image space.
    :param kwargs: Additional keyword arguments passed to :class:`RenderingOptics`.
    """
    _inherent = RenderingOptics._inherent + ['pupil_diameter', 'fl1', 'fl2']

    def __init__(
        self,
        image_distance: float,
        pupil_diameter: float,
        fl1: float,
        fl2: float = None,
        sensor: Sensor = None,
        **kwargs,
    ):
        if fl2 is None:
            fl2 = fl1

        super().__init__(image_distance, sensor, **kwargs)
        self.pupil_diameter: float = pupil_diameter  #: Diameter of the light-passing pupil on principal planes.
        self.fl1: float = fl1  #: Focal length in object space.
        self.fl2: float = fl2  #: Focal length in image space.

    def psf(self, origins: Ts, size: Size2d = None, wl: Vector = None, **kwargs) -> Ts:
        size = self.pick('psf_size', size)

        if len(kwargs) != 0:
            raise RuntimeError(f'Unknown keyword arguments for {self.__class__.__name__}: '
                               f'{", ".join(kwargs.keys())}')

        obj_d = origins[..., 2]  # ...
        img_d = _func.imgd(obj_d, self.fl1, self.fl2)  # ...
        coc = _func.circle_of_confusion(self.pupil_diameter, self.sensor_distance, img_d)  # ...
        radius = coc[..., None, None] / 2  # ... x 1 x 1

        psf = torch.zeros(*coc.shape, *size, device=origins.device, dtype=origins.dtype)  # ... x H x W
        y, x = utils.grid(size, self.sensor.pixel_size, device=origins.device, dtype=origins.dtype)
        r2 = x.square() + y.square()  # H x W
        psf[r2 <= radius.square()] = 1
        psf = _func.norm_psf(psf)
        psf = psf.unsqueeze(-3)  # ... x 1 x H x W
        return psf

    def pointwise_render(self, *args, **kwargs):
        warnings.warn(f'PSF of {IdealOptics} is always space-invariant so point-wise rendering '
                      f'is virtually equivalent to vanilla convolution but far more inefficient. '
                      f'Consider using {self.conv_render.__name__} instead.')
        return super().pointwise_render(*args, **kwargs)

    def patchwise_render(self, *args, **kwargs):
        warnings.warn(f'PSF of {IdealOptics} is always space-invariant so patch-wise rendering '
                      f'is virtually equivalent to vanilla convolution but far more inefficient. '
                      f'Consider using {self.conv_render.__name__} instead.')
        return super().patchwise_render(*args, **kwargs)

    @property
    def sensor_distance(self) -> float:
        """
        Distance between image principal plane and image plane (sensor plane).

        :rtype:
        """
        if self.fl1 == self.fl2:
            return self.image_distance
        else:
            return self.image_distance * self.fl2 / self.fl1


class Pinhole(StandardOptics):
    def __init__(self, focal_length: float, sensor: Sensor):
        super().__init__(focal_length, sensor)

    def forward(self, scene: _sc.Scene) -> Ts:
        if not isinstance(scene, _sc.ImageScene):
            raise RuntimeError(f'{self.__class__.__name__} only supports ImageScene at present')
        if (si := scene.intrinsic) is not None:
            if not torch.allclose(si, self.intrinsic().broadcast_to(si.size(0), -1, -1)):
                raise NotImplementedError()
        pn = self.sensor.pixel_num
        if (scene.height, scene.width) != pn:
            raise RuntimeError(f'Got {_sc.ImageScene.__name__} with shape {scene.height}x{scene.width}, '
                               f'but size of sensor is {pn[0]}x{pn[1]}')
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
        i[0, 0] = self.focal_length / self.sensor.pixel_size[1]
        i[1, 1] = self.focal_length / self.sensor.pixel_size[0]
        i[2, 2] = 1
        i[0, 2] = self.sensor.w / 2
        i[1, 2] = self.sensor.h / 2
        return i

    @property
    def focal_length(self) -> float:
        return self.image_distance

    @property
    def fov_full(self) -> float:
        return self.fov_half * 2

    @property
    def fov_half(self) -> float:
        half_h, half_w = self.sensor.h / 2, self.sensor.w / 2
        tan = math.sqrt(half_h * half_h + half_w * half_w) / self.focal_length
        return math.atan(tan)

    @property
    def fov_half_x(self) -> float:
        return math.atan(self.sensor.w / (2 * self.focal_length))

    @property
    def fov_half_y(self) -> float:
        return math.atan(self.sensor.h / (2 * self.focal_length))

    @property
    def reference(self) -> 'Pinhole':
        return self
