from dnois.base import ShapeError
from dnois.base.typing import Ts

__all__ = [
    'ImageScene',
    'Scene',
]


class Scene:  # not used at present, just a dummy base class
    pass


class ImageScene(Scene):
    """
    Class for scenes represented by its pinhole image, i.e. perspective projection
    to the image plane of a pinhole camera.

    Given the intrinsic parameters of a pinhole camera and corresponding depth map,
    the image can be projected backward to reconstruct a point cloud.
    """

    def __init__(
        self,
        image: Ts,  # (B x )(P x )C x H x W
        depth: Ts = None,  # (B x )H x W
        intrinsic: Ts = None,  # (B|1 x )3 x 3
        polarized: bool = False,
        batched: bool = True
    ):
        i_ndim = 3 + bool(polarized) + bool(batched)
        i_shape = f'({"B, " if batched else ""}{"P, " if polarized else ""}C, H, W)'
        if image.ndim != i_ndim:
            raise ShapeError(f'Image with shape {i_shape} expected, got {image.shape}')
        if depth is not None:
            d_ndim = 2 + bool(batched)
            d_shape = f'({"B, " if batched else ""}C, H, W)'
            if depth.ndim != d_ndim:
                raise ShapeError(f'Depth map with shape {d_shape} expected, got {depth.shape}')
            if image.shape[-2:] != depth.shape[-2:]:
                raise ShapeError(f'The spatial dimensions of image ({image.shape[-2:]}) '
                                 f'and depth map ({depth.shape[-2:]}) do not match')
            if batched and image.size(0) != depth.size(0):
                raise ShapeError(f'Batch sizes of image ({image.size(0)}) and depth map '
                                 f'({depth.size(0)}) are different')
        if intrinsic is not None:
            if intrinsic.shape[-2:] != (3, 3):
                raise ShapeError(f'The last two dimensions of intrinsic must be (3, 3)')
            if batched:
                if intrinsic.ndim != 3:
                    raise ShapeError(f'Intrinsic matrix of shape (B, 3, 3) expected, got{intrinsic.shape}')
                if intrinsic.size(0) not in (image.size(0), 1):
                    raise ShapeError(f'Batch size of intrinsic matrix must be equal to '
                                     f'that of the image ({image.size(0)}) or 1, got {intrinsic.shape}')
            elif intrinsic.ndim != 2:
                raise ShapeError(f'Intrinsic matrix of shape (3, 3) expected, got {intrinsic.shape}')

        self._image = image
        self._depth = depth
        self._intrinsic = intrinsic
        self._polarized = polarized
        self._batched = batched

    @property
    def image(self) -> Ts:
        return self._image

    @property
    def depth(self) -> Ts | None:
        return self._depth

    @property
    def intrinsic(self) -> Ts | None:
        return self._intrinsic

    @property
    def batch_size(self) -> int:
        return self._image.size(0) if self._batched else 0

    @property
    def n_plr(self) -> int:
        return self._image.size(-4) if self._polarized else 0

    @property
    def n_wl(self) -> int:
        return self._image.size(-3)

    @property
    def height(self) -> int:
        return self._image.size(-2)

    @property
    def width(self) -> int:
        return self._image.size(-1)

    @property
    def depth_aware(self) -> bool:
        return self._depth is not None
