import torch

from .. import utils, fourier, base
from ..base.typing import Ts, Size2d, size2d, cast

__all__ = [
    'depth_aware',
    'simple',
    'space_variant',
    'superpose',
]


def simple(obj: Ts, psf: Ts, pad: Size2d | str = 'linear') -> Ts:
    r"""
    Simplest image formation model. Blurred image is computed by a convolution between the
    object (sharp image) and PSF, implemented by FFT.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,H_p,W_p)`.
    :param pad: Padding width used to mitigate aliasing. See :func:`dnois.fourier.dconv2`
        for more details. Default: ``'linear'``.
    :type pad: int, tuple[int, int] or str
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor
    """
    if psf.size(-2) > obj.size(-2) or psf.size(-1) > obj.size(-1):
        raise base.ShapeError(f'Spatial dimension of PSF ({psf.shape[-2:]}) cannot '
                              f'be larger than that of object ({obj.shape[-2:]})')
    return fourier.dconv2(obj, psf, out='same', padding=pad)


def depth_aware(
    obj: Ts,
    mask: Ts,
    psf: Ts,
    pad: Size2d | str = 'linear',
    occlusion_aware: bool = False,
) -> Ts:
    r"""
    Depth-aware image formation model. The ``obj`` (sharp image) is first segmented into
    depth planes w.r.t. depth according to ``mask``. Then the portion in each plane is
    convolved with the PSF of corresponding depth. Final image (blurred image) is the
    superposition of all the planes.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    Smaller indices in :math:`D` dimension in ``mask`` and ``psf`` are expected to
    represent smaller depths.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor mask: A tensor of shape :math:`(\cdots,D,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,D,H_p,W_p)`. PSF should be normalized,
        see :func:`dnois.optics.norm_psf` for details.
    :param pad: Padding width used to mitigate aliasing. See :func:`dnois.fourier.dconv2`
        for more details. Default: ``'linear'``.
    :type pad: int, tuple[int, int] or str
    :param bool occlusion_aware: Whether to use the occlusion-aware image formation algorithm
        proposed in `Depth from Defocus with Learned Optics for Imaging and Occlusion-aware
        Depth Estimation - Ikoma et al. <https://ieeexplore.ieee.org/abstract/document/9466261>`__.
        Default: ``False``.
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor
    """
    if psf.size(-2) > obj.size(-2) or psf.size(-1) > obj.size(-1):
        raise base.ShapeError(f'Spatial dimension of PSF ({psf.shape[-2:]}) cannot '
                              f'be larger than that of object ({obj.shape[-2:]})')
    if mask.gt(1).any() or mask.lt(0).any():
        raise ValueError(f'Value of mask must lie in [0, 1]')

    slices = obj.unsqueeze(-3) * mask  # ... x D x H x W
    if occlusion_aware:
        accum = torch.flip(torch.cumsum(torch.flip(mask, (-3,)), -3), (-3,))
        blr_accum, blr_mask, blr_img = fourier.dconv_mult([accum, mask, slices], psf, (-2, -1), 'same', pad)
        blr_accum = blr_accum.clamp(min=1e-5)
        blr_mask = blr_mask / blr_accum

        acc_prod = torch.cumprod(1 - blr_mask, -3)
        acc_prod = torch.roll(acc_prod, 1, -3)
        acc_prod[..., 0, :, :] = 1

        blr_img = blr_img / blr_accum
        blurred = torch.sum(acc_prod * blr_img, dim=-3)
    else:
        blurred_slices = fourier.dconv2(slices, psf, out='same', padding=pad)  # ... x D x H x W
        blurred = blurred_slices.sum(dim=-3)  # ... x H x W
    return blurred


def superpose(obj: Ts, psf: Ts) -> Ts:
    r"""
    Point-wise image formation model. The image is the superposition of PSFs
    of all object points.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,H_o,W_o,H_p,W_p)`.
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor
    """
    raise NotImplementedError()


def space_variant(
    obj: Ts,
    psf: Ts,
    pad: Size2d = 0,
    linear_conv: bool = False,
    merge: utils.PatchMerging = 'avg',
) -> Ts:
    r"""
    Space-variant image formation model. The image plane is partitioned into non-overlapping
    patches. The PSF in each patch is assumed to be space-invariant and convolved with the
    corresponding patch to obtain the blurred patch. Number of patches is determined by the
    dimensionality of ``psf``. This is typically used to blur an image with FoV-dependent PSF.

    To mitigate the abrupt change between PSFs in different patches, each patch can be
    padded with pixels from neighboring patches where padding amount is specified by ``pad``.
    Adjacent patches overlap after padding and overlapping regions are merged.
    Note that merging method must be ``crop`` if ``linear_conv`` is ``False``.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    :param Tensor obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param Tensor psf: A tensor of shape :math:`(\cdots,N_h,N_w,H_p,W_p)`.
    :param pad: Padding width in vertical and horizontal directions. Default: 0.
    :type pad: int | tuple[int, int]
    :param bool linear_conv: Whether to use linear convolution in each patch. Default: False.
    :param str merge: Method to merge patches (see :func:`dnois.utils.merge_patches`). Default: ``avg``.
    :return: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :rtype: Tensor
    """
    pad = size2d(pad)
    patches = utils.partition_padded(obj, psf.shape[-4:-2], pad, 'replicate')
    # ... x N_h x N_w x H_p x W_p
    patches = torch.stack([torch.stack(cast(list[Ts], row), -3) for row in patches], -4)
    if psf.size(-2) > patches.size(-2) or psf.size(-1) > patches.size(-1):
        psf = utils.resize(psf, (min(psf.size(-2), patches.size(-2)), min(psf.size(-1), patches.size(-1))))

    # ... x N_h x N_w x H_p x W_p
    blurred = fourier.dconv2(psf, patches, out='same', padding='linear' if linear_conv else 'none')

    blurred = blurred.transpose(-4, 0).transpose(-3, 1)  # N_h x N_w x ... x H_p x W_p
    blurred = utils.merge_patches(blurred, (2 * pad[0], 2 * pad[1]), merge)
    blurred = utils.crop(blurred, pad)

    return blurred
