import warnings

import torch

from dnois import utils, fourier
from dnois.base.typing import Ts, Size2d, Literal, size2d, cast

__all__ = [
    # 'depth_aware',
    # 'simple',
    'space_variant',
]


def _depth_aware_linear(obj: Ts, mask: Ts, psf: Ts, pad: Size2d):
    planes = obj.unsqueeze(-3) * mask  # ...xDxHxW
    img = fourier.conv2(psf, planes, pad=pad)
    img = torch.sum(img, dim=-3)
    return img


def _depth_aware_occlusion(obj: Ts, mask: Ts, psf: Ts, pad: Size2d):
    """
    `Depth from Defocus with Learned Optics for Imaging and Occlusion-aware
    Depth Estimation - Ikoma et al.
    <https://ieeexplore.ieee.org/abstract/document/9466261>`__ TODO

    :param obj:
    :param mask:
    :param psf:
    :param pad:
    :return:
    """
    if obj.is_complex() or psf.is_complex():
        raise NotImplementedError()

    planes = obj.unsqueeze(-3) * mask
    accum = torch.flip(torch.cumsum(torch.flip(mask, (-3,)), -3), (-3,))
    blr_accum, blr_mask, img = fourier.conv2_mult(psf, [accum, mask, planes], pad=pad)
    blr_accum = blr_accum + blr_accum.max() * 1e-4  # avoid division by 0
    blr_mask = blr_mask / blr_accum

    acc_prod = torch.cumprod(1 - blr_mask, -3)
    acc_prod = torch.roll(acc_prod, 1, -3)
    acc_prod[..., 0, :, :] = 1

    img = img / blr_accum
    return torch.sum(acc_prod * img, dim=-3)


def simple(obj: Ts, psf: Ts, output_size: Size2d = None, pad: Size2d = 0) -> Ts:
    r"""
    Simplest image formation model. Blurred image is computed by a convolution between the
    object (sharp image) and PSF, implemented by FFT.

    To mitigate aliasing caused by FFT-based convolution, an ``output_size`` smaller than
    the spatial size of object can be specified to crop out the edges with alias artifact.
    Alternatively ``obj`` can be zero-padded with width ``pad``.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    :param obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param psf: A tensor of shape :math:`(\cdots,H_p,W_p)`.
    :param output_size: Final output size. Default: same as input size.
    :param pad: Padding width. Default: 0.
    :return: A tensor of shape :math:`(\cdots,H_i,W_i)`.
    """
    obj_size = obj.shape[-2:]
    if output_size is not None:
        output_size = size2d(output_size)
        if output_size[0] > obj_size[0] or output_size[1] > obj_size[1]:
            raise ValueError(f'output_size ({output_size}) cannot exceed object size ({obj_size}')
    if psf.size(-2) > obj_size[0] or psf.size(-1) > obj_size[1]:
        warnings.warn(f'The size of PSF ({psf.shape[-2:]}) is larger than that of '
                      f'object ({obj.shape[-2:]}) and thus is cropped.')
    # TODO: aliasing detection
    psf = utils.resize(psf, obj.shape[-2:])

    img = fourier.conv2(obj, psf, pad=pad)

    if output_size is not None:
        img = utils.resize(img, output_size)  # must be cropping
    return img


def depth_aware(
    obj: Ts,
    mask: Ts,
    psf: Ts,
    output_size: Size2d = None,
    pad: Size2d = 0,
    mode: Literal['linear', 'occlusion'] = 'linear',
) -> Ts:
    r"""
    Depth-aware image formation model. The ``obj`` (sharp image) is first segmented into
    depth planes w.r.t. depth according to ``mask``. Then the portion in each plane is
    convolved with the PSF of corresponding depth. Final image (blurred image) is the
    superposition of all the planes.

    To mitigate aliasing caused by FFT-based convolution, an ``output_size`` smaller than
    the spatial size of object can be specified to crop out the edges with alias artifact.
    Alternatively ``obj`` can be zero-padded with width ``pad``.

    Note that both ``obj`` and ``psf`` can be real representing incoherent imaging and
    complex representing coherent imaging. If just one of them is complex, the other will
    be cast as complex. The blurred image is real if and only if they are both real.

    Smaller indices in :math:`D` dimension in ``mask`` and ``psf`` are expected to
    represent smaller depths.

    :param obj: A tensor of shape :math:`(\cdots,H_o,W_o)`.
    :param mask: A tensor of shape :math:`(\cdots,D,H_o,W_o)`.
    :param psf: A tensor of shape :math:`(\cdots,D,H_p,W_p)`.
    :param output_size: Final output size. Default: same as input size.
    :param pad: Padding width. Default: 0.
    :param mode: TODO
    :return: A tensor of shape :math:`(\cdots,H_i,W_i)`.
    """
    obj_size = obj.shape[-2:]
    if output_size is not None:
        output_size = size2d(output_size)
        if output_size[0] > obj_size[0] or output_size[1] > obj_size[1]:
            raise ValueError(f'output_size ({output_size}) cannot exceed object size ({obj_size}')
    if psf.size(-2) > obj_size[0] or psf.size(-1) > obj_size[1]:
        warnings.warn(f'The size of PSF ({psf.shape[-2:]}) is larger than that of '
                      f'object ({obj.shape[-2:]}) and thus is cropped.')
    # TODO: aliasing detection
    psf = utils.resize(psf, obj.shape[-2:])

    if mode == 'linear':
        img = _depth_aware_linear(obj, mask, psf, pad)
    elif mode == 'occlusion':
        img = _depth_aware_occlusion(obj, mask, psf, pad)
    else:
        raise ValueError(f'Unknown mode for depth-aware imaging: {mode}.')

    if output_size is not None:
        img = utils.resize(img, output_size)  # must be cropping
    return img


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
    :return: A tensor of shape :math:`(\cdots,H_i,W_i)`.
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
