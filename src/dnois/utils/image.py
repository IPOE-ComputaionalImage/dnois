import torch
import torch.nn.functional as functional

from dnois.base.typing import Size2d, Union, Literal, Ts, size2d

__all__ = [
    'crop',
    'merge_patches',
    'pad',
    'partition',
    'partition_padded',
    'resize',

    'PatchMerging',
]

PatchMerging = Literal['avg', 'crop', 'slope']


def _average_merge(img, patch_n, patch_sz, overlap, patches):
    for i in range(patch_n[0]):
        upper = i * (patch_sz[0] - overlap[0])
        for j in range(patch_n[1]):
            left = j * (patch_sz[1] - overlap[1])
            img[..., upper:upper + patch_sz[0], left:left + patch_sz[1]] += patches[i][j]

    for i in range(1, patch_n[0]):
        upper = i * (patch_sz[0] - overlap[0])
        img[..., upper:upper + overlap[0], :] /= 2
    for j in range(1, patch_n[1]):
        left = j * (patch_sz[1] - overlap[1])
        img[..., left:left + overlap[1]] /= 2

    return img


def _crop_merge(img: Ts, patch_n, patch_sz, overlap, patches):
    uppers = [i * (patch_sz[0] - overlap[0]) + overlap[0] // 2 for i in range(patch_n[0])]
    lefts = [i * (patch_sz[1] - overlap[1]) + overlap[1] // 2 for i in range(patch_n[1])]
    uppers[0] = 0
    lefts[0] = 0
    uppers.append(img.size(-2))
    lefts.append(img.size(-1))
    for i in range(patch_n[0]):
        crop_up = 0 if i == 0 else overlap[0] // 2
        for j in range(patch_n[1]):
            crop_left = 0 if j == 0 else overlap[1] // 2
            patch = patches[i][j].narrow(-2, crop_up, uppers[i + 1] - uppers[i])
            patch = patch.narrow(-1, crop_left, lefts[j + 1] - lefts[j])
            img[..., uppers[i]:uppers[i + 1], lefts[j]:lefts[j + 1]] += patch
    return img


def _slope_merge(img, patch_n, patch_sz, overlap, patches):
    shape = patches[0][0].shape
    i = torch.arange(shape[-2]).reshape(-1, 1) / (overlap[0] - 1)
    j = torch.arange(shape[-1]).reshape(1, -1) / (overlap[1] - 1)
    i, j = torch.broadcast_tensors(i, j)
    upper_m, lower_m, left_m, right_m = [
        torch.clamp(mask, 0, 1).to(patches[0][0]) for mask in (i, i.max() - i, j, j.max() - j)
    ]

    for i in range(patch_n[0]):
        upper = i * (patch_sz[0] - overlap[0])
        for j in range(patch_n[1]):
            left = j * (patch_sz[1] - overlap[1])

            p = patches[i][j]
            if i != 0:
                p = p * upper_m
            if i != patch_n[0] - 1:
                p = p * lower_m
            if j != 0:
                p = p * left_m
            if j != patch_n[1] - 1:
                p = p * right_m
            img[..., upper:upper + patch_sz[0], left:left + patch_sz[1]] += p

    return img


def partition(
    image: Ts,
    n_patches: Size2d,
    overlap: Size2d = 0,
    sequential: bool = False
) -> list[Ts] | list[list[Ts]]:
    """
    Partition an image into patches, either overlapping or not.

    Neighboring patches have an overlapping boundary of width ``overlap``. Therefore, the
    size of single patch is: ``PH = (H + overlap[0]) / n_patches[0]``, ``PW = (W +
    overlap[1]) / n_patches[1]``.

    :param Tensor image: One or more images of shape ... x H x W.
    :param n_patches: Number of patches in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :param overlap: Overlapping width in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :param bool sequential: Whether to arrange all patches into one dimension.
    :return: Image patches of shape ``patches[0]`` x ``patches[1]`` x ... x PH x PW
        or (``patches[0]`` x ``patches[1]``) x ... x PH x PW, depending on ``sequential``
        wherein PH and PW are height and width of each patch, respectively.
    :rtype: list[Tensor] if ``sequential`` is ``True``, list[list[Tensor]] otherwise.
    """
    n_patches = size2d(n_patches)
    overlap = size2d(overlap)
    img_sz = image.shape[-2:]
    if any((img_sz[i] - overlap[i]) % n_patches[i] != 0 for i in (0, 1)):
        raise ValueError(f'patches must be divisible by the dimension of image minus overlap'
                         f'in each direction')
    non_overlap_sz = [(img_sz[i] - overlap[i]) // n_patches[i] for i in (0, 1)]
    patch_sz = [non_overlap_sz[i] + overlap[i] for i in (0, 1)]

    img_patches = [... for _ in range(n_patches[0] * n_patches[1])]
    for i in range(n_patches[0]):
        upper = non_overlap_sz[0] * i
        for j in range(n_patches[1]):
            left = non_overlap_sz[1] * j
            patch = image[..., upper:upper + patch_sz[0], left:left + patch_sz[1]]
            img_patches[i * n_patches[1] + j] = patch

    if sequential:
        return img_patches
    else:
        return [
            [img_patches[i * n_patches[1] + j] for j in range(n_patches[1])]
            for i in range(n_patches[0])
        ]


def partition_padded(
    image: Ts,
    n_patches: Size2d,
    padding: Size2d = 0,
    mode: str = 'constant',
    value: float | int = 0,
    sequential: bool = False,
) -> list[Ts] | list[list[Ts]]:
    """
    Partition an image into patches, each of which is padded with pixels from neighbouring patches.
    Paddings of marginal patches depend on the parameters ``mode`` and ``value``.

    :param Tensor image: One or more images of shape ... x H x W.
    :param n_patches: Number of patches in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :param padding: Padding width in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :param str mode: See :func:``torch.nn.functional.pad``.
    :param value: See :func:``torch.nn.functional.pad``.
    :type: int | float
    :param bool sequential: Whether to arrange all patches into one dimension.
    :return: Image patches of shape patches[0] x patches[1] x ... x PH x PW
        or (patches[0] x patches[1]) x ... x PH x PW, depending on ``sequential``
        wherein PH and PW are height and width of each patch, respectively.
    :rtype: list[Tensor] if ``sequential`` is ``True``, list[list[Tensor]] otherwise.
    """
    n_patches = size2d(n_patches)
    padding = size2d(padding)
    image = functional.pad(image, (padding[1], padding[1], padding[0], padding[0]), mode, value)
    img_patches = partition(image, n_patches, (2 * padding[0], 2 * padding[1]), sequential)
    return img_patches


def merge_patches(
    patches: Union[Ts, list[list[Ts]]],
    overlap: Size2d = 0,
    merge_method: PatchMerging = 'avg'
) -> Ts:
    """
    Merge a set of patches into an image.

    :param patches: Either a tensor of shape PR x PC x ... x PH x PW or a 2d list composed of
        tensors of shape ... x PH x PW, wherein PR, PC are numbers of patches in vertical and
        horizontal direction and PH, PW are height and width of each patch, respectively.
    :type: Tensor | list[list[Tensor]]
    :param overlap: Overlapping width in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :param merge_method: The method used to determine the values of overlapping pixels.

        ``avg``
            In each overlapping position, pixels of involved patches are averaged.

        ``crop``
            Both of two overlapping patches fill half the overlapping region.

        ``slope``
            In each overlapping position, pixels of involved patches are averaged
            where the component closer to its patches has larger weights.
    :type: Literal['avg', 'crop', 'slope']
    :return: A resulted image of shape ... x H x W.
    :rtype: Tensor
    """
    overlap = size2d(overlap)
    first = patches[0][0]
    if isinstance(patches, Ts):
        patch_sz = patches.shape[-2:]
        patch_n = patches.shape[:2]
    else:
        patch_sz = first.shape[-2:]
        patch_n = (len(patches), len(patches[0]))
    img_sz = tuple([patch_n[i] * patch_sz[i] - (patch_n[i] - 1) * overlap[i] for i in (0, 1)])

    img = first.new_zeros(first.shape[:-2] + img_sz)
    if merge_method == 'avg':
        return _average_merge(img, patch_n, patch_sz, overlap, patches)
    elif merge_method == 'crop':
        return _crop_merge(img, patch_n, patch_sz, overlap, patches)
    elif merge_method == 'slope':
        return _slope_merge(img, patch_n, patch_sz, overlap, patches)
    else:
        raise ValueError(f'Unknown merging method: {merge_method}')


def crop(image: Ts, cropping: Size2d) -> Ts:
    """
    Crop an image with given cropping width.

    :param Tensor image: The image to be cropped.
    :param cropping: Cropping width in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :return: Cropped image.
    :rtype: Tensor
    """
    cropping = size2d(cropping)
    if cropping == (0, 0):
        return image
    return image[..., cropping[0]:-cropping[0], cropping[1]:-cropping[1]]


def pad(image: Ts, padding: Size2d, mode: str = 'constant', value: float = 0) -> Ts:
    """
    Pad an image with given padding width.

    :param Tensor image: The image to be padded.
    :param padding: Padding width in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :param str mode: See :py:func:`torch.nn.functional.pad`.
    :param float value: See :py:func:`torch.nn.functional.pad`.
    :return: Padded image.
    :rtype: Tensor
    """
    ph, pw = size2d(padding)
    if ph == 0 and pw == 0:
        return image
    return functional.pad(image, (pw, pw, ph, ph), mode, value)


def resize(image: Ts, target_size: Size2d, mode: str = 'constant', value: float = 0) -> Ts:
    """
    Resize an image to given size by padding or cropping around its edges.

    :param Tensor image: The image to be resized.
    :param target_size: Target size in vertical and horizontal direction.
    :type: int | tuple[int, int]
    :param str mode: See :py:func:`torch.nn.functional.pad`.
    :param float value: See :py:func:`torch.nn.functional.pad`.
    :return: Resized image with given size.
    :rtype: Tensor
    """
    target_size = size2d(target_size)
    ph = target_size[-2] - image.size(-2)
    pw = target_size[-1] - image.size(-1)
    if ph == 0 and pw == 0:
        return image
    return functional.pad(image, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode, value)
