import torch
from torch import nn

from dnois.base.typing import Size2d, Literal, Ts, size2d

from .noise import gaussian

__all__ = [
    'cfa_collect',
    'cfa_flatten',
    'quantize',
    'rgb2raw',
    'spectral_integrate_cfa',

    'BayerPattern',
    'SimpleSensor',
]

_MSG1 = 'Number of channels of input radiance is not {0} for an {1} sensor without SRF'
BayerPattern = Literal['RGGB', 'GRBG', 'BGGR', 'BGRG']


def _make_srf(srf: Ts | tuple[Ts, Ts, Ts], pattern: BayerPattern) -> Ts | None:
    if srf is None:
        return None
    r, g, b = srf
    if pattern == 'RGGB':
        srf = (r, g, g, b)
    elif pattern == 'GRBG':
        srf = (g, r, b, g)
    elif pattern == 'BGGR':
        srf = (b, g, g, r)
    elif pattern == 'BGRG':
        srf = (g, b, r, g)
    else:
        raise ValueError(f'Wrong bayer pattern: {pattern}')
    return torch.stack(srf)


def cfa_flatten(image: Ts, unit_size: Size2d = 1) -> Ts:
    r"""
    Flatten the pixels in a channel-wise image with shape :math:`(\cdots,C,H,W)` into
    `CFA <https://en.wikipedia.org/wiki/Color_filter_array>`_ units with size :math:`(h,w)`
    to form an image with shape :math:`(\cdots,H\times h,W\times w)`, where :math:`C=hw`.
    This is the inverse of :py:func:`~dnois.sensor.cfa_collect`.

    .. note::

        This function is similar to :py:func:`~torch.nn.functional.pixel_shuffle` but
        removes a dimension of the input and supports unequal unit height and width.

    :param Tensor image: The image to be flattened, a tensor shape :math:`(\cdots,C,H,W)`.
    :param unit_size: Height and width of a pixel group :math:`(h,w)`. Default: ``(1, 1)``.
    :type: int or tuple[int, int]
    :return: Flattened image with shape :math:`(\cdots,H\times h,W\times w)`.
    :rtype: Tensor
    """
    us = size2d(unit_size)
    if image.size(-3) != us[0] * us[1]:
        raise ValueError(f'Number of channels must be equal to the product of unit height and width')
    if us[0] == us[1] == 1:
        return image.squeeze(-3)
    if us[0] == us[1]:
        return nn.functional.pixel_shuffle(image, us[0]).squeeze(-3)

    image = image.reshape(*image.shape[:-3], image.size(-3) // (us[0] * us[1]), *us, *image.shape[-2:])
    n = image.ndim
    image = image.permute(*list(range(n - 4)), n - 2, n - 4, n - 1, n - 3)
    image = image.reshape(*image.shape[:-4], image.size(-4) * us[0], image.size(-2) * us[1])
    return image.squeeze(-3)


def cfa_collect(image: Ts, unit_size: Size2d = 1) -> Ts:
    r"""
    Rearrange all the pixels in an image with shape :math:`(\cdots,H,W)` into channels-wise form
    :math:`(\cdots,C,H/h,W/w)`, where :math:`C=hw` is the number of channels and :math:`h` and
    :math:`w` are the size of a unit of a regular `color filter array (CFA)
    <https://en.wikipedia.org/wiki/Color_filter_array>`_. In this way, the vanilla 2D pixel
    array is divided into contiguous and non-overlapping pixel groups, or units.
    'Regular' means all the units have same size.

    The pixels in a unit will be rearranged into a single pixel with :math:`C=hw` channels
    in a row-major manner.

    .. note::

        This function is similar to :py:func:`~torch.nn.functional.pixel_unshuffle` but
        adds a new dimension to the input and supports unequal unit height and width.

    :param Tensor image: The image to be rearranged, a tensor shape :math:`(\cdots,H,W)`.
    :param unit_size: Height and width of a pixel group :math:`(h,w)`. Default: ``(1, 1)``.
    :type: int or tuple[int, int]
    :return: Rearranged image with shape :math:`(\cdots,hw,H/h,W/w)`.
    :rtype: Tensor
    """
    us = size2d(unit_size)
    image = image.unsqueeze(-3)  # ... x 1 x H x W
    if us[0] == us[1] == 1:
        return image
    if us[0] == us[1]:
        return nn.functional.pixel_unshuffle(image, us[0])

    image = image.reshape(
        *image.shape[:-2], image.size(-2) // us[0], us[0], image.size(-1) // us[1], us[1]
    )
    n = image.ndim
    image = image.permute(*list(range(n - 4)), n - 3, n - 1, n - 4, n - 2)
    image = image.reshape(*image.shape[:-5], image.size(-5) * us[0] * us[1], *image.shape[-2:])
    return image


# kornia.color
def rgb2raw(image: Ts, pattern: BayerPattern) -> Ts:
    """
    Convert an RGB image into a single-channel image using Bayer CFA pattern.

    :param Tensor image: The RGB image, a tensor of shape ``(..., 3, H, W)``.
    :param BayerPattern pattern: Bayer CFA pattern, either ``'RGGB'``, ``'GRBG'``,
        ``'BGGR'`` or ``'BGRG'``, specifying how are pixels arranged in the order
        of upper left, upper right, lower left, lower right.
    :return: A single-channel image with shape ``(..., 1, H, W)``.
    :rtype: Tensor
    """
    output: Ts = image[..., 1:2, :, :].clone()

    if pattern == 'RGGB':
        output[..., :, ::2, ::2] = image[..., 0:1, ::2, ::2]  # red
        output[..., :, 1::2, 1::2] = image[..., 2:3, 1::2, 1::2]  # blue
    elif pattern == 'GRBG':
        output[..., :, ::2, 1::2] = image[..., 0:1, ::2, 1::2]  # red
        output[..., :, 1::2, ::2] = image[..., 2:3, 1::2, ::2]  # blue
    elif pattern == 'BGGR':
        output[..., :, 1::2, 1::2] = image[..., 0:1, 1::2, 1::2]  # red
        output[..., :, ::2, ::2] = image[..., 2:3, ::2, ::2]  # blue
    elif pattern == 'BGRG':
        output[..., :, 1::2, ::2] = image[..., 0:1, 1::2, ::2]  # red
        output[..., :, ::2, 1::2] = image[..., 2:3, ::2, 1::2]  # blue

    return output


def spectral_integrate_cfa(
    radiance: Ts,
    srf: Ts,
    unit_size: Size2d = 1,
    channel_dim: bool = False
) -> Ts:
    r"""
    Integrate given radiance field across wavelengths with given spectral response
    function (SRF).

    This function supports regular `color filter array (CFA)
    <https://en.wikipedia.org/wiki/Color_filter_array>`_, where the vanilla 2D pixel
    array is divided into contiguous and non-overlapping pixel groups. 'Regular' means
    all the pixel groups are identical.
    See :py:func:`cfa_collect` and :py:func:`cfa_flatten` for more details.

    :param Tensor radiance: A tensor of shape :math:`(\cdots,N_\lambda,H,W)`.
    :param Tensor srf: A tensor of shape :math:`(N_C, N_\lambda)`.
    :param unit_size: Height and width of a pixel group, in pixels. Note that
        their product should be equal to :math:`N_C`.
        Default: ``(1, 1)``.
    :type: int or tuple[int, int]
    :param bool channel_dim: If ``True``, last three dimension of returned tensor will be
        ``(N_C, H // unit_size[0], W // unit_size[1])``; otherwise ``(H, W)``.
        Default: ``False``.
    :return: Integrated radiance field, of shape
        ``(..., N_C, H // unit_size[0], W // unit_size[1])`` or ``(..., H, W)``.
    :rtype: Tensor
    """
    unit_size = size2d(unit_size)
    if radiance.size(-3) != srf.size(-1):
        raise ValueError(f'N_wl of radiance is {radiance.size(-3)} but that of srf is {srf.size(-1)}')
    if unit_size[0] * unit_size[1] != srf.size(-2):
        raise ValueError(f'Pixel group size is {unit_size} but number of channels is {srf.size(-2)}')
    if radiance.size(-2) % unit_size[0] != 0 or radiance.size(-1) % unit_size[1] != 0:
        raise ValueError(f'Spatial size of radiance ({radiance.shape[-2:]}) '
                         f'must be divisible by pixel group size ({unit_size})')

    unit_size = size2d(unit_size)
    radiance = cfa_collect(radiance, unit_size)  # ... x N_wl x N_C x H' x W'
    radiance = radiance.transpose(-4, -3)  # ... x N_C x N_wl x H' x W'
    t = torch.einsum('...wij,...w->...ij', radiance, srf)  # ... x N_C x H' x W'

    if channel_dim:
        return t
    else:
        return cfa_flatten(t, unit_size).squeeze(-3)  # ... x H x W


def quantize(signal: Ts, levels: int = 256, differentiable: bool = False) -> Ts:
    """
    Quantize continuous-valued signal, emulating an analogous-to-digital conversion.

    :param Tensor signal: The signal to be quantized whose value must be
        in :math:`[0,1]`.
    :param int levels: Quantization levels. 256 for example, which is the number
        of levels for most image sensors. Default: 256.
    :param bool differentiable: Whether to perform quantization
        in a differentiable manner. Specifically, if ``True``,
        a quantization noise will be added to signal to simulate quantization.
        Default: ``False``.
    :return: Quantized signal.
    :rtype: Tensor
    """
    if signal.min().item() < 0 or signal.max().item() > 1:
        raise ValueError(f'Value of signal must be in [0, 1]')
    v_max = levels - 1
    qt = signal * v_max
    qt = torch.round(qt) / v_max
    if differentiable:
        qt_noise = qt - signal.detach
        return signal + qt_noise
    else:
        return qt


class SimpleSensor(nn.Module):
    """
    A simple RGB or grayscale sensor model, which processes the radiance field reaching the sensor
    plane as follows:

    #.  Divide pixels into channels according to Bayer CFA.
    #.  Spectral integral by ``srf``.
    #.  Apply additional Gaussian white noise.
    #.  Restrict signal values to a given range.
    #.  Quantize signal to a given level.

    :param bool rgb: RGB sensor if ``True``, grayscale sensor otherwise.
        Default: ``True``.
    :param srf: SRF tensor of shape ``(N_C, N_wl)`` where ``N_C`` is 1 or 3,
        or a 3-tuple of tensors of length ``N_wl``, corresponding to the SRF of
        R, G, B channels. Default: do not perform spectral integral.
    :type srf: Tensor or tuple[Tensor, Tensor, Tensor]
    :param BayerPattern bayer_pattern: See :py:func:`rgb2raw`. Default: ``'RGGB'``.
    :param float noise_std: Standard deviation of the Gaussian white noise.
        Default: 0.
    :param float max_value: Maximum possible value of output signal.
        Default: 1.
    :param int _quantize: Quantization level. The output signal will not be
        quantized if a negative or zero value is given. Default: 256.
    :param bool differentiable_quant: See :py:func:`quantize`. Default: ``True``.
    """

    #: Spectral response functions. Either ``None`` or a tensor of shape
    #: ``(1, N_wl)`` or ``(4, N_wl)`` (two identical green SRF in Bayer CFA)
    srf: Ts

    def __init__(
        self,
        rgb: bool = True,
        srf: Ts | tuple[Ts, Ts, Ts] = None,
        bayer_pattern: BayerPattern = 'RGGB',
        noise_std: float = 0.,
        max_value: float = 1.,
        _quantize: int = 256,
        differentiable_quant: bool = True,
    ):
        super().__init__()
        self.rgb: bool = rgb  #: RGB sensor or not.
        #: Bayer CFA pattern. See :py:func:`rgb2raw`
        self.bayer_pattern: BayerPattern = bayer_pattern
        self.noise_std: float = noise_std  #: Standard deviation of Gaussian noise.
        self.max_value: float = max_value  #: Maximum possible value of signal.
        self.quantize: int = _quantize  #: Quantization level.
        #: Whether to perform differentiable quantization.
        self.differentiable_quantization: bool = differentiable_quant

        if not (
            srf is None or
            (isinstance(srf, tuple) and len(srf) == 3) or
            (torch.is_tensor(srf) and srf.size(-2) not in (1, 3))
        ):
            raise ValueError(
                f'Number of channels of SRF must be one or three in {self.__class__.__name__}'
            )
        if rgb:
            srf = _make_srf(srf, bayer_pattern)  # 4 x N_wl
        self.register_buffer('srf', srf)

    def forward(self, radiance: Ts) -> Ts:
        """
        Simulate the process of conversion from optical signal ``radiance`` to
        electric signal, an 2D image.

        :param Tensor radiance: A tensor representing the received radiance field,
            of shape ``(..., N_C, H, W)`` if ``srf`` is ``None`` or
            ``(..., N_wl, H, W)`` otherwise.
        :return: Output image signal, a tensor of shape ``(..., H, W)``.
        :rtype: Tensor
        """
        if self.srf is None:
            if self.rgb:  # radiance: ... x 3 x H x W
                if radiance.size(-3) != 3:
                    raise ValueError(_MSG1.format('3', 'RGB'))
                transmitted = rgb2raw(radiance, self.bayer_pattern).squeeze(-3)
            else:
                if radiance.size(-3) != 1:
                    raise ValueError(_MSG1.format('1', 'grayscale'))
                transmitted = radiance
        else:
            if radiance.size(-3) != self.srf.size(-1):
                raise ValueError(
                    'Number of wavelengths of input radiance is not equal to '
                    'that of the sensor\'s SRF'
                )
            unit_size = 2 if self.rgb else 1
            transmitted = spectral_integrate_cfa(radiance, self.srf, unit_size)

        signal = gaussian(transmitted, self.noise_std)
        signal = signal.clip(0., self.max_value)

        if self.quantize <= 0:
            return signal
        return quantize(
            signal.detach / self.max_value, self.quantize, self.differentiable_quantization
        )
