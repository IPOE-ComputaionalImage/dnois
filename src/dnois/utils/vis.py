from ..base.typing import RGBFormat, Ts, cast

__all__ = [
    't4plot',
    'wl2rgb',

    'RGBTriplet',
]

RGBTriplet = tuple[float, float, float] | str


def t4plot(tensor: Ts) -> Ts:
    return tensor.detach().cpu()


def wl2rgb(wl: float, gamma: float = 0.8, output_format: RGBFormat = 'floats') -> RGBTriplet:
    wl *= 1e9
    if 380 <= wl <= 440:
        red, green, blue = -(wl - 440) / (440 - 380), 0., 1.
    elif 440 <= wl <= 490:
        red, green, blue = 0.0, (wl - 440) / (490 - 440), 1.
    elif 490 <= wl <= 510:
        red, green, blue = 0.0, 1., -(wl - 510) / (510 - 490)
    elif 510 <= wl <= 580:
        red, green, blue = (wl - 510) / (580 - 510), 1., 0.
    elif 580 <= wl <= 645:
        red, green, blue = 1.0, -(wl - 645) / (645 - 580), 0.
    elif 645 <= wl <= 780:
        red, green, blue = 1.0, 0., 0.
    else:
        red, green, blue = 0.0, 0., 0.

    if 380 <= wl <= 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 420 <= wl <= 700:
        factor = 1.0
    elif 700 <= wl <= 780:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    else:
        factor = 0.0

    rgb = ((red * factor) ** gamma, (green * factor) ** gamma, (blue * factor) ** gamma)
    if output_format == 'floats':
        return rgb
    else:
        rgb = tuple(int(v * 255) for v in rgb)
        if output_format == 'ints':
            return cast(RGBTriplet, rgb)
        elif output_format == 'hex':
            return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
        else:
            raise ValueError(f'Unknown output format: {output_format}')
