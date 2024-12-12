import torch

from .base.typing import Ts

__all__ = [
    'center_depth',
    'depth2ips',
    'depth2slope',
    'depth_range',
    'ips2depth',
    'ips2slope',
    'slope2depth',
    'slope2ips',
    'slope_lim',
    'slope_range'
]


def depth2ips(depth, min_depth, max_depth):
    return max_depth * (depth - min_depth) / (depth * (max_depth - min_depth))


def ips2depth(ips, min_depth, max_depth):
    return max_depth * min_depth / (max_depth - ips * (max_depth - min_depth))


def depth2slope(depth, central_depth):
    return (depth - central_depth) / depth


def slope2depth(slope, central_depth):
    return central_depth / (1 - slope)


def ips2slope(ips, slope_range_):
    return slope_range_ * (ips - 0.5)


def slope2ips(slope, slope_range_):
    return slope / slope_range_ + 0.5


def center_depth(min_depth, max_depth):
    return 2 * min_depth * max_depth / (min_depth + max_depth)


def slope_lim(min_depth, max_depth):
    return (max_depth - min_depth) / (max_depth + min_depth)


def slope_range(min_depth, max_depth):
    return 2 * slope_lim(min_depth, max_depth)


def depth_range(central_depth, slope_range_):
    _center_depth2 = 2 * central_depth
    return _center_depth2 / (2 + slope_range_), _center_depth2 / (2 - slope_range_)


def quantize_depth_map(dmap: Ts, min_depth, max_depth, n: int) -> list[Ts]:
    dmap = dmap.clamp(min_depth, max_depth)
    imap = depth2ips(dmap, min_depth, max_depth)
    imap = imap.clamp(0, 1)
    imap = imap * n
    idx_map = imap.floor().int().clamp(max=n - 1)
    masks = [torch.zeros_like(idx_map, dtype=torch.bool) for _ in range(n)]
    for i in range(n):
        masks[i][idx_map == i] = True
    return masks
