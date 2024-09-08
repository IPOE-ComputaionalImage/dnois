import torch

__all__ = [
    'gaussian',
]

_Ts = torch.Tensor


def gaussian(signal: _Ts, sigma: float | _Ts) -> _Ts:
    """
    Adding gaussian noise to signal.

    :param Tensor signal: Input signal.
    :param sigma: Standard deviation of the gaussian noise.
        Must be broadcastable with ``signal`` if a Tensor.
    :type sigma: float | Tensor
    :return: Noisy signal.
    :rtype: Tensor
    """
    return signal + sigma * torch.randn_like(signal)
