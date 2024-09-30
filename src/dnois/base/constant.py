import torch

__all__ = [
    'DPI',
    'FRAUNHOFER_LINES',
]

DPI = 2 * torch.pi
FRAUNHOFER_LINES = {
    'F': 486.134e-9,
    'd': 587.5618e-9,
    'C': 656.281,
}
