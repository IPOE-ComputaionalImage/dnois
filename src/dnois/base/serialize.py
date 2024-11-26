import json
from pathlib import Path
import warnings

import torch

from .typing import Any

__all__ = [
    'AsJsonMixIn',
]


class AsDictMixIn:
    def to_dict(self) -> dict[str, Any]:
        """
        Converts ``self`` into a ``dict`` which recursively contains only primitive Python objects.

        :rtype: dict
        """
        raise NotImplementedError(f'Class {self.__class__.__name__} does not '
                                  f'implement {self.to_dict.__name__} method')

    @classmethod
    def from_dict(cls, d: dict):
        cls: type
        return cls(**d)

    def _attr2dictitem(self, name: str):
        converter = getattr(self, '_todict_' + name, None)
        if converter is not None and callable(converter):
            return converter()

        attr = getattr(self, name)
        if isinstance(attr, AsDictMixIn):
            return attr.to_dict()

        if torch.is_tensor(attr):
            if attr.numel() > 100:
                warnings.warn(f'Trying to convert a too large tensor (numel={attr.numel()}) to a list')
            attr = attr.tolist()

        return attr


class AsJsonMixIn(AsDictMixIn):
    def save_json(self, fp, **kwargs):
        """
        Save ``self`` into a JSON file-like object ``fp``.

        :keyword kwargs: Keyword arguments passed to :func:`json.dump`.
        """
        return json.dump(self.to_dict(), fp, **kwargs)

    @classmethod
    def load_json(cls, file, **kwargs):
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            with file.open('r', encoding='utf-8') as f:
                d = json.load(f, **kwargs)
        else:
            d = json.load(file, **kwargs)
        return cls.from_dict(d)
