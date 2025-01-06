import json
from pathlib import Path
import warnings

import torch

from .typing import Any, cast

__all__ = [
    'AsDictMixIn',
    'AsJsonMixIn',
]


class AsDictMixIn:
    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        """
        Converts ``self`` into a ``dict`` which recursively contains only primitive Python objects.

        :rtype: dict
        """
        raise NotImplementedError(f'Class {self.__class__.__name__} does not '
                                  f'implement {self.to_dict.__name__} method')

    @classmethod
    def from_dict(cls, d: dict):
        cls: type
        d = cast(type[AsDictMixIn], cls)._pre_from_dict(d)
        return cls(**d)

    def _attr2dictitem(self, name: str, keep_tensor: bool = True):
        converter = getattr(self, '_todict_' + name, None)
        if converter is not None and callable(converter):
            return converter(keep_tensor)

        attr = getattr(self, name)
        if isinstance(attr, AsDictMixIn):
            return attr.to_dict(keep_tensor)

        if torch.is_tensor(attr) and not keep_tensor:
            if attr.numel() > 100:
                warnings.warn(f'Trying to convert a too large tensor (numel={attr.numel()}) to a list')
            attr = attr.tolist()

        return attr

    @classmethod
    def _pre_from_dict(cls, d: dict) -> dict:
        return d.copy()


class AsJsonMixIn(AsDictMixIn):
    def to_json(self, **kwargs) -> str:
        """
        Converts ``self`` into a JSON string.

        :keyword kwargs: Keyword arguments passed to :func:`json.dumps`.
        :rtype: str
        """
        return json.dumps(self.to_dict(False), **kwargs)

    def save_json(self, fp, **kwargs):
        """
        Save ``self`` into a JSON file-like object ``fp``.

        :keyword kwargs: Keyword arguments passed to :func:`json.dump`.
        """
        return json.dump(self.to_dict(False), fp, **kwargs)

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
