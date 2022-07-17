import numpy as np
from dataclasses import dataclass, field
from typing import Any
from astropy.io import fits
import re

__all__ = ['Image', 'Metadatum']


_standard_keywords = {'SIMPLE', 'EXTEND', 'BSCALE', 'BZERO', 'XTENSION', 'BITPIX', 'PCOUNT', 'GCOUNT', 'EXTNAME'}
_standard_patterns = re.compile(r'NAXIS\d*')


@dataclass
class Metadatum:
    key: str
    value: Any
    comment: str = None

    def to_tuple(self) -> tuple:
        return self.key, self.value, self.comment

    @classmethod
    def from_card(cls, card: fits.Card):
        return cls(card.keyword, card.value, card.comment)


@dataclass
class Image:
    name: str
    data: np.ndarray
    unit: str = None
    metadata: list[Metadatum] = field(default_factory=list)

    def __eq__(self, other):
        return self.name.upper() == other.name.upper() and \
               self.unit == other.unit and \
               self.metadata == other.metadata and \
               np.allclose(self.data, other.data)

    def __post_init__(self):
        md = next((x for x in self.metadata if x.key == 'BUNIT'), None)  # try to find 'BUNIT' in metadata
        if md and self.unit:
            if md.value != self.unit:
                raise RuntimeError
        elif md and not self.unit:
            self.unit = md.value
        elif not md and self.unit:
            self.metadata.append(Metadatum('BUNIT', self.unit))

        md = next((x for x in self.metadata if x.key == 'EXTNAME'), None)  # try to find 'EXTNAME' in metadata
        if md:
            if md.value != self.name:
                raise RuntimeError
            self.metadata.remove(md)

    @classmethod
    def from_fits(cls, path, index: int = None, name: str = None):
        with fits.open(path) as hdus:
            if index:
                hdu = hdus[index]
            else:
                for hdu in hdus:
                    if hdu.is_image and hdu.data is not None:
                        break
                else:
                    raise RuntimeError
            return cls.from_hdu(hdu, name)

    @classmethod
    def from_hdu(cls, hdu: fits.ImageHDU, name: str = None):
        return cls(name=hdu.name if name is None else name,
                   data=hdu.data,
                   metadata=[Metadatum.from_card(card) for card in hdu.header.cards
                             if card.keyword not in _standard_keywords
                             and not _standard_patterns.fullmatch(card.keyword)])

    def to_hdu(self) -> fits.ImageHDU:
        return fits.ImageHDU(data=self.data,
                             header=fits.Header([md.to_tuple() for md in self.metadata]),
                             name=self.name)
