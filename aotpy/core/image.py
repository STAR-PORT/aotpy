"""
This module contains classes that define multidimensional data in AOT and their respective metadata.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .time import Time

__all__ = ['Image', 'Metadatum']


@dataclass
class Metadatum:
    """Contains a set of key, value and (optionally) comment which describe the Image data in some aspect."""
    key: str
    "Identifier of the value."

    value: Any
    "Information that describes the Image data."

    comment: str = None
    "Optional description of the metadatum."

    def to_tuple(self) -> tuple:
        """ Return the `Metadatum` object expressed as a tuple (key, value, comment)."""
        return self.key, self.value, self.comment


@dataclass
class Image:
    """Contains multidimensional data and the metadata related to it."""

    name: str
    """Unique name that identifies the data."""

    data: np.ndarray
    """The multi-dimensional data itself."""

    # _: KW_ONLY

    unit: str = None
    """Unit of measurement used for the data."""

    time: Time = None

    metadata: list[Metadatum] = field(default_factory=list)
    """List of Metadatum objects that describe the Image data."""

    def __eq__(self, other):
        return self.name.upper() == other.name.upper() and \
            np.allclose(self.data, other.data) and \
            self.unit == other.unit and \
            self.time == other.time and \
            self.metadata == other.metadata

    def __post_init__(self):
        self.name = self.name.upper()

    def metadata_to_dict(self) -> dict[str, Any]:
        """Return the metadata as a dictionary key->value (comments are ignored)."""
        return {metadatum.key: metadatum.value for metadatum in self.metadata}
