"""
This module contains classes that define multidimensional data in AOT and their respective metadata.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Metadatum
from .time import Time

__all__ = ['Image']


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
