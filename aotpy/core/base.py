"""
This module contains a classes used as a base for the rest of the core of aotpy.
"""

from dataclasses import dataclass
from typing import Any

__all__ = ['Referenceable', 'Coordinates', 'Metadatum']


@dataclass
class Coordinates:
    """Contains a set of horizontal (x) and vertical (y) Cartesian coordinates in a plane."""

    x: float = None
    "Horizontal Cartesian coordinate."

    y: float = None
    "Vertical Cartesian coordinate."


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
class Referenceable:
    """Abstract class for all classes which can be referenced via a UID."""
    uid: str
    """Unique identifier of the object, which allows unambiguous referencing."""

    def __post_init__(self):
        if self.__class__ == Referenceable:
            raise TypeError("Cannot instantiate abstract class.")
