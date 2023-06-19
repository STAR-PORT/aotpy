"""
This module contains a classes used as a base for the rest of the core of aotpy.
"""

from collections import namedtuple
from dataclasses import dataclass

__all__ = ['Referenceable', 'Coordinates']

Coordinates = namedtuple('Coordinates', 'x y')


@dataclass
class Referenceable:
    """Abstract class for all classes which can be referenced via a UID."""
    uid: str
    """Unique identifier of the object, which allows unambiguous referencing."""

    def __post_init__(self):
        if self.__class__ == Referenceable:
            raise TypeError("Cannot instantiate abstract class.")
