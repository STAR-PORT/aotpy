"""
This module contains a class for optical aberrations.
"""

from dataclasses import dataclass, field

from .base import Referenceable, Coordinates
from .image import Image

__all__ = ['Aberration']


@dataclass(kw_only=True)
class Aberration(Referenceable):
    """Represents an optical aberration that may exist in a certain part of the AO system."""
    modes: Image
    """Set of :math:`m` different :math:`h \\times w` arrays, each representing the orthonormal basis of the
    corresponding mode. (Dimensions :math:`m \\times h \\times w`, dimensionless quantity, using data type flt)"""

    coefficients: Image
    """Set of :math:`m` coefficients (one for each of the orthonormal basis of modes) for each :math:`n` pupil offset.
    (Dimensions :math:`m \\times n`, in user defined units, using data type flt)"""

    offsets: list[Coordinates] = field(default_factory=list)
    """List of horizontal/vertical offsets from the centre of the field, defined as `Coordinates`. 
    There must be an offset for each set of coefficients."""
