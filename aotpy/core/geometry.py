"""
This module contains classes that describe the geometry of a system over time.
"""

from dataclasses import dataclass, field

from .base import Referenceable, Coordinates
from .time import Time

__all__ = ['GeometryInstant', 'Geometry']


@dataclass(kw_only=True)
class GeometryInstant:
    """Contains the geometric parameters of the respective object at one instant."""

    rotation: float = None
    '2D rotation on the reference origin point. (in rad units)'

    translation: Coordinates = None
    """Translation in the horizontal/vertical axis from the reference origin point, represented as `Coordinates`.
    (in m units)"""

    magnification: Coordinates = None
    """Optical magnification ratio in the horizontal/vertical axis from the reference origin point, represented as
    `Coordinates`. (dimensionless quantity)"""


@dataclass(kw_only=True)
class Geometry(Referenceable):
    """Contains the geometric parameters of the respective object at different points in time."""

    time: Time = None

    sequence: list[GeometryInstant] = field(default_factory=list)
    """The sequence of geometric parameters itself."""
