"""
This module contains classes that describe different types of light sources in an AO system.
"""

from dataclasses import dataclass, field

from .base import Referenceable
from .image import Image
from .telescope import LaserLaunchTelescope

__all__ = ['Source', 'ScienceStar', 'NaturalGuideStar', 'SodiumLaserGuideStar', 'RayleighLaserGuideStar']


@dataclass(kw_only=True)
class Source(Referenceable):
    """Abstract class that contains data regarding a light source used by the system."""

    right_ascension: float = None
    'Right ascension of the light source, epoch J2000 (equatorial coordinate system). (in deg units)'

    declination: float = None
    'Declination of the light source, epoch J2000 (equatorial coordinate system). (in deg units)'

    elevation_offset: float = None
    "Offset from the Main Telescope's `elevation`. (in deg units)"

    azimuth_offset: float = None
    "Offset from the Main Telescope's `azimuth`. (in deg units)"

    width: float = None
    'Effective width at zenith. (in rad units)'

    def __post_init__(self):
        if self.__class__ == Source:
            raise TypeError("Cannot instantiate abstract class.")


@dataclass(kw_only=True)
class ScienceStar(Source):
    """Contains data regarding a science star being observed by the system."""
    pass


@dataclass(kw_only=True)
class NaturalGuideStar(Source):
    """Contains data regarding a natural guide star being observed by the system."""
    pass


@dataclass(kw_only=True)
class LaserGuideStar(Source):
    """Abstract class that contains data regarding a laser guide star being observed by the system."""

    laser_launch_telescope: LaserLaunchTelescope = None

    def __post_init__(self):
        if self.__class__ == LaserGuideStar:
            raise TypeError("Cannot instantiate abstract class.")


@dataclass(kw_only=True)
class SodiumLaserGuideStar(LaserGuideStar):
    """Contains data regarding a sodium laser guide star being observed by the system."""

    height: float = None
    'Mean LGS height at zenith. (in m units)'

    profile: Image = None
    """Normalised LGS profile (each set of :math:`l` layers :math:`\\sum = 1`) at zenith, over time.
    (Dimensions :math:`t \\times l`, dimensionless quantity, using data type flt)"""

    altitudes: list[float] = field(default_factory=list)
    """LGS layer profile altitudes at zenith. Must be the same length as the number of layers defined in `profile`.
    (in m units)"""


@dataclass(kw_only=True)
class RayleighLaserGuideStar(LaserGuideStar):
    """Contains data regarding a Rayleigh laser guide star being observed by the system."""

    distance: float = None
    'Fixed distance of the LGS from the telescope. (in m units)'

    depth: float = None
    'TODO: Depth (in m units)'
