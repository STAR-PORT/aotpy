from dataclasses import dataclass, field

from .telescope import LaserLaunchTelescope

__all__ = ['Source', 'NaturalGuideStar', 'LaserGuideStar']


@dataclass
class Source:
    name: str
    right_ascension: float = None
    declination: float = None
    zenith_angle: float = None
    azimuth: float = None


@dataclass
class NaturalGuideStar(Source):
    pass

@dataclass
class LaserGuideStar(Source):
    laser_launch_telescope: LaserLaunchTelescope = None
    sodium_height: float = None
    sodium_width: float = None
    sodium_profile: list[float] = field(default_factory=list)
    sodium_altitudes: list[float] = field(default_factory=list)
