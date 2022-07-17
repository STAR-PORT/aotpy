from dataclasses import dataclass

__all__ = ['Telescope', 'MainTelescope', 'LaserLaunchTelescope']

from .image import Image


@dataclass
class Telescope:
    name: str
    d_hex: float = None
    d_circle: float = None
    d_eq: float = None
    cobs: float = None
    pupil: Image = None
    pupil_angle: float = None
    elevation: float = None
    azimuth: float = None
    static_map: Image = None


@dataclass
class MainTelescope(Telescope):
    pass


@dataclass
class LaserLaunchTelescope(Telescope):
    pass
