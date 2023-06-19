"""
This module contains classes for describing telescopes in a system. These may be the main telescope itself of laser
launch telescopes.
"""

from dataclasses import dataclass, field

from .aberration import Aberration
from .base import Referenceable, Coordinates

__all__ = ['Segments', 'Monolithic', 'CircularSegments', 'HexagonalSegments', 'Telescope', 'MainTelescope',
           'LaserLaunchTelescope']

from .image import Image


@dataclass(kw_only=True)
class Segments:
    """Abstract class that describes the segments that constitute the pupil."""
    size: float = None
    """Size of the segments that constitute the pupil, measured as the diameter of the smallest circle that contains the
    entirety of a segment. If no segments exist (monolithic) this should be null. (in m units)"""

    coordinates: list[Coordinates] = field(default_factory=list)
    'List of horizontal/vertical coordinates of each segment in the pupil. (in m units)'

    def __post_init__(self):
        if self.__class__ == Segments:
            raise TypeError("Cannot instantiate abstract class.")


class Monolithic(Segments):
    """Describes a monolithic pupil."""

    def __init__(self):
        super().__init__()  # Use default initialization values for Segments


class HexagonalSegments(Segments):
    """Describes the hexagonal segments that constitute the pupil."""
    pass


class CircularSegments(Segments):
    """Describes the circular segments that constitute the pupil."""
    pass


@dataclass(kw_only=True)
class Telescope(Referenceable):
    """Abstract class that contains data related to a telescope in the system."""

    latitude: float = None
    'Latitude of the telescope (World Geodetic System). (in deg units)'

    longitude: float = None
    'Longitude of the telescope (World Geodetic System). (in deg units)'

    elevation: float = None
    """Elevation of the telescope at start, the angle between the object and the observer's local horizon with 0° being
    the horizon and 90° being zenith (horizontal coordinate system). (in deg units)"""

    azimuth: float = None
    """Azimuth of the telescope at start, the angle of the object around the horizon with 0° being North, increasing
    eastward (horizontal coordinate system). (in deg units)"""

    parallactic: float = None
    """Parallactic angle, the spherical angle between the hour circle and the great circle through a celestial object
    and the zenith. (in deg units)"""

    pupil_mask: Image = None
    """Binary image that describes the shape of the pupil. A 1 indicates the presence of the pupil, while a 0 indicates
    the opposite. (Dimensions :math:`h \\times w`, dimensionless quantity, using data type int)"""

    pupil_angle: float = None
    'Clockwise rotation of the pupil mask. (in rad units)'

    enclosing_diameter: float = None
    'Diameter of the smallest circle that contains the entirety of the pupil (enclosing circle). (in m units)'

    inscribed_diameter: float = None
    """Diameter of the largest circle that can be contained in the pupil (inscribed circle).
    On monolithic circular pupils this is equivalent to `enclosing_diameter`. (in m units)"""

    obstruction_diameter: float = None
    'Diameter of the smallest circle that fully contains the central obstruction. (in m units)'

    segments: Segments = field(default_factory=Monolithic)

    transformation_matrix: Image = None
    r"""Matrix that defines 2-dimensional affine transformations over time (:math:`t`) using homogeneous coordinates.
    Any combination of translation, reflection, scale, rotation and shearing can be described via a single
    :math:`3 \times 3` matrix :math:`M` such that :math:`P' = MP`, where :math:`P` is a
    :math:`\begin{bmatrix}x & y & 1 \end{bmatrix}` vector (with :math:`x` and :math:`y` being the original horizontal 
    and vertical coordinates, respectively) and :math:`P'` is a :math:`\begin{bmatrix}x' & y' & 1 \end{bmatrix}`, where
    :math:`x'` and :math:`y'` are the transformed coordinates. All geometry information must be described relative to
    the same reference origin point, from which transformations may occur.
    (Dimensions :math:`3 \times 3 \times t`, dimensionless quantity, using data type flt)"""

    aberration: Aberration = None  # static_map

    def __post_init__(self):
        if self.__class__ == Telescope:
            raise TypeError("Cannot instantiate abstract class.")


class MainTelescope(Telescope):
    """Contains data related to the main telescope in the system. There is only one main telescope in each system."""
    pass


class LaserLaunchTelescope(Telescope):
    """Contains data related to a laser launch telescope in the system. These telescopes produce a laser guide star."""
    pass
