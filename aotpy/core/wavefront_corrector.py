"""
This module contains classes that describe different types of wavefront correctors in a system.
"""


from dataclasses import dataclass, field

from .aberration import Aberration
from .base import Referenceable, Coordinates
from .image import Image
from .telescope import Telescope

__all__ = ['WavefrontCorrector', 'DeformableMirror', 'TipTiltMirror', 'LinearStage']


@dataclass(kw_only=True)
class WavefrontCorrector(Referenceable):
    """Abstract class that contains data related to a wavefront corrector in the system."""
    telescope: Telescope
    n_valid_actuators: int
    """Number of valid actuators in the corrector. (in count units)"""

    pupil_mask: Image = None
    """Binary image that describes the shape of the pupil. A 1 indicates the presence of the pupil, while a 0 indicates
    the opposite. (Dimensions :math:`h \\times w`, dimensionless quantity, using data type int)"""

    tfz_num: list[float] = field(default_factory=list)
    'List of numerators of the transfer function Z. (dimensionless quantity)'

    tfz_den: list[float] = field(default_factory=list)
    'List of denominators of the transfer function Z. (dimensionless quantity)'

    transformation_matrix: Image = None
    r"""Matrix that defines 2-dimensional affine transformations over time (:math:`t`) using homogeneous coordinates.
    Any combination of translation, reflection, scale, rotation and shearing can be described via a single
    :math:`3 \times 3` matrix :math:`M` such that :math:`P' = MP`, where :math:`P` is a
    :math:`\begin{bmatrix}x & y & 1 \end{bmatrix}` vector (with :math:`x` and :math:`y` being the original horizontal 
    and vertical coordinates, respectively) and :math:`P'` is a :math:`\begin{bmatrix}x' & y' & 1 \end{bmatrix}`, where
    :math:`x'` and :math:`y'` are the transformed coordinates. All geometry information must be described relative to
    the same reference origin point, from which transformations may occur.
    (Dimensions :math:`3 \times 3 \times t`, dimensionless quantity, using data type flt)"""

    aberration: Aberration = None

    def __post_init__(self):
        if self.__class__ == WavefrontCorrector:
            raise TypeError("Cannot instantiate abstract class.")


@dataclass(kw_only=True)
class DeformableMirror(WavefrontCorrector):
    """Contains data related to a deformable mirror in the system."""
    actuator_coordinates: list[Coordinates] = field(default_factory=list)
    'List of horizontal/vertical coordinates of the valid actuators of the DM. (in m units)'

    influence_function: Image = None
    """A set of 2D images, one for each valid actuator, where each image represents the displacement of the surface of
    the deformable mirror after poking the respective actuator.
    (Dimensions :math:`a_v \\times h \\times w`, in m units, using data type flt)"""

    stroke: float = None
    'Maximum possible actuator displacement, measured as an excursion from a central null position. (in m units)'


@dataclass(kw_only=True)
class TipTiltMirror(WavefrontCorrector):
    """Contains data related to a tip-tilt mirror in the system. It is assumed it has a two-axis movement."""
    n_valid_actuators: int = field(init=False, default=2)
    pass


@dataclass(kw_only=True)
class LinearStage(WavefrontCorrector):
    """Contains data related to a linear stage in the system. It is assumed it has a single axis of movement."""
    n_valid_actuators: int = field(init=False, default=1)
    pass
