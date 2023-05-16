"""
This module contains a class for describing atmospheric parameters.
"""

from dataclasses import dataclass, field

from .base import Referenceable
from .image import Image
from .time import Time

__all__ = ['AtmosphericParameters']


@dataclass(kw_only=True)
class AtmosphericParameters(Referenceable):
    """Contains atmospheric data relevant to the telemetry data recorded."""

    wavelength: float = None
    'Reference wavelength. (in m units)'

    time: Time = None

    r0: list[float] = field(default_factory=list)
    'List of Fried parameters at reference wavelength at zenith, over time. (in m units)'

    fwhm: list[float] = field(default_factory=list)
    'List of full width at half maximum measures of the seeing disc at zenith, over time. (in arcsec units)'

    tau0: list[float] = field(default_factory=list)
    'List of atmospheric coherence times, over time. (in s units)'

    theta0: list[float] = field(default_factory=list)
    'List of isoplanatic angles, over time. (in rad units)'

    layers_weight: Image = None
    """Fractional weight of each :math:`l` turbulence layer (sums to 1), for each time instant.
    (Dimensions :math:`t \\times l`, dimensionless quantity, using data type flt)"""

    layers_height: Image = None
    """Height above observatory at zenith of each :math:`l` turbulence layer, for each time instant.
    (Dimensions :math:`t \\times l`, in m units, using data type flt)"""

    layers_l0: Image = None
    """Outer scale of turbulence at reference wavelength at zenith of each :math:`l` turbulence layer, for each time
    instant. (Dimensions :math:`t \\times l`, in m units, using data type flt)"""

    layers_wind_speed: Image = None
    r"""Wind speed of each :math:`l` turbulence layer, for each time instant.
    (Dimensions :math:`t \times l`, in ms\ :math:`^{-1}` units, using data type flt)"""

    layers_wind_direction: Image = None
    """Wind direction of each :math:`l` turbulence layer, for each time instant, with 0° being North, increasing
    eastward. (Dimensions :math:`t \\times l`, in deg units, using data type flt)"""

    transformation_matrix: Image = None
    r"""Matrix that defines 2-dimensional affine transformations over time (:math:`t`) using homogeneous coordinates.
    Any combination of translation, reflection, scale, rotation and shearing can be described via a single
    :math:`3 \times 3` matrix :math:`M` such that :math:`P' = MP`, where :math:`P` is a
    :math:`\begin{bmatrix}x & y & 1 \end{bmatrix}` vector (with :math:`x` and :math:`y` being the original horizontal 
    and vertical coordinates, respectively) and :math:`P'` is a :math:`\begin{bmatrix}x' & y' & 1 \end{bmatrix}`, where
    :math:`x'` and :math:`y'` are the transformed coordinates. All geometry information must be described relative to
    the same reference origin point, from which transformations may occur.
    (Dimensions :math:`3 \times 3 \times t`, dimensionless quantity, using data type flt)"""
