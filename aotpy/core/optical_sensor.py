"""
This module contains classes that describe different optical sensors in the system.

This includes scoring cameras and wavefront sensors, and their respective detectors.
"""

from dataclasses import dataclass, field

from .aberration import Aberration
from .base import Referenceable, Coordinates
from .image import Image
from .source import Source

__all__ = ['Detector', 'ScoringCamera', 'WavefrontSensor', 'ShackHartmann', 'Pyramid']


@dataclass(kw_only=True)
class Detector(Referenceable):
    """Contains data regarding a detector in the system.
    Detectors are a part of wavefront sensors and scoring cameras."""

    type: str = None
    "Identifies the type of detector, such as ``'CMOS'`` or ``'CCD'``. (dimensionless quantity)"

    sampling_technique: str = None
    """Identifies the sampling technique, for example ``'Single Reset Read'``, ``'Fowler'``, ``'Double Correlated'``,
     ``'Uncorrelated'``. (dimensionless quantity)"""

    shutter_type: str = None
    "Identifies the shutter type, typically ``'Rolling'`` or ``'Global'``. (dimensionless quantity)"

    flat_field: Image = None
    """Inverse of the detector pixel-to-pixel sensitivity
    (Dimensions :math:`h \\times w`, dimensionless quantity, using data type flt)"""

    readout_noise: float = None
    r'Readout noise. (in e\ :math:`^-` s\ :math:`^{-1}` pix\ :math:`^{-1}` units)'

    pixel_intensities: Image = None
    """Intensity detected in each pixel, for each data frame. This is a sequence of :math:`t` images, each spanning 
    :math:`x` pixels horizontally and :math:`y` pixels vertically.
    (Dimensions :math:`t \\times h \\times w`, in ADU units, using data type flt)"""

    integration_time: float = None
    'TODO: Integration time. (in s units)'

    coadds: int = None
    'Number of frame co-additions.  (in count units)'

    dark: Image = None
    """Intensity detected in each pixel when there is no light being observed. This is an image spanning 
    :math:`x` pixels horizontally and :math:`y` pixels vertically.
    (Dimensions :math:`h \\times w`, in ADU units, using data type flt)"""

    weight_map: Image = None
    """Pixel weight map, where each detector pixel is associated with a number that represents its relative value.
    Summing up to 1. (Dimensions :math:`h \\times w`, dimensionless quantity, using data type flt)"""

    quantum_efficiency: float = None
    'TODO: Quantum efficiency. (dimensionless quantity)'

    pixel_scale: float = None
    r'TODO: Pixel scale. (in rad pix\ :math:`^{-1}` units)'

    binning: int = None
    'TODO: Binning. (in count units)'

    bandwidth: float = None
    'TODO: Spectral bandwidth. (in m units)'

    transmission_wavelength: list[float] = field(default_factory=list)
    'List of wavelengths that describe a transmission profile. (in m units)'

    transmission: list[float] = field(default_factory=list)
    'List of transmission percentages that describe a transmission profile. (dimensionless quantity)'

    sky_background: Image = None
    """detector pixel intensities from a source-less direction in the sky
    (Dimensions :math:`h \\times w`, in ADU units, using data type flt)"""

    gain: float = None
    r'TODO: Gain. (in e\ :math:`^-` units)'

    excess_noise: float = None
    r'TODO: Excess noise. (in e\ :math:`^-` units)'

    filter: str = None
    'Name of filter in use. (dimensionless quantity)'

    bad_pixel_map: Image = None
    """Binary image which identifies the bad pixels. Pixels identified with 1 are considered bad, while 0 is considered
    normal.  (Dimensions :math:`h \\times w`, dimensionless quantity, using data type int)"""

    dynamic_range: float = None
    """Ratio of the maximum signal that can be integrated to the r.m.s. noise floor.
    If this ratio is R, then dynamic range in decibels is 20 log R. (in dB units)"""

    readout_rate: float = None
    r'Inverse of the time required to digitize a single pixel (in px s\ :math:`^{-1}` units)'

    frame_rate: float = None
    r"""Inverse of the time needed for the detector to acquire an image and then completely read it out.
    (in frame s\ :math:`^{-1}` units)"""

    transformation_matrix: Image = None
    r"""Matrix that defines 2-dimensional affine transformations over time (:math:`t`) using homogeneous coordinates.
    Any combination of translation, reflection, scale, rotation and shearing can be described via a single
    :math:`3 \times 3` matrix :math:`M` such that :math:`P' = MP`, where :math:`P` is a
    :math:`\begin{bmatrix}x & y & 1 \end{bmatrix}` vector (with :math:`x` and :math:`y` being the original horizontal 
    and vertical coordinates, respectively) and :math:`P'` is a :math:`\begin{bmatrix}x' & y' & 1 \end{bmatrix}`, where
    :math:`x'` and :math:`y'` are the transformed coordinates. All geometry information must be described relative to
    the same reference origin point, from which transformations may occur.
    (Dimensions :math:`3 \times 3 \times t`, dimensionless quantity, using data type flt)"""


@dataclass(kw_only=True)
class ScoringCamera(Referenceable):
    """Contains data regarding a scoring camera in the system."""

    pupil_mask: Image = None
    """Binary image that describes the shape of the pupil. A 1 indicates the presence of the pupil, while a 0 indicates
    the opposite. (Dimensions :math:`h \\times w`, dimensionless quantity, using data type int)"""

    wavelength: float = None
    """'Observation wavelength  (in m units)'"""

    transformation_matrix: Image = None
    r"""Matrix that defines 2-dimensional affine transformations over time (:math:`t`) using homogeneous coordinates.
    Any combination of translation, reflection, scale, rotation and shearing can be described via a single
    :math:`3 \times 3` matrix :math:`M` such that :math:`P' = MP`, where :math:`P` is a
    :math:`\begin{bmatrix}x & y & 1 \end{bmatrix}` vector (with :math:`x` and :math:`y` being the original horizontal 
    and vertical coordinates, respectively) and :math:`P'` is a :math:`\begin{bmatrix}x' & y' & 1 \end{bmatrix}`, where
    :math:`x'` and :math:`y'` are the transformed coordinates. All geometry information must be described relative to
    the same reference origin point, from which transformations may occur.
    (Dimensions :math:`3 \times 3 \times t`, dimensionless quantity, using data type flt)"""

    detector: Detector = None
    aberration: Aberration = None


@dataclass(kw_only=True)
class WavefrontSensor(Referenceable):
    """Abstract class that contains data related to one wavefront sensor in the system."""

    source: Source

    dimensions: int
    """Number of dimensions being measured by each subaperture. (in count units)"""

    n_valid_subapertures: int
    'Number of valid subapertures (must coincide with `subaperture_mask` data). (in count units)'

    measurements: Image = None
    """Measurements from the sensor over time. Each of its :math:`s_v` subapertures is able to measure in :math:`d`
    dimensions.  (Dimensions :math:`t \\times d \\times s_v`, in user defined units, using data type flt)"""

    ref_measurements: Image = None
    'Reference measurements. (Dimensions :math:`d \\times s_v`, in user defined units, using data type flt)'

    subaperture_mask: Image = None
    r"""Representation of the subaperture grid, where the cells corresponding to invalid subapertures are marked as
    :math:`-1` and the cells corresponding to valid subapertures contain their respective index in the sequence of
    valid subapertures (using zero-based numbering, that is, the initial element is assigned the index 0).
    (Dimensions :math:`s \times s`, dimensionless quantity, using data type int)"""

    mask_offsets: list[Coordinates] = field(default_factory=list)
    """List of horizontal/vertical offsets in detector pixels, represented as `Coordinates`.
    Each offset defines the lowest horizontal/vertical position occupied by the respective mask. (in pix units)"""

    subaperture_size: float = None
    'Size of each subaperture in detector pixels. (in pix units)'

    subaperture_intensities: Image = None
    """Detected average intensity (flux) of each of the :math:`s_v` valid subapertures, over :math:`t` time.
    (Dimensions :math:`t \\times s_v`, in ADU units, using data type flt)"""

    wavelength: float = None
    'Wavelength being sensed. (in m units)'

    optical_gain: Image = None
    'WFS optical gain over time. (Dimensions :math:`t`, dimensionless quantity, using data type flt)'

    transformation_matrix: Image = None
    r"""Matrix that defines 2-dimensional affine transformations over time (:math:`t`) using homogeneous coordinates.
    Any combination of translation, reflection, scale, rotation and shearing can be described via a single
    :math:`3 \times 3` matrix :math:`M` such that :math:`P' = MP`, where :math:`P` is a
    :math:`\begin{bmatrix}x & y & 1 \end{bmatrix}` vector (with :math:`x` and :math:`y` being the original horizontal 
    and vertical coordinates, respectively) and :math:`P'` is a :math:`\begin{bmatrix}x' & y' & 1 \end{bmatrix}`, where
    :math:`x'` and :math:`y'` are the transformed coordinates. All geometry information must be described relative to
    the same reference origin point, from which transformations may occur.
    (Dimensions :math:`3 \times 3 \times t`, dimensionless quantity, using data type flt)"""

    detector: Detector = None
    aberration: Aberration = None
    non_common_path_aberration: Aberration = None

    def __post_init__(self):
        if self.__class__ == WavefrontSensor:
            raise TypeError("Cannot instantiate abstract class.")


@dataclass(kw_only=True)
class ShackHartmann(WavefrontSensor):
    """Contains data related to one Shack-Hartmann wavefront sensor used by the system."""
    dimensions: int = field(init=False, default=2)
    """Number of dimensions being measured by each subaperture. For `ShackHartmann` this must be equal to 2
    (horizontal and vertical offset). (in count units)"""

    centroiding_algorithm: str = None
    'Name of the centroiding algorithm used. (dimensionless quantity)'

    centroid_gains: Image = None
    """Centroid gain factors for each of :math:`s_v` subapertures and :math:`d` dimensions.
    (Dimensions :math:`d \\times s_v`, dimensionless quantity, using data type flt)"""

    spot_fwhm: Image = None
    """Spot full width half maximum for each of :math:`s_v` subapertures and  :math:`d` dimensions.
    (Dimensions :math:`d \\times s_v`, in arcsec units, using data type flt)"""


@dataclass(kw_only=True)
class Pyramid(WavefrontSensor):
    """Contains data related to one Pyramid wavefront sensor used by the system."""
    dimensions: int = 4
    """Number of dimensions being measured by each subaperture. For `Pyramid` this may be equal to 2 (if the signals are
    interpreted as horizontal and vertical offsets), 1 (if the subapertures overlap and are interpreted as a single
    signal) or as the number of sides of the pyramid (that is, `n_sides` signals). (in count units)"""

    n_sides: int
    'Number of pyramid sides (typically 4).  (in count units)'

    modulation: float = None
    'Modulation amplitude.  (in m units)'
