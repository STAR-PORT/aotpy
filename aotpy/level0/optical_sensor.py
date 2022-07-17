from dataclasses import dataclass, field


from .source import Source
from .image import Image

__all__ = ['Detector', 'OpticalRelay', 'OpticalAberration', 'ScoringCamera', 'WavefrontSensor', 'ShackHartmann',
           'Pyramid']


@dataclass
class Detector:
    name: str
    flat_field: Image = None
    readout_noise: float = None
    pixel_intensities: Image = None
    integration_time: float = None
    coadds: int = None
    dark: Image = None
    weight_map: Image = None
    quantum_efficiency: float = None
    pixel_scale: float = None
    binning: int = None
    bandwidth: float = None
    transmission_wavelength: list[float] = field(default_factory=list)
    transmission: list[float] = field(default_factory=list)
    sky_background: Image = None
    gain: float = None
    excess_noise: float = None
    filter: str = None
    bad_pixel_map: Image = None


@dataclass
class OpticalRelay:
    name: str
    field_of_view: float = None
    focal_length: float = None


@dataclass
class OpticalAberration:
    name: str
    coefficients: list[float] = field(default_factory=list)
    modes: Image = None
    pupil: Image = None


@dataclass
class ScoringCamera:
    name: str
    pupil: Image = None
    theta: float = None
    wavelength: float = None

    frame: Image = None
    field_static_map: Image = None
    x_stat: list[float] = field(default_factory=list)
    y_stat: list[float] = field(default_factory=list)

    detector: Detector = None
    optical_relay: OpticalRelay = None
    optical_aberration: OpticalAberration = None


@dataclass
class WavefrontSensor:
    name: str
    source: Source
    slopes: Image = None
    ref_slopes: Image = None
    theta: float = None
    wavelength: float = None
    d_wavelength: float = None
    n_subapertures: int = None
    valid_subapertures: Image = None
    subaperture_size: float = None
    subaperture_intensities: Image = None
    pupil_angle: float = None
    algorithm: str = None
    optical_gain: float = None
    centroid_gains: Image = None

    detector: Detector = None
    optical_relay: OpticalRelay = None
    optical_aberration: OpticalAberration = None


@dataclass
class ShackHartmann(WavefrontSensor):
    spot_fwhm: Image = None


@dataclass
class Pyramid(WavefrontSensor):
    modulation: float = None
