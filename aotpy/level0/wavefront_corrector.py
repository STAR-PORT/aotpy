from dataclasses import dataclass, field

from .optical_sensor import OpticalAberration
from .telescope import Telescope
from .image import Image

__all__ = ['WavefrontCorrector', 'DeformableMirror', 'TipTiltMirror', 'LinearStage']


@dataclass
class WavefrontCorrector:
    name: str
    telescope: Telescope
    pitch: float = None
    n_actuators: int = None

    tfz_num: list[float] = field(default_factory=list)
    tfz_den: list[float] = field(default_factory=list)

    optical_aberration: OpticalAberration = None



@dataclass
class DeformableMirror(WavefrontCorrector):
    valid_actuators: Image = None
    influence_function: Image = None


@dataclass
class TipTiltMirror(WavefrontCorrector):
    n_actuator: int = field(init=False, default=2)
    pass


@dataclass
class LinearStage(WavefrontCorrector):
    n_actuator: int = field(init=False, default=1)
    pass
