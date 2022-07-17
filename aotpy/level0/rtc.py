from dataclasses import dataclass, field

from .optical_sensor import WavefrontSensor
from .wavefront_corrector import WavefrontCorrector
from .image import Image

__all__ = ['Loop', 'ControlLoop', 'OffloadLoop', 'RTC']


@dataclass
class Loop:
    name: str
    commanded_corrector: WavefrontCorrector
    commands: Image = None
    ref_commands: Image = None
    timestamps: list[float] = field(default_factory=list)
    framerate: float = None
    delay: float = None
    time_filter_num: Image = None
    time_filter_den: Image = None


@dataclass
class ControlLoop(Loop):
    input_wfs: WavefrontSensor = None    # TODO this should be mandatory
    residual_wavefront: Image = None
    control_matrix: Image = None
    interaction_matrix: Image = None


@dataclass
class OffloadLoop(Loop):
    input_corrector: WavefrontCorrector = None     # TODO this should be mandatory
    offload_matrix: Image = None


@dataclass
class RTC:
    lookup_tables: list[Image] = field(default_factory=list)
    non_common_path_aberrations: list[Image] = field(default_factory=list)

    loops: list[Loop] = field(default_factory=list)
