from dataclasses import dataclass, field
from datetime import datetime

from .atmosphere import AtmosphericParameters
from .optical_sensor import WavefrontSensor, ScoringCamera
from .rtc import RTC
from .source import Source
from .telescope import MainTelescope
from .wavefront_corrector import WavefrontCorrector

__all__ = ['AOSystem']


@dataclass
class AOSystem:
    level: int = field(init=False, default=0)
    start_datetime: datetime = None
    end_datetime: datetime = None
    ao_mode: str = None

    sources: list[Source] = field(default_factory=list)
    atmosphere_params: list[AtmosphericParameters] = field(default_factory=list)
    telescope: MainTelescope = None
    scoring_cameras: list[ScoringCamera] = field(default_factory=list)
    wavefront_sensors: list[WavefrontSensor] = field(default_factory=list)
    rtc: RTC = None
    wavefront_correctors: list[WavefrontCorrector] = field(default_factory=list)
