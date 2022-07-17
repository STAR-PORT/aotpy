from dataclasses import dataclass, field
from datetime import datetime

__all__ = ['AtmosphericParameters', 'AtmosphereLayer']


@dataclass
class AtmosphereLayer:
    weight: float = None
    height: float = None
    wind_speed: float = None
    wind_direction: float = None


@dataclass
class AtmosphericParameters:
    data_source: str
    timestamp: datetime = None
    wavelength: float = None
    r0: float = None  # Fried Parameter
    l0: float = None  # Outer scale of turbulence
    layers: list[AtmosphereLayer] = field(default_factory=list)
