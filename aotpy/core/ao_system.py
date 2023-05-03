"""
This module contains a class that defines an adaptive optics system.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .atmosphere import AtmosphericParameters
from .loop import Loop
from .optical_sensor import WavefrontSensor, ScoringCamera
from .source import Source
from .telescope import MainTelescope
from .wavefront_corrector import WavefrontCorrector
from .. import _AVAILABLE_WRITERS, _AVAILABLE_READERS

__all__ = ['AOSystem']


@dataclass(kw_only=True)
class AOSystem:
    """Contains all the relevant information about an adaptive optics system.
    Different parts of the system can be accessed via the lists of objects it contains."""

    ao_mode: str = None
    """Describes the system's AO configuration.
    
    Must be one of ``'SCAO'``, ``'SLAO'``, ``'GLAO'``, ``'MOAO'``, ``'LTAO'`` or ``'MCAO'`` (which respectively stand
    for Single Conjugate, Single Laser, Ground Layer, Multi-Object, Laser Tomography and Multi-Conjugate Adaptive
    Optics)."""

    date_beginning: datetime = None
    'Start time of data acquisition.'

    date_end: datetime = None
    'Stop time of data acquisition.'

    strehl_ratio: float = None
    'Estimated strehl ratio (arcsec).'

    temporal_error: float = None
    'Estimated temporal error TODO: (units?).'

    config: str = None
    'Free-form text that describes configuration parameters of the system.'

    atmosphere_params: list[AtmosphericParameters] = field(default_factory=list)
    """List of all atmospheric data related to the system."""

    main_telescope: MainTelescope = None
    """The main telescope of the system."""

    sources: list[Source] = field(default_factory=list)
    """List of all the light sources in the system."""

    scoring_cameras: list[ScoringCamera] = field(default_factory=list)
    """List of all the scoring cameras in the system."""

    wavefront_sensors: list[WavefrontSensor] = field(default_factory=list)
    """List of all wavefront sensors in the system."""

    wavefront_correctors: list[WavefrontCorrector] = field(default_factory=list)
    """List of all wavefront correctors in the system."""

    loops: list[Loop] = field(default_factory=list)
    """List of all loops in the system."""

    def write_to_file(self, filename: str | os.PathLike, **kwargs) -> None:
        """
        Writes `AOSystem` to a file. The writing function is deduced by the extension in the specified `filename`.

        Parameters
        ----------
        filename
            Path to the file to be written.
        kwargs
            Optional keyword arguments passed on as options to the writer function.
        """
        ext = Path(filename).suffix[1:]
        try:
            WriterClass = _AVAILABLE_WRITERS[ext.lower()]
        except KeyError:
            raise ValueError(f"No available writer for extension '{ext}'. "
                             f"Available extensions: {str(list(_AVAILABLE_WRITERS.keys()))[1:-1]}")
        WriterClass(self).write(filename, **kwargs)

    @staticmethod
    def read_from_file(filename: str | os.PathLike, **kwargs) -> 'AOSystem':
        """
        Reads `AOSystem` from a file. The reading function is deduced by the extension in the specified `filename`.

        Parameters
        ----------
        filename
            Path to the file to be read.
        kwargs
            Optional keyword arguments passed on as options to the reader function.
        """
        ext = Path(filename).suffix[1:]
        try:
            ReaderClass = _AVAILABLE_READERS[ext.lower()]
        except KeyError:
            raise ValueError(f"No available reader for extension '{ext}'. "
                             f"Available extensions: {str(list(_AVAILABLE_READERS.keys()))[1:-1]}")
        return ReaderClass(filename, **kwargs).get_system()
