"""
This module contains a class that defines an adaptive optics system.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .atmosphere import AtmosphericParameters
from .base import Metadatum
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

    name: str = None
    'Name of the AO system that produced the data.'

    strehl_ratio: float = None
    'Estimated mean Strehl ratio for the duration of the observation (arcsec).'

    strehl_wavelength: float = None
    'Reference wavelength for the Strehl ratio estimation (m).'

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

    metadata: list[Metadatum] = field(default_factory=list)
    """List of Metadatum objects that are associated with the overall system."""

    def write_to_file(self, filename: str | os.PathLike, file_type: str = None, **kwargs) -> None:
        """
        Writes `AOSystem` to a file using the correct writing function for that file type. The file type is deduced by
        the extension in the specified `filename` but it may instead be directly set via the `file_type` parameter.

        Parameters
        ----------
        filename
            Path to the file to be written.
        file_type
            The type of the file to be written. By default, it attempts to determine the type from the file extension.
        kwargs
            Optional keyword arguments passed on as options to the writer function.
        """
        if file_type is None:
            p = Path(filename)
            ext = p.suffix[1:]
            if not ext:
                raise ValueError(f"'{filename}' does not have a file extension. If you know the correct file type, you "
                                 f"may specify it through the 'file_type' argument. "
                                 f"Available types: {str(list(_AVAILABLE_WRITERS.keys()))[1:-1]}")
            if ext.lower() in ['.zip', '.tar', '.gz', '.tgz'] and len(p.suffixes) > 1:
                ext = p.suffixes[0]
        else:
            ext = file_type
        if (e := ext.lower()) in _AVAILABLE_WRITERS:
            _AVAILABLE_WRITERS[e](self).write(filename, **kwargs)
        else:
            raise ValueError(f"No available writer for file type '{ext}'. "
                             f"Available types: {str(list(_AVAILABLE_WRITERS.keys()))[1:-1]}")

    @staticmethod
    def read_from_file(filename: str | os.PathLike, file_type: str = None, **kwargs) -> 'AOSystem':
        """
        Reads `AOSystem` from a file using the correct reading function for that file type. The file type is deduced by
        the extension in the specified `filename` but it may instead be directly set via the `file_type` parameter.

        Parameters
        ----------
        filename
            Path to the file to be read.
        file_type
            The type of the file to be read. By default, it attempts to determine the type from the file extension.
        kwargs
            Optional keyword arguments passed on as options to the reader function.
        """
        if file_type is None:
            ext = Path(filename).suffix[1:]
            if not ext:
                raise ValueError(f"'{filename}' does not have a file extension. You may pick a filename with an "
                                 f"extension or specify the type through the 'file_type' argument. "
                                 f"Available types: {str(list(_AVAILABLE_READERS.keys()))[1:-1]}")
        else:
            ext = file_type
        if (e := ext.lower()) in _AVAILABLE_READERS:
            return _AVAILABLE_READERS[e](filename, **kwargs).get_system()
        else:
            raise ValueError(f"No available reader for file type '{ext}'. "
                             f"Available types: {str(list(_AVAILABLE_READERS.keys()))[1:-1]}")

    def __str__(self) -> str:
        if self.name:
            out = self.name
        else:
            out = 'Unnamed AOSystem'
        out += " telemetry data recording "

        if self.date_beginning is not None:
            out += f"started at {self.date_beginning} "
        if self.date_end is not None:
            out += f"ended at {self.date_beginning} "
        out += f"(configuration '{self.ao_mode}')."
        out += f"\n\tMain Telescope: '{self.main_telescope.uid}'"
        aux = (
            (self.atmosphere_params, 'Atmospheric Parameters'),
            (self.sources, 'Light Sources'),
            (self.scoring_cameras, 'Scoring Cameras'),
            (self.wavefront_sensors, 'Wavefront Sensors'),
            (self.wavefront_correctors, 'Wavefront Correctors'),
            (self.loops, 'System Loops')
        )
        for refs, name in aux:
            if refs:
                out += f'\n\t{name} ({len(refs)}):'
                for i, ref in enumerate(refs):
                    out += f"\n\t\t{i}: '{ref.uid}' ({type(ref).__name__})"

        return out
