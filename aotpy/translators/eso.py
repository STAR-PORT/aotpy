"""
This module contains a base class for translating data produced by ESO systems.

It also provides tools that enable users to automatically add the necessary metadata for compatibility with ESO's
science archive. This is done by querying the archive via programmatic access (pyvo is necessary for this
functionality).

"""

import os
import pathlib
import re
import warnings
from abc import abstractmethod
from datetime import timedelta, datetime, timezone

import astropy.table
import numpy as np
from astropy.io import fits
from astropy.time import Time

try:
    from pyvo.dal import tap
except (ImportError, ModuleNotFoundError):
    tap = None

import aotpy
from aotpy.io.fits import image_from_fits_file, metadata_from_hdu
from .base import BaseTranslator

ESO_TAP_OBS = "https://archive.eso.org/tap_obs"
LOG_PATTERN = re.compile(r'(?P<event_id>\d+) (?P<object>[^ ]+) (?P<event>[^ ]+) (?P<frame>\d+)\n')


class ESOTranslator(BaseTranslator):
    """Abstract class for translators for ESO systems.

    Translators are able to convert non-standard AO telemetry data files into an `AOSystem` object.
    """

    @abstractmethod
    def _get_eso_telescope_name(self) -> str:
        """
        Get value for TELESCOP keyword in ESO's science archive.

        Possible values listed below, according to ESO's Data Interface Control Document.

        --- ESO telescopes ---

        =====================   =============================================================================
        TELESCOP                Telescope
        =====================   =============================================================================
        ESO-NTT                 ESO 3.5-m New Technology Telescope
        ESO-3.6 or ESO-3P6      ESO 3.6-m Telescope
        ESO-VLT-Ui              ESO VLT, Unit Telescope i
        ESO-VLT-Uijkl           ESO VLT, incoherent combination of Unit Telescopes ijkl
        ESO-VLTI-Uijkl          ESO VLT, coherent combination of Unit Telescopes ijkl
        ESO-VLTI-Amnop          ESO VLT, coherent combination of Auxiliary Telescopes mnop
        ESO-VLTI-Uijkl-Amnop    ESO VLT, coherent combination of UTs ijkl and Auxiliary Telescopes mnop
        ESO-VST                 ESO 2.6-m VLT Survey Telescope
        VISTA                   ESO 4-meter Visible and Infrared Telescope for Astronomy
        Sky Monitor             All-Sky Monitor of the Paranal Observatory (MASCOT)
        APEX-12m                Atacama Pathfinder Experiment
        ESO-ELT                 ESO Extremely Large Telescope
        =====================   =============================================================================

        --- Hosted telescopes ---

        =====================   =============================================================================
        TELESCOP                Telescope
        =====================   =============================================================================
        MPI-2.2                 MPI 2.2-m Telescope
        SPECULOOS-<name>        1-m SPECULOOS Telescopes, <name> is one of Galilean moons of Jupiter
        TRAPPIST-S              TRAPPIST South 60-cm Telescope
        APICAM                  ApiCam-3 Fisheye Telescope
        =====================   =============================================================================

        --- Non-ESO telescopes ---

        =====================   =============================================================================
        TELESCOP                Telescope
        =====================   =============================================================================
        UKIRT                   3.6-m United Kingdom Infra-Red Telescope
        WHT                     4.2-m William Herschel Telescope
        =====================   =============================================================================
        """
        pass

    @abstractmethod
    def _get_eso_ao_name(self) -> str:
        """
        Get abbreviation of the adaptive optics system's name, as defined by ESO.
        """
        pass

    @abstractmethod
    def _get_run_id(self) -> str:
        """
        Get the run ID (prog ID) that refers to the data, as defined by ESO.
        """
        pass

    @abstractmethod
    def _get_chip_id(self) -> str:
        """
        Get ESO DET CHIP1 ID for the specific data. This is an ESO archive requirement to ensure compatibility
        when there are simultaneous recordings of the same instrument on different telescopes.
        """
        pass

    def add_archive_metadata(self, query_archive: bool = False) -> None:
        """
        Adds necessary metadata for ESO Archive to AOSystem.

        Parameters
        ----------
        query_archive : default = False
            Whether the ESO archive should be queried to find relevant metadata already in the archive. Requires pyvo.
        """
        telescope = self._get_eso_telescope_name()
        metadata = {
            'ORIGIN': 'ESO-PARANAL',
            'DATE': datetime.now().isoformat(timespec='milliseconds'),
            'TELESCOP': telescope.replace('%', ''),
            'INSTRUME': self._get_eso_ao_name(),
            'ESO DET CHIP1 ID': self._get_chip_id(),
            'ESO OBS PROG ID': self._get_run_id(),
            'OBJECT': 'AO-TELEM',
            'OBSERVER': 'I, Condor',
            'DATE-OBS': self.system.date_beginning.astimezone(timezone.utc).replace(tzinfo=None).isoformat(
                timespec='milliseconds'),
            'MJD-OBS': Time(self.system.date_beginning, scale='utc').mjd,
            'MJD-END': Time(self.system.date_end, scale='utc').mjd,
            'EXPTIME': (self.system.date_end - self.system.date_beginning).total_seconds(),
            'ESO DPR CATG': 'CALIB',
            'ESO DPR TYPE': f'AO-TELEM,AOT,{self._get_eso_ao_name()}',
            'ESO DPR TECH': self.system.ao_mode,
        }
        if self.system.main_telescope.azimuth is not None:
            metadata['ESO TEL AZ'] = self.system.main_telescope.azimuth
        if self.system.main_telescope.elevation is not None:
            metadata['ESO TEL ALT'] = self.system.main_telescope.elevation

        if query_archive:
            if tap is None:
                raise ImportError("Querying the ESO archive requires the pyvo module."
                                  "You can set the 'query_archive' option to False to skip querying the archive.")
            beg = Time(self.system.date_beginning, scale='utc')

            if '%' in telescope:
                tel_comparator = " LIKE "
            else:
                tel_comparator = "="

            delta = timedelta(hours=1)
            query = f"""
            SELECT TOP 1 ABS(mjd_obs - {beg.mjd}) as diff, *
            FROM dbo.raw
            WHERE mjd_obs between {(beg - delta).mjd} and {(beg + delta).mjd}
                and telescope{tel_comparator}'{telescope}'
                and not dp_type LIKE '%AOT%'
            ORDER BY 1 ASC
            """
            res = tap.TAPService(ESO_TAP_OBS).search(query).to_table()
            if res:
                res = res[0]
                metadata |= {
                    'INSTRUME': res['instrument'],
                    'RA': res['ra'],
                    'DEC': res['dec'],
                    # 'PI-COI': res['pi_coi'],
                    # we do not store PI-COI because it is not needed and can cause issues with special chars
                    'ESO OBS PROG ID': res['prog_id'],
                    'ESO INS MODE': res['ins_mode'],
                }
                if 'ESO TEL AZ' not in metadata:
                    metadata['ESO TEL AZ'] = res['tel_az']
                if 'ESO TEL ALT' not in metadata:
                    metadata['ESO TEL ALT'] = res['tel_alt']
            else:
                warnings.warn(f"Could not find data from telescope '{telescope}' near mjd_obs {beg.mjd} at the "
                              f"ESO Archive")
        self.system.metadata.extend([aotpy.Metadatum(k, v) for k, v in metadata.items()])

    def get_atmospheric_parameters_from_archive(self) -> astropy.table.Table:
        """
        Get atmospheric data from ESO's Science Archive within the recording period.

        Requires pyvo.
        """
        if tap is None:
            raise ImportError("Querying the ESO archive requires the pyvo module.")

        delta = timedelta(minutes=1)
        beg = (self.system.date_beginning - delta).isoformat(timespec='milliseconds')
        end = (self.system.date_end + delta).isoformat(timespec='milliseconds')

        query = f"""
        SELECT *
        FROM asm.meteo_paranal
        WHERE midpoint_date between '{beg}' and '{end}'
        AND valid=1
        """
        res = tap.TAPService(ESO_TAP_OBS).search(query).to_table()
        return res

    @staticmethod
    def _azimuth_conversion(az: float):
        """
        Convert azimuth from ESO's reference frame to AOT's reference frame.

        ESO's azimuth is measured westwards from the south, while in AOT it is defined as being measured from the
        eastward from the north.

        Parameters
        ----------
        az
            ESO azimuth to be converted.
        """
        # We need to subtract the angle between north and south and then apply symmetry.
        return -(az - 180) % 360

    @staticmethod
    def _get_pixel_data_from_table(pix_frame: fits.FITS_rec) -> np.ndarray:
        """
        Get properly reshaped pixel data from FITS binary table data.

        Parameters
        ----------
        pix_frame
            Binary table data containing pixel image.
        """
        sizes_x = pix_frame['WindowSizeX']
        sizes_y = pix_frame['WindowSizeY']
        if np.any(sizes_x != sizes_x[0]) or np.any(sizes_y != sizes_y[0]):
            warnings.warn('Pixel window size seems to change over time.')
        sizes_x = sizes_x[0]
        sizes_y = sizes_y[0]

        return pix_frame['Pixels'][:, :sizes_x * sizes_y].reshape(-1, sizes_x, sizes_y)

    @staticmethod
    def _stack_slopes(data: np.ndarray, subap_axis: int) -> np.ndarray:
        # ESO slopes are ordered tip1, tilt1, tip2, tilt2, etc., so even numbers are tip and odd numbers are tilt.
        # We separate and then stack them.
        if subap_axis == 0:
            tip = data[::2]
            tilt = data[1::2]
        elif subap_axis == 1:
            tip = data[:, ::2]
            tilt = data[:, 1::2]
        else:
            raise NotImplementedError
        return np.stack([tip, tilt], axis=1)

    @staticmethod
    def _get_time_from_table(name: str, frames: fits.FITS_rec) -> aotpy.Time:
        timestamps_list = []
        try:
            timestamps = frames['Seconds'] + frames['USeconds'] / 1.e6
            if not np.all(timestamps == 0):
                # For some reason, some files seem to have all Seconds/USeconds set to 0
                # In that case we do not consider them to be valid
                timestamps_list = timestamps.tolist()
        except KeyError:
            # Some frames don't have timestamps, only frame counters (typically pixels)
            pass
        try:
            # If there's a "HOFrameCounter" we are on a LO loop, and we want to use the HO fcs
            fcs = frames['HOFrameCounter']
        except KeyError:
            fcs = frames['FrameCounter']

        return aotpy.Time(name, timestamps=timestamps_list, frame_numbers=fcs.tolist())

    @staticmethod
    def _fix_pointing_format(time: float):
        # TODO test this and add to translators
        # Old ESO pointing data was stored using a pseudo-float format such that e.g. 100526.90157 -> +10h 05m 26.90157s
        # This function converts from this format into decimal degrees
        t = str(time)
        return int(t[:2]) + int(t[2:4]) / 60 + float(t[4:]) / 3600

    @staticmethod
    def _image_from_eso_file(filename: str | os.PathLike) -> aotpy.Image:
        image = image_from_fits_file(filename)
        image.metadata = [md for md in image.metadata if not md.key.startswith("ESO DPR") and md.key != 'ORIGIN' and
                          md.key != 'RA' and md.key != 'DEC']
        return image

    @classmethod
    def _image_from_object(cls, path: pathlib.Path, obj: str, time: aotpy.Time, *,
                           subap_axis: int = None, subap_mask: np.ndarray = None,
                           act_axis: int = None, act_mask: np.ndarray = None) -> aotpy.Image:
        pattern = re.compile(fr'{obj}(_(?P<event_id>\d+))?.fits')

        obj_files = [(f.name, match.group('event_id')) for f in path.iterdir() if (match := pattern.fullmatch(f.name))]
        first_fc = None
        if len(obj_files) > 1:
            if time is None:
                raise NotImplementedError
            first_fc = time.frame_numbers[0]

        metadata = None
        arrays = []
        event_ids = []
        subap_axis_after = 0 if subap_axis == 0 else 2
        for file, event_id in obj_files:
            with fits.open(path / file) as hdus:
                data: np.ndarray = hdus[0].data
                if act_axis is not None:
                    data = data[tuple(act_mask if i == act_axis else slice(None) for i in range(data.ndim))]
                if subap_axis is not None:
                    data = cls._stack_slopes(data, subap_axis)
                    data = data[tuple(subap_mask if i == subap_axis_after else slice(None) for i in range(data.ndim))]
                arrays.append(np.squeeze(data)) # remove dimensions of size 1
                if metadata is None:
                    # Only get metadata from the first
                    metadata = [md for md in metadata_from_hdu(hdus[0]) if not md.key.startswith("ESO DPR") and
                                md.key != 'ORIGIN' and md.key != 'RA' and md.key != 'DEC']
            event_ids.append(event_id)
        if len(arrays) > 1:
            logs: dict[str, list[dict]] = {}
            for filepath in path.iterdir():
                if filepath.name.endswith('.log'):
                    with open(filepath) as f:
                        for line in f:
                            if match := LOG_PATTERN.fullmatch(line):
                                logs.setdefault(match.group('event_id'), []).append(match.groupdict())
                            else:
                                warnings.warn(f"Invalid log line '{line}' in file '{filepath.name}'")
                                continue

            fcs = []
            for event_id in event_ids:
                if event_id is None:
                    fcs.append(first_fc)
                else:
                    try:
                        for log in logs[event_id]:
                            if log['object'] == obj and log['event'] == 'SCHEDULED_CONFIG':
                                fcs.append(int(log['frame']))
                                break
                        else:
                            fcs.append(None)
                    except KeyError:
                        fcs.append(None)

            return aotpy.Image(obj, np.stack(arrays), metadata=metadata, time=aotpy.Time(f'{obj} time',
                                                                                         frame_numbers=fcs))
        else:
            return aotpy.Image(obj, arrays[0], metadata=metadata)
