"""
This module contains a base class for translating data produced by ESO systems.

It also provides tools that enable users to automatically add the necessary metadata for compatibility with ESO's
science archive. This is done by querying the archive via programmatic access (pyvo is necessary for this
functionality).

"""

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

from ..core.base import Metadatum
from .base import BaseTranslator

ESO_TAP_OBS = "https://archive.eso.org/tap_obs"


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
            'INSTRUME': self._get_eso_ao_name(),
            'HIERARCH ESO OBS PROG ID': self._get_run_id(),
            'TELESCOP': telescope.replace('%', ''),
            'DATE': datetime.now().isoformat(timespec='milliseconds'),
            'OBJECT': 'AO-TELEM',
            'OBSERVER': 'I, Condor',
            'DATE-OBS': self.system.date_beginning.astimezone(timezone.utc).replace(tzinfo=None).isoformat(
                timespec='milliseconds'),
            'MJD-OBS': Time(self.system.date_beginning, scale='utc').mjd,
            'MJD-END': Time(self.system.date_end, scale='utc').mjd,
            'EXPTIME': (self.system.date_end - self.system.date_beginning).total_seconds(),
            'HIERARCH ESO DPR CATG': 'CALIB',
            'HIERARCH ESO DPR TYPE': f'AO-TELEM,AOT,{self._get_eso_ao_name()}',
            'HIERARCH ESO DPR TECH': self.system.ao_mode,
        }
        if self.system.main_telescope.azimuth is not None:
            metadata['HIERARCH ESO TEL AZ'] = self.system.main_telescope.azimuth
        if self.system.main_telescope.elevation is not None:
            metadata['HIERARCH ESO TEL ALT'] = self.system.main_telescope.elevation

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
                    'HIERARCH ESO OBS PROG ID': res['prog_id'],
                    'HIERARCH ESO INS MODE': res['ins_mode'],
                }
                if 'HIERARCH ESO TEL AZ' not in metadata:
                    metadata['HIERARCH ESO TEL AZ'] = res['tel_az']
                if 'HIERARCH ESO TEL ALT' not in metadata:
                    metadata['HIERARCH ESO TEL ALT'] = res['tel_alt']
            else:
                warnings.warn(f"Could not find data from telescope '{telescope}' near mjd_obs {beg.mjd} at the "
                              f"ESO Archive")
        self.system.metadata.extend([Metadatum(k, v) for k, v in metadata.items()])

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
            ESO azimuth to be converted
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
    def _stack_slopes(data: np.ndarray, slope_axis: int) -> np.ndarray:
        # ESO slopes are ordered tip1, tilt1, tip2, tilt2, etc., so even numbers are tip and odd numbers are tilt.
        # We separate and then stack them.
        if slope_axis == 0:
            tip = data[::2]
            tilt = data[1::2]
        elif slope_axis == 1:
            tip = data[:, ::2]
            tilt = data[:, 1::2]
        else:
            raise NotImplementedError
        return np.stack([tip, tilt], axis=1)
