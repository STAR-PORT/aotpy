"""
This module contains a class for translating data produced by ESO's NAOMI system.
"""

import importlib.resources
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

import aotpy
from .eso import ESOTranslator


# TODO set image units


class NAOMITranslator(ESOTranslator):
    """Contains functions for translating telemetry data produced by ESO's NAOMI system.

    Parameters
    ----------
    path
        Path to folder containing telemetry data.
    at_number : {1, 2, 3, 4}
        Number of the AT that produced the data.
    """

    def __init__(self, path: str, at_number: int):
        path = Path(path)
        self._at_number = at_number

        with fits.open(path / 'NAOMI_LOOP_0001.fits', extname='LoopFrame') as hdus:
            main_hdr = hdus[0].header
            main_loop_frame: fits.FITS_rec = hdus['LoopFrame'].data

        self.system = aotpy.AOSystem(ao_mode='SCAO', name='NAOMI')
        self.system.main_telescope = aotpy.MainTelescope(
            uid=f'ESO VLT AT{at_number}',
            elevation=main_hdr['ESO TEL ALT'],
            azimuth=self._azimuth_conversion(main_hdr['ESO TEL AZ']),
            parallactic=main_hdr['ESO TEL PRLTIC'],
            enclosing_diameter=1.82,
            inscribed_diameter=1.82
        )
        naomi_data_path = importlib.resources.files('aotpy.data') / 'NAOMI'
        with importlib.resources.as_file(naomi_data_path / 'zernike_control_modes.fits') as p:
            # Load file with the representation of the modes controlled in NAOMI (Zernike modes 2 to 15)
            control_modes = aotpy.Image('CONTROL MODES', fits.getdata(p))

        if main_hdr['ESO AOS CM MODES CONTROLLED'] != control_modes.data.shape[0]:
            warnings.warn("Keyword 'ESO AOS CM MODES CONTROLLED' does not match expected number of control modes.")

        ngs = aotpy.NaturalGuideStar(uid='NGS',
                                     right_ascension=main_hdr['RA'],
                                     declination=main_hdr['DEC'])

        main_timestamps = main_loop_frame['Seconds'] + main_loop_frame['USeconds'] / 1.e6
        self.system.date_beginning = datetime.utcfromtimestamp(main_timestamps[0])
        self.system.date_end = datetime.utcfromtimestamp(main_timestamps[-1])
        main_frame_numbers = main_loop_frame['FrameCounter']
        loop_time = aotpy.Time('Loop Time', timestamps=main_timestamps.tolist(),
                               frame_numbers=main_frame_numbers.tolist())

        gradients = self._stack_slopes(main_loop_frame['Gradients'], slope_axis=1)
        reference = self._stack_slopes(fits.getdata(path / 'Acq.DET1.REFSLP_WITH_OFFSETS_0001.fits'), slope_axis=1)[0]
        wfs = aotpy.ShackHartmann(
            uid='WFS',
            source=ngs,
            n_valid_subapertures=12,
            measurements=aotpy.Image('Gradients', gradients, time=loop_time),
            ref_measurements=aotpy.Image('Acq.DET1.REFSLP_WITH_OFFSETS', reference),
            subaperture_intensities=aotpy.Image(f'Intensities', main_loop_frame['Intensities'], time=loop_time)
        )

        wfs.non_common_path_aberration = aotpy.Aberration(
            uid='NCPA',
            modes=control_modes,
            coefficients=self._image_from_eso_file(path / 'Ctr.MODAL_OFFSETS_ROTATED_0001.fits')  # in DM modal space
        )

        wfs.detector = aotpy.Detector(
            uid='DET',
            weight_map=self._image_from_eso_file(path / 'Acq.DET1.WEIGHT_0001.fits'),
            dark=self._image_from_eso_file(path / 'Acq.DET1.DARK_0001.fits'),
            flat_field=self._image_from_eso_file(path / 'Acq.DET1.FLAT_0001.fits'),
            bad_pixel_map=self._image_from_eso_file(path / 'Acq.DET1.DEAD_0001.fits'),
            sky_background=self._image_from_eso_file(path / 'Acq.DET1.BACKGROUND_0001.fits')
        )

        pix_loop_frame = fits.getdata(path / 'NAOMI_PIXELS_0001.fits')
        wfs.detector.pixel_intensities = aotpy.Image(
            'Pixels',
            data=self._get_pixel_data_from_table(pix_loop_frame),
            time=aotpy.Time('Pixel time', frame_numbers=pix_loop_frame['FrameCounter'].tolist())
        )

        dm = aotpy.DeformableMirror('DM', telescope=self.system.main_telescope, n_valid_actuators=241)

        modal_coefficients = main_loop_frame['ModalCoefficients']
        modal_coefficients += fits.getdata(path / 'Ctr.MODAL_OFFSETS_ROTATED_0001.fits') * 2
        # These are saved in the DM modal space. Need to add the rotated offsets to get the real coefficients that are
        # then sent to the DM after M2DM conversion.
        s2m = self._stack_slopes(fits.getdata(path / 'Recn.REC1.CM_0001.fits'), slope_axis=1)
        # The S2M matrix is already rotated to to DM modes
        m2s = self._stack_slopes(fits.getdata(path / 'ModalRecnCalibrat.REF_IM_0001.fits'), slope_axis=0)

        try:
            ref_commands = aotpy.Image('Ctr.ACT_POS_REF_MAP', fits.getdata(path / 'Ctr.ACT_POS_REF_MAP_0001.fits')[0])
        except FileNotFoundError:
            ref_commands = None
            warnings.warn("Reference commands file not found ('Ctr.ACT_POS_REF_MAP_0001.fits').")

        loop = aotpy.ControlLoop(
            uid='Main Loop',
            input_sensor=wfs,
            commanded_corrector=dm,
            time=loop_time,
            time_filter_num=aotpy.Image('Ctr.TERM_A', fits.getdata(path / 'Ctr.TERM_A_0001.fits')),
            time_filter_den=aotpy.Image('Ctr.TERM_B', fits.getdata(path / 'Ctr.TERM_B_0001.fits')),
            commands=aotpy.Image('DM positions', main_loop_frame['Positions'], time=loop_time),
            ref_commands=ref_commands,
            modes=control_modes,
            modal_coefficients=aotpy.Image('Modal Coefficients', modal_coefficients, time=loop_time),
            measurements_to_modes=aotpy.Image('Recn.REC1.CM', s2m),
            modes_to_commands=self._image_from_eso_file(path / 'RTC.M2DM_SCALED_0001.fits'),
            commands_to_modes=self._image_from_eso_file(path / 'RTC.DM2M_SCALED_0001.fits'),
            modes_to_measurements=aotpy.Image('ModalRecnCalibrat.REF_IM', m2s),
            closed=main_hdr['ESO AOS LOOP ST'],
            framerate=main_hdr['ESO AOS LOOP RATE']
        )

        asm = aotpy.AtmosphericParameters(
            'ESO ASM (Astronomical Site Monitor)',
            wavelength=500e-9,
            fwhm=[main_hdr['ESO TEL AMBI FWHM']],
            tau0=[main_hdr['ESO TEL AMBI TAU0']],
            theta0=[main_hdr['ESO TEL AMBI THETA0']],
            layers_wind_direction=aotpy.Image('ESO TEL AMBI WINDDIR', np.array([[main_hdr['ESO TEL AMBI WINDDIR']]])),
            layers_wind_speed=aotpy.Image('ESO TEL AMBI WINDSP', np.array([[main_hdr['ESO TEL AMBI WINDSP']]]))
        )

        self.system.sources = [ngs]
        self.system.wavefront_sensors = [wfs]
        self.system.wavefront_correctors = [dm]
        self.system.loops = [loop]
        self.system.atmosphere_params = [asm]

    def _get_eso_telescope_name(self) -> str:
        # Allow for both:
        #   ESO-VLTI-Amnop
        #   ESO-VLTI-Uijkl-Amnop
        # as long as mnop contains the correct number
        return f"ESO-VLTI-%A%{self._at_number}%"

    def _get_eso_ao_name(self) -> str:
        return 'NAOMI'

    def _get_run_id(self) -> str:
        return '60.A-9278(D)'

    def _get_chip_id(self) -> str:
        return f'NAOMI{self._at_number}'
