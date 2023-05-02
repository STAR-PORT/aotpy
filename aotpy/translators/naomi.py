"""
This module contains a class for translating data produced by ESO's NAOMI system.
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

import aotpy
from aotpy.io import image_from_file
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
            main_loop_frame = hdus['LoopFrame'].data

        self.system = aotpy.AOSystem(ao_mode='SCAO')
        self.system.main_telescope = aotpy.MainTelescope(
            uid=f'ESO VLT AT{at_number}',
            elevation=main_hdr['ESO TEL ALT'],
            azimuth=self.azimuth_conversion(main_hdr['ESO TEL AZ']),
            parallactic=main_hdr['ESO TEL PRLTIC']
        )

        ngs = aotpy.NaturalGuideStar('NGS',
                                     right_ascension=main_hdr['RA'],
                                     declination=main_hdr['DEC'])

        main_timestamps = main_loop_frame['Seconds'] + main_loop_frame['USeconds'] / 1.e6
        self.system.date_beginning = datetime.utcfromtimestamp(main_timestamps[0])
        self.system.date_end = datetime.utcfromtimestamp(main_timestamps[-1])
        main_frame_numbers = main_loop_frame['FrameCounter']
        loop_time = aotpy.Time('Loop Time', timestamps=main_timestamps.tolist(),
                               frame_numbers=main_frame_numbers.tolist())

        wfs = aotpy.ShackHartmann('WFS', source=ngs, n_valid_subapertures=12,
                                  detector=aotpy.Detector('DET'))

        gradients = main_loop_frame['Gradients']
        # AOF gradients are ordered tip1, tilt1, tip2, tilt2, etc., so even numbers are tip and odd numbers are tilt
        # We separate them, select the valid subapertures and then stack them
        tip = gradients[:, ::2]
        tilt = gradients[:, 1::2]
        gradients = np.stack([tip, tilt], axis=1)

        reference = fits.getdata(path / 'Acq.DET1.REFSLP_WITH_OFFSETS_0001.fits')[0]
        tip = reference[::2]
        tilt = reference[1::2]
        reference = np.stack([tip, tilt], axis=0)

        wfs.measurements = aotpy.Image('Gradients', gradients, time=loop_time)
        wfs.ref_measurements = aotpy.Image('Acq.DET1.REFSLP_WITH_OFFSETS', reference)
        wfs.subaperture_intensities = aotpy.Image(f'Intensities', main_loop_frame['Intensities'], time=loop_time)

        wfs.detector.weight_map = image_from_file(path / 'Acq.DET1.WEIGHT_0001.fits')
        wfs.detector.dark = image_from_file(path / 'Acq.DET1.DARK_0001.fits')
        wfs.detector.flat_field = image_from_file(path / 'Acq.DET1.FLAT_0001.fits')
        wfs.detector.bad_pixel_map = image_from_file(path / 'Acq.DET1.DEAD_0001.fits')
        wfs.detector.sky_background = image_from_file(path / 'Acq.DET1.BACKGROUND_0001.fits')

        pix_loop_frame = fits.getdata(path / 'NAOMI_PIXELS_0001.fits')
        wfs.detector.pixel_intensities = aotpy.Image(
            'Pixels',
            data=self.get_pixel_data_from_table(pix_loop_frame),
            time=aotpy.Time('Pixel time', frame_numbers=pix_loop_frame['FrameCounter'].tolist())
        )

        dm = aotpy.DeformableMirror('DM', telescope=self.system.main_telescope, n_valid_actuators=241)

        loop = aotpy.ControlLoop('Main Loop', input_sensor=wfs, commanded_corrector=dm, time=loop_time)
        loop.time_filter_num = aotpy.Image('Ctr.TERM_A', fits.getdata(path / 'Ctr.TERM_A_0001.fits'))
        loop.time_filter_den = aotpy.Image('Ctr.TERM_B', fits.getdata(path / 'Ctr.TERM_B_0001.fits'))

        loop.commands = aotpy.Image('DM positions', main_loop_frame['Positions'], time=loop_time)
        # loop.ref_commands = image_from_file(path / 'Ctr.ACT_POS_REF_MAP_0001.fits')
        # loop.ref_commands.data = loop.ref_commands.data[0]
        loop.ref_commands = aotpy.Image('Ctr.ACT_POS_REF_MAP', fits.getdata(path / 'Ctr.ACT_POS_REF_MAP_0001.fits'))

        derotation_matrix = fits.getdata(path / 'RecnOptimiser.ROTATION_MATRIX_0001.fits').T

        s2m = fits.getdata(path / 'Recn.REC1.CM_0001.fits')
        s2m = s2m.T @ derotation_matrix
        tip = s2m[::2]
        tilt = s2m[1::2]
        s2m = np.stack([tip, tilt], axis=1).T
        loop.measurements_to_modes = aotpy.Image('Recn.REC1.CM', s2m)

        loop.modes_to_commands = image_from_file(path / 'RTC.M2DM_SCALED_0001.fits')

        if main_hdr['ESO AOS CM MODES CONTROLLED'] != s2m.shape[0]:
            warnings.warn("Keyword 'ESO AOS CM MODES CONTROLLED' does not match modes in control matrix")

        loop.commands_to_modes = image_from_file(path / 'RTC.DM2M_SCALED_0001.fits')
        m2s = fits.getdata(path / 'ModalRecnCalibrat.REF_IM_0001.fits')
        tip = m2s[::2]
        tilt = m2s[1::2]
        m2s = np.stack([tip, tilt], axis=1)
        loop.modes_to_measurements = aotpy.Image('ModalRecnCalibrat.REF_IM', m2s)

        loop.closed = main_hdr['ESO AOS LOOP ST']
        loop.framerate = main_hdr['ESO AOS LOOP RATE']

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
        return f"ESO-VLTI-A{self._at_number}"

    def _get_eso_ao_name(self) -> str:
        return 'NAOMI'
