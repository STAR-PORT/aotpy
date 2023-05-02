"""
This module contains a class for translating data produced by ESO's CIAO system.
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

import aotpy
from .eso import ESOTranslator

# TODO set image units


class CIAOTranslator(ESOTranslator):
    """Contains functions for translating telemetry data produced by ESO's CIAO system.

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

        with fits.open(path / 'CIAO_LOOP_0001.fits', extname='LoopFrame') as hdus:
            main_hdr = hdus[0].header
            main_loop_frame = hdus['LoopFrame'].data

        self.system = aotpy.AOSystem(
            ao_mode='SCAO',
            strehl_ratio=main_hdr['ESO AOS ATM SR'],
            temporal_error=main_hdr['ESO AOS ATM TERR']
        )
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

        wfs = aotpy.ShackHartmann('WFS', source=ngs, n_valid_subapertures=68,
                                  detector=aotpy.Detector('SAPHIRA'))

        gradients = main_loop_frame['Gradients']
        # AOF gradients are ordered tip1, tilt1, tip2, tilt2, etc., so even numbers are tip and odd numbers are tilt
        # We separate them, select the valid subapertures and then stack them
        tip = gradients[:, ::2]
        tilt = gradients[:, 1::2]
        gradients = np.stack([tip, tilt], axis=1)
        wfs.measurements = aotpy.Image('Gradients', gradients, time=loop_time)

        reference = fits.getdata(path / 'Acq.DET1.REFSLP_0001.fits')[0]
        tip = reference[::2]
        tilt = reference[1::2]
        reference = np.stack([tip, tilt], axis=0)
        wfs.ref_measurements = aotpy.Image('Acq.DET1.REFSLP', reference)

        wfs.subaperture_intensities = aotpy.Image(f'Intensities', main_loop_frame['Intensities'], time=loop_time)

        pix_loop_frame = fits.getdata(path / 'CIAO_PIXELS_0001.fits')
        wfs.detector.pixel_intensities = aotpy.Image(
            'Pixels',
            data=self.get_pixel_data_from_table(pix_loop_frame),
            time=aotpy.Time('Pixel time', frame_numbers=pix_loop_frame['FrameCounter'].tolist())
        )

        wfs.centroiding_algorithm = main_hdr['ESO AOS ACQ CENTROID ALGO']

        ho_dm = aotpy.DeformableMirror('High Order Deformable Mirror (HODM)', telescope=self.system.main_telescope,
                                       n_valid_actuators=60)
        ittm = aotpy.TipTiltMirror('Image Tip-Tilt Mirror (ITTM)', telescope=self.system.main_telescope)

        cm = fits.getdata(path / 'Recn.REC1.CM_0001.fits')
        tip = cm[:, ::2]
        tilt = cm[:, 1::2]
        cm = np.stack([tip, tilt], axis=1)
        ho_cm = cm[: ho_dm.n_valid_actuators]
        tt_cm = cm[ho_dm.n_valid_actuators:]

        s2m = fits.getdata(path / 'RecnOptimiser.S2M_0001.fits')
        if main_hdr['ESO AOS CM MODES CONTROLLED'] != s2m.shape[0]:
            warnings.warn("Keyword 'ESO AOS CM MODES CONTROLLED' does not match modes in measurements to modes matrix")
        tip = s2m[:, ::2]
        tilt = s2m[:, 1::2]
        s2m = np.stack([tip, tilt], axis=1)
        s2m = aotpy.Image('RecnOptimiser.S2M', s2m)

        m2c = fits.getdata(path / 'RecnOptimiser.M2V_0001.fits')
        ho_m2c = m2c[:ho_dm.n_valid_actuators]
        tt_m2c = m2c[ho_dm.n_valid_actuators:]

        ho_im = fits.getdata(path / 'RecnOptimiser.HO_IM_0001.fits')
        tip = ho_im[::2]
        tilt = ho_im[1::2]
        ho_im = np.stack([tip, tilt], axis=1)
        ho_im = aotpy.Image('RecnOptimiser.HO_IM', ho_im)

        ho_loop = aotpy.ControlLoop(
            'HO Loop',
            input_sensor=wfs,
            commanded_corrector=ho_dm,
            time=loop_time,
            commands=aotpy.Image('HODM positions', main_loop_frame['HODM_Positions'], time=loop_time),
            ref_commands=aotpy.Image('HOCtr.ACT_POS_REF_MAP',
                                     fits.getdata(path / 'HOCtr.ACT_POS_REF_MAP_0001.fits')[0]),
            control_matrix=aotpy.Image('HO Control Matrix', ho_cm),
            measurements_to_modes=s2m,
            modes_to_commands=aotpy.Image('HO modes to commands', ho_m2c),
            interaction_matrix=ho_im,
            framerate=main_hdr['ESO AOS LOOP RATE'],
            closed=main_hdr['ESO AOS HO LOOP ST']
        )

        tt_loop = aotpy.ControlLoop(
            'TT Loop',
            input_sensor=wfs,
            commanded_corrector=ittm,
            time=loop_time,
            commands=aotpy.Image('ITTM positions', main_loop_frame['ITTM_Positions'], time=loop_time),
            ref_commands=aotpy.Image('ESO AOS TTM REFPOS',
                                     np.array([main_hdr['ESO AOS TTM REFPOS X'], main_hdr['ESO AOS TTM REFPOS Y']])),
            control_matrix=aotpy.Image('TT Control Matrix', tt_cm),
            measurements_to_modes=s2m,
            modes_to_commands=aotpy.Image('TT modes to commands', tt_m2c),
            framerate=main_hdr['ESO AOS LOOP RATE'],
            closed=main_hdr['ESO AOS TT LOOP ST']
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

        aos = aotpy.AtmosphericParameters(
            'AO System',
            fwhm=[main_hdr['ESO AOS ATM SEEING']],
            tau0=[main_hdr['ESO AOS ATM TAU0']],
        )

        self.system.sources = [ngs]
        self.system.wavefront_sensors = [wfs]
        self.system.wavefront_correctors = [ho_dm, ittm]
        self.system.loops = [ho_loop, tt_loop]
        self.system.atmosphere_params = [asm, aos]

    def _get_eso_telescope_name(self) -> str:
        return f"ESO-VLTI-A{self._at_number}"

    def _get_eso_ao_name(self) -> str:
        return 'CIAO'
