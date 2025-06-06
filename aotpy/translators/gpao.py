"""
This module contains a class for translating data produced by ESO's GRAVITY+ AO system.
"""

import re
import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

import aotpy
from .eso import ESOTranslator


# TODO set image units


class GPAOTranslator(ESOTranslator):
    """Contains functions for translating telemetry data produced by ESO's GRAVITY+ AO system.

    Parameters
    ----------
    path_ngs_loop
        Path to folder containing NGS loop data.
    path_ngs_pix
        Path to folder containing NGS pixel data.
    ut_number : {1, 2, 3, 4}
        Number of the UT that produced the data.
    """

    def __init__(self, path_ngs_loop: str, path_ngs_pix: str, ut_number: int):
        self._ut_number = ut_number
        # TODO support LGS modes
        """
        GPAO has 4 possible modes: NGS_VIS, NGS_IR, LGS_VIS, LGS_IR.
        Depending on the mode we are using different pipeline and different SH configurations
        All LGS modes have a 30x30 LGS SH + one NGS SH (depending on VIS/IR see below)
        All IR modes have one 9x9 NGS SH (HOIR/LOIR pipelines both use the CIAO WFS)
        VIS modes either have a 40x40 NGS SH (if NGS mode, HO pipeline) or 4x4 (if LGS mode, LO pipeline)
        """
        # Get the NGS pipeline, should be HO/HOIR/LO/LOIR
        pipeline = re.match(r".*\d{8}T\d{6}-(?P<Pipeline>.*)LoopRecorder", path_ngs_loop).group("Pipeline")
        path_ngs_loop = Path(path_ngs_loop)

        # TODO ao_mode differs if LGS
        self.system = aotpy.AOSystem(name='GPAO', ao_mode='SCAO')
        self.system.main_telescope = aotpy.MainTelescope(
            uid=f'ESO VLT UT{ut_number}',
            enclosing_diameter=8.2,
            inscribed_diameter=8.2
        )
        act_valid_map = fits.getdata(path_ngs_loop / 'RTC.ACT_VALID_MAP.fits')[0]
        dm = aotpy.DeformableMirror(
            uid='DM',
            telescope=self.system.main_telescope,
            n_valid_actuators=act_valid_map.size,
        )
        self.system.wavefront_correctors.append(dm)

        ngs_loop_frame = fits.getdata(path_ngs_loop / f'{path_ngs_loop.name}.fits', extname=f'{pipeline}LoopFrame')
        ngs_time = self._get_time_from_table(f"{pipeline} Loop Time", ngs_loop_frame)
        self.system.date_beginning = datetime.datetime.fromtimestamp(ngs_time.timestamps[0], datetime.UTC)
        self.system.date_end = datetime.datetime.fromtimestamp(ngs_time.timestamps[-1], datetime.UTC)

        ngs = aotpy.NaturalGuideStar('NGS')
        self.system.sources.append(ngs)

        subap_valid_map = fits.getdata(path_ngs_loop / f'{pipeline}RecnOptimiser.SUBAP_VALID_MAP.fits')[0]
        gradients = self._stack_slopes(ngs_loop_frame['Gradients'], subap_axis=1)[:, :, subap_valid_map]
        ngs_wfs = aotpy.ShackHartmann(
            uid='NGS WFS',
            source=ngs,
            n_valid_subapertures=subap_valid_map.size,
            measurements=aotpy.Image('NGS Gradients', gradients),
            ref_measurements=self._image_from_object(path_ngs_loop, f'{pipeline}Acq.DET1.REFSLP_WITH_OFFSETS', ngs_time,
                                                     subap_axis=1, subap_mask=subap_valid_map),
            subaperture_intensities=aotpy.Image('NGS Intensities', ngs_loop_frame['Intensities'][:, subap_valid_map])
        )
        self.system.wavefront_sensors.append(ngs_wfs)

        path_ngs_pix = Path(path_ngs_pix)
        ngs_pix_frame = fits.getdata(path_ngs_pix / f'{path_ngs_pix.name}.fits', extname=f'{pipeline}PixelFrame')
        ngs_wfs.detector = aotpy.Detector(
            uid='DET1',
            pixel_intensities=aotpy.Image(name=f'{pipeline} Pixels',
                                          data=self._get_pixel_data_from_table(ngs_pix_frame),
                                          time=aotpy.Time(f'{pipeline} Pixel Time',
                                                          frame_numbers=ngs_pix_frame['FrameCounter'].tolist()))
        )
        # ngs_wfs.subaperture_size = ngs_wfs.detector.pixel_intensities.data.shape[0] // ngs_wfs.subaperture_mask.data.shape[0]

        loop_freq = fits.getheader(path_ngs_loop / f'{pipeline}Ctr.CFG.DYNAMIC.fits')['LOOP_FREQ']
        self.system.loops.append(aotpy.ControlLoop(
            uid=f'{pipeline} loop',
            input_sensor=ngs_wfs,
            commanded_corrector=dm,
            commands=aotpy.Image('DM_Positions', ngs_loop_frame['DM_Positions'][:, act_valid_map]),
            ref_commands=self._image_from_object(path_ngs_loop, 'RTC.ACT_POS_REF_MAP_WITH_OFFSETS', ngs_time,
                                                 act_axis=0, act_mask=act_valid_map),
            time=ngs_time,
            framerate=loop_freq,
            time_filter_num=aotpy.Image(f'{pipeline}Ctr.A_TERMS',
                                        fits.getdata(path_ngs_loop / f'{pipeline}Ctr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image(f'{pipeline}Ctr.B_TERMS',
                                        fits.getdata(path_ngs_loop / f'{pipeline}Ctr.B_TERMS.fits').T),
            control_matrix=self._image_from_object(path_ngs_loop, f'{pipeline}Recn.REC1.HOCM', ngs_time, subap_axis=1,
                                                   subap_mask=subap_valid_map, act_axis=0, act_mask=act_valid_map),
            measurements_to_modes=self._image_from_object(path_ngs_loop, 'CLMatrixOptimiser.S2M', ngs_time,
                                                          subap_axis=1, subap_mask=subap_valid_map),
            modes_to_commands=self._image_from_object(path_ngs_loop, 'CLMatrixOptimiser.M2V', ngs_time,
                                                      act_axis=0, act_mask=act_valid_map),
            interaction_matrix=self._image_from_object(path_ngs_loop, f'{pipeline}SynthIMComp.IM', ngs_time,
                                                       subap_axis=0, subap_mask=subap_valid_map, act_axis=1,
                                                       act_mask=act_valid_map),
            commands_to_modes=self._image_from_object(path_ngs_loop, 'CLMatrixOptimiser.V2M', ngs_time,
                                                      act_axis=1, act_mask=act_valid_map),
            modes_to_measurements=self._image_from_object(path_ngs_loop, 'CLMatrixOptimiser.M2S', ngs_time,
                                                          subap_axis=0, subap_mask=subap_valid_map),
        ))

        m2 = aotpy.TipTiltMirror(
            uid='M2',
            telescope=self.system.main_telescope
        )
        self.system.wavefront_correctors.append(m2)

        ttol_pos = ngs_loop_frame['TTOL_Positions']
        ttol_updated = ngs_loop_frame['TTOL_OUpdated'] == 1
        ttol_updated[0] = True  # always keep the first position
        ttol_pos = ttol_pos[ttol_updated]
        self.system.loops.append(aotpy.OffloadLoop(
            uid=f'TTOL (M2 tip-tilt offload)',
            input_corrector=dm,
            commanded_corrector=m2,
            commands=aotpy.Image('TTOL_Positions', ttol_pos),
            time=aotpy.Time('TTOL time', frame_numbers=ngs_loop_frame['FrameCounter'][ttol_updated].tolist())
        ))

        avc_term = ngs_loop_frame['AVC_Term']
        # Only add AVC loop if it was enabled
        if not np.all(avc_term == 0):
            self.system.loops.append(aotpy.ControlLoop(
                uid=f'{pipeline} AVC loop',
                input_sensor=ngs_wfs,
                commanded_corrector=dm,
                commands=aotpy.Image('AVC_Term', avc_term),
                time=ngs_time,
                framerate=loop_freq
            ))

    @staticmethod
    def _get_pixscale_from_pipeline(pipeline):
        return {'HO': 0.42, 'LO': 0.21, 'HOIR': 0.51, 'LOIR': 0.51, 'LGS': 0.8}[pipeline]

    @staticmethod
    def _get_subapsize_from_pipeline(pipeline):
        return {'HO': 6, 'LO': 12, 'HOIR': 8, 'LOIR': 8, 'LGS': 8}[pipeline]

    @staticmethod
    def _get_wavelength_from_pipeline(pipeline):
        return {'HO': 0.850, 'LO': 0.850, 'HOIR': 2.100, 'LOIR': 2.100, 'LGS': 0.589}[pipeline]

    def _get_eso_telescope_name(self) -> str:
        return f'ESO-VLT-U{self._ut_number}'

    def _get_eso_ao_name(self) -> str:
        return 'GPAO'

    def _get_run_id(self) -> str:
        # It should be something like '60.A-9278(F)'
        raise NotImplementedError("Run ID not yet decided")

    def _get_chip_id(self) -> str:
        return f'GPAO{self._ut_number}'
