"""
This module contains a class for translating data produced by ESO's AOF system.
"""

import importlib.resources
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

import aotpy
from aotpy.io import image_from_file
from .eso import ESOTranslator

# TODO set image units


class ERISTranslator(ESOTranslator):
    """Contains functions for translating telemetry data produced by ESO's ERIS system.

    Parameters
    ----------
    path_lgs
        Path to folder containing LGS data (LGSAcq, LGSCtr, LGSRecn, JitCtr, JitRecnOptimiser, RTC).
    path_ir
        Path to folder containing IR data (IRAcq, IRCtr, IRLoopMonitor).
    path_pix
        Path to folder containing pixel data.
    """

    def __init__(self, path_lgs: str, path_ir: str, path_pix: str):
        self.system = aotpy.AOSystem(ao_mode='')
        self.system.main_telescope = aotpy.MainTelescope(
            uid='ESO VLT UT4',
            enclosing_diameter=8.2,
            inscribed_diameter=8.2
        )

        self._handle_lgs_data(path_lgs)
        self._handle_ngs_data(path_ir, path_pix)

    def _handle_lgs_data(self, path_lgs):
        path_lgs = Path(path_lgs)
        lgs_loop_frame = fits.getdata(path_lgs / f'{path_lgs.name}.fits', extname='LGSLoopFrame')

        self.dsm = aotpy.DeformableMirror(
            uid='DSM',
            telescope=self.system.main_telescope,
            n_valid_actuators=1156,
        )
        self.system.wavefront_correctors.append(self.dsm)

        lgs_timestamps = lgs_loop_frame['Seconds'] + lgs_loop_frame['USeconds'] / 1.e6
        self.system.date_beginning = datetime.utcfromtimestamp(lgs_timestamps[0])
        self.system.date_end = datetime.utcfromtimestamp(lgs_timestamps[-1])
        lgs_frame_numbers = lgs_loop_frame['FrameCounter']
        lgs_time = aotpy.Time('LGS Loop Time', timestamps=lgs_timestamps.tolist(),
                              frame_numbers=lgs_frame_numbers.tolist())

        llt = aotpy.LaserLaunchTelescope('LLT')
        lgs = aotpy.SodiumLaserGuideStar(uid='LGS', laser_launch_telescope=llt)
        self.system.sources.append(lgs)

        eris_data_path = importlib.resources.files('data') / 'ERIS'
        with importlib.resources.as_file(eris_data_path / 'subap.fits') as p:
            subaperture_mask = image_from_file(p)
        n_valid_subapertures = np.count_nonzero(subaperture_mask.data != -1)
        if n_valid_subapertures != fits.getdata(path_lgs / 'LGSRecnOptimiser.SUBAP_VALID_MAP.fits')[0].size:
            warnings.warn('Unexpected number of valid subapertures')

        reference = self._stack_slopes(fits.getdata(path_lgs / 'LGSAcq.DET1.REFSLP_WITH_OFFSETS.fits'), slope_axis=1)[0]
        lgs_wfs = aotpy.ShackHartmann(
            uid='LGS WFS',
            source=lgs,
            n_valid_subapertures=n_valid_subapertures,
            measurements=aotpy.Image('Gradients', self._stack_slopes(lgs_loop_frame['Gradients'], slope_axis=1)),
            ref_measurements=aotpy.Image('LGSAcq.DET1.REFSLP_WITH_OFFSETS', reference),
            subaperture_mask=subaperture_mask,
            subaperture_intensities=aotpy.Image('Intensities', lgs_loop_frame['Intensities'])
        )

        lgs_wfs.detector = aotpy.Detector(
            uid='LGS DET1',
            dark=image_from_file(path_lgs / 'LGSAcq.DET1.DARK.fits'),
            weight_map=image_from_file(path_lgs / 'LGSAcq.DET1.WEIGHT.fits'),
            sky_background=image_from_file(path_lgs / 'LGSAcq.DET1.BACKGROUND.fits')
        )
        self.system.wavefront_sensors.append(lgs_wfs)

        s2m = self._stack_slopes(fits.getdata(path_lgs / 'CLMatrixOptimiser.S2M.fits'), slope_axis=1)
        m2s = self._stack_slopes(fits.getdata(path_lgs / 'CLMatrixOptimiser.M2S.fits'), slope_axis=0)
        self.system.loops.append(aotpy.ControlLoop(
            uid='High-order loop',
            input_sensor=lgs_wfs,
            commanded_corrector=self.dsm,
            commands=aotpy.Image('DSM_positions', lgs_loop_frame['DSM_Positions']),
            ref_commands=aotpy.Image('LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS',
                                     fits.getdata(path_lgs / 'LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]),
            time=lgs_time,
            # TODO framerate=1000,
            time_filter_num=aotpy.Image('LGSCtr.A_TERMS', fits.getdata(path_lgs / 'LGSCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('LGSCtr.B_TERMS', fits.getdata(path_lgs / 'LGSCtr.B_TERMS.fits').T),
            # control_matrix=aotpy.Image(f'LGSRecn.REC{i}.HOCM', cm),
            measurements_to_modes=aotpy.Image('CLMatrixOptimiser.S2M', s2m),
            modes_to_commands=image_from_file(path_lgs / 'CLMatrixOptimiser.M2V.fits'),
            commands_to_modes=image_from_file(path_lgs / 'CLMatrixOptimiser.V2M.fits'),
            modes_to_measurements=aotpy.Image('CLMatrixOptimiser.M2S', m2s),
            # interaction_matrix=aotpy.Image(f'Interaction_Matrix_LGS_WFS{i}', im)
        ))

        jit = aotpy.TipTiltMirror(
            uid='Jitter',
            telescope=llt
        )

        cm = self._stack_slopes(fits.getdata(path_lgs / 'JitRecnOptimiser.JitCM.fits'), slope_axis=1)
        im = self._stack_slopes(fits.getdata(path_lgs / 'JitRecnCalibrat.IM.fits'), slope_axis=0)
        # jit_ref = fits.getdata(path_lgs / 'JitCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]
        self.system.loops.append(aotpy.ControlLoop(
            uid='Jitter loop',
            input_sensor=lgs_wfs,
            commanded_corrector=jit,
            commands=aotpy.Image('Jitter_Positions', lgs_loop_frame['Jitter_Positions']),
            # ref_commands=aotpy.Image(f'Jit{i}Ctr.ACT_POS_REF_MAP_WITH_OFFSETS', jit_ref[(i - 1) * 2: i * 2]),
            time=lgs_time,
            framerate=1000,
            control_matrix=aotpy.Image('JitRecnOptimiser.JitCM', cm),
            interaction_matrix=aotpy.Image('JitRecnCalibrat.IM', im)
        ))

        fsm = aotpy.TipTiltMirror(
            uid='Field Steering Mirror',
            telescope=llt
        )

        # off_ref = fits.getdata(path_lgs / 'JitCtr.OACT_POS_REF_MAP.fits')[0, :]
        # proj_map = fits.getdata(path_lgs / f'JitCtr.PROJ_MAP_SCALED.fits')
        self.system.loops.append(aotpy.OffloadLoop(
            uid='Jitter Offload loop',
            input_corrector=jit,
            commanded_corrector=fsm,
            commands=aotpy.Image('Jitter_Offload', lgs_loop_frame['Jitter_Offload']),
            # ref_commands=aotpy.Image(f'Jit{i}Ctr.OACT_POS_REF_MAP', off_ref[(i - 1) * 2: i * 2]),
            time=lgs_time,
            # offload_matrix=aotpy.Image(f'Jitter{i}_Offload_Matrix', proj_map[(i - 1) * 2:i * 2, (i - 1) * 2:i * 2])
        ))

        self.system.wavefront_correctors.extend([jit, fsm])

        ngs = aotpy.NaturalGuideStar(uid='NGS')
        self.system.sources.append(ngs)

        reference = self._stack_slopes(fits.getdata(path_lgs / 'LOAcq.DET1.REFSLP_WITH_OFFSETS.fits'), slope_axis=1)[0]
        lo_wfs = aotpy.ShackHartmann(
            uid='LO WFS',
            source=ngs,
            n_valid_subapertures=fits.getdata(path_lgs / 'LORecnOptimiser.SUBAP_VALID_MAP.fits')[0].size,
            # measurements=aotpy.Image('Gradients', gradients),
            ref_measurements=aotpy.Image('LOAcq.DET1.REFSLP_WITH_OFFSETS', reference),
            # subaperture_mask=subaperture_mask,
            # subaperture_intensities=aotpy.Image('Intensities', lgs_loop_frame['Intensities'])
        )

        lo_wfs.detector = aotpy.Detector(
            uid='LO DET1',
            dark=image_from_file(path_lgs / 'LOAcq.DET1.DARK.fits'),
            weight_map=image_from_file(path_lgs / 'LOAcq.DET1.WEIGHT.fits'),
            sky_background=image_from_file(path_lgs / 'LOAcq.DET1.BACKGROUND.fits')
        )
        self.system.wavefront_sensors.append(lo_wfs)

        # LO and LGS don't have the same frequency. We have to find in which LGS frames we actually have an update on
        # the LO loop. So we check whenever LO_FrameCounter is incremented
        lo_frame_numbers = lgs_loop_frame['LO_FrameCounter']
        lo_mask = np.empty_like(lo_frame_numbers)
        lo_mask[1:] = np.diff(lo_frame_numbers)
        lo_mask[0] = 1
        lo_timestamps = lgs_timestamps[np.where(lo_mask)]
        lo_frame_numbers = lgs_frame_numbers[np.where(lo_mask)]
        lo_time = aotpy.Time('LO Loop Time', timestamps=lo_timestamps.tolist(), frame_numbers=lo_frame_numbers.tolist())

        s2m = self._stack_slopes(fits.getdata(path_lgs / 'LOCtr.SENSOR_2_MODES.fits'), slope_axis=1)
        self.system.loops.append(aotpy.ControlLoop(
            uid='Low-order loop',
            input_sensor=lo_wfs,
            commanded_corrector=self.dsm,
            commands=aotpy.Image('DSM_LowOrderTerm', lgs_loop_frame['DSM_LowOrderTerm'][np.where(lo_mask)]),
            # ref_commands=aotpy.Image('LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS',
            #                         fits.getdata(path_lgs / 'LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]),
            time=lo_time,
            # TODO framerate,
            time_filter_num=aotpy.Image('LOCtr.A_TERMS', fits.getdata(path_lgs / 'LOCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('LOCtr.B_TERMS', fits.getdata(path_lgs / 'LOCtr.B_TERMS.fits').T),
            measurements_to_modes=aotpy.Image('LOCtr.SENSOR_2_MODES', s2m),
        ))

        # TODO what is the truth loop exactly?
        s2m = self._stack_slopes(fits.getdata(path_lgs / 'TruthCtr.SENSOR_2_MODES.fits'), slope_axis=1)
        self.system.loops.append(aotpy.ControlLoop(
            uid='Truth loop',
            input_sensor=lo_wfs,
            commanded_corrector=self.dsm,
            time_filter_num=aotpy.Image('TruthCtr.A_TERMS', fits.getdata(path_lgs / 'TruthCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('TruthCtr.B_TERMS', fits.getdata(path_lgs / 'TruthCtr.B_TERMS.fits').T),
            measurements_to_modes=aotpy.Image('TruthCtr.SENSOR_2_MODES', s2m),
        ))

    def _handle_ngs_data(self, path_ir, path_pix):
        path_ir = Path(path_ir)
        ir_loop_frame = fits.getdata(path_ir / f'{path_ir.name}.fits', extname='IRLoopFrame')
        path_pix = Path(path_pix)
        pix_loop_frame = fits.getdata(path_pix / f'{path_pix.name}.fits', extname='IRPixelFrame')

        ngs_timestamps = ir_loop_frame['Seconds'] + ir_loop_frame['USeconds'] / 1.e6
        if np.all(ngs_timestamps == 0):
            # The file has no timestamps
            ngs_timestamps_list = []
        else:
            ngs_timestamps_list = ngs_timestamps.tolist()
        ho_frame_numbers = ir_loop_frame['HOFrameCounter']
        ir_time = aotpy.Time('NGS Loop Time', timestamps=ngs_timestamps_list, frame_numbers=ho_frame_numbers.tolist())

        ngs = aotpy.NaturalGuideStar('NGS')
        self.system.sources.append(ngs)

        gradients = ir_loop_frame['WFS_Gradients']
        tip = gradients[:, ::2]
        tilt = gradients[:, 1::2]
        gradients = np.stack([tip, tilt], axis=1)

        reference = fits.getdata(path_ir / 'IRAcq.DET1.REFSLP_WITH_OFFSETS.fits')[0]
        tip = reference[::2]
        tilt = reference[1::2]
        reference = np.stack([tip, tilt], axis=0)
        ngs_wfs = aotpy.ShackHartmann(
            uid='NGS WFS1',
            n_valid_subapertures=4,  # All subapertures are valid
            subaperture_mask=aotpy.Image('NGS_WFS_SUBAPERTURE_MASK', np.array([[1, 3], [2, 4]])),
            source=ngs,
            measurements=aotpy.Image('NGS_WFS_Gradients', gradients),
            ref_measurements=aotpy.Image('IRAcq.DET1.REFSLP_WITH_OFFSETS', reference),
            subaperture_intensities=aotpy.Image('WFS_Intensities', ir_loop_frame['WFS_Intensities'])
        )
        self.system.wavefront_sensors.append(ngs_wfs)

        # Find the indexes where the counter for pixels matches the counter for ngs
        # Assumes the frames for pixel are contained in the frames for ngs
        pix_time_mask = np.searchsorted(ir_loop_frame['FrameCounter'], pix_loop_frame['FrameCounter'])
        pix_timestamps = ngs_timestamps[pix_time_mask]
        if np.all(pix_timestamps == 0):
            # The file has no timestamps
            pix_timestamps_list = []
        else:
            pix_timestamps_list = pix_timestamps.tolist()

        pix_time = aotpy.Time('Pixel Time', timestamps=pix_timestamps_list,
                              frame_numbers=ho_frame_numbers[pix_time_mask].tolist())

        ngs_wfs.detector = aotpy.Detector(
            uid='NGS DET1',
            dark=image_from_file(path_ir / 'IRAcq.DET1.DARK.fits'),
            weight_map=image_from_file(path_ir / 'IRAcq.DET1.WEIGHT.fits'),
            pixel_intensities=aotpy.Image(name='NGS Pixels',
                                          data=self._get_pixel_data_from_table(pix_loop_frame),
                                          time=pix_time)
        )

        s2m = fits.getdata(path_ir / 'IRCtr.SENSOR_2_MODES.fits')
        tip = s2m[:, ::2]
        tilt = s2m[:, 1::2]
        s2m = np.stack([tip, tilt], axis=1)

        m2c = fits.getdata(path_ir / 'IRCtr.MODES_2_ACT.fits')[self.dsm_valid]
        self.system.loops.append(aotpy.ControlLoop(
            uid='Low-order loop',
            input_sensor=ngs_wfs,
            commanded_corrector=self.dsm,
            commands=aotpy.Image('LO_Positions', ir_loop_frame['LO_Positions'][:, self.dsm_valid]),
            measurements_to_modes=aotpy.Image('IRCtr.SENSOR_2_MODES', s2m),
            modes_to_commands=aotpy.Image('IRCtr.MODES_2_ACT', m2c),
            time=ir_time,
            time_filter_num=aotpy.Image('IRCtr.A_TERMS', fits.getdata(path_ir / 'IRCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('IRCtr.B_TERMS', fits.getdata(path_ir / 'IRCtr.B_TERMS.fits').T)
        ))

    def _get_eso_telescope_name(self) -> str:
        return 'ESO-VLT-U4'

    def _get_eso_ao_name(self) -> str:
        return 'ERIS'
