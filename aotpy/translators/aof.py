"""
This module contains a class for translating data produced by ESO's AOF system.
"""

import importlib.resources
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

import aotpy
from aotpy.io import image_from_file
from .eso import ESOTranslator

# TODO set image units


class AOFTranslator(ESOTranslator):
    """Contains functions for translating telemetry data produced by ESO's AOF system.

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
        self.system = aotpy.AOSystem(ao_mode='LTAO')
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

        self.dsm_valid = fits.getdata(path_lgs / 'RTC.USED_ACT_MAP.fits')[0] - 1
        # We have to subtract one because the array uses one-based indexing unlike Python

        self.dsm = aotpy.DeformableMirror(
            uid=f'DSM',
            telescope=self.system.main_telescope,
            n_valid_actuators=self.dsm_valid.size,
        )
        self.system.wavefront_correctors.append(self.dsm)

        lgs_timestamps = lgs_loop_frame['Seconds'] + lgs_loop_frame['USeconds'] / 1.e6
        self.system.date_beginning = datetime.utcfromtimestamp(lgs_timestamps[0])
        self.system.date_end = datetime.utcfromtimestamp(lgs_timestamps[-1])
        lgs_frame_numbers = lgs_loop_frame['FrameCounter']
        lgs_time = aotpy.Time('LGS Loop Time', timestamps=lgs_timestamps.tolist(),
                              frame_numbers=lgs_frame_numbers.tolist())

        aof_data_path = importlib.resources.files('data') / 'AOF'
        subaperture_mask = image_from_file(aof_data_path / 'subap.fits')
        n_valid_subapertures = np.count_nonzero(subaperture_mask.data != -1)

        dsm_positions = aotpy.Image('DSM_positions', lgs_loop_frame['DSM_Positions'][:, self.dsm_valid])
        m2c = image_from_file(path_lgs / 'LGSCtr.ACT_POS_MODAL_PROJECTION.fits')
        lgs_tfz_num = aotpy.Image('LGSCtr.A_TERMS', fits.getdata(path_lgs / 'LGSCtr.A_TERMS.fits').T)
        lgs_tfz_den = aotpy.Image('LGSCtr.B_TERMS', fits.getdata(path_lgs / 'LGSCtr.B_TERMS.fits').T)
        jit_tfz_num = fits.getdata(path_lgs / 'JitCtr.A_TERMS.fits').T
        jit_tfz_den = fits.getdata(path_lgs / 'JitCtr.B_TERMS.fits').T
        jit_ref = fits.getdata(path_lgs / 'JitCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]
        off_ref = fits.getdata(path_lgs / 'JitCtr.OACT_POS_REF_MAP.fits')[0, :]
        proj_map = fits.getdata(path_lgs / f'JitCtr.PROJ_MAP_SCALED.fits')
        im_list = np.split(fits.getdata(path_lgs / f'RTC.IMref4Atm.fits')[:, self.dsm_valid], 4, axis=0)
        for i in range(1, 5):
            llt = aotpy.LaserLaunchTelescope(f'LLT{i}')
            lgs = aotpy.SodiumLaserGuideStar(uid=f'LGS{i}', laser_launch_telescope=llt)
            self.system.sources.append(lgs)

            gradients = lgs_loop_frame[f'WFS{i}_Gradients']
            # AOF gradients are ordered tip1, tilt1, tip2, tilt2, etc., so even numbers are tip and odd numbers are tilt
            # We separate them, select the valid subapertures and then stack them
            tip = gradients[:, ::2]
            tilt = gradients[:, 1::2]
            gradients = np.stack([tip, tilt], axis=1)

            reference = fits.getdata(path_lgs / f'LGSAcq.DET{i}.REFSLP_WITH_OFFSETS.fits')[0]
            tip = reference[::2]
            tilt = reference[1::2]
            reference = np.stack([tip, tilt], axis=0)
            wfs = aotpy.ShackHartmann(
                uid=f'LGS WFS{i}',
                source=lgs,
                n_valid_subapertures=n_valid_subapertures,
                measurements=aotpy.Image(f'WFS{i}_Gradients', gradients),
                ref_measurements=aotpy.Image(f'LGSAcq.DET{i}.REFSLP_WITH_OFFSETS', reference),
                subaperture_mask=subaperture_mask,
                subaperture_intensities=aotpy.Image(f'WFS{i}_Intensities', lgs_loop_frame[f'WFS{i}_Intensities'])
            )

            wfs.detector = aotpy.Detector(
                uid=f'LGS DET{i}',
                dark=image_from_file(path_lgs / f'LGSAcq.DET{i}.DARK.fits'),
                weight_map=image_from_file(path_lgs / f'LGSAcq.DET{i}.WEIGHT.fits')
            )
            self.system.wavefront_sensors.append(wfs)

            cm = fits.getdata(path_lgs / f'LGSRecn.REC{i}.HOCM.fits')[self.dsm_valid]
            tip = cm[:, ::2]
            tilt = cm[:, 1::2]
            cm = np.stack([tip, tilt], axis=1)

            im = im_list[i - 1]
            tip = im[::2]
            tilt = im[1::2]
            im = np.stack([tip, tilt], axis=1)
            self.system.loops.append(aotpy.ControlLoop(
                uid=f'High-order loop {i}',
                input_sensor=wfs,
                commanded_corrector=self.dsm,
                commands=dsm_positions,
                time=lgs_time,
                framerate=1000,
                time_filter_num=lgs_tfz_num,
                time_filter_den=lgs_tfz_den,
                control_matrix=aotpy.Image(f'LGSRecn.REC{i}.HOCM', cm),
                interaction_matrix=aotpy.Image(f'Interaction_Matrix_LGS_WFS{i}', im)
            ))

            jit = aotpy.TipTiltMirror(
                uid=f'Jitter{i}',
                telescope=llt
            )

            cm = fits.getdata(path_lgs / f'JitRecnOptimiser.JitCM{i}.fits')
            tip = cm[:, ::2]
            tilt = cm[:, 1::2]
            cm = np.stack([tip, tilt], axis=1)
            self.system.loops.append(aotpy.ControlLoop(
                uid=f'Jitter loop {i}',
                input_sensor=wfs,
                commanded_corrector=jit,
                commands=aotpy.Image(f'Jitter{i}_Positions', lgs_loop_frame[f'Jitter{i}_Positions']),
                ref_commands=aotpy.Image(f'Jit{i}Ctr.ACT_POS_REF_MAP_WITH_OFFSETS', jit_ref[(i - 1) * 2: i * 2]),
                time=lgs_time,
                framerate=1000,
                time_filter_num=aotpy.Image(f'Jit{i}Ctr.A_TERMS', jit_tfz_num[i - 1:i + 1, :]),
                time_filter_den=aotpy.Image(f'Jit{i}Ctr.B_TERMS', jit_tfz_den[i - 1:i + 1, :]),
                control_matrix=aotpy.Image(f'JitRecnOptimiser.JitCM{i}', cm),
                modes_to_commands=m2c
            ))

            fsm = aotpy.TipTiltMirror(
                uid=f'Field Steering Mirror {i}',
                telescope=llt
            )

            self.system.loops.append(aotpy.OffloadLoop(
                uid=f'Jitter Offload loop {i}',
                input_corrector=jit,
                commanded_corrector=fsm,
                commands=aotpy.Image(f'Jitter{i}_Offload', lgs_loop_frame[f'Jitter{i}_Offload']),
                ref_commands=aotpy.Image(f'Jit{i}Ctr.OACT_POS_REF_MAP', off_ref[(i - 1) * 2: i * 2]),
                time=lgs_time,
                offload_matrix=aotpy.Image(f'Jitter{i}_Offload_Matrix', proj_map[(i - 1) * 2:i * 2, (i - 1) * 2:i * 2])
            ))

            self.system.wavefront_correctors.extend([jit, fsm])

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
                                          data=self.get_pixel_data_from_table(pix_loop_frame),
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
        return 'AOF'
