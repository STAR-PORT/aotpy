import numpy as np
from astropy.io import fits
from pathlib import Path

from aotpy import *

__all__ = ['load_from_aof']


def load_from_aof(folder_path: str) -> AOSystem:
    """This function assumes that it is pointed to a folder with the LGS loop data, and that in the same directory
    there is a folder with IR data and another with IR Pixel data
    For example, for the file structure:
        data/example_folder/
        data/example_folder_IR/
        data/example_folder_IR_PIX/
    this function assumes it recieves "data/example_folder" as the folder_path
    """

    path_lgs = Path(folder_path)
    lgs_loop_frame = fits.getdata(path_lgs.joinpath(f'{path_lgs.name}.fits'), extname='LGSLoopFrame')

    path_ir = Path(f'{folder_path}_IR')
    ir_loop_frame = fits.getdata(path_ir.joinpath(f'{path_ir.name}.fits'), extname='IRLoopFrame')

    path_pix = Path(f'{folder_path}_IR_PIX')
    pix_loop_frame = fits.getdata(path_pix.joinpath(f'{path_pix.name}.fits'), extname='IRPixelFrame')

    lgs_timestamps, ir_timestamps = _handle_timestamps(lgs_loop_frame, ir_loop_frame)

    system = AOSystem()
    system.rtc = RTC()
    system.telescope = MainTelescope(
        name='VLT UT4',
        d_circle=8.2
    )

    dsm = DeformableMirror(
        name=f'DSM',
        telescope=system.telescope,
        valid_actuators=Image.from_fits(path_lgs.joinpath('RTC.USED_ACT_MAP.fits')),
        n_actuators=1156
    )
    system.wavefront_correctors.append(dsm)

    dsm_positions = Image('DSM_positions', lgs_loop_frame['DSM_Positions'])
    lgs_tfz_num = Image('LGSCtr.A_TERMS', fits.getdata(path_lgs.joinpath('LGSCtr.A_TERMS.fits')).T)
    lgs_tfz_den = Image('LGSCtr.B_TERMS', fits.getdata(path_lgs.joinpath('LGSCtr.B_TERMS.fits')).T)
    jit_tfz_num = fits.getdata(path_lgs.joinpath('JitCtr.A_TERMS.fits')).T
    jit_tfz_den = fits.getdata(path_lgs.joinpath('JitCtr.B_TERMS.fits')).T
    jit_ref = fits.getdata(path_lgs.joinpath('JitCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits'))[:, 0]
    off_ref = fits.getdata(path_lgs.joinpath('JitCtr.OACT_POS_REF_MAP.fits'))[0, :]
    proj_map = fits.getdata(path_lgs.joinpath(f'JitCtr.PROJ_MAP_SCALED.fits'))
    for i in range(1, 5):
        llt = LaserLaunchTelescope(f'LLT{i}')
        lgs = LaserGuideStar(name=f'LGS{i}', laser_launch_telescope=llt)
        system.sources.append(lgs)

        wfs = ShackHartmann(
            name=f'LGS WFS{i}',
            source=lgs,
            slopes=Image(f'WFS{i}_Gradients', lgs_loop_frame[f'WFS{i}_Gradients']),  # TODO: reshape this?
            ref_slopes=Image.from_fits(path_lgs.joinpath(f'LGSAcq.DET{i}.REFSLP_WITH_OFFSETS.fits')),
            subaperture_intensities=Image(f'WFS{i}_Intensities', lgs_loop_frame[f'WFS{i}_Intensities']),
            n_subapertures=1240
        )
        # TODO: we have sourced the list of valid subapertures per WFS, but we are working on how to best package it

        wfs.detector = Detector(
            name=f'LGS DET{i}',
            dark=Image.from_fits(path_lgs.joinpath(f'LGSAcq.DET{i}.DARK.fits')),
            weight_map=Image.from_fits(path_lgs.joinpath(f'LGSAcq.DET{i}.WEIGHT.fits'))
        )
        system.wavefront_sensors.append(wfs)

        system.rtc.loops.append(ControlLoop(
            name=f'High-order loop {i}',
            input_wfs=wfs,
            commanded_corrector=dsm,
            commands=dsm_positions,
            timestamps=lgs_timestamps,
            framerate=1000,
            time_filter_num=lgs_tfz_num,
            time_filter_den=lgs_tfz_den,
            control_matrix=Image.from_fits(path_lgs.joinpath(f'LGSRecn.REC{i}.HOCM.fits'))
        ))

        jit = TipTiltMirror(
            name=f'Jitter{i}',
            telescope=llt
        )

        system.rtc.loops.append(ControlLoop(
            name=f'Jitter loop {i}',
            input_wfs=wfs,
            commanded_corrector=jit,
            commands=Image(f'Jitter{i}_Positions', lgs_loop_frame[f'Jitter{i}_Positions']),
            ref_commands=Image(f'Jit{i}Ctr.ACT_POS_REF_MAP_WITH_OFFSETS', jit_ref[(i-1)*2: i*2]),
            timestamps=lgs_timestamps,
            framerate=1000,
            time_filter_num=Image(f'Jit{i}Ctr.A_TERMS', jit_tfz_num[i-1:i+1, :]),
            time_filter_den=Image(f'Jit{i}Ctr.B_TERMS', jit_tfz_den[i-1:i+1, :]),
            control_matrix=Image.from_fits(path_lgs.joinpath(f'JitRecnOptimiser.JitCM{i}.fits'))
        ))

        fsm = TipTiltMirror(
            name=f'Field Steering Mirror {i}',
            telescope=llt
        )

        system.rtc.loops.append(OffloadLoop(
            name=f'Jitter Offload loop {i}',
            input_corrector=jit,
            commanded_corrector=fsm,
            commands=Image(f'Jitter{i}_Offload', lgs_loop_frame[f'Jitter{i}_Offload']),
            ref_commands=Image(f'Jit{i}Ctr.OACT_POS_REF_MAP', off_ref[(i-1)*2: i*2]),
            timestamps=lgs_timestamps,
            offload_matrix=Image(f'Jitter{i}_Offload_Matrix', proj_map[(i-1)*2:i*2, (i-1)*2:i*2])
        ))

        system.wavefront_correctors.extend([jit, fsm])

    ngs = NaturalGuideStar('NGS1')
    system.sources.append(ngs)

    wfs = ShackHartmann(
        name='NGS WFS1',
        source=ngs,
        slopes=Image('WFS_Gradients', ir_loop_frame['WFS_Gradients']),
        ref_slopes=Image.from_fits(path_ir.joinpath('IRAcq.DET1.REFSLP_WITH_OFFSETS.fits')),
        subaperture_intensities=Image('WFS_Intensities', ir_loop_frame['WFS_Intensities'])
    )
    wfs.n_subapertures = wfs.slopes.data.shape[1]

    sizes_x = pix_loop_frame['WindowSizeX']
    sizes_y = pix_loop_frame['WindowSizeY']
    if np.any(sizes_x != sizes_x[0]) or np.any(sizes_y != sizes_y[0]):
        raise RuntimeError  # Window size shouldn't change over time
    sizes_x = sizes_x[0]
    sizes_y = sizes_y[0]

    wfs.detector = Detector(
        name='NGS DET1',
        pixel_intensities=Image('NGS Pixels', pix_loop_frame['Pixels'][:, :sizes_x*sizes_y].reshape(-1, sizes_x, sizes_y)),
        dark=Image.from_fits(path_ir.joinpath('IRAcq.DET1.DARK.fits')),
        weight_map=Image.from_fits(path_ir.joinpath('IRAcq.DET1.WEIGHT.fits'))
    )
    system.wavefront_sensors.append(wfs)

    system.rtc.loops.append(ControlLoop(
        name='Low-order loop',
        input_wfs=wfs,
        commanded_corrector=dsm,
        commands=Image('LO_Positions', ir_loop_frame['LO_Positions']),
        control_matrix=Image('LO Control Matrix',
                             np.matmul(fits.getdata(path_ir.joinpath('IRCtr.SENSOR_2_MODES.fits')).T,
                                       fits.getdata(path_ir.joinpath('IRCtr.MODES_2_ACT.fits')).T)),
        timestamps=ir_timestamps,
        time_filter_num=Image('IRCtr.A_TERMS', fits.getdata(path_ir.joinpath('IRCtr.A_TERMS.fits')).T),
        time_filter_den=Image('IRCtr.B_TERMS', fits.getdata(path_ir.joinpath('IRCtr.B_TERMS.fits')).T)
    ))

    return system


def _handle_timestamps(lgs_loop_frame, ir_loop_frame):
    # In some cases 'Seconds' and 'USeconds' are emtpy instead of containing the frame timestamps
    # However, since we know that the LGS framerate is 1000hz, we can deduce LGS timestamps from the LGS frame counter
    # And since "HOFrameCounter" refers to the LGS counter, we can also deduce IR timestamps

    lgs_timestamps = lgs_loop_frame['Seconds'] + lgs_loop_frame['USeconds'] / 1.e6
    ir_timestamps = ir_loop_frame['Seconds'] + ir_loop_frame['USeconds'] / 1.e6
    if np.all(lgs_timestamps == 0) and np.all(ir_timestamps == 0):
        lgs_counter = lgs_loop_frame['FrameCounter']
        ir_counter = ir_loop_frame['HOFrameCounter']

        first = min(lgs_counter[0], ir_counter[0])
        last = max(lgs_counter[-1], ir_counter[-1])

        diff = last - first
        step = 1 / 1000
        d = {i + first: i * step for i in range(diff + 1)}

        return [d[c] for c in lgs_counter], [d[c] for c in ir_counter]
    else:
        lgs_timestamps -= lgs_timestamps[0]
        ir_timestamps -= ir_timestamps[0]
        return lgs_timestamps.tolist(), ir_timestamps.tolist()