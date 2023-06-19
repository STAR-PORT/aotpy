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
    path
        Path to folder containing all system data.
    """

    def __init__(self, path):
        self._path = Path(path)
        self.system = aotpy.AOSystem()
        self.system.main_telescope = aotpy.MainTelescope(
            uid='ESO VLT UT4',
            enclosing_diameter=8.2,
            inscribed_diameter=8.2
        )
        self.dsm = aotpy.DeformableMirror(
            uid='DSM',
            telescope=self.system.main_telescope,
            n_valid_actuators=1156,
        )
        self.system.wavefront_correctors.append(self.dsm)

        if (ho_loop_file := self._find_bintable_files('hoLoopData')) is not None \
                and (ho_pixel_file := self._find_bintable_files('hoPixelData')) is not None:
            self._handle_ngs_data(ho_loop_file, ho_pixel_file)
        elif (lgs_loop_file := self._find_bintable_files('lgsLoopData')) is not None \
                and (lo_loop_file := self._find_bintable_files('loLoopData')) is not None \
                and (lgs_pixel_file := self._find_bintable_files('lgsPixelData')) is not None \
                and (lo_pixel_file := self._find_bintable_files('loPixelData')) is not None:
            self._handle_lgs_data(lgs_loop_file, lgs_pixel_file, lo_loop_file, lo_pixel_file)
        else:
            raise ValueError('Path does not contain necessary telemetry data.')

    def _find_bintable_files(self, name):
        file = list(self._path.glob(f'{name}_*.fits'))
        if (size := len(file)) == 0:
            return None
        else:
            file = file[0]
            if size > 1:
                warnings.warn(f"Found more than one '{name}' file. Picking '{file.name}'.")
            return file

    def _handle_lgs_data(self, lgs_loop_file, lgs_pixel_file, lo_loop_file, lo_pixel_file):
        self.system.ao_mode = 'LTAO'
        lgs_loop_frame = fits.getdata(lgs_loop_file, extname='LGSLoopFrame')
        lgs_pix_frame = fits.getdata(lgs_pixel_file, extname='LGSPixelFrame')

        lgs_timestamps = lgs_loop_frame['Seconds'] + lgs_loop_frame['USeconds'] / 1.e6
        self.system.date_beginning = datetime.utcfromtimestamp(lgs_timestamps[0])
        self.system.date_end = datetime.utcfromtimestamp(lgs_timestamps[-1])
        lgs_frame_numbers = lgs_loop_frame['FrameCounter']
        lgs_time = aotpy.Time('LGS Loop Time', timestamps=lgs_timestamps.tolist(),
                              frame_numbers=lgs_frame_numbers.tolist())

        active_laser = fits.getheader(self._path / 'JitCtr.CFG.DYNAMIC.fits')['ACTIVE_JITTER']
        llt = aotpy.LaserLaunchTelescope(f'LLT{active_laser}')
        lgs = aotpy.SodiumLaserGuideStar(uid='LGS', laser_launch_telescope=llt)
        self.system.sources.append(lgs)

        eris_data_path = importlib.resources.files('aotpy.data') / 'ERIS'
        with importlib.resources.as_file(eris_data_path / 'ho_subap.fits') as p:
            subaperture_mask = image_from_file(p, name='LGS WFS SUBAPERTURE MASK')
        n_valid_subapertures = np.count_nonzero(subaperture_mask.data != -1)

        reference = self._stack_slopes(fits.getdata(self._path / 'LGSAcq.DET1.REFSLP_WITH_OFFSETS.fits'),
                                       slope_axis=1)[0]
        lgs_wfs = aotpy.ShackHartmann(
            uid='LGS WFS',
            source=lgs,
            n_valid_subapertures=n_valid_subapertures,
            measurements=aotpy.Image('LGS Gradients', self._stack_slopes(lgs_loop_frame['Gradients'], slope_axis=1)),
            ref_measurements=aotpy.Image('LGSAcq.DET1.REFSLP_WITH_OFFSETS', reference),
            subaperture_mask=subaperture_mask,
            mask_offsets=[aotpy.Coordinates(0, 0)],
            subaperture_intensities=aotpy.Image('LGS Intensities', lgs_loop_frame['Intensities'])
        )
        self.system.wavefront_sensors.append(lgs_wfs)

        lgs_wfs.detector = aotpy.Detector(
            uid='LGS DET1',
            dark=image_from_file(self._path / 'LGSAcq.DET1.DARK.fits'),
            weight_map=image_from_file(self._path / 'LGSAcq.DET1.WEIGHT.fits'),
            sky_background=image_from_file(self._path / 'LGSAcq.DET1.BACKGROUND.fits'),
            pixel_intensities=aotpy.Image(name='LGS Pixels',
                                          data=self._get_pixel_data_from_table(lgs_pix_frame),
                                          time=aotpy.Time('LGS Pixel Time',
                                                          frame_numbers=lgs_pix_frame['FrameCounter'].tolist()))
        )
        lgs_wfs.subaperture_size = \
            lgs_wfs.detector.pixel_intensities.data.shape[0] // lgs_wfs.subaperture_mask.data.shape[0]

        s2m = self._stack_slopes(fits.getdata(self._path / 'CLMatrixOptimiser.S2M.fits'), slope_axis=1)
        m2s = self._stack_slopes(fits.getdata(self._path / 'CLMatrixOptimiser.M2S.fits'), slope_axis=0)

        lgs_freq = fits.getheader(self._path / 'LGSDet.CFG.DYNAMIC.fits')['FREQ']
        self.system.loops.append(aotpy.ControlLoop(
            uid='High-order loop',
            input_sensor=lgs_wfs,
            commanded_corrector=self.dsm,
            commands=aotpy.Image('DSM_positions', lgs_loop_frame['DSM_Positions']),
            ref_commands=aotpy.Image('LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS',
                                     fits.getdata(self._path / 'LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]),
            time=lgs_time,
            framerate=lgs_freq,
            time_filter_num=aotpy.Image('LGSCtr.A_TERMS', fits.getdata(self._path / 'LGSCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('LGSCtr.B_TERMS', fits.getdata(self._path / 'LGSCtr.B_TERMS.fits').T),
            measurements_to_modes=aotpy.Image('CLMatrixOptimiser.S2M', s2m),
            modes_to_commands=image_from_file(self._path / 'CLMatrixOptimiser.M2V.fits'),
            commands_to_modes=image_from_file(self._path / 'CLMatrixOptimiser.V2M.fits'),
            modes_to_measurements=aotpy.Image('CLMatrixOptimiser.M2S', m2s),
        ))

        jit = aotpy.TipTiltMirror(
            uid='Jitter',
            telescope=llt
        )

        cm = self._stack_slopes(fits.getdata(self._path / 'JitRecnOptimiser.JitCM.fits'), slope_axis=1)
        im = self._stack_slopes(fits.getdata(self._path / 'JitRecnCalibrat.IM.fits'), slope_axis=0)
        # jit_ref = fits.getdata(path_lgs / 'JitCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]
        self.system.loops.append(aotpy.ControlLoop(
            uid='Jitter loop',
            input_sensor=lgs_wfs,
            commanded_corrector=jit,
            commands=aotpy.Image('Jitter_Positions', lgs_loop_frame['Jitter_Positions']),
            # ref_commands=aotpy.Image(f'Jit{i}Ctr.ACT_POS_REF_MAP_WITH_OFFSETS', jit_ref[(i - 1) * 2: i * 2]),
            time=lgs_time,
            framerate=lgs_freq,
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
            framerate=lgs_freq,
            # offload_matrix=aotpy.Image(f'Jitter{i}_Offload_Matrix', proj_map[(i - 1) * 2:i * 2, (i - 1) * 2:i * 2])
        ))

        self.system.wavefront_correctors.extend([jit, fsm])

        ngs = aotpy.NaturalGuideStar(uid='NGS')
        self.system.sources.append(ngs)

        lo_loop_frame = fits.getdata(lo_loop_file, extname='LOLoopFrame')
        lo_pix_frame = fits.getdata(lo_pixel_file, extname='LOPixelFrame')

        lo_timestamps = lo_loop_frame['Seconds'] + lo_loop_frame['USeconds'] / 1.e6
        if np.all(lo_timestamps == 0):
            # The file has no timestamps
            lo_timestamps_list = []
        else:
            lo_timestamps_list = lo_timestamps.tolist()
        ho_frame_numbers = lo_loop_frame['HO_FrameCounter']
        lo_time = aotpy.Time('LO Loop Time', timestamps=lo_timestamps_list, frame_numbers=ho_frame_numbers.tolist())

        with importlib.resources.as_file(eris_data_path / 'lo_subap.fits') as p:
            subaperture_mask = image_from_file(p, name='LO WFS SUBAPERTURE MASK')
        n_valid_subapertures = np.count_nonzero(subaperture_mask.data != -1)

        reference = self._stack_slopes(fits.getdata(self._path / 'LOAcq.DET1.REFSLP_WITH_OFFSETS.fits'),
                                       slope_axis=1)[0]
        lo_wfs = aotpy.ShackHartmann(
            uid='LO WFS',
            source=ngs,
            n_valid_subapertures=n_valid_subapertures,
            measurements=aotpy.Image('LO Gradients', self._stack_slopes(lo_loop_frame['Gradients'], slope_axis=1)),
            ref_measurements=aotpy.Image('LOAcq.DET1.REFSLP_WITH_OFFSETS', reference),
            subaperture_mask=subaperture_mask,
            mask_offsets=[aotpy.Coordinates(0, 0)],
            subaperture_intensities=aotpy.Image('LO Intensities', lo_loop_frame['Intensities'])
        )

        lo_pix_fc = lo_pix_frame['FrameCounter']
        aux_fc = np.full_like(lo_pix_fc, -1, dtype=np.int32)
        x = 0
        for i, fc_loop in enumerate(lo_loop_frame['FrameCounter']):
            for j, fc_pix in enumerate(lo_pix_fc[x:]):
                if fc_pix > fc_loop:
                    # we're already past the place
                    x += j
                    break
                if fc_pix == fc_loop:
                    x += j
                    aux_fc[x] = i
                    break
            else:
                break
        where = np.where(aux_fc != -1)
        mask = aux_fc[where]
        masked_lgs = lgs_frame_numbers[mask]
        step = int((masked_lgs[-1] - masked_lgs[0]) / (masked_lgs.size - 1))
        aux_fc[where] = masked_lgs

        first = where[0][0]
        diff = 0
        while True:
            diff += 1
            cur = first - diff
            if cur < 0:
                break
            aux_fc[cur] = masked_lgs[0] - diff * step

        last = where[0][-1]
        diff = 0
        while True:
            diff += 1
            cur = last + diff
            if cur > aux_fc.size - 1:
                break
            aux_fc[cur] = masked_lgs[-1] + diff * step

        lo_wfs.detector = aotpy.Detector(
            uid='LO DET1',
            dark=image_from_file(self._path / 'LOAcq.DET1.DARK.fits'),
            weight_map=image_from_file(self._path / 'LOAcq.DET1.WEIGHT.fits'),
            sky_background=image_from_file(self._path / 'LOAcq.DET1.BACKGROUND.fits'),
            pixel_intensities=aotpy.Image(name='LO Pixels',
                                          data=self._get_pixel_data_from_table(lo_pix_frame),
                                          time=aotpy.Time('LO Pixel Time',
                                                          frame_numbers=aux_fc.tolist()))
        )
        self.system.wavefront_sensors.append(lo_wfs)
        lo_wfs.subaperture_size = \
            lo_wfs.detector.pixel_intensities.data.shape[0] // lo_wfs.subaperture_mask.data.shape[0]

        s2m = self._stack_slopes(fits.getdata(self._path / 'LOCtr.SENSOR_2_MODES.fits'), slope_axis=1)

        self.system.loops.append(aotpy.ControlLoop(
            uid='Low-order loop',
            input_sensor=lo_wfs,
            commanded_corrector=self.dsm,
            commands=aotpy.Image('LO_DSM_positions', lo_loop_frame['LO_DSM_positions']),
            modal_coefficients=aotpy.Image('LO_Modal', lo_loop_frame['LO_Modal']),
            # ref_commands=aotpy.Image('LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS',
            #                         fits.getdata(path_lgs / 'LGSCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]),
            time=lo_time,
            framerate=fits.getheader(self._path / 'LODet.CFG.DYNAMIC.fits')['FREQ'],
            time_filter_num=aotpy.Image('LOCtr.A_TERMS', fits.getdata(self._path / 'LOCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('LOCtr.B_TERMS', fits.getdata(self._path / 'LOCtr.B_TERMS.fits').T),
            measurements_to_modes=aotpy.Image('LOCtr.SENSOR_2_MODES', s2m),
        ))

        trombone = aotpy.LinearStage(
            uid='Trombone',
            telescope=self.system.main_telescope
        )
        self.system.wavefront_correctors.append(trombone)

        s2m = self._stack_slopes(fits.getdata(self._path / 'TruthCtr.SENSOR_2_MODES.fits'), slope_axis=1)
        self.system.loops.append(aotpy.ControlLoop(
            uid='Truth loop',
            input_sensor=lo_wfs,
            commanded_corrector=trombone,
            time_filter_num=aotpy.Image('TruthCtr.A_TERMS', fits.getdata(self._path / 'TruthCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('TruthCtr.B_TERMS', fits.getdata(self._path / 'TruthCtr.B_TERMS.fits').T),
            measurements_to_modes=aotpy.Image('TruthCtr.SENSOR_2_MODES', s2m),
        ))

    def _handle_ngs_data(self, ho_loop_file, ho_pixel_file):
        self.system.ao_mode = 'SCAO'
        ho_loop_frame = fits.getdata(ho_loop_file, extname='HOLoopFrame')
        ho_pix_frame = fits.getdata(ho_pixel_file, extname='HOPixelFrame')

        ho_timestamps = ho_loop_frame['Seconds'] + ho_loop_frame['USeconds'] / 1.e6
        self.system.date_beginning = datetime.utcfromtimestamp(ho_timestamps[0])
        self.system.date_end = datetime.utcfromtimestamp(ho_timestamps[-1])
        ho_frame_numbers = ho_loop_frame['FrameCounter']
        ho_time = aotpy.Time('HO Loop Time', timestamps=ho_timestamps.tolist(),
                             frame_numbers=ho_frame_numbers.tolist())

        ngs = aotpy.NaturalGuideStar('NGS')
        self.system.sources.append(ngs)

        eris_data_path = importlib.resources.files('aotpy.data') / 'ERIS'
        with importlib.resources.as_file(eris_data_path / 'ho_subap.fits') as p:
            subaperture_mask = image_from_file(p, name='LO WFS SUBAPERTURE MASK')
        n_valid_subapertures = np.count_nonzero(subaperture_mask.data != -1)

        reference = self._stack_slopes(fits.getdata(self._path / 'HOAcq.DET1.REFSLP_WITH_OFFSETS.fits'),
                                       slope_axis=1)[0]
        ho_wfs = aotpy.ShackHartmann(
            uid='HO WFS',
            source=ngs,
            n_valid_subapertures=n_valid_subapertures,
            measurements=aotpy.Image('Gradients', self._stack_slopes(ho_loop_frame['Gradients'], slope_axis=1)),
            ref_measurements=aotpy.Image('HOAcq.DET1.REFSLP_WITH_OFFSETS.fits', reference),
            subaperture_mask=subaperture_mask,
            mask_offsets=[aotpy.Coordinates(0, 0)],
            subaperture_intensities=aotpy.Image('Intensities', ho_loop_frame['Intensities'])
        )
        self.system.wavefront_sensors.append(ho_wfs)

        ho_wfs.detector = aotpy.Detector(
            uid='DET1',
            dark=image_from_file(self._path / 'HOAcq.DET1.DARK.fits'),
            weight_map=image_from_file(self._path / 'HOAcq.DET1.WEIGHT.fits'),
            sky_background=image_from_file(self._path / 'HOAcq.DET1.BACKGROUND.fits'),
            pixel_intensities=aotpy.Image(name='HO Pixels',
                                          data=self._get_pixel_data_from_table(ho_pix_frame),
                                          time=aotpy.Time('HO Pixel Time',
                                                          frame_numbers=ho_pix_frame['FrameCounter'].tolist()))
        )
        ho_wfs.subaperture_size = \
            ho_wfs.detector.pixel_intensities.data.shape[0] // ho_wfs.subaperture_mask.data.shape[0]

        s2m = self._stack_slopes(fits.getdata(self._path / 'CLMatrixOptimiser.S2M.fits'), slope_axis=1)
        m2s = self._stack_slopes(fits.getdata(self._path / 'CLMatrixOptimiser.M2S.fits'), slope_axis=0)

        self.system.loops.append(aotpy.ControlLoop(
            uid='High-order loop',
            input_sensor=ho_wfs,
            commanded_corrector=self.dsm,
            commands=aotpy.Image('DSM_positions', ho_loop_frame['DSM_Positions']),
            ref_commands=aotpy.Image('HOCtr.ACT_POS_REF_MAP_WITH_OFFSETS',
                                     fits.getdata(self._path / 'HOCtr.ACT_POS_REF_MAP_WITH_OFFSETS.fits')[:, 0]),
            time=ho_time,
            framerate=fits.getheader(self._path / 'HODet.CFG.DYNAMIC.fits')['FREQ'],
            time_filter_num=aotpy.Image('HOCtr.A_TERMS', fits.getdata(self._path / 'HOCtr.A_TERMS.fits').T),
            time_filter_den=aotpy.Image('HOCtr.B_TERMS', fits.getdata(self._path / 'HOCtr.B_TERMS.fits').T),
            measurements_to_modes=aotpy.Image('CLMatrixOptimiser.S2M', s2m),
            modes_to_commands=image_from_file(self._path / 'CLMatrixOptimiser.M2V.fits'),
            commands_to_modes=image_from_file(self._path / 'CLMatrixOptimiser.V2M.fits'),
            modes_to_measurements=aotpy.Image('CLMatrixOptimiser.M2S', m2s),
        ))

    def _get_eso_telescope_name(self) -> str:
        return 'ESO-VLT-U4'

    def _get_eso_ao_name(self) -> str:
        return 'ERIS'
