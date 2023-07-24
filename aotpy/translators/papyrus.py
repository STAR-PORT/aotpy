"""
This module contains a class for translating telemetry data produced by the ALPAO RTC, part of the PAPYRUS system. 
It assumes the MATLAB files produced by ALPAO RTC have been converted to use struct objects instead of classes. 
The custom MATLAB function makeTelemetryFileReadable is made for this.
Note that the use of scipy.io.loadmat to read the .mat files in python suppose that these .mat files were 
saved with the version -v7 of matlab or anterior. scipy.io.loadmat does not work for files saved with matlab -v7.3.
"""

import importlib.resources

import datetime
import numpy as np

import aotpy
from aotpy.io import image_from_file
from .base import BaseTranslator

try:
    from scipy.io import loadmat
except (ImportError, ModuleNotFoundError):
    loadmat = None


class PAPYRUSTranslator(BaseTranslator):
    def __init__(self, file) -> None:
        if loadmat is None:
            raise ImportError("Translating PAPYRUS data requires the scipy module.")
        data = loadmat(file, simplify_cells=True)['data']

        # common fields between SH and PYWFS telemetry

        self.system = aotpy.AOSystem(ao_mode='SCAO')

        papyrus_data_path = importlib.resources.files('aotpy.data') / 'PAPYRUS'
        with importlib.resources.as_file(papyrus_data_path / 'T152_pupil.fits') as p:
            pupil_mask = image_from_file(p, name='T152 PUPIL')

        self.system.main_telescope = aotpy.MainTelescope(
            uid="T152",
            enclosing_diameter=1.5,
            inscribed_diameter=1.5,
            pupil_mask=pupil_mask)

        actuator_coordinates = [aotpy.Coordinates(coor[0], coor[1]) for coor in data['wfcCommand']['coordinates']]
        dm = aotpy.DeformableMirror(uid="DM_ALPAO_241",
                                    telescope=self.system.main_telescope,
                                    n_valid_actuators=data['wfc']['offset']['values'].size,
                                    actuator_coordinates=actuator_coordinates,
                                    pupil_mask=self.system.main_telescope.pupil_mask)
        self.system.wavefront_correctors.append(dm)

        # read timestamps
        frame_number_measurements = np.array(data['wfsSlopesMetaData']['frameid'], dtype=float).tolist()
        time_stamp_measurements = (np.array(data['wfsSlopesMetaData']['timestamp'], dtype=float) / 1e9).tolist()
        frame_number_pixel_intensities = np.array(data['wfsImagesMetaData']['frameid'], dtype=float).tolist()
        time_stamp_pixel_intensities = (np.array(data['wfsImagesMetaData']['timestamp'], dtype=float) / 1e9).tolist()
        frame_number_commands = np.array(data['wfcCommandMetaData']['frameid'], dtype=float).tolist()
        time_stamp_commands = (np.array(data['wfcCommandMetaData']['timestamp'], dtype=float) / 1e9).tolist()

        # extract acquisition date from time stamps
        self.system.date_beginning = datetime.datetime.fromtimestamp(time_stamp_commands[0])
        self.system.date_end = datetime.datetime.fromtimestamp(time_stamp_commands[-1])

        time_measurements = aotpy.Time(uid='Measurements time',
                                       timestamps=time_stamp_measurements,
                                       frame_numbers=frame_number_measurements)

        time_pixel_intensities = aotpy.Time(uid='WFS detector frames time',
                                            timestamps=time_stamp_pixel_intensities,
                                            frame_numbers=frame_number_pixel_intensities)

        time_commands = aotpy.Time(uid='Commands time',
                                   timestamps=time_stamp_commands,
                                   frame_numbers=frame_number_commands)

        if data['wfsUid'] == 'ACE.REMOTE.AlpaoSHS.16':  # if SH telemetry, run SH translator script
            self.system.config = 'WFS : SH'
            self.system.sources = [aotpy.NaturalGuideStar(uid=data['source'])]

            slope_x = data['wfsSlopes']['sx']
            slope_y = data['wfsSlopes']['sy']
            measurements = np.empty((slope_x.shape[1], 2, slope_x.shape[0]))
            measurements[:, 0, :] = np.transpose(slope_x)
            measurements[:, 1, :] = np.transpose(slope_y)

            subaperture_mask = np.array(data['wfsSlopes']['mask'], dtype=int)
            subaperture_mask[np.where(subaperture_mask == 0)] = -1
            index = 0

            for j in range(subaperture_mask.shape[1]):
                for i in range(subaperture_mask.shape[0]):
                    if subaperture_mask[i, j] != -1:
                        subaperture_mask[i, j] = index
                        index += 1

            sh = aotpy.ShackHartmann(uid=data['wfsUid'],
                                     source=self.system.sources[0],
                                     n_valid_subapertures=data['wfsSlopes']['mask'].sum(),
                                     measurements=aotpy.Image(name="PAPYRUS Sh Slopes",
                                                              data=measurements,
                                                              time=time_measurements),
                                     subaperture_mask=aotpy.Image('SH subaperture mask', subaperture_mask))

            dark = data['detector']['dark']
            detector = aotpy.Detector(uid='cblue One',
                                      readout_noise=3,
                                      pixel_intensities=aotpy.Image(name="cblue Frames",
                                                                    data=np.moveaxis(data['wfsImages'], -1, 0),
                                                                    unit='Cblue One ADU',
                                                                    time=time_pixel_intensities),
                                      frame_rate=float(data['detector']['frameRate']),
                                      integration_time=data['detector']['exposureTime'],
                                      gain=data['detector']['gain'],
                                      binning=np.max(np.array(data['detector']['binning'])),
                                      dark=aotpy.Image('Ocam2K dark', dark),
                                      pixel_scale=5.8 / 3600 / 16 * np.pi / 180,
                                      # on-sky angular size of one pixel of SH detector [rad]
                                      type='CMOS')
            sh.detector = detector
            self.system.wavefront_sensors.append(sh)

            im = data['calibrator']['interactionMatrix']
            interaction_matrix = np.empty((im.shape[0] // 2, 2, im.shape[1]))
            interaction_matrix[:, 0, :] = im[::2]
            interaction_matrix[:, 1, :] = im[1::2]

            cm = data['loop']['commandMatrix']
            control_matrix = np.empty((cm.shape[0], 2, cm.shape[1] // 2))
            control_matrix[:, 0, :] = cm[:, ::2]
            control_matrix[:, 1, :] = cm[:, 1::2]

            modes_to_commands = data['wfc']['modeToCommand']

            modal_coefficients = np.transpose(data['loop']['calibrator']['ModeGains'])

            gain = data['loop']['controller']['Gain']
            if gain == 0:
                closed = False
            else:
                closed = True

            ref_commands = data['wfc']['offset']['values']

            loop = aotpy.ControlLoop(
                uid=data['uid'],
                commanded_corrector=dm,
                input_sensor=sh,
                commands=aotpy.Image(name='PAPYRUS DM Commands',
                                     data=np.transpose(data['wfcCommand']['values']),
                                     unit=data['wfcCommand']['unit'],
                                     time=time_commands),
                interaction_matrix=aotpy.Image("PAPYRUS Interaction Matrix", interaction_matrix),
                control_matrix=aotpy.Image("PAPYRUS Control Matrix", control_matrix),
                modes_to_commands=aotpy.Image("M2C", modes_to_commands),
                modal_coefficients=aotpy.Image('loop modal gains', modal_coefficients),
                closed=closed,
                ref_commands=aotpy.Image('dm flat(i.e closed on static aberations)', ref_commands),
                time_filter_num=aotpy.Image("Loop gain", np.array([float(gain)])))
            self.system.loops.append(loop)

        else:  # run pyramid telemetry translator script
            self.system.config = 'WFS : Pyramid'
            self.system.sources = [aotpy.NaturalGuideStar(uid='NGS')]

            slope_x = data['wfsSlopes']['sx']
            slope_y = data['wfsSlopes']['sy']
            measurements = np.empty((slope_x.shape[1], 2, slope_x.shape[0]))
            measurements[:, 0, :] = np.transpose(slope_x)
            measurements[:, 1, :] = np.transpose(slope_y)

            subaperture_mask = np.array(data['wfsSlopes']['mask'], dtype=int)
            subaperture_mask[np.where(subaperture_mask == 0)] = -1
            index = 0

            for j in range(subaperture_mask.shape[1]):
                for i in range(subaperture_mask.shape[0]):
                    if subaperture_mask[i, j] != -1:
                        subaperture_mask[i, j] = int(index)
                        index += 1

            pyramid = aotpy.Pyramid(uid=data['wfsUid'],
                                    source=self.system.sources[0],
                                    dimensions=2,
                                    n_valid_subapertures=data['wfsSlopes']['mask'].sum(),
                                    measurements=aotpy.Image("PAPYRUS Sh Slopes", measurements),
                                    n_sides=4,
                                    subaperture_mask=aotpy.Image('SH subaperture mask', subaperture_mask))

            dark = data['detector']['dark']
            detector = aotpy.Detector(uid='Ocam2K',
                                      readout_noise=0,
                                      pixel_intensities=aotpy.Image("Ocam2K frames",
                                                                    np.moveaxis(data['wfsImages'], -1, 0)),
                                      frame_rate=1500.0,
                                      integration_time=data['detector']['exposureTime'],
                                      gain=data['detector']['gain'],
                                      binning=np.max(np.array(data['detector']['binning'])),
                                      dark=aotpy.Image('Ocam2K dark', dark),
                                      type='EMCCD')
            pyramid.detector = detector
            self.system.wavefront_sensors.append(pyramid)

            im = data['calibrator']['interactionMatrix']
            interaction_matrix = np.empty((im.shape[0] // 2, 2, im.shape[1]))
            interaction_matrix[:, 0, :] = im[::2]
            interaction_matrix[:, 1, :] = im[1::2]

            cm = data['loop']['commandMatrix']
            control_matrix = np.empty((cm.shape[0], 2, cm.shape[1] // 2))
            control_matrix[:, 0, :] = cm[:, ::2]
            control_matrix[:, 1, :] = cm[:, 1::2]

            modes_to_commands = data['wfc']['modeToCommand']

            modal_coefficients = np.transpose(data['loop']['calibrator']['ModeGains'])

            gain = data['loop']['controller']['Gain']
            if gain == 0:
                closed = False
            else:
                closed = True

            ref_commands = data['wfc']['offset']['values']

            loop = aotpy.ControlLoop(
                uid=data['uid'],
                commanded_corrector=dm,
                input_sensor=pyramid,
                commands=aotpy.Image('PAPYRUS DM Commands', np.transpose(data['wfcCommand']['values'])),
                interaction_matrix=aotpy.Image("PAPYRUS Interaction Matrix", interaction_matrix),
                control_matrix=aotpy.Image("PAPYRUS Control Matrix", control_matrix),
                modes_to_commands=aotpy.Image("M2C", modes_to_commands),
                modal_coefficients=aotpy.Image('Loop modal gains', modal_coefficients),
                closed=closed,
                ref_commands=aotpy.Image('DM flat (i.e loop closed on static aberrations)', ref_commands),
                time_filter_num=aotpy.Image("Loop gain", np.array([float(gain)])))
            self.system.loops.append(loop)
