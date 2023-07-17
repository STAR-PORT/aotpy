"""
This module contains a class for translating data produced by ALPAO's PAPYRUS system. It assumes the MATLAB files
produced by PAPYRUS have been converted to use structs instead of classes.
"""

import importlib.resources

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

        self.system = aotpy.AOSystem(ao_mode='SCAO', name='PAPYRUS')

        self.system.sources = [aotpy.NaturalGuideStar(uid=data['source'])]

        papyrus_data_path = importlib.resources.files('aotpy.data') / 'PAPYRUS'
        with importlib.resources.as_file(papyrus_data_path / 'T152_pupil.fits') as p:
            pupil_mask = image_from_file(p, name='T152 PUPIL')

        self.system.main_telescope = aotpy.MainTelescope(
            uid="T152",
            enclosing_diameter=1.5,
            inscribed_diameter=1.5,
            pupil_mask=pupil_mask
        )

        actuator_coordinates = [aotpy.Coordinates(coor[0], coor[1]) for coor in data['wfcCommandStruct']['coordinates']]
        dm = aotpy.DeformableMirror(uid="DM_ALPAO_241",
                                    telescope=self.system.main_telescope,
                                    n_valid_actuators=data['wfc']['offsetStruct']['values'].size,
                                    actuator_coordinates=actuator_coordinates,
                                    pupil_mask=self.system.main_telescope.pupil_mask)
        self.system.wavefront_correctors.append(dm)

        slope_x = data['wfsSlopesStruct']['sx']
        slope_y = data['wfsSlopesStruct']['sy']
        measurements = np.empty((slope_x.shape[1], 2, slope_x.shape[0]))
        measurements[:, 0, :] = np.transpose(slope_x)
        measurements[:, 1, :] = np.transpose(slope_y)
        sh = aotpy.ShackHartmann(uid=data['wfsUid'],
                                 source=self.system.sources[0],
                                 n_valid_subapertures=data['wfsSlopesStruct']['mask'].sum(),
                                 measurements=aotpy.Image("papyrusShSlopes", measurements))

        detector = aotpy.Detector(uid='cblueOne',
                                  readout_noise=3,
                                  pixel_intensities=aotpy.Image("cblueFrames", np.moveaxis(data['wfsImages'], -1, 0)))
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

        loop = aotpy.ControlLoop(
            uid=data['uid'],
            commanded_corrector=dm,
            input_sensor=sh,
            commands=aotpy.Image('papyrusDmCommands', np.transpose(data['wfcCommandStruct']['values'])),
            interaction_matrix=aotpy.Image("papyrusInteractionMatrix", interaction_matrix),
            control_matrix=aotpy.Image("papyruscontrolMatrix", control_matrix))
        self.system.loops.append(loop)
