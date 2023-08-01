"""
This module contains classes and functions that enable writing AOT FITS files.
"""

import numbers
import os

import numpy as np
from astropy.io import fits

import aotpy
from . import _keywords as kw
from .utils import FITSFileImage, FITSURLImage, datetime_to_iso, card_from_metadatum
from ..base import SystemWriter

# NaN is defined here as a single precision float (32-bits), which is the lowest possible float precision in FITS.
# The goal is to ensure that low precision numpy arrays aren't unnecessarily upcasted just because of NaN.
_nan = np.single(np.nan)

# Given that integer values in AOT cannot be negative, we assume the lowest integer in a signed 16-bit integer is null.
# Implicitly, this means aotpy does not support unsigned integers.
_int_min = np.iinfo(np.int16).min


def write_system_to_fits(filename: str, system: aotpy.AOSystem, **kwargs) -> None:
    """
    Write `system` to file specified by `filename` using the FITS format.

    Parameters
    ----------
    filename
        Path to the file that will be written.
    system
        `AOSystem` to be written into a file.
    **kwargs
        Keyword arguments passed on as options to the file handling function.
    """
    FITSWriter(system).write(filename, **kwargs)


class FITSWriter(SystemWriter):
    def __init__(self, system: aotpy.AOSystem) -> None:
        self._system = system

        self._tables: dict[str, dict[
            str, tuple[aotpy.Referenceable, dict[str, numbers.Integral | numbers.Real | list | np.ndarray | str]]]] = {}
        """The outer dictionary converts from table name to the table dictionary. The table dictionary converts from the
        object uid to a tuple. This tuple contains the object itself and a dictionary that converts from the field names
        to the values present in the object."""

        self._images: dict[str, aotpy.Image] = {}

        self._handle_data()

        primary_hdu = self._create_primary_hdu()
        bintable_hdus = self._create_bintable_hdus()
        image_hdus = self._create_image_hdus()
        self._hdus = fits.HDUList([primary_hdu, *bintable_hdus, *image_hdus])

    def write(self, filename: str | os.PathLike, **kwargs) -> None:
        self.get_hdus().writeto(filename, **kwargs)

    def get_hdus(self) -> fits.HDUList:
        """
        Get the list of HDUs that compose the AOT FITS file for the initialized system.
        Returns
        -------
            `HDUList` that composes the AOT FITS file for the initialized system.
        """
        return self._hdus

    def _handle_data(self):
        for atm in self._system.atmosphere_params:
            self._handle_atmospheric_parameters(atm)

        self._handle_telescope(self._system.main_telescope, False)

        for src in self._system.sources:
            self._handle_source(src, False)

        for cam in self._system.scoring_cameras:
            self._handle_scoring_camera(cam)

        for wfs in self._system.wavefront_sensors:
            self._handle_wavefront_sensor(wfs, False)

        for cor in self._system.wavefront_correctors:
            self._handle_wavefront_corrector(cor, False)

        for loop in self._system.loops:
            self._handle_loops(loop)

    def _add_to_table(self, table_name: str, obj: aotpy.Referenceable, row: dict) -> bool:
        fields = kw.TABLE_FIELDS[table_name]
        table = self._tables.setdefault(table_name, {})

        if obj.uid in table:
            if obj is table[obj.uid][0]:
                return False  # already added this exact object, no need to add again
            raise ValueError(f"Repeated value '{obj.uid}' in '{kw.REFERENCE_UID}' column in {table_name}")

        if len(fields) != len(row):
            raise RuntimeError  # This should never happen

        converted = {}
        for field in fields.values():
            value = row[field.name]
            if value is None and field.mandatory:
                raise ValueError(
                    f"'{obj.uid}' missing value in mandatory column '{field.name}' on table '{table_name}'.")

            if field.format == kw.STRING_FORMAT:
                if value is None:
                    value = ''
                elif not isinstance(value, str) and not isinstance(value, np.ndarray) \
                        and np.issubdtype(value.dtype, np.character):
                    raise ValueError(f"Unxpected value in table '{table_name}' column '{field.name}'. "
                                     f"Expected '{field.format}' format, got: {value}.")

            elif field.format == kw.FLOAT_FORMAT:
                if value is None:
                    value = _nan
                elif not isinstance(value, numbers.Real):
                    raise ValueError(f"Unxpected value in table '{table_name}' column '{field.name}'. "
                                     f"Expected '{field.format}' format, got: {value}.")

            elif field.format == kw.INTEGER_FORMAT:
                if value is None:
                    value = _int_min
                elif not isinstance(value, numbers.Integral):
                    raise ValueError(f"Unxpected value in table '{table_name}' column '{field.name}'. "
                                     f"Expected '{field.format}' format, got: {value}.")

            elif field.format == kw.LIST_FORMAT:
                if value is None:
                    value = []
                else:
                    try:
                        value = [_nan if v is None else v for v in value]
                    except TypeError:
                        # Not iterable
                        raise ValueError(f"Unxpected value in table '{table_name}' column '{field.name}'. "
                                         f"Expected '{field.format}' format, got: {value}.") from None
            else:
                raise NotImplementedError

            converted[field.name] = value

        table[obj.uid] = (obj, converted)
        return True

    @staticmethod
    def _create_row_reference(uid: str) -> str:
        return f'{kw.ROW_REFERENCE}<{uid}>'

    def _handle_time(self, time: aotpy.Time) -> str | None:
        if time is None:
            return None
        if time.timestamps and time.frame_numbers and len(time.timestamps) != len(time.frame_numbers):
            raise ValueError(f"Error in Time '{time.uid}': If both 'timestamps' and 'frame_numbers' are non-null, they "
                             f"must have the same length.")
        row = {
            kw.REFERENCE_UID: time.uid,
            kw.TIME_TIMESTAMPS: time.timestamps,
            kw.TIME_FRAME_NUMBERS: time.frame_numbers
        }
        self._add_to_table(kw.TIME_TABLE, time, row)
        return self._create_row_reference(time.uid)

    def _handle_image(self, image: aotpy.Image) -> str | None:
        if image is None:
            return None
        self._handle_time(image.time)
        if isinstance(image, FITSFileImage):
            if image.index is not None:
                return f'{kw.FILE_REFERENCE}<{image.filename}>{image.index}'
            else:
                return f'{kw.FILE_REFERENCE}<{image.filename}>'
        if isinstance(image, FITSURLImage):
            if image.index is not None:
                return f'{kw.URL_REFERENCE}<{image.url}>{image.index}'
            else:
                return f'{kw.URL_REFERENCE}<{image.url}>'
        else:
            reference = f'{kw.INTERNAL_REFERENCE}<{image.name}>'
            if image.name in self._images:
                if image is self._images[image.name]:
                    # Image has already been handled before
                    return reference
                raise ValueError(f'Repeated image name {image.name}.')
            self._images[image.name] = image
            return reference

    def _handle_atmospheric_parameters(self, atm: aotpy.AtmosphericParameters) -> None:
        if atm is None:
            raise ValueError("'AOSystem.atmosphere_params' list cannot contain 'None' items.")

        row = {
            kw.REFERENCE_UID: atm.uid,
            kw.ATMOSPHERIC_PARAMETERS_WAVELENGTH: atm.wavelength,
            kw.TIME_REFERENCE: self._handle_time(atm.time),
            kw.ATMOSPHERIC_PARAMETERS_R0: atm.r0,
            kw.ATMOSPHERIC_PARAMETERS_FWHM: atm.fwhm,
            kw.ATMOSPHERIC_PARAMETERS_TAU0: atm.tau0,
            kw.ATMOSPHERIC_PARAMETERS_THETA0: atm.theta0,
            kw.ATMOSPHERIC_PARAMETERS_LAYERS_WEIGHT: self._handle_image(atm.layers_weight),
            kw.ATMOSPHERIC_PARAMETERS_LAYERS_HEIGHT: self._handle_image(atm.layers_height),
            kw.ATMOSPHERIC_PARAMETERS_LAYERS_L0: self._handle_image(atm.layers_l0),
            kw.ATMOSPHERIC_PARAMETERS_LAYERS_WIND_SPEED: self._handle_image(atm.layers_wind_speed),
            kw.ATMOSPHERIC_PARAMETERS_LAYERS_WIND_DIRECTION: self._handle_image(atm.layers_wind_direction),
            kw.TRANSFORMATION_MATRIX: self._handle_image(atm.transformation_matrix),
        }
        self._add_to_table(kw.ATMOSPHERIC_PARAMETERS_TABLE, atm, row)

    def _handle_aberration(self, abr: aotpy.Aberration) -> str | None:
        if abr is None:
            return None

        row = {
            kw.REFERENCE_UID: abr.uid,
            kw.ABERRATION_MODES: self._handle_image(abr.modes),
            kw.ABERRATION_COEFFICIENTS: self._handle_image(abr.coefficients),
            kw.ABERRATION_X_OFFSETS: [off.x for off in abr.offsets],
            kw.ABERRATION_Y_OFFSETS: [off.y for off in abr.offsets]
        }
        self._add_to_table(kw.ABERRATIONS_TABLE, abr, row)
        return self._create_row_reference(abr.uid)

    def _handle_telescope(self, tel: aotpy.Telescope, referenced: bool, llt: bool = False) -> str | None:
        if tel is None:
            if referenced:
                return None
            raise ValueError("'AOSystem.main_telescope' must not be 'None'.")

        main = False
        if isinstance(tel, aotpy.MainTelescope):
            if llt:
                raise ValueError("Referenced telescope must have type 'LaserLaunchTelescope'.")
            main = True
            tel_type = kw.TELESCOPE_TYPE_MAIN
        else:
            if not referenced:
                raise ValueError("'AOSystem.main_telescope' must have type 'MainTelescope'.")
            if isinstance(tel, aotpy.LaserLaunchTelescope):
                tel_type = kw.TELESCOPE_TYPE_LLT
            else:
                if llt:
                    raise ValueError("Referenced telescope must have type 'LaserLaunchTelescope'.")
                raise NotImplementedError

        if tel.segments is None:
            raise ValueError("Telescope 'segments' variable cannot be 'None'.")
        else:
            if isinstance(tel.segments, aotpy.Monolithic):
                segments_type = kw.TELESCOPE_SEGMENT_TYPE_MONOLITHIC
            elif isinstance(tel.segments, aotpy.HexagonalSegments):
                segments_type = kw.TELESCOPE_SEGMENT_TYPE_HEXAGON
            elif isinstance(tel.segments, aotpy.CircularSegments):
                segments_type = kw.TELESCOPE_SEGMENT_TYPE_CIRCLE
            else:
                raise NotImplementedError
            segments_size = tel.segments.size
            segments_x = [coord.x for coord in tel.segments.coordinates]
            segments_y = [coord.y for coord in tel.segments.coordinates]

        row = {
            kw.REFERENCE_UID: tel.uid,
            kw.TELESCOPE_TYPE: tel_type,
            kw.TELESCOPE_LATITUDE: tel.latitude,
            kw.TELESCOPE_LONGITUDE: tel.longitude,
            kw.TELESCOPE_ELEVATION: tel.elevation,
            kw.TELESCOPE_AZIMUTH: tel.azimuth,
            kw.TELESCOPE_PARALLACTIC: tel.parallactic,
            kw.TELESCOPE_PUPIL_MASK: self._handle_image(tel.pupil_mask),
            kw.TELESCOPE_PUPIL_ANGLE: tel.pupil_angle,
            kw.TELESCOPE_ENCLOSING_D: tel.enclosing_diameter,
            kw.TELESCOPE_INSCRIBED_D: tel.inscribed_diameter,
            kw.TELESCOPE_OBSTRUCTION_D: tel.obstruction_diameter,
            kw.TELESCOPE_SEGMENTS_TYPE: segments_type,
            kw.TELESCOPE_SEGMENTS_SIZE: segments_size,
            kw.TELESCOPE_SEGMENTS_X: segments_x,
            kw.TELESCOPE_SEGMENTS_Y: segments_y,
            kw.TRANSFORMATION_MATRIX: self._handle_image(tel.transformation_matrix),
            kw.ABERRATION_REFERENCE: self._handle_aberration(tel.aberration)
        }
        if self._add_to_table(kw.TELESCOPES_TABLE, tel, row) and referenced and main:
            raise ValueError(
                "Telescope references cannot reference a 'MainTelescope' object that is not AOSystem.main_telescope.")
        return self._create_row_reference(tel.uid)

    def _handle_source(self, src: aotpy.Source, referenced: bool) -> str | None:
        if src is None:
            if referenced:
                return None
            raise ValueError("'AOSystem.sources' list cannot contain 'None' items.")

        if isinstance(src, aotpy.ScienceStar):
            src_type = kw.SOURCE_TYPE_SCIENCE_STAR
        elif isinstance(src, aotpy.NaturalGuideStar):
            src_type = kw.SOURCE_TYPE_NATURAL_GUIDE_STAR
        elif isinstance(src, aotpy.SodiumLaserGuideStar):
            src_type = kw.SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR
            row = {
                kw.REFERENCE_UID: src.uid,
                kw.SOURCE_SODIUM_LGS_HEIGHT: src.height,
                kw.SOURCE_SODIUM_LGS_PROFILE: self._handle_image(src.profile),
                kw.SOURCE_SODIUM_LGS_ALTITUDES: src.altitudes,
                kw.LASER_LAUNCH_TELESCOPE_REFERENCE: self._handle_telescope(src.laser_launch_telescope, True)
            }
            self._add_to_table(kw.SOURCES_SODIUM_LGS_TABLE, src, row)
        elif isinstance(src, aotpy.RayleighLaserGuideStar):
            src_type = kw.SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR
            row = {
                kw.REFERENCE_UID: src.uid,
                kw.SOURCE_RAYLEIGH_LGS_DISTANCE: src.distance,
                kw.SOURCE_RAYLEIGH_LGS_DEPTH: src.depth,
                kw.LASER_LAUNCH_TELESCOPE_REFERENCE: self._handle_telescope(src.laser_launch_telescope, True)
            }
            self._add_to_table(kw.SOURCES_RAYLEIGH_LGS_TABLE, src, row)
        else:
            raise NotImplementedError

        row = {
            kw.REFERENCE_UID: src.uid,
            kw.SOURCE_TYPE: src_type,
            kw.SOURCE_RIGHT_ASCENSION: src.right_ascension,
            kw.SOURCE_DECLINATION: src.declination,
            kw.SOURCE_ELEVATION_OFFSET: src.elevation_offset,
            kw.SOURCE_AZIMUTH_OFFSET: src.azimuth_offset,
            kw.SOURCE_WIDTH: src.width
        }
        if self._add_to_table(kw.SOURCES_TABLE, src, row) and referenced:
            raise ValueError(f"Source '{src.uid}' is referenced but not present in 'AOSystem.sources'")
        return self._create_row_reference(src.uid)

    def _handle_detector(self, det: aotpy.Detector) -> str | None:
        if det is None:
            return None

        row = {
            kw.REFERENCE_UID: det.uid,
            kw.DETECTOR_TYPE: det.type,
            kw.DETECTOR_SAMPLING_TECHNIQUE: det.sampling_technique,
            kw.DETECTOR_SHUTTER_TYPE: det.shutter_type,
            kw.DETECTOR_FLAT_FIELD: self._handle_image(det.flat_field),
            kw.DETECTOR_READOUT_NOISE: det.readout_noise,
            kw.DETECTOR_PIXEL_INTENSITIES: self._handle_image(det.pixel_intensities),
            kw.DETECTOR_INTEGRATION_TIME: det.integration_time,
            kw.DETECTOR_COADDS: det.coadds,
            kw.DETECTOR_DARK: self._handle_image(det.dark),
            kw.DETECTOR_WEIGHT_MAP: self._handle_image(det.weight_map),
            kw.DETECTOR_QUANTUM_EFFICIENCY: det.quantum_efficiency,
            kw.DETECTOR_PIXEL_SCALE: det.pixel_scale,
            kw.DETECTOR_BINNING: det.binning,
            kw.DETECTOR_BANDWIDTH: det.bandwidth,
            kw.DETECTOR_TRANSMISSION_WAVELENGTH: det.transmission_wavelength,
            kw.DETECTOR_TRANSMISSION: det.transmission,
            kw.DETECTOR_SKY_BACKGROUND: self._handle_image(det.sky_background),
            kw.DETECTOR_GAIN: det.gain,
            kw.DETECTOR_EXCESS_NOISE: det.excess_noise,
            kw.DETECTOR_FILTER: det.filter,
            kw.DETECTOR_BAD_PIXEL_MAP: self._handle_image(det.bad_pixel_map),
            kw.DETECTOR_DYNAMIC_RANGE: det.dynamic_range,
            kw.DETECTOR_READOUT_RATE: det.readout_rate,
            kw.DETECTOR_FRAME_RATE: det.frame_rate,
            kw.TRANSFORMATION_MATRIX: self._handle_image(det.transformation_matrix),
        }
        self._add_to_table(kw.DETECTORS_TABLE, det, row)
        return self._create_row_reference(det.uid)

    def _handle_scoring_camera(self, cam: aotpy.ScoringCamera) -> None:
        if cam is None:
            raise ValueError("'AOSystem.scoring_cameras' list cannot contain 'None' items.")

        row = {
            kw.REFERENCE_UID: cam.uid,
            kw.SCORING_CAMERA_PUPIL_MASK: self._handle_image(cam.pupil_mask),
            kw.SCORING_CAMERA_WAVELENGTH: cam.wavelength,
            kw.TRANSFORMATION_MATRIX: self._handle_image(cam.transformation_matrix),
            kw.DETECTOR_REFERENCE: self._handle_detector(cam.detector),
            kw.ABERRATION_REFERENCE: self._handle_aberration(cam.aberration)
        }
        self._add_to_table(kw.SCORING_CAMERAS_TABLE, cam, row)

    def _handle_wavefront_sensor(self, wfs: aotpy.WavefrontSensor, referenced: bool) -> str | None:
        if wfs is None:
            if referenced:
                return None
            raise ValueError("'AOSystem.wavefront_sensors' list cannot contain 'None' items.")

        if isinstance(wfs, aotpy.ShackHartmann):
            wfs_type = kw.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN
            row = {
                kw.REFERENCE_UID: wfs.uid,
                kw.WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROIDING_ALGORITHM: wfs.centroiding_algorithm,
                kw.WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROID_GAINS: self._handle_image(wfs.centroid_gains),
                kw.WAVEFRONT_SENSOR_SHACK_HARTMANN_SPOT_FWHM: self._handle_image(wfs.spot_fwhm)
            }
            self._add_to_table(kw.WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE, wfs, row)
        elif isinstance(wfs, aotpy.Pyramid):
            wfs_type = kw.WAVEFRONT_SENSOR_TYPE_PYRAMID
            row = {
                kw.REFERENCE_UID: wfs.uid,
                kw.WAVEFRONT_SENSOR_PYRAMID_N_SIDES: wfs.n_sides,
                kw.WAVEFRONT_SENSOR_PYRAMID_MODULATION: wfs.modulation
            }
            self._add_to_table(kw.WAVEFRONT_SENSORS_PYRAMID_TABLE, wfs, row)
        else:
            raise NotImplementedError

        row = {
            kw.REFERENCE_UID: wfs.uid,
            kw.WAVEFRONT_SENSOR_TYPE: wfs_type,
            kw.SOURCE_REFERENCE: self._handle_source(wfs.source, True),
            kw.WAVEFRONT_SENSOR_DIMENSIONS: wfs.dimensions,
            kw.WAVEFRONT_SENSOR_N_VALID_SUBAPERTURES: wfs.n_valid_subapertures,
            kw.WAVEFRONT_SENSOR_MEASUREMENTS: self._handle_image(wfs.measurements),
            kw.WAVEFRONT_SENSOR_REF_MEASUREMENTS: self._handle_image(wfs.ref_measurements),
            kw.WAVEFRONT_SENSOR_SUBAPERTURE_MASK: self._handle_image(wfs.subaperture_mask),
            kw.WAVEFRONT_SENSOR_MASK_X_OFFSETS: [off.x for off in wfs.mask_offsets],
            kw.WAVEFRONT_SENSOR_MASK_Y_OFFSETS: [off.y for off in wfs.mask_offsets],
            kw.WAVEFRONT_SENSOR_SUBAPERTURE_SIZE: wfs.subaperture_size,
            kw.WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES: self._handle_image(wfs.subaperture_intensities),
            kw.WAVEFRONT_SENSOR_WAVELENGTH: wfs.wavelength,
            kw.WAVEFRONT_SENSOR_OPTICAL_GAIN: self._handle_image(wfs.optical_gain),
            kw.TRANSFORMATION_MATRIX: self._handle_image(wfs.transformation_matrix),
            kw.DETECTOR_REFERENCE: self._handle_detector(wfs.detector),
            kw.ABERRATION_REFERENCE: self._handle_aberration(wfs.aberration),
            kw.NCPA_REFERENCE: self._handle_aberration(wfs.non_common_path_aberration)
        }
        if self._add_to_table(kw.WAVEFRONT_SENSORS_TABLE, wfs, row) and referenced:
            raise ValueError(
                f"Wavefront Sensor '{wfs.uid}' is referenced but not present in 'AOSystem.wavefront_sensors'")
        return self._create_row_reference(wfs.uid)

    def _handle_wavefront_corrector(self, cor: aotpy.WavefrontCorrector, referenced: bool) -> str | None:
        if cor is None:
            if referenced:
                return None
            raise ValueError("'AOSystem.wavefront_correctors' list cannot contain 'None' items.")

        if isinstance(cor, aotpy.DeformableMirror):
            cor_type = kw.WAVEFRONT_CORRECTOR_TYPE_DM
            row = {
                kw.REFERENCE_UID: cor.uid,
                kw.WAVEFRONT_CORRECTOR_DM_ACTUATORS_X: [coord.x for coord in cor.actuator_coordinates],
                kw.WAVEFRONT_CORRECTOR_DM_ACTUATORS_Y: [coord.y for coord in cor.actuator_coordinates],
                kw.WAVEFRONT_CORRECTOR_DM_INFLUENCE_FUNCTION: self._handle_image(cor.influence_function),
                kw.WAVEFRONT_CORRECTOR_DM_STROKE: cor.stroke
            }
            self._add_to_table(kw.WAVEFRONT_CORRECTORS_DM_TABLE, cor, row)
        elif isinstance(cor, aotpy.TipTiltMirror):
            cor_type = kw.WAVEFRONT_CORRECTOR_TYPE_TTM
        elif isinstance(cor, aotpy.LinearStage):
            cor_type = kw.WAVEFRONT_CORRECTOR_TYPE_LS
        else:
            raise NotImplementedError
        row = {
            kw.REFERENCE_UID: cor.uid,
            kw.WAVEFRONT_CORRECTOR_TYPE: cor_type,
            kw.TELESCOPE_REFERENCE: self._handle_telescope(cor.telescope, True),
            kw.WAVEFRONT_CORRECTOR_N_VALID_ACTUATORS: cor.n_valid_actuators,
            kw.WAVEFRONT_CORRECTOR_PUPIL_MASK: self._handle_image(cor.pupil_mask),
            kw.WAVEFRONT_CORRECTOR_TFZ_NUM: cor.tfz_num,
            kw.WAVEFRONT_CORRECTOR_TFZ_DEN: cor.tfz_den,
            kw.TRANSFORMATION_MATRIX: self._handle_image(cor.transformation_matrix),
            kw.ABERRATION_REFERENCE: self._handle_aberration(cor.aberration)
        }
        if self._add_to_table(kw.WAVEFRONT_CORRECTORS_TABLE, cor, row) and referenced:
            raise ValueError(
                f"Wavefront Corrector '{cor.uid}' is referenced but not present in 'AOSystem.wavefront_correctors'")
        return self._create_row_reference(cor.uid)

    def _handle_loops(self, loop: aotpy.Loop) -> None:
        if loop is None:
            raise ValueError("'AOSystem.loops' list cannot contain 'None' items.")

        if isinstance(loop, aotpy.ControlLoop):
            loop_type = kw.LOOPS_TYPE_CONTROL
            row = {
                kw.REFERENCE_UID: loop.uid,
                kw.LOOPS_CONTROL_INPUT_SENSOR: self._handle_wavefront_sensor(loop.input_sensor, True),
                kw.LOOPS_CONTROL_MODES: self._handle_image(loop.modes),
                kw.LOOPS_CONTROL_MODAL_COEFFICIENTS: self._handle_image(loop.modal_coefficients),
                kw.LOOPS_CONTROL_CONTROL_MATRIX: self._handle_image(loop.control_matrix),
                kw.LOOPS_CONTROL_MEASUREMENTS_TO_MODES: self._handle_image(loop.measurements_to_modes),
                kw.LOOPS_CONTROL_MODES_TO_COMMANDS: self._handle_image(loop.modes_to_commands),
                kw.LOOPS_CONTROL_INTERACTION_MATRIX: self._handle_image(loop.interaction_matrix),
                kw.LOOPS_CONTROL_COMMANDS_TO_MODES: self._handle_image(loop.commands_to_modes),
                kw.LOOPS_CONTROL_MODES_TO_MEASUREMENTS: self._handle_image(loop.modes_to_measurements),
                kw.LOOPS_CONTROL_RESIDUAL_COMMANDS: self._handle_image(loop.residual_commands)
            }
            self._add_to_table(kw.LOOPS_CONTROL_TABLE, loop, row)
        elif isinstance(loop, aotpy.OffloadLoop):
            loop_type = kw.LOOPS_TYPE_OFFLOAD
            row = {
                kw.REFERENCE_UID: loop.uid,
                kw.LOOPS_OFFLOAD_INPUT_CORRECTOR: self._handle_wavefront_corrector(loop.input_corrector, True),
                kw.LOOPS_OFFLOAD_OFFLOAD_MATRIX: self._handle_image(loop.offload_matrix)
            }
            self._add_to_table(kw.LOOPS_OFFLOAD_TABLE, loop, row)
        else:
            raise NotImplementedError
        row = {
            kw.REFERENCE_UID: loop.uid,
            kw.LOOPS_TYPE: loop_type,
            kw.LOOPS_COMMANDED: self._handle_wavefront_corrector(loop.commanded_corrector, True),
            kw.TIME_REFERENCE: self._handle_time(loop.time),
            kw.LOOPS_STATUS: kw.LOOPS_STATUS_CLOSED if loop.closed else kw.LOOPS_STATUS_OPEN,
            kw.LOOPS_COMMANDS: self._handle_image(loop.commands),
            kw.LOOPS_REF_COMMANDS: self._handle_image(loop.ref_commands),
            kw.LOOPS_FRAMERATE: loop.framerate,
            kw.LOOPS_DELAY: loop.delay,
            kw.LOOPS_TIME_FILTER_NUM: self._handle_image(loop.time_filter_num),
            kw.LOOPS_TIME_FILTER_DEN: self._handle_image(loop.time_filter_den)
        }
        self._add_to_table(kw.LOOPS_TABLE, loop, row)

    def _create_primary_hdu(self) -> fits.PrimaryHDU:
        hdr = fits.Header()
        hdr[kw.AOT_VERSION] = kw.CURRENT_AOT_VERSION
        hdr[kw.AOT_TIMESYS] = kw.AOT_TIMESYS_UTC

        if self._system.ao_mode not in kw.AOT_AO_MODE_SET:
            raise ValueError(f"'AOSystem.ao_mode' must be one of: {str(kw.AOT_AO_MODE_SET)[1:-1]}")
        hdr[kw.AOT_AO_MODE] = self._system.ao_mode

        hdr[kw.AOT_DATE_BEG] = datetime_to_iso(self._system.date_beginning)
        hdr[kw.AOT_DATE_END] = datetime_to_iso(self._system.date_end)

        if self._system.name is not None:
            hdr[kw.AOT_SYSTEM_NAME] = self._system.name
        if self._system.strehl_ratio is not None:
            hdr[kw.AOT_STREHL_RATIO] = self._system.strehl_ratio
        if self._system.temporal_error is not None:
            hdr[kw.AOT_TEMPORAL_ERROR] = self._system.temporal_error
        if self._system.config is not None:
            hdr[kw.AOT_CONFIG] = self._system.config

        hdr.extend([card_from_metadatum(md) for md in self._system.metadata])
        return fits.PrimaryHDU(header=hdr)

    def _create_bintable_hdus(self) -> list[fits.BinTableHDU]:
        hdus = []
        for table_name in kw.TABLE_SEQUENCE:
            try:
                table = self._tables[table_name]
            except KeyError:
                if table_name in kw.SECONDARY_TABLE_SET:
                    # If the table is not mandatory and it is not being used, just don't create it
                    continue
                # Otherwise, we need to create an empty table
                table = {}

            columns = []
            for field in kw.TABLE_FIELDS[table_name].values():
                values = [tup[1][field.name] for tup in table.values()]

                if field.format == kw.LIST_FORMAT:
                    array = np.empty(len(values), dtype=np.object_)

                    # Convert every entry to numpy array, if any 64-bit floats are detected we need to use 'D' format.
                    flag = False
                    for i, l in enumerate(values):
                        aux = np.array(l)
                        if aux.dtype != np.float32:
                            aux = aux.astype(np.float64, casting='safe')
                            flag = True
                        array[i] = aux

                    # We always use the 'Q' format, meaning the VLAs use a 64-bit descriptor.
                    # If we used the 'P' format we could potentially save some storage (64-bits per row per VLA column).
                    # However, in scenarios with very large amounts of VLA data, the heap offset could overflow.
                    # This is hard to calculate ahead of time, so we just prefer to take the small bump to file size.
                    # Realistically, this size increase is insignificant when compared to the actual data being stored.
                    col = fits.Column(name=field.name, format=f"Q{'D' if flag else 'E'}", unit=field.unit, array=array)
                else:
                    # Convert to numpy array and try as much as possible to keep the resulting dtype
                    array = np.array(values)
                    if field.format == kw.STRING_FORMAT:
                        col = fits.Column(name=field.name, format=f'{np.char._get_num_chars(array)}A', unit=field.unit,
                                          array=array)
                    elif field.format == kw.INTEGER_FORMAT:
                        if (t := array.dtype) == np.int16:
                            f = '1I'
                        elif t == np.int32:
                            f = '1J'
                        else:
                            if t != np.int64 and array.size > 0:
                                # If not int16, int32 or int64 make one last-ditch effort to convert to int64 by default
                                array = array.astype(np.int64, casting='safe')
                            f = '1K'
                        col = fits.Column(name=field.name, format=f, null=_int_min, unit=field.unit, array=array)
                    elif field.format == kw.FLOAT_FORMAT:
                        if (t := array.dtype) == np.float32:
                            f = '1E'
                        else:
                            if t != np.float64 and array.size > 0:
                                # If not float32 or float64 make one last-ditch effort to convert to float64 by default
                                array = array.astype(np.float64, casting='safe')
                            f = '1D'
                        col = fits.Column(name=field.name, format=f, unit=field.unit, array=array)
                    else:
                        raise NotImplementedError
                columns.append(col)
            hdus.append(fits.BinTableHDU.from_columns(name=table_name, columns=columns))
        return hdus

    def _create_image_hdus(self) -> list[fits.ImageHDU]:
        hdus = []
        for image in self._images.values():
            hdr = fits.Header([card_from_metadatum(md) for md in image.metadata])
            if image.time is not None:
                hdr[kw.TIME_REFERENCE] = self._create_row_reference(image.time.uid)
            if image.unit is not None:
                hdr[kw.IMAGE_UNIT] = image.unit
            hdus.append(fits.ImageHDU(name=image.name, data=image.data, header=hdr))
        return hdus
