import datetime

import numpy as np
from astropy.io import fits
from datetime import timezone
from dataclasses import dataclass, field

from aotpy import level0
from . import _keywords as kw

# TODO: eventually add units and comments
# TODO: eventually Q vs P / I vs J vs K / D vs E
# TODO: validate mandatory fields on write, insert_in_column argument


@dataclass
class _TempColumn:
    name: str
    format: str
    null: int = None
    array: list = field(default_factory=list)

    def __post_init__(self):
        if self.format == 'K' and self.null is None:
            self.null = np.iinfo(np.int64).min


@dataclass
class _TempTable:
    name: str
    columns: list[_TempColumn]
    hdr: fits.Header = None

    def __post_init__(self):
        self._columns = {col.name: col for col in self.columns}

    def __getitem__(self, item):
        return self._columns[item]


def get_hdus(system: level0.AOSystem) -> fits.HDUList:
    return _Writer(system).create_hdus()


def write_to_fits(system: level0.AOSystem, filename: str) -> None:
    _Writer(system).create_hdus().writeto(filename, overwrite=True)


def _to_object_array(ls: list) -> np.ndarray:
    # Weird workaround to avoid issues with creating object arrays
    # See https://github.com/numpy/numpy/issues/19113
    array = np.empty(len(ls), dtype=np.object_)
    array[:] = ls
    return array


class _Writer:
    def __init__(self, system: level0.AOSystem) -> None:
        self.system = system

        self.tables = [
            _TempTable(name=kw.ATMOSPHERIC_PARAMETERS_TABLE, columns=[
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_DATA_SOURCE, format='64A'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_TIMESTAMP, format='64A'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_WAVELENGTH, format='D'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_R0, format='D'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_L0, format='D'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_LAYER_WEIGHT, format='QD'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_LAYER_HEIGHT, format='QD'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_LAYER_WIND_SPEED, format='QD'),
                _TempColumn(name=kw.ATMOSPHERIC_PARAMETERS_LAYER_WIND_DIRECTION, format='QD')
            ]),
            _TempTable(name=kw.DETECTORS_TABLE, columns=[
                _TempColumn(name=kw.DETECTOR_NAME, format='64A'),
                _TempColumn(name=kw.DETECTOR_FLAT_FIELD, format='64A'),
                _TempColumn(name=kw.DETECTOR_READOUT_NOISE, format='D'),
                _TempColumn(name=kw.DETECTOR_PIXEL_INTENSITIES, format='64A'),
                _TempColumn(name=kw.DETECTOR_INTEGRATION_TIME, format='D'),
                _TempColumn(name=kw.DETECTOR_COADDS, format='K', null=-1),
                _TempColumn(name=kw.DETECTOR_DARK, format='64A'),
                _TempColumn(name=kw.DETECTOR_WEIGHT_MAP, format='64A'),
                _TempColumn(name=kw.DETECTOR_QUANTUM_EFFICIENCY, format='D'),
                _TempColumn(name=kw.DETECTOR_PIXEL_SCALE, format='D'),
                _TempColumn(name=kw.DETECTOR_BINNING, format='K', null=-1),
                _TempColumn(name=kw.DETECTOR_BANDWIDTH, format='D'),
                _TempColumn(name=kw.DETECTOR_TRANSMISSION_WAVELENGTH, format='QD'),
                _TempColumn(name=kw.DETECTOR_TRANSMISSION, format='QD'),
                _TempColumn(name=kw.DETECTOR_SKY_BACKGROUND, format='64A'),
                _TempColumn(name=kw.DETECTOR_GAIN, format='D'),
                _TempColumn(name=kw.DETECTOR_EXCESS_NOISE, format='D'),
                _TempColumn(name=kw.DETECTOR_FILTER, format='64A'),
                _TempColumn(name=kw.DETECTOR_BAD_PIXEL_MAP, format='64A')
            ]),
            _TempTable(name=kw.OPTICAL_RELAYS_TABLE, columns=[
                _TempColumn(name=kw.OPTICAL_RELAY_NAME, format='64A'),
                _TempColumn(name=kw.OPTICAL_RELAY_FIELD_OF_VIEW, format='D'),
                _TempColumn(name=kw.OPTICAL_RELAY_FOCAL_LENGTH, format='D')
            ]),
            _TempTable(name=kw.OPTICAL_ABERRATIONS_TABLE, columns=[
                _TempColumn(name=kw.OPTICAL_ABERRATION_NAME, format='64A'),
                _TempColumn(name=kw.OPTICAL_ABERRATION_COEFFICIENTS, format='QD'),
                _TempColumn(name=kw.OPTICAL_ABERRATION_MODES, format='64A'),
                _TempColumn(name=kw.OPTICAL_ABERRATION_PUPIL, format='64A')
            ]),
            _TempTable(name=kw.SCORING_CAMERAS_TABLE, columns=[
                _TempColumn(name=kw.SCORING_CAMERA_NAME, format='64A'),
                _TempColumn(name=kw.SCORING_CAMERA_PUPIL, format='64A'),
                _TempColumn(name=kw.SCORING_CAMERA_THETA, format='D'),
                _TempColumn(name=kw.SCORING_CAMERA_WAVELENGTH, format='D'),
                _TempColumn(name=kw.SCORING_CAMERA_FRAME, format='64A'),
                _TempColumn(name=kw.SCORING_CAMERA_FIELD_STATIC_MAP, format='64A'),
                _TempColumn(name=kw.SCORING_CAMERA_X_STAT, format='QD'),
                _TempColumn(name=kw.SCORING_CAMERA_Y_STAT, format='QD'),
                _TempColumn(name=kw.SCORING_CAMERA_DETECTOR_NAME, format='64A'),
                _TempColumn(name=kw.SCORING_CAMERA_OPTICAL_RELAY_NAME, format='64A'),
                _TempColumn(name=kw.SCORING_CAMERA_OPTICAL_ABERRATION_NAME, format='64A')
            ]),
            _TempTable(name=kw.WAVEFRONT_SENSORS_TABLE, columns=[
                _TempColumn(name=kw.WAVEFRONT_SENSOR_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_TYPE, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_SOURCE_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_SLOPES, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_REF_SLOPES, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_THETA, format='D'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_WAVELENGTH, format='D'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_D_WAVELENGTH, format='D'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_N_SUBAPERTURES, format='K', null=-1),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_VALID_SUBAPERTURES, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_SUBAPERTURE_SIZE, format='D'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_PUPIL_ANGLE, format='D'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_ALGORITHM, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_OPTICAL_GAIN, format='D'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_CENTROID_GAINS, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_DETECTOR_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_OPTICAL_RELAY_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_OPTICAL_ABERRATION_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_SPOT_FWHM, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_SENSOR_MODULATION, format='D')
            ]),
            _TempTable(name=kw.RTC_TABLE, columns=[
                _TempColumn(name=kw.RTC_LOOPS_NAME, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_TYPE, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_INPUT, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_COMMANDED_COR_NAME, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_COMMANDS, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_REF_COMMANDS, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_TIMESTAMPS, format='QD'),
                _TempColumn(name=kw.RTC_LOOPS_FRAMERATE, format='D'),
                _TempColumn(name=kw.RTC_LOOPS_DELAY, format='D'),
                _TempColumn(name=kw.RTC_LOOPS_TIME_FILTER_NUM, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_TIME_FILTER_DEN, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_RESIDUAL_WAVEFRONT, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_CONTROL_MATRIX, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_INTERACTION_MATRIX, format='64A'),
                _TempColumn(name=kw.RTC_LOOPS_OFFLOAD_MATRIX, format='64A')
            ]),
            _TempTable(name=kw.SOURCES_TABLE, columns=[
                _TempColumn(name=kw.SOURCE_NAME, format='64A'),
                _TempColumn(name=kw.SOURCE_TYPE, format='64A'),
                _TempColumn(name=kw.SOURCE_RIGHT_ASCENSION, format='D'),
                _TempColumn(name=kw.SOURCE_DECLINATION, format='D'),
                _TempColumn(name=kw.SOURCE_ZENITH_ANGLE, format='D'),
                _TempColumn(name=kw.SOURCE_AZIMUTH, format='D'),
                _TempColumn(name=kw.SOURCE_LASER_LAUNCH_TELESCOPE_NAME, format='64A'),
                _TempColumn(name=kw.SOURCE_SODIUM_HEIGHT, format='D'),
                _TempColumn(name=kw.SOURCE_SODIUM_WIDTH, format='D'),
                _TempColumn(name=kw.SOURCE_SODIUM_PROFILE, format='QD'),
                _TempColumn(name=kw.SOURCE_SODIUM_ALTITUDES, format='QD')
            ]),
            _TempTable(name=kw.TELESCOPES_TABLE, columns=[
                _TempColumn(name=kw.TELESCOPE_NAME, format='64A'),
                _TempColumn(name=kw.TELESCOPE_TYPE, format='64A'),
                _TempColumn(name=kw.TELESCOPE_D_HEX, format='D'),
                _TempColumn(name=kw.TELESCOPE_D_CIRCLE, format='D'),
                _TempColumn(name=kw.TELESCOPE_D_EQ, format='D'),
                _TempColumn(name=kw.TELESCOPE_COBS, format='D'),
                _TempColumn(name=kw.TELESCOPE_PUPIL, format='64A'),
                _TempColumn(name=kw.TELESCOPE_PUPIL_ANGLE, format='D'),
                _TempColumn(name=kw.TELESCOPE_ELEVATION, format='D'),
                _TempColumn(name=kw.TELESCOPE_AZIMUTH, format='D'),
                _TempColumn(name=kw.TELESCOPE_STATIC_MAP, format='64A')
            ]),
            _TempTable(name=kw.WAVEFRONT_CORRECTORS_TABLE, columns=[
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_TYPE, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_TELESCOPE_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_PITCH, format='D'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_N_ACTUATORS, format='K', null=-1),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_TFZ_NUM, format='QD'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_TFZ_DEN, format='QD'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_OPTICAL_ABERRATION_NAME, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_VALID_ACTUATORS, format='64A'),
                _TempColumn(name=kw.WAVEFRONT_CORRECTOR_INFLUENCE_FUNCTION, format='64A')
            ])
        ]
        # Create a dictionary of dictionaries, where the column name is the key for each column
        self.tables: dict[str, _TempTable] = {table.name: table for table in self.tables}

        self.names = set()

        self.laser_launch_telescopes = {}
        self.images: dict[str, level0.Image] = {}

        for atm in self.system.atmosphere_params:
            self.handle_atmospheric_parameters(atm)

        for cam in self.system.scoring_cameras:
            self.handle_scoring_camera(cam)

        for wfs in self.system.wavefront_sensors:
            self.handle_wavefront_sensor(wfs)

        self.handle_rtc(system.rtc)

        for src in self.system.sources:
            self.handle_source(src)

        self.handle_telescope(system.telescope)

        for cor in self.system.wavefront_correctors:
            self.handle_wavefront_corrector(cor)

    def create_hdus(self) -> fits.HDUList:
        if self.system.level != 0:
            raise NotImplementedError

        cards = {
            kw.AOT_VERSION: '0.1',
            kw.AOT_LEVEL: self.system.level,
            kw.AOT_DATE: '' if (start := self.system.start_datetime) is None else start.astimezone(timezone.utc).date().isoformat(),
            kw.AOT_EXPOSURE_START: '' if (start := self.system.start_datetime) is None else start.astimezone(timezone.utc).time().isoformat(),
            kw.AOT_EXPOSURE_END: '' if (end := self.system.end_datetime) is None else end.astimezone(timezone.utc).time().isoformat(),
            kw.AOT_AO_MODE: self.system.ao_mode,
        }

        hdus = [fits.PrimaryHDU(header=fits.Header(cards))]
        for table in self.tables.values():
            c = [fits.Column(name=col.name, format=col.format, null=col.null, array=col.array) if col.format != 'QD'
                 else fits.Column(name=col.name, format=col.format, null=col.null, array=_to_object_array(col.array))
                 for col in table.columns]
            hdus.append(fits.BinTableHDU.from_columns(c, header=table.hdr, name=table.name))
        hdus.extend([image.to_hdu() for image in self.images.values()])

        return fits.HDUList([hdu for hdu in hdus if hdu])

    def insert_in_column(self, col: _TempColumn, value):
        if 'A' in col.format:
            if value is None:
                value = ''
            elif isinstance(value, level0.Image):
                value = self.handle_image(value)
            elif isinstance(value, datetime.datetime):
                value = value.replace(tzinfo=None).isoformat()
            elif isinstance(value, str):
                if len(value) > 64:
                    print(f'Value "{value}" is too long for column {col.name}, shortening to {value[:64]}')
                    value = value[:64]
            else:
                raise RuntimeError  # Don't know how to handle this string

        elif col.format == 'D':
            if value is None:
                value = np.nan
        elif col.format == 'K':
            if value is None:
                value = col.null
        elif col.format == 'QD':
            if value is None:
                value = []
            else:
                value = [np.nan if v is None else v for v in value]
        else:
            raise RuntimeError
        col.array.append(value)

    def check_name(self, name: str) -> str:
        if len(name) > 64:
            print(f'Name "{name}" is too long, shortening to {name[:64]}')
            name = name[:64]
        if name in self.names:
            raise RuntimeError  # TODO: Repeated name
        self.names.add(name)
        return name

    def handle_image(self, image: level0.Image) -> str:
        if len(name := image.name) > 64:
            name = name[:64]
            print(f'Image name "{image.name}" is too long, shortening to {name}')
        if name in self.images:
            if (other := self.images[name]) is not image and image != other:
                # the image already stored is not the same object, nor they have the same contents
                raise RuntimeError  # can't have two images with the same name if they don't have the same data
        else:
            self.images[name] = image
        return name

    def handle_atmospheric_parameters(self, atm: level0.AtmosphericParameters):
        table = self.tables[kw.ATMOSPHERIC_PARAMETERS_TABLE]
        cols = {
            kw.ATMOSPHERIC_PARAMETERS_DATA_SOURCE: atm.data_source,
            kw.ATMOSPHERIC_PARAMETERS_TIMESTAMP: atm.timestamp,
            kw.ATMOSPHERIC_PARAMETERS_WAVELENGTH: atm.wavelength,
            kw.ATMOSPHERIC_PARAMETERS_R0: atm.r0,
            kw.ATMOSPHERIC_PARAMETERS_L0: atm.l0,
            kw.ATMOSPHERIC_PARAMETERS_LAYER_WEIGHT: [layer.weight for layer in atm.layers],
            kw.ATMOSPHERIC_PARAMETERS_LAYER_HEIGHT: [layer.height for layer in atm.layers],
            kw.ATMOSPHERIC_PARAMETERS_LAYER_WIND_SPEED: [layer.wind_speed for layer in atm.layers],
            kw.ATMOSPHERIC_PARAMETERS_LAYER_WIND_DIRECTION: [layer.wind_direction for layer in atm.layers]
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)

    def handle_detector(self, det: level0.Detector) -> str:
        if det is None:
            return ''
        table = self.tables[kw.DETECTORS_TABLE]
        cols = {
            kw.DETECTOR_NAME: self.check_name(det.name),
            kw.DETECTOR_FLAT_FIELD: det.flat_field,
            kw.DETECTOR_READOUT_NOISE: det.readout_noise,
            kw.DETECTOR_PIXEL_INTENSITIES: det.pixel_intensities,
            kw.DETECTOR_INTEGRATION_TIME: det.integration_time,
            kw.DETECTOR_COADDS: det.coadds,
            kw.DETECTOR_DARK: det.dark,
            kw.DETECTOR_WEIGHT_MAP: det.weight_map,
            kw.DETECTOR_QUANTUM_EFFICIENCY: det.quantum_efficiency,
            kw.DETECTOR_PIXEL_SCALE: det.pixel_scale,
            kw.DETECTOR_BINNING: det.binning,
            kw.DETECTOR_BANDWIDTH: det.bandwidth,
            kw.DETECTOR_TRANSMISSION_WAVELENGTH: det.transmission_wavelength,
            kw.DETECTOR_TRANSMISSION: det.transmission,
            kw.DETECTOR_SKY_BACKGROUND: det.sky_background,
            kw.DETECTOR_GAIN: det.gain,
            kw.DETECTOR_EXCESS_NOISE: det.excess_noise,
            kw.DETECTOR_FILTER: det.filter,
            kw.DETECTOR_BAD_PIXEL_MAP: det.bad_pixel_map
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)

        return det.name

    def handle_optical_relay(self, rel: level0.OpticalRelay) -> str:
        if rel is None:
            return ''
        table = self.tables[kw.OPTICAL_RELAYS_TABLE]
        cols = {
            kw.OPTICAL_RELAY_NAME: self.check_name(rel.name),
            kw.OPTICAL_RELAY_FIELD_OF_VIEW: rel.field_of_view,
            kw.OPTICAL_RELAY_FOCAL_LENGTH: rel.focal_length
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)

        return cols[kw.OPTICAL_RELAY_NAME]

    def handle_optical_aberration(self, ab: level0.OpticalAberration) -> str:
        if ab is None:
            return ''
        table = self.tables[kw.OPTICAL_ABERRATIONS_TABLE]
        cols = {
            kw.OPTICAL_ABERRATION_NAME: self.check_name(ab.name),
            kw.OPTICAL_ABERRATION_COEFFICIENTS: ab.coefficients,
            kw.OPTICAL_ABERRATION_MODES: ab.modes,
            kw.OPTICAL_ABERRATION_PUPIL: ab.pupil
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)

        return cols[kw.OPTICAL_ABERRATION_NAME]

    def handle_scoring_camera(self, cam: level0.ScoringCamera):
        table = self.tables[kw.SCORING_CAMERAS_TABLE]
        cols = {
            kw.SCORING_CAMERA_NAME: self.check_name(cam.name),
            kw.SCORING_CAMERA_PUPIL: cam.pupil,
            kw.SCORING_CAMERA_THETA: cam.theta,
            kw.SCORING_CAMERA_WAVELENGTH: cam.wavelength,
            kw.SCORING_CAMERA_FRAME: cam.frame,
            kw.SCORING_CAMERA_FIELD_STATIC_MAP: cam.field_static_map,
            kw.SCORING_CAMERA_X_STAT: cam.x_stat,
            kw.SCORING_CAMERA_Y_STAT: cam.y_stat,
            kw.SCORING_CAMERA_DETECTOR_NAME: self.handle_detector(cam.detector),
            kw.SCORING_CAMERA_OPTICAL_RELAY_NAME: self.handle_optical_relay(cam.optical_relay),
            kw.SCORING_CAMERA_OPTICAL_ABERRATION_NAME: self.handle_optical_aberration(cam.optical_aberration)
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)

    def handle_wavefront_sensor(self, wfs: level0.WavefrontSensor):
        if isinstance(wfs, level0.ShackHartmann):
            wfs_type = kw.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN
            spot_fwhm = wfs.spot_fwhm
            modulation = None
        elif isinstance(wfs, level0.Pyramid):
            wfs_type = kw.WAVEFRONT_SENSOR_TYPE_PYRAMID
            spot_fwhm = None
            modulation = wfs.modulation
        else:
            raise NotImplementedError

        table = self.tables[kw.WAVEFRONT_SENSORS_TABLE]
        cols = {
            kw.WAVEFRONT_SENSOR_NAME: self.check_name(wfs.name),
            kw.WAVEFRONT_SENSOR_TYPE: wfs_type,
            kw.WAVEFRONT_SENSOR_SOURCE_NAME: wfs.source.name,
            kw.WAVEFRONT_SENSOR_SLOPES: wfs.slopes,
            kw.WAVEFRONT_SENSOR_REF_SLOPES: wfs.ref_slopes,
            kw.WAVEFRONT_SENSOR_THETA: wfs.theta,
            kw.WAVEFRONT_SENSOR_WAVELENGTH: wfs.wavelength,
            kw.WAVEFRONT_SENSOR_D_WAVELENGTH: wfs.d_wavelength,
            kw.WAVEFRONT_SENSOR_N_SUBAPERTURES: wfs.n_subapertures,
            kw.WAVEFRONT_SENSOR_VALID_SUBAPERTURES: wfs.valid_subapertures,
            kw.WAVEFRONT_SENSOR_SUBAPERTURE_SIZE: wfs.subaperture_size,
            kw.WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES: wfs.subaperture_intensities,
            kw.WAVEFRONT_SENSOR_PUPIL_ANGLE: wfs.pupil_angle,
            kw.WAVEFRONT_SENSOR_ALGORITHM: wfs.algorithm,
            kw.WAVEFRONT_SENSOR_OPTICAL_GAIN: wfs.optical_gain,
            kw.WAVEFRONT_SENSOR_CENTROID_GAINS: wfs.centroid_gains,
            kw.WAVEFRONT_SENSOR_DETECTOR_NAME: self.handle_detector(wfs.detector),
            kw.WAVEFRONT_SENSOR_OPTICAL_RELAY_NAME: self.handle_optical_relay(wfs.optical_relay),
            kw.WAVEFRONT_SENSOR_OPTICAL_ABERRATION_NAME: self.handle_optical_aberration(wfs.optical_aberration),
            kw.WAVEFRONT_SENSOR_SPOT_FWHM: spot_fwhm,
            kw.WAVEFRONT_SENSOR_MODULATION: modulation
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)

    def handle_rtc(self, rtc: level0.RTC):
        table = self.tables[kw.RTC_TABLE]
        table.hdr = fits.Header({
            kw.RTC_N_LOOKUP_TABLES: len(rtc.lookup_tables),
            **{f'{kw.RTC_LOOKUP_TABLE}{i}': self.handle_image(lut) for i, lut in enumerate(rtc.lookup_tables, start=1)},
            kw.RTC_N_NON_COMMON_PATH_ABERRATIONS: len(rtc.non_common_path_aberrations),
            **{f'{kw.RTC_NON_COMMON_PATH_ABERRATION}{i}': self.handle_image(ncpa) for i, ncpa in enumerate(rtc.non_common_path_aberrations, start=1)},
        })

        for loop in rtc.loops:
            if isinstance(loop, level0.ControlLoop):
                loop_type = kw.RTC_LOOPS_TYPE_CONTROL
                loop_input = loop.input_wfs.name
                residual_wavefront = loop.residual_wavefront
                control_matrix = loop.control_matrix
                interaction_matrix = loop.interaction_matrix
                offload_matrix = None
            elif isinstance(loop, level0.OffloadLoop):
                loop_type = kw.RTC_LOOPS_TYPE_OFFLOAD
                loop_input = loop.input_corrector.name
                residual_wavefront = None
                control_matrix = None
                interaction_matrix = None
                offload_matrix = loop.offload_matrix
            else:
                raise NotImplementedError

            cols = {
                kw.RTC_LOOPS_NAME: self.check_name(loop.name),
                kw.RTC_LOOPS_TYPE: loop_type,
                kw.RTC_LOOPS_INPUT: loop_input,
                kw.RTC_LOOPS_COMMANDED_COR_NAME: loop.commanded_corrector.name,
                kw.RTC_LOOPS_COMMANDS: loop.commands,
                kw.RTC_LOOPS_REF_COMMANDS: loop.ref_commands,
                kw.RTC_LOOPS_TIMESTAMPS: loop.timestamps,
                kw.RTC_LOOPS_FRAMERATE: loop.framerate,
                kw.RTC_LOOPS_DELAY: loop.delay,
                kw.RTC_LOOPS_TIME_FILTER_NUM: loop.time_filter_num,
                kw.RTC_LOOPS_TIME_FILTER_DEN: loop.time_filter_den,
                kw.RTC_LOOPS_RESIDUAL_WAVEFRONT: residual_wavefront,
                kw.RTC_LOOPS_CONTROL_MATRIX: control_matrix,
                kw.RTC_LOOPS_INTERACTION_MATRIX: interaction_matrix,
                kw.RTC_LOOPS_OFFLOAD_MATRIX: offload_matrix
            }
            for col, value in cols.items():
                self.insert_in_column(table[col], value)

    def handle_source(self, src: level0.Source):
        if isinstance(src, level0.NaturalGuideStar):
            src_type = kw.SOURCE_TYPE_NGS
            laser_launch_telescope_name = None
            sodium_height = None
            sodium_width = None
            sodium_profile = []
            sodium_altitudes = []
        elif isinstance(src, level0.LaserGuideStar):
            src_type = kw.SOURCE_TYPE_LGS
            laser_launch_telescope_name = self.handle_telescope(src.laser_launch_telescope)
            sodium_height = src.sodium_height
            sodium_width = src.sodium_width
            sodium_profile = src.sodium_profile
            sodium_altitudes = src.sodium_altitudes
        else:
            raise NotImplementedError

        table = self.tables[kw.SOURCES_TABLE]
        cols = {
            kw.SOURCE_NAME: self.check_name(src.name),
            kw.SOURCE_TYPE: src_type,
            kw.SOURCE_RIGHT_ASCENSION: src.right_ascension,
            kw.SOURCE_DECLINATION: src.declination,
            kw.SOURCE_ZENITH_ANGLE: src.zenith_angle,
            kw.SOURCE_AZIMUTH: src.azimuth,
            kw.SOURCE_LASER_LAUNCH_TELESCOPE_NAME: laser_launch_telescope_name,
            kw.SOURCE_SODIUM_HEIGHT: sodium_height,
            kw.SOURCE_SODIUM_WIDTH: sodium_width,
            kw.SOURCE_SODIUM_PROFILE: sodium_profile,
            kw.SOURCE_SODIUM_ALTITUDES: sodium_altitudes
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)

    def handle_telescope(self, tel: level0.Telescope) -> str:
        if tel is None:
            return ''
        if isinstance(tel, level0.MainTelescope):
            tel_type = kw.TELESCOPE_TYPE_MAIN
        elif isinstance(tel, level0.LaserLaunchTelescope):
            tel_type = kw.TELESCOPE_TYPE_LLT
            if tel.name in self.laser_launch_telescopes:
                if tel is self.laser_launch_telescopes[tel.name]:
                    return tel.name  # LLT is already in table
                raise RuntimeError  # TODO: raise error for non-unique name
        else:
            raise NotImplementedError

        table = self.tables[kw.TELESCOPES_TABLE]
        cols = {
            kw.TELESCOPE_NAME: self.check_name(tel.name),
            kw.TELESCOPE_TYPE: tel_type,
            kw.TELESCOPE_D_HEX: tel.d_hex,
            kw.TELESCOPE_D_CIRCLE: tel.d_circle,
            kw.TELESCOPE_D_EQ: tel.d_eq,
            kw.TELESCOPE_COBS: tel.cobs,
            kw.TELESCOPE_PUPIL: tel.pupil,
            kw.TELESCOPE_PUPIL_ANGLE: tel.pupil_angle,
            kw.TELESCOPE_ELEVATION: tel.elevation,
            kw.TELESCOPE_AZIMUTH: tel.azimuth,
            kw.TELESCOPE_STATIC_MAP: tel.static_map
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)
        return tel.name

    def handle_wavefront_corrector(self, cor: level0.WavefrontCorrector):
        if isinstance(cor, level0.DeformableMirror):
            cor_type = kw.WAVEFRONT_CORRECTOR_TYPE_DM
            valid_actuators = cor.valid_actuators
            influence_function = cor.influence_function
        elif isinstance(cor, level0.TipTiltMirror):
            cor_type = kw.WAVEFRONT_CORRECTOR_TYPE_TT
            valid_actuators = None
            influence_function = None
        elif isinstance(cor, level0.LinearStage):
            cor_type = kw.WAVEFRONT_CORRECTOR_TYPE_LS
            valid_actuators = None
            influence_function = None
        else:
            raise NotImplementedError

        table = self.tables[kw.WAVEFRONT_CORRECTORS_TABLE]
        cols = {
            kw.WAVEFRONT_CORRECTOR_NAME: self.check_name(cor.name),
            kw.WAVEFRONT_CORRECTOR_TYPE: cor_type,
            kw.WAVEFRONT_CORRECTOR_TELESCOPE_NAME: cor.telescope.name,
            kw.WAVEFRONT_CORRECTOR_PITCH: cor.pitch,
            kw.WAVEFRONT_CORRECTOR_N_ACTUATORS: cor.n_actuators,
            kw.WAVEFRONT_CORRECTOR_TFZ_NUM: cor.tfz_num,
            kw.WAVEFRONT_CORRECTOR_TFZ_DEN: cor.tfz_den,
            kw.WAVEFRONT_CORRECTOR_OPTICAL_ABERRATION_NAME: self.handle_optical_aberration(cor.optical_aberration),
            kw.WAVEFRONT_CORRECTOR_VALID_ACTUATORS: valid_actuators,
            kw.WAVEFRONT_CORRECTOR_INFLUENCE_FUNCTION: influence_function
        }
        for col, value in cols.items():
            self.insert_in_column(table[col], value)
