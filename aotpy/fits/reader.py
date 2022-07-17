from astropy.io import fits

from aotpy import level0
from . import _keywords as kw

from datetime import date, time, datetime, timezone
import numpy as np


def read_from_fits(filename: str) -> level0.AOSystem:
    r = _Reader(filename)
    return r.read()


def _get_value(columns: fits.ColDefs, row: fits.FITS_record, column: str):
    col = columns[column]
    value = row[column]

    if 'A' in col.format:
        if value == '':
            return None
    elif 'Q' in col.format or 'P' in col.format:
        return [None if np.isnan(v) else v for v in value]
    elif 'D' in col.format or 'E' in col.format:
        if np.isnan(value):
            return None
    elif 'K' in col.format or 'J' in col.format or 'I' in col.format:
        if col.null and value == col.null:
            return None
    else:
        raise RuntimeError
    return value


# TODO in the future it would be nice if "extra" stuff would still be read but kept on a special variable
# but only do this if an option is set on reader. then also have an option on writer to write it back
# extra stuff includes: extra hdus, extra headers in existing tables, extra columns in existing tables

# TODO check name uniqueness


class _Reader:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.primary_header = None
        self.hdus = None
        self.images: dict[str, level0.Image] = {}

    def get_from_primary_header(self, keyword: str):
        try:
            return self.primary_header[keyword]
        except KeyError:
            print(f'Keyword {keyword} not found')

    def get_image(self, extname: str):
        if extname == '':
            return None
        try:
            return self.images[extname.upper()]
            # extension names are always saved in upper case
        except KeyError:
            print(f'ImageHDU {extname} not found')
            return None

    def read(self) -> level0.AOSystem:
        with fits.open(self.filename) as hdus:
            self.primary_header = hdus[0].header
            self.hdus = hdus

            if kw.AOT_VERSION not in self.primary_header:
                raise RuntimeError  # If it doesn't exist then this is not an AOT file
                # TODO do something with the version

            if self.get_from_primary_header(kw.AOT_LEVEL) != 0:
                # TODO other levels
                raise NotImplementedError

            system = level0.AOSystem()
            d = self.get_from_primary_header(kw.AOT_DATE)
            start = self.get_from_primary_header(kw.AOT_EXPOSURE_START)
            end = self.get_from_primary_header(kw.AOT_EXPOSURE_END)
            if d and start and end:
                system.start_datetime = datetime.combine(date.fromisoformat(d),
                                                         time.fromisoformat(start), tzinfo=timezone.utc)
                system.end_datetime = datetime.combine(date.fromisoformat(d),
                                                       time.fromisoformat(end), tzinfo=timezone.utc)
            if ao_mode := self.get_from_primary_header(kw.AOT_AO_MODE):
                system.ao_mode = ao_mode

            table_found = {table: False for table in kw.TABLE_SET}
            for hdu in hdus:
                if hdu.name in kw.TABLE_SET:
                    table_found[hdu.name] = True
                else:
                    if isinstance(hdu, fits.ImageHDU):
                        self.images[hdu.name] = level0.Image.from_hdu(hdu)
                    # TODO: else, do something with extra HDUs
            if not np.all(table_found.values()):
                raise RuntimeError  # not all standard tables were found!

            table = hdus[kw.ATMOSPHERIC_PARAMETERS_TABLE]
            columns = table.columns
            for row in table.data:
                layer_weights = _get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_LAYER_WEIGHT)
                layer_heights = _get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_LAYER_HEIGHT)
                layer_wind_speeds = _get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_LAYER_WIND_SPEED)
                layer_wind_directions = _get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_LAYER_WIND_DIRECTION)

                if timestamp := _get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_TIMESTAMP):
                    timestamp = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)

                system.atmosphere_params.append(
                    level0.AtmosphericParameters(
                        data_source=_get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_DATA_SOURCE),
                        timestamp=timestamp,
                        wavelength=_get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_WAVELENGTH),
                        r0=_get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_R0),
                        l0=_get_value(columns, row, kw.ATMOSPHERIC_PARAMETERS_L0),
                        layers=[
                            level0.AtmosphereLayer(
                                weight=layer_weights[i],
                                height=layer_heights[i],
                                wind_speed=layer_wind_speeds[i],
                                wind_direction=layer_wind_directions[i]
                            ) for i in range(len(layer_weights))
                        ]
                    )
                )

            telescopes = {'': None}
            table = hdus[kw.TELESCOPES_TABLE]
            columns = table.columns
            for row in table.data:
                name = _get_value(columns, row, kw.TELESCOPE_NAME)
                type = _get_value(columns, row, kw.TELESCOPE_TYPE)
                if type == kw.TELESCOPE_TYPE_MAIN:
                    tel = level0.MainTelescope(name)
                    system.telescope = tel
                elif type == kw.TELESCOPE_TYPE_LLT:
                    tel = level0.LaserLaunchTelescope(name)
                else:
                    raise NotImplementedError
                tel.d_hex = _get_value(columns, row, kw.TELESCOPE_D_HEX)
                tel.d_circle = _get_value(columns, row, kw.TELESCOPE_D_CIRCLE)
                tel.d_eq = _get_value(columns, row, kw.TELESCOPE_D_EQ)
                tel.cobs = _get_value(columns, row, kw.TELESCOPE_COBS)
                tel.pupil = self.get_image(row[kw.TELESCOPE_PUPIL])
                tel.elevation = _get_value(columns, row, kw.TELESCOPE_ELEVATION)
                tel.azimuth = _get_value(columns, row, kw.TELESCOPE_AZIMUTH)
                tel.pupil_angle = _get_value(columns, row, kw.TELESCOPE_PUPIL_ANGLE)
                tel.static_map = self.get_image(row[kw.TELESCOPE_STATIC_MAP])

                telescopes[name] = tel

            sources = {'': None}
            table = hdus[kw.SOURCES_TABLE]
            columns = table.columns
            for row in hdus[kw.SOURCES_TABLE].data:
                name = _get_value(columns, row, kw.SOURCE_NAME)
                type = _get_value(columns, row, kw.SOURCE_TYPE)
                if type == kw.SOURCE_TYPE_NGS:
                    src = level0.NaturalGuideStar(name)
                elif type == kw.SOURCE_TYPE_LGS:
                    src = level0.LaserGuideStar(
                        name=name,
                        laser_launch_telescope=telescopes[row[kw.SOURCE_LASER_LAUNCH_TELESCOPE_NAME]],
                        sodium_height=_get_value(columns, row, kw.SOURCE_SODIUM_HEIGHT),
                        sodium_width=_get_value(columns, row, kw.SOURCE_SODIUM_WIDTH),
                        sodium_profile=_get_value(columns, row, kw.SOURCE_SODIUM_PROFILE),
                        sodium_altitudes=_get_value(columns, row, kw.SOURCE_SODIUM_ALTITUDES)
                    )
                else:
                    raise NotImplementedError
                src.right_ascension = _get_value(columns, row, kw.SOURCE_RIGHT_ASCENSION)
                src.declination = _get_value(columns, row, kw.SOURCE_DECLINATION)
                src.zenith_angle = _get_value(columns, row, kw.SOURCE_ZENITH_ANGLE)
                src.azimuth = _get_value(columns, row, kw.SOURCE_AZIMUTH)
                system.sources.append(src)
                sources[name] = src

            table = hdus[kw.DETECTORS_TABLE]
            columns = table.columns
            detectors = {
                row[kw.DETECTOR_NAME]: level0.Detector(
                    name=_get_value(columns, row, kw.DETECTOR_NAME),
                    flat_field=self.get_image(row[kw.DETECTOR_FLAT_FIELD]),
                    readout_noise=_get_value(columns, row, kw.DETECTOR_READOUT_NOISE),
                    pixel_intensities=self.get_image(row[kw.DETECTOR_PIXEL_INTENSITIES]),
                    integration_time=_get_value(columns, row, kw.DETECTOR_INTEGRATION_TIME),
                    coadds=_get_value(columns, row, kw.DETECTOR_COADDS),
                    dark=self.get_image(row[kw.DETECTOR_DARK]),
                    weight_map=self.get_image(row[kw.DETECTOR_WEIGHT_MAP]),
                    quantum_efficiency=_get_value(columns, row, kw.DETECTOR_QUANTUM_EFFICIENCY),
                    pixel_scale=_get_value(columns, row, kw.DETECTOR_PIXEL_SCALE),
                    binning=_get_value(columns, row, kw.DETECTOR_BINNING),
                    bandwidth=_get_value(columns, row, kw.DETECTOR_BANDWIDTH),
                    transmission_wavelength=_get_value(columns, row, kw.DETECTOR_TRANSMISSION_WAVELENGTH),
                    transmission=_get_value(columns, row, kw.DETECTOR_TRANSMISSION),
                    sky_background=self.get_image(row[kw.DETECTOR_SKY_BACKGROUND]),
                    gain=_get_value(columns, row, kw.DETECTOR_GAIN),
                    excess_noise=_get_value(columns, row, kw.DETECTOR_EXCESS_NOISE),
                    filter=_get_value(columns, row, kw.DETECTOR_FILTER),
                    bad_pixel_map=self.get_image(row[kw.DETECTOR_BAD_PIXEL_MAP])
                ) for row in table.data
            }
            detectors[''] = None

            table = hdus[kw.OPTICAL_RELAYS_TABLE]
            columns = table.columns
            optical_relays = {
                row[kw.OPTICAL_RELAY_NAME]:
                    level0.OpticalRelay(
                        name=_get_value(columns, row, kw.OPTICAL_RELAY_NAME),
                        field_of_view=_get_value(columns, row, kw.OPTICAL_RELAY_FIELD_OF_VIEW),
                        focal_length=_get_value(columns, row, kw.OPTICAL_RELAY_FOCAL_LENGTH)
                    ) for row in table.data
            }
            optical_relays[''] = None

            table = hdus[kw.OPTICAL_ABERRATIONS_TABLE]
            columns = table.columns
            optical_aberrations = {
                row[kw.OPTICAL_ABERRATION_NAME]:
                    level0.OpticalAberration(
                        name=_get_value(columns, row, kw.OPTICAL_ABERRATION_NAME),
                        coefficients=_get_value(columns, row, kw.OPTICAL_ABERRATION_COEFFICIENTS),
                        modes=self.get_image(row[kw.OPTICAL_ABERRATION_MODES]),
                        pupil=self.get_image(row[kw.OPTICAL_ABERRATION_PUPIL])
                ) for row in table.data
            }
            optical_aberrations[''] = None

            table = hdus[kw.SCORING_CAMERAS_TABLE]
            columns = table.columns
            system.scoring_cameras = [
                level0.ScoringCamera(
                    name=_get_value(columns, row, kw.SCORING_CAMERA_NAME),
                    pupil=self.get_image(row[kw.SCORING_CAMERA_PUPIL]),
                    theta=_get_value(columns, row, kw.SCORING_CAMERA_THETA),
                    wavelength=_get_value(columns, row, kw.SCORING_CAMERA_WAVELENGTH),
                    frame=self.get_image(row[kw.SCORING_CAMERA_FRAME]),
                    field_static_map=self.get_image(row[kw.SCORING_CAMERA_FIELD_STATIC_MAP]),
                    x_stat=_get_value(columns, row, kw.SCORING_CAMERA_X_STAT),
                    y_stat=_get_value(columns, row, kw.SCORING_CAMERA_Y_STAT),
                    detector=detectors[row[kw.SCORING_CAMERA_DETECTOR_NAME]],
                    optical_relay=optical_relays[row[kw.SCORING_CAMERA_OPTICAL_RELAY_NAME]],
                    optical_aberration=optical_aberrations[row[kw.SCORING_CAMERA_OPTICAL_ABERRATION_NAME]],
                ) for row in table.data
            ]

            table = hdus[kw.WAVEFRONT_SENSORS_TABLE]
            columns = table.columns
            wfss = {'': None}
            for row in table.data:
                name = _get_value(columns, row, kw.WAVEFRONT_SENSOR_NAME)
                type = _get_value(columns, row, kw.WAVEFRONT_SENSOR_TYPE)
                source = sources[row[kw.WAVEFRONT_SENSOR_SOURCE_NAME]]
                if type == kw.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN:
                    wfs = level0.ShackHartmann(
                        name=name,
                        source=source,
                        spot_fwhm=self.get_image(row[kw.WAVEFRONT_SENSOR_SPOT_FWHM])
                    )
                elif type == kw.WAVEFRONT_SENSOR_TYPE_PYRAMID:
                    wfs = level0.Pyramid(
                        name=name,
                        source=source,
                        modulation=_get_value(columns, row, kw.WAVEFRONT_SENSOR_MODULATION)
                    )
                else:
                    raise NotImplementedError

                wfs.slopes = self.get_image(row[kw.WAVEFRONT_SENSOR_SLOPES])
                wfs.ref_slopes = self.get_image(row[kw.WAVEFRONT_SENSOR_REF_SLOPES])
                wfs.theta = _get_value(columns, row, kw.WAVEFRONT_SENSOR_THETA)
                wfs.wavelength = _get_value(columns, row, kw.WAVEFRONT_SENSOR_WAVELENGTH)
                wfs.d_wavelength = _get_value(columns, row, kw.WAVEFRONT_SENSOR_D_WAVELENGTH)
                wfs.n_subapertures = _get_value(columns, row, kw.WAVEFRONT_SENSOR_N_SUBAPERTURES)
                wfs.valid_subapertures = self.get_image(row[kw.WAVEFRONT_SENSOR_VALID_SUBAPERTURES])
                wfs.subaperture_size = _get_value(columns, row, kw.WAVEFRONT_SENSOR_SUBAPERTURE_SIZE)
                wfs.subaperture_intensities = self.get_image(row[kw.WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES])
                wfs.pupil_angle = _get_value(columns, row, kw.WAVEFRONT_SENSOR_PUPIL_ANGLE)
                wfs.algorithm = _get_value(columns, row, kw.WAVEFRONT_SENSOR_ALGORITHM)
                wfs.optical_gain = _get_value(columns, row, kw.WAVEFRONT_SENSOR_OPTICAL_GAIN)
                wfs.centroid_gains = self.get_image(row[kw.WAVEFRONT_SENSOR_CENTROID_GAINS])
                wfs.detector = detectors[row[kw.WAVEFRONT_SENSOR_DETECTOR_NAME]]
                wfs.optical_relay = optical_relays[row[kw.WAVEFRONT_SENSOR_OPTICAL_RELAY_NAME]]
                wfs.optical_aberration = optical_aberrations[row[kw.WAVEFRONT_SENSOR_OPTICAL_ABERRATION_NAME]]

                system.wavefront_sensors.append(wfs)
                wfss[name] = wfs

            table = hdus[kw.WAVEFRONT_CORRECTORS_TABLE]
            columns = table.columns
            wfcs = {'': None}
            for row in table.data:
                name = _get_value(columns, row, kw.WAVEFRONT_CORRECTOR_NAME)
                type = _get_value(columns, row, kw.WAVEFRONT_CORRECTOR_TYPE)
                telescope = telescopes[row[kw.WAVEFRONT_CORRECTOR_TELESCOPE_NAME]]

                if type == kw.WAVEFRONT_CORRECTOR_TYPE_DM:
                    cor = level0.DeformableMirror(
                        name=name,
                        telescope=telescope,
                        n_actuators=_get_value(columns, row, kw.WAVEFRONT_CORRECTOR_N_ACTUATORS),
                        valid_actuators=self.get_image(row[kw.WAVEFRONT_CORRECTOR_VALID_ACTUATORS]),
                        influence_function=self.get_image(row[kw.WAVEFRONT_CORRECTOR_INFLUENCE_FUNCTION])
                    )
                elif type == kw.WAVEFRONT_CORRECTOR_TYPE_TT:
                    cor = level0.TipTiltMirror(
                        name=name,
                        telescope=telescope
                    )
                elif type == kw.WAVEFRONT_CORRECTOR_TYPE_LS:
                    cor = level0.LinearStage(
                        name=name,
                        telescope=telescope
                    )
                else:
                    raise NotImplementedError

                cor.pitch = _get_value(columns, row, kw.WAVEFRONT_CORRECTOR_PITCH)
                cor.tfz_num = _get_value(columns, row, kw.WAVEFRONT_CORRECTOR_TFZ_NUM)
                cor.tfz_den = _get_value(columns, row, kw.WAVEFRONT_CORRECTOR_TFZ_DEN)
                cor.optical_aberration = optical_aberrations[row[kw.WAVEFRONT_CORRECTOR_OPTICAL_ABERRATION_NAME]]

                system.wavefront_correctors.append(cor)
                wfcs[name] = cor

            table = hdus[kw.RTC_TABLE]
            rtc_hdr = table.header
            system.rtc = level0.RTC(
                lookup_tables=[self.get_image(rtc_hdr[f'{kw.RTC_LOOKUP_TABLE}{i}'])
                               for i in range(1, rtc_hdr[kw.RTC_N_LOOKUP_TABLES] + 1)],
                non_common_path_aberrations=[self.get_image(rtc_hdr[f'{kw.RTC_NON_COMMON_PATH_ABERRATION}{i}'])
                                             for i in range(1, rtc_hdr[kw.RTC_N_NON_COMMON_PATH_ABERRATIONS] + 1)]
            )
            columns = table.columns
            for row in table.data:
                name = _get_value(columns, row, kw.RTC_LOOPS_NAME)
                type = _get_value(columns, row, kw.RTC_LOOPS_TYPE)
                loop_input = row[kw.RTC_LOOPS_INPUT]
                commanded = row[kw.RTC_LOOPS_COMMANDED_COR_NAME]
                if type == kw.RTC_LOOPS_TYPE_CONTROL:
                    loop = level0.ControlLoop(
                        name=name,
                        input_wfs=wfss[loop_input],
                        commanded_corrector=wfcs[commanded],
                        residual_wavefront=self.get_image(row[kw.RTC_LOOPS_RESIDUAL_WAVEFRONT]),
                        control_matrix=self.get_image(row[kw.RTC_LOOPS_CONTROL_MATRIX]),
                        interaction_matrix=self.get_image(row[kw.RTC_LOOPS_INTERACTION_MATRIX])
                    )
                elif type == kw.RTC_LOOPS_TYPE_OFFLOAD:
                    loop = level0.OffloadLoop(
                        name=name,
                        input_corrector=wfcs[loop_input],
                        commanded_corrector=wfcs[commanded],
                        offload_matrix=self.get_image(row[kw.RTC_LOOPS_OFFLOAD_MATRIX])
                    )
                else:
                    raise NotImplementedError

                loop.commands = self.get_image(row[kw.RTC_LOOPS_COMMANDS])
                loop.ref_commands = self.get_image(row[kw.RTC_LOOPS_REF_COMMANDS])
                loop.timestamps = _get_value(columns, row, kw.RTC_LOOPS_TIMESTAMPS)
                loop.framerate = _get_value(columns, row, kw.RTC_LOOPS_FRAMERATE)
                loop.delay = _get_value(columns, row, kw.RTC_LOOPS_DELAY)
                loop.time_filter_num = self.get_image(row[kw.RTC_LOOPS_TIME_FILTER_NUM])
                loop.time_filter_den = self.get_image(row[kw.RTC_LOOPS_TIME_FILTER_DEN])

                system.rtc.loops.append(loop)

        return system
