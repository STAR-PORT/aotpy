"""
This module contains classes and functions that enable reading AOT FITS files.
"""


import re
import warnings
from datetime import datetime

import numpy as np
from astropy.io import fits

import aotpy
from . import _keywords as kw
from .utils import FITSURLImage, FITSFileImage, image_from_hdu, keyword_is_relevant
from ..base import SystemReader

_reference_pattern = re.compile(r'([^<]+)<(.+)>(\d+)?')


def _type_matches(aot_format: str, fits_type: str) -> bool:
    try:
        return aot_format == kw.fits_type_to_aot(fits_type)
    except ValueError:
        return False


def _convert_row(columns: fits.ColDefs, row: fits.FITS_record, fields: kw.AOTFieldDict) -> dict:
    res = {}
    for name, field in fields.items():
        try:
            value = row[name]
        except KeyError:
            value = None

        match field.format:
            case kw.STRING_FORMAT:
                if value == '':
                    value = None
            case kw.LIST_FORMAT:
                if value is None:
                    value = []
                value = [None if np.isnan(v) else v for v in value]
            case kw.FLOAT_FORMAT:
                if np.isnan(value):
                    value = None
            case kw.INTEGER_FORMAT:
                if value is not None:
                    col = columns[name]
                    if col.null and value == col.null:
                        value = None
            case _:
                raise RuntimeError  # This should never happen
        if value is None and field.mandatory:
            # This will never trigger in list fields, but it's ok because we don't have mandatory list fields
            raise ValueError(f"Found null value in mandatory field '{name}'")
        res[name] = value
    return res


def read_system_from_fits(filename: str, extra_data: bool = False, **kwargs) -> aotpy.AOSystem:
    """
    Get `AOSystem` from FITS file specified by `filename`.

    Parameters
    ----------
    filename
        Path to file to be read into an `AOSystem`.
    extra_data : default = False
        Whether it is expected that the file contains some data that does not fit the AOT standard. If `extra_data` is
        not `True`, user will be warned if extra data is detected.
    **kwargs
        Keyword arguments passed on as options to the file handling function.
    """
    r = FITSReader(filename, extra_data=extra_data, **kwargs)
    return r.get_system()


class FITSReader(SystemReader):
    def _initialize_data(self) -> None:
        """
        Initialize data structures necessary for reading the file.
        """
        self._images: dict[str, list] = {}
        self._time: dict[str, list] = {}
        self._aberrations: dict[str, list] = {}
        self._telescopes: dict[str, list] = {}
        self._sources: dict[str, list] = {}
        self._detectors: dict[str, list] = {}
        self._wfss: dict[str, list] = {}
        self._wfcs: dict[str, list] = {}

        self._table_to_dict = {
            kw.TIME_TABLE: self._time,
            kw.ABERRATIONS_TABLE: self._aberrations,
            kw.TELESCOPES_TABLE: self._telescopes,
            kw.SOURCES_TABLE: self._sources,
            kw.DETECTORS_TABLE: self._detectors,
            kw.WAVEFRONT_SENSORS_TABLE: self._wfss,
            kw.WAVEFRONT_CORRECTORS_TABLE: self._wfcs
        }

        self._extra_header: list[fits.Card] = []
        self._extra_hdus: fits.HDUList = fits.HDUList()
        self._extra_columns: dict[str, list[fits.Column]] = {}
        self._extra_objects: list[aotpy.Referenceable] = []

    def _read(self, **kwargs) -> tuple[aotpy.AOSystem, list]:
        with fits.open(self._filename, **kwargs) as hdus:
            self._primary_header: fits.Header = hdus[0].header
            self._system: aotpy.AOSystem = self._check_header()

            if hdus[0].data is not None:
                raise ValueError('Primary HDU must have no data.')
            for table in kw.MANDATORY_TABLE_SET:
                if table not in hdus:
                    raise ValueError(f"Missing mandatory binary table '{table}'.")

            table_count = {table: 0 for table in kw.TABLE_SET}
            # Skip PrimaryHDU
            for hdu in hdus[1:]:
                if hdu.name in table_count:
                    table_count[hdu.name] += 1
                else:
                    if hdu.is_image:
                        if not hdu.name:
                            raise ValueError('All image extensions in file must have a name.')
                        if hdu.name in self._images:
                            raise ValueError(f"Image name '{hdu.name}' appears repeated in file.")
                        if hdu.data is None:
                            warnings.warn(f"Image HDU '{hdu.name}' was ignored for having no data.")
                        self._images[hdu.name] = [image_from_hdu(hdu), False]
                    else:
                        self._extra_hdus.append(hdu)
            if self._extra_hdus and not self._extra_data_flag:
                warnings.warn(f"""File contains non-AOT HDUs that were ignored: """
                              f"""{', '.join([f"'{x.name}'" for x in self._extra_hdus])}""")

            for name, count in table_count.items():
                if name in kw.MANDATORY_TABLE_SET and count != 1:
                    raise ValueError(f"Mandatory table '{name}' must appear exactly once in file.")
                if count > 1:
                    raise ValueError(f"Secondary table '{name}' cannot appear repeated in file.")
            seq = [table for table in kw.TABLE_SEQUENCE if table_count[table] == 1]
            if seq != [hdu.name for hdu, _ in zip(hdus[1:], seq)]:
                warnings.warn('File does not follow the standard AOT table sequence.')

            self._handle_time(hdus)

            for tup in self._images.values():
                image = tup[0]
                image.time = self._handle_reference(image._time, kw.TIME_TABLE)

            self._handle_atmosphere(hdus)
            self._handle_aberrations(hdus)
            self._handle_telescopes(hdus)
            self._handle_sources(hdus)
            self._handle_detectors(hdus)
            self._handle_scoring_cameras(hdus)
            self._handle_wavefront_sensors(hdus)
            self._handle_wavefront_correctors(hdus)
            self._handle_loops(hdus)

            self._check_usage()
            aux = [self._extra_header, self._extra_hdus, self._extra_columns, self._extra_objects]
            extra = aux if [x for x in aux if x] else None  # extra is None if everything is empty
            return self._system, extra

    def _check_header(self) -> aotpy.AOSystem:
        if kw.AOT_VERSION not in self._primary_header:
            raise ValueError(f"File is not in the AOT format or the mandatory '{kw.AOT_VERSION}' keyword is missing")
            # TODO do something with the version
        try:
            if self._primary_header[kw.AOT_TIMESYS] != kw.AOT_TIMESYS_UTC:
                raise ValueError(f"Keyword '{kw.AOT_TIMESYS}' must have the value '{kw.AOT_TIMESYS_UTC}'")
        except KeyError:
            raise ValueError(f"Mandatory keyword '{kw.AOT_TIMESYS}' is missing") from None

        ao_mode = None
        try:
            aux = self._primary_header[kw.AOT_AO_MODE]
            if aux in kw.AOT_AO_MODE_SET:
                ao_mode = aux
            else:
                warnings.warn(f"Ignored unrecognized value '{aux}' for keyword '{kw.AOT_AO_MODE}'. "
                              f"Value should be one of: {str(kw.AOT_AO_MODE_SET)[1:-1]}")
        except KeyError:
            warnings.warn(f"Mandatory keyword '{kw.AOT_AO_MODE}' is missing")

        beg = None
        try:
            aux = self._primary_header[kw.AOT_DATE_BEG]
            if aux:
                try:
                    beg = datetime.fromisoformat(aux)
                except (ValueError, TypeError):
                    warnings.warn(f"'{kw.AOT_DATE_BEG}' keyword not properly formatted.")
        except KeyError:
            pass

        end = None
        try:
            aux = self._primary_header[kw.AOT_DATE_END]
            if aux:
                try:
                    end = datetime.fromisoformat(aux)
                except (ValueError, TypeError):
                    warnings.warn(f"'{kw.AOT_DATE_END}' keyword not properly formatted.")
        except KeyError:
            pass

        try:
            strehl_ratio = self._primary_header[kw.AOT_STREHL_RATIO]
        except KeyError:
            strehl_ratio = None

        try:
            temporal_error = self._primary_header[kw.AOT_TEMPORAL_ERROR]
        except KeyError:
            temporal_error = None

        try:
            config = self._primary_header[kw.AOT_CONFIG]
        except KeyError:
            config = None

        for card in self._primary_header.cards:
            if card.keyword not in kw.AOT_HEADER_SET and keyword_is_relevant(card.keyword):
                self._extra_header.append(card)
        if self._extra_header and not self._extra_data_flag:
            warnings.warn(f"""Header contains non-AOT keywords that were ignored: """
                          f"""{', '.join([f"'{x.keyword}'" for x in self._extra_header])}""")

        return aotpy.AOSystem(ao_mode=ao_mode, date_beginning=beg, date_end=end, strehl_ratio=strehl_ratio,
                              temporal_error=temporal_error, config=config)

    def _check_bintable(self, hdus: fits.HDUList, table_name: str):
        fields = kw.TABLE_FIELDS[table_name]
        table = hdus[table_name]
        if not isinstance(table, fits.BinTableHDU):
            raise ValueError(f"HDU '{table.name}' must be a binary table")

        field_count = {field: 0 for field in fields}
        for col in table.columns.columns:
            if col.name in field_count:
                field_count[col.name] += 1
                field = fields[col.name]
                if not _type_matches(field.format, col.format):
                    raise ValueError(f"Column '{col.name}' in table '{table_name}' must be a '{field.format}'"
                                     f" field (found '{col.format}')")
                if col.unit != field.unit:
                    warnings.warn(f"Column '{col.name}' in table '{table_name}' does not have a standard unit. "
                                  f"Found '{col.unit}', expected '{field.unit}'")
                if field.unique:
                    u = table.data[col.name]
                    if len(np.unique(u)) != len(u):
                        raise ValueError(
                            f"Column '{col.name}' in table '{table_name}' must have unique values.")
            else:
                self._extra_columns.setdefault(table_name, []).append(col)
        if table_name in self._extra_columns and not self._extra_data_flag:
            warnings.warn(f"""Table '{table_name}' contains non-AOT columns that were ignored: """
                          f"""{', '.join([f"'{x.name}'" for x in self._extra_columns[table_name]])}""")

        for name, count in field_count.items():
            if count < 1:
                warnings.warn(f"Column '{name}' missing in table '{table_name}'.")
            if count > 1:
                raise ValueError(f"Column '{name}' cannot appear repeated in table '{table_name}'.")

        seq = [field_name for field_name, count in field_count.items() if count > 0]
        if seq != [col.name for col, _ in zip(table.columns.columns, seq)]:
            warnings.warn(f"Non-AOT column sequence in table '{table_name}'.")

    def _handle_image(self, ref: str):
        if ref is None:
            return None
        fullmatch = _reference_pattern.fullmatch(ref)
        if fullmatch is None:
            warnings.warn(f"Image reference '{ref}' was ignored: not properly formatted.")
            return None
        prefix, name, index = fullmatch.groups()

        match prefix:
            case kw.INTERNAL_REFERENCE:
                try:
                    aux = self._images[name]
                    aux[1] = True
                    return aux[0]
                except KeyError:
                    warnings.warn(f"Could not find internal image referenced by '{ref}'.")
                    return None
            case kw.FILE_REFERENCE | kw.URL_REFERENCE:
                if index is not None:
                    try:
                        index = int(index)
                    except ValueError:
                        warnings.warn(f"Index in file reference '{ref}' was ignored: not properly formatted.")
                        index = None
                if prefix == kw.FILE_REFERENCE:
                    image = FITSFileImage(name, index)
                else:
                    image = FITSURLImage(name, index)

                image.time = self._handle_reference(image._time, kw.TIME_TABLE)
                return image
            case _:
                warnings.warn(f"Reference '{ref}' was ignored: expected an image reference.")
                return None

    def _handle_reference(self, ref: str, table: str):
        if ref is None:
            return None
        fullmatch = _reference_pattern.fullmatch(ref)
        if fullmatch is None:
            warnings.warn(f"Row reference '{ref}' was ignored: not properly formatted.")
            return None
        prefix, name, index = fullmatch.groups()

        if prefix != kw.ROW_REFERENCE:
            warnings.warn(f"Reference '{ref}' was ignored: expected a row reference.")
            return None

        d = self._table_to_dict[table]
        try:
            aux = d[name]
            aux[1] = True
            return aux[0]
        except KeyError:
            warnings.warn(f"Could not find row referenced by '{ref}'. Ignoring reference.")
            return None

    def _check_usage(self):
        # We don't need to check the usage of atmosphere parameters, sources, scoring cameras, wavefront sensors,
        # wavefront correctors or loops, since they are always referenced by the AOSystem class
        a = [(self._images, 'image extensions'),
             (self._time, 'time rows'),
             (self._aberrations, 'aberration rows'),
             (self._telescopes, 'telescope rows'),
             (self._detectors, 'detector rows')]

        for d, name in a:
            unused = {k: v[0] for k, v in d.items() if not v[1]}
            if unused:
                self._extra_objects.extend(unused.values())
                if not self._extra_data_flag:
                    warnings.warn(f"""File contains some {name} were ignored for never being referenced: """
                                  f"""{', '.join([f"'{x}'" for x in unused])}""")

    def _handle_time(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.TIME_TABLE)

        table = hdus[kw.TIME_TABLE]
        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.TIME_FIELDS)

            self._time[data[kw.REFERENCE_UID]] = [aotpy.Time(
                uid=data[kw.REFERENCE_UID],
                timestamps=data[kw.TIME_TIMESTAMPS],
                frame_numbers=data[kw.TIME_FRAME_NUMBERS]
            ), False]

    def _handle_atmosphere(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.ATMOSPHERIC_PARAMETERS_TABLE)

        table = hdus[kw.ATMOSPHERIC_PARAMETERS_TABLE]
        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.ATMOSPHERIC_PARAMETERS_FIELDS)

            self._system.atmosphere_params.append(
                aotpy.AtmosphericParameters(
                    uid=data[kw.REFERENCE_UID],
                    wavelength=data[kw.ATMOSPHERIC_PARAMETERS_WAVELENGTH],
                    time=self._handle_reference(data[kw.TIME_REFERENCE], kw.TIME_TABLE),
                    r0=data[kw.ATMOSPHERIC_PARAMETERS_R0],
                    fwhm=data[kw.ATMOSPHERIC_PARAMETERS_FWHM],
                    tau0=data[kw.ATMOSPHERIC_PARAMETERS_TAU0],
                    theta0=data[kw.ATMOSPHERIC_PARAMETERS_THETA0],
                    layers_weight=self._handle_image(data[kw.ATMOSPHERIC_PARAMETERS_LAYERS_WEIGHT]),
                    layers_height=self._handle_image(data[kw.ATMOSPHERIC_PARAMETERS_LAYERS_HEIGHT]),
                    layers_l0=self._handle_image(data[kw.ATMOSPHERIC_PARAMETERS_LAYERS_L0]),
                    layers_wind_speed=self._handle_image(data[kw.ATMOSPHERIC_PARAMETERS_LAYERS_WIND_SPEED]),
                    layers_wind_direction=self._handle_image(data[kw.ATMOSPHERIC_PARAMETERS_LAYERS_WIND_DIRECTION]),
                    transformation_matrix=self._handle_image(data[kw.TRANSFORMATION_MATRIX])
                )
            )

    def _handle_aberrations(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.ABERRATIONS_TABLE)

        table = hdus[kw.ABERRATIONS_TABLE]
        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.ABERRATION_FIELDS)

            self._aberrations[data[kw.REFERENCE_UID]] = [aotpy.Aberration(
                uid=data[kw.REFERENCE_UID],
                modes=self._handle_image(data[kw.ABERRATION_MODES]),
                coefficients=self._handle_image(data[kw.ABERRATION_COEFFICIENTS]),
                offsets=[aotpy.Coordinates(x, y)
                         for x, y in zip(data[kw.ABERRATION_X_OFFSETS], data[kw.ABERRATION_Y_OFFSETS])],
            ), False]

    def _handle_telescopes(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.TELESCOPES_TABLE)

        table = hdus[kw.TELESCOPES_TABLE]
        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.TELESCOPE_FIELDS)

            uid = data[kw.REFERENCE_UID]
            t = data[kw.TELESCOPE_TYPE]
            if t == kw.TELESCOPE_TYPE_MAIN:
                if self._system.main_telescope is not None:
                    raise ValueError(f"There must only be one telescope of type '{kw.TELESCOPE_TYPE_MAIN}'.")
                tel = aotpy.MainTelescope(uid)
                self._system.main_telescope = tel
                self._telescopes[uid] = [tel, True]  # The main telescope is always referenced by the AOSystem
            elif t == kw.TELESCOPE_TYPE_LLT:
                tel = aotpy.LaserLaunchTelescope(uid)
                self._telescopes[uid] = [tel, False]
            else:
                warnings.warn(f"Skipped telescope {uid}: unknown type '{t}'.")
                continue

            tel.latitude = data[kw.TELESCOPE_LATITUDE]
            tel.longitude = data[kw.TELESCOPE_LONGITUDE]
            tel.elevation = data[kw.TELESCOPE_ELEVATION]
            tel.azimuth = data[kw.TELESCOPE_AZIMUTH]
            tel.parallactic = data[kw.TELESCOPE_PARALLACTIC]
            tel.pupil_mask = self._handle_image(data[kw.TELESCOPE_PUPIL_MASK])
            tel.pupil_angle = data[kw.TELESCOPE_PUPIL_ANGLE]
            tel.enclosing_diameter = data[kw.TELESCOPE_ENCLOSING_D]
            tel.inscribed_diameter = data[kw.TELESCOPE_INSCRIBED_D]
            tel.obstruction_diameter = data[kw.TELESCOPE_OBSTRUCTION_D]

            t = data[kw.TELESCOPE_SEGMENTS_TYPE]
            if t == kw.TELESCOPE_SEGMENT_TYPE_MONOLITHIC:
                seg = aotpy.Monolithic()
            else:
                if t == kw.TELESCOPE_SEGMENT_TYPE_CIRCLE:
                    seg = aotpy.CircularSegments()
                elif t == kw.TELESCOPE_SEGMENT_TYPE_HEXAGON:
                    seg = aotpy.HexagonalSegments()
                else:
                    warnings.warn(f"Ignored unknown segment type '{t}'.")
                    seg = aotpy.Monolithic()
                seg.size = data[kw.TELESCOPE_SEGMENTS_SIZE]
                seg.coordinates = [aotpy.Coordinates(x, y)
                                   for x, y in zip(data[kw.TELESCOPE_SEGMENTS_X], data[kw.TELESCOPE_SEGMENTS_Y])]

            tel.segments = seg
            tel.transformation_matrix = self._handle_image(data[kw.TRANSFORMATION_MATRIX])
            tel.aberration = self._handle_reference(data[kw.ABERRATION_REFERENCE], kw.ABERRATIONS_TABLE)

    def _handle_sources(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.SOURCES_TABLE)

        table = hdus[kw.SOURCES_TABLE]
        existing_types = table.data['TYPE']
        if kw.SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR in existing_types:
            if kw.SOURCES_SODIUM_LGS_TABLE in hdus:
                self._check_bintable(hdus, kw.SOURCES_SODIUM_LGS_TABLE)
            else:
                raise ValueError(f"Missing table '{kw.SOURCES_SODIUM_LGS_TABLE}' must exist when "
                                 f"'{kw.SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR}' type sources exist")
        if kw.SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR in existing_types:
            if kw.SOURCES_RAYLEIGH_LGS_TABLE in hdus:
                self._check_bintable(hdus, kw.SOURCES_RAYLEIGH_LGS_TABLE)
            else:
                raise ValueError(f"Missing table '{kw.SOURCES_SODIUM_LGS_TABLE}' must exist when "
                                 f"'{kw.SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR}' type sources exist")

        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.SOURCE_FIELDS)

            uid = data[kw.REFERENCE_UID]
            t = data[kw.SOURCE_TYPE]
            if t == kw.SOURCE_TYPE_SCIENCE_STAR:
                src = aotpy.ScienceStar(uid)
            elif t == kw.SOURCE_TYPE_NATURAL_GUIDE_STAR:
                src = aotpy.NaturalGuideStar(uid)
            elif t == kw.SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR:
                # Try to find uid in secondary table
                f = next((x for x in hdus[kw.SOURCES_SODIUM_LGS_TABLE].data if x[kw.REFERENCE_UID] == uid), None)
                if f is None:
                    raise ValueError(f"Source '{uid}' not found in table '{kw.SOURCES_SODIUM_LGS_TABLE}' even"
                                     f" though it is of type '{t}'")

                other_data = _convert_row(hdus[kw.SOURCES_SODIUM_LGS_TABLE].columns, f, kw.SOURCE_SODIUM_LGS_FIELDS)
                src = aotpy.SodiumLaserGuideStar(
                    uid=uid,
                    height=other_data[kw.SOURCE_SODIUM_LGS_HEIGHT],
                    profile=self._handle_image(other_data[kw.SOURCE_SODIUM_LGS_PROFILE]),
                    altitudes=other_data[kw.SOURCE_SODIUM_LGS_ALTITUDES],
                    laser_launch_telescope=self._handle_reference(other_data[kw.LASER_LAUNCH_TELESCOPE_REFERENCE],
                                                                  kw.TELESCOPES_TABLE)
                )
            elif t == kw.SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR:
                # Try to find uid in secondary table
                f = next((x for x in hdus[kw.SOURCES_RAYLEIGH_LGS_TABLE].data if x[kw.REFERENCE_UID] == uid), None)
                if f is None:
                    raise ValueError(f"Source '{uid}' not found in table '{kw.SOURCES_RAYLEIGH_LGS_TABLE}' even"
                                     f" though it is of type '{t}'")

                other_data = _convert_row(hdus[kw.SOURCES_RAYLEIGH_LGS_TABLE].columns, f, kw.SOURCE_RAYLEIGH_LGS_FIELDS)
                src = aotpy.RayleighLaserGuideStar(
                    uid=uid,
                    distance=other_data[kw.SOURCE_RAYLEIGH_LGS_DISTANCE],
                    depth=other_data[kw.SOURCE_RAYLEIGH_LGS_DEPTH],
                    laser_launch_telescope=self._handle_reference(other_data[kw.LASER_LAUNCH_TELESCOPE_REFERENCE],
                                                                  kw.TELESCOPES_TABLE)
                )
            else:
                warnings.warn(f"Skipped source '{uid}': unknown type '{t}'.")
                continue

            src.right_ascension = data[kw.SOURCE_RIGHT_ASCENSION]
            src.declination = data[kw.SOURCE_DECLINATION]
            src.elevation_offset = data[kw.SOURCE_ELEVATION_OFFSET]
            src.azimuth_offset = data[kw.SOURCE_AZIMUTH_OFFSET]
            src.width = data[kw.SOURCE_WIDTH]

            self._sources[uid] = [src, True]  # sources are always referenced by AOSystem
            self._system.sources.append(src)

    def _handle_detectors(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.DETECTORS_TABLE)

        table = hdus[kw.DETECTORS_TABLE]
        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.DETECTOR_FIELDS)

            self._detectors[data[kw.REFERENCE_UID]] = [aotpy.Detector(
                uid=data[kw.REFERENCE_UID],
                type=data[kw.DETECTOR_TYPE],
                sampling_technique=data[kw.DETECTOR_SAMPLING_TECHNIQUE],
                shutter_type=data[kw.DETECTOR_SHUTTER_TYPE],
                flat_field=self._handle_image(data[kw.DETECTOR_FLAT_FIELD]),
                readout_noise=data[kw.DETECTOR_READOUT_NOISE],
                pixel_intensities=self._handle_image(data[kw.DETECTOR_PIXEL_INTENSITIES]),
                integration_time=data[kw.DETECTOR_INTEGRATION_TIME],
                coadds=data[kw.DETECTOR_COADDS],
                dark=self._handle_image(data[kw.DETECTOR_DARK]),
                weight_map=self._handle_image(data[kw.DETECTOR_WEIGHT_MAP]),
                quantum_efficiency=data[kw.DETECTOR_QUANTUM_EFFICIENCY],
                pixel_scale=data[kw.DETECTOR_PIXEL_SCALE],
                binning=data[kw.DETECTOR_BINNING],
                bandwidth=data[kw.DETECTOR_BANDWIDTH],
                transmission_wavelength=data[kw.DETECTOR_TRANSMISSION_WAVELENGTH],
                transmission=data[kw.DETECTOR_TRANSMISSION],
                sky_background=self._handle_image(data[kw.DETECTOR_SKY_BACKGROUND]),
                gain=data[kw.DETECTOR_GAIN],
                excess_noise=data[kw.DETECTOR_EXCESS_NOISE],
                filter=data[kw.DETECTOR_FILTER],
                bad_pixel_map=self._handle_image(data[kw.DETECTOR_BAD_PIXEL_MAP]),
                dynamic_range=data[kw.DETECTOR_DYNAMIC_RANGE],
                readout_rate=data[kw.DETECTOR_READOUT_RATE],
                frame_rate=data[kw.DETECTOR_FRAME_RATE],
                transformation_matrix=self._handle_image(data[kw.TRANSFORMATION_MATRIX])
            ), False]

    def _handle_scoring_cameras(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.SCORING_CAMERAS_TABLE)

        table = hdus[kw.SCORING_CAMERAS_TABLE]
        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.SCORING_CAMERA_FIELDS)

            self._system.scoring_cameras.append(aotpy.ScoringCamera(
                uid=data[kw.REFERENCE_UID],
                pupil_mask=self._handle_image(data[kw.SCORING_CAMERA_PUPIL_MASK]),
                wavelength=data[kw.SCORING_CAMERA_WAVELENGTH],
                transformation_matrix=self._handle_image(data[kw.TRANSFORMATION_MATRIX]),
                detector=self._handle_reference(data[kw.DETECTOR_REFERENCE], kw.DETECTORS_TABLE),
                aberration=self._handle_reference(data[kw.ABERRATION_REFERENCE], kw.ABERRATIONS_TABLE),
            ))

    def _handle_wavefront_sensors(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.WAVEFRONT_SENSORS_TABLE)

        table = hdus[kw.WAVEFRONT_SENSORS_TABLE]
        existing_types = table.data['TYPE']
        if kw.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN in existing_types:
            if kw.WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE in hdus:
                self._check_bintable(hdus, kw.WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE)
            else:
                raise ValueError(f"Missing table '{kw.WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE}' must exist when "
                                 f"'{kw.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN}' type wavefront sensors exist")
        if kw.WAVEFRONT_SENSOR_TYPE_PYRAMID in existing_types:
            if kw.WAVEFRONT_SENSORS_PYRAMID_TABLE in hdus:
                self._check_bintable(hdus, kw.WAVEFRONT_SENSORS_PYRAMID_TABLE)
            else:
                raise ValueError(f"Missing table '{kw.WAVEFRONT_SENSORS_PYRAMID_TABLE}' must exist when "
                                 f"'{kw.WAVEFRONT_SENSOR_TYPE_PYRAMID}' type wavefront sensors exist")

        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.WAVEFRONT_SENSOR_FIELDS)

            uid = data[kw.REFERENCE_UID]
            t = data[kw.WAVEFRONT_SENSOR_TYPE]
            source = self._handle_reference(data[kw.SOURCE_REFERENCE], kw.SOURCES_TABLE)
            dimensions = data[kw.WAVEFRONT_SENSOR_DIMENSIONS]
            n_valid_subapertures = data[kw.WAVEFRONT_SENSOR_N_VALID_SUBAPERTURES]
            if t == kw.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN:
                # Try to find uid in secondary table
                f = next(
                    (x for x in hdus[kw.WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE].data if x[kw.REFERENCE_UID] == uid),
                    None)
                if f is None:
                    raise ValueError(f"Wavefront sensor '{uid}' not found in table "
                                     f"'{kw.WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE}' even though it is of type '{t}'")

                other_data = _convert_row(hdus[kw.WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE].columns, f,
                                          kw.WAVEFRONT_SENSOR_SHACK_HARTMANN_FIELDS)
                if dimensions != 2:
                    warnings.warn(f"Unexpected value for '{kw.WAVEFRONT_SENSOR_DIMENSIONS}' in wavefront sensor '{uid}'"
                                  f" of type '{t}'. Expected 2, got {dimensions}.")
                wfs = aotpy.ShackHartmann(
                    uid=uid,
                    source=source,
                    n_valid_subapertures=n_valid_subapertures,
                    centroiding_algorithm=other_data[kw.WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROIDING_ALGORITHM],
                    centroid_gains=self._handle_image(other_data[kw.WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROID_GAINS]),
                    spot_fwhm=self._handle_image(other_data[kw.WAVEFRONT_SENSOR_SHACK_HARTMANN_SPOT_FWHM])
                )
            elif t == kw.WAVEFRONT_SENSOR_TYPE_PYRAMID:
                # Try to find uid in secondary table
                f = next((x for x in hdus[kw.WAVEFRONT_SENSORS_PYRAMID_TABLE].data if x[kw.REFERENCE_UID] == uid), None)
                if f is None:
                    raise ValueError(f"Wavefront sensor '{uid}' not found in table "
                                     f"'{kw.WAVEFRONT_SENSORS_PYRAMID_TABLE}' even though it is of type '{t}'")

                other_data = _convert_row(hdus[kw.WAVEFRONT_SENSORS_PYRAMID_TABLE].columns, f,
                                          kw.WAVEFRONT_SENSOR_PYRAMID_FIELDS)
                wfs = aotpy.Pyramid(
                    uid=uid,
                    source=source,
                    dimensions=dimensions,
                    n_valid_subapertures=n_valid_subapertures,
                    n_sides=other_data[kw.WAVEFRONT_SENSOR_PYRAMID_N_SIDES],
                    modulation=other_data[kw.WAVEFRONT_SENSOR_PYRAMID_MODULATION]
                )
            else:
                warnings.warn(f"Skipped wavefront sensor '{uid}': unknown type '{t}'.")
                continue

            wfs.measurements = self._handle_image(data[kw.WAVEFRONT_SENSOR_MEASUREMENTS])
            wfs.ref_measurements = self._handle_image(data[kw.WAVEFRONT_SENSOR_REF_MEASUREMENTS])
            wfs.subaperture_mask = self._handle_image(data[kw.WAVEFRONT_SENSOR_SUBAPERTURE_MASK])
            wfs.mask_offsets = [aotpy.Coordinates(x, y) for x, y in zip(data[kw.WAVEFRONT_SENSOR_MASK_X_OFFSETS],
                                                                        data[kw.WAVEFRONT_SENSOR_MASK_Y_OFFSETS])]
            wfs.subaperture_size = data[kw.WAVEFRONT_SENSOR_SUBAPERTURE_SIZE]
            wfs.subaperture_intensities = self._handle_image(data[kw.WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES])
            wfs.wavelength = data[kw.WAVEFRONT_SENSOR_WAVELENGTH]
            wfs.optical_gain = self._handle_image(data[kw.WAVEFRONT_SENSOR_OPTICAL_GAIN])
            wfs.transformation_matrix = self._handle_image(data[kw.TRANSFORMATION_MATRIX])
            wfs.detector = self._handle_reference(data[kw.DETECTOR_REFERENCE], kw.DETECTORS_TABLE)
            wfs.aberration = self._handle_reference(data[kw.ABERRATION_REFERENCE], kw.ABERRATIONS_TABLE)
            wfs.non_common_path_aberration = self._handle_reference(data[kw.NCPA_REFERENCE], kw.ABERRATIONS_TABLE)

            self._wfss[uid] = [wfs, True]  # wavefront sensors are always referenced by AOSystem
            self._system.wavefront_sensors.append(wfs)

    def _handle_wavefront_correctors(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.WAVEFRONT_CORRECTORS_TABLE)

        table = hdus[kw.WAVEFRONT_CORRECTORS_TABLE]
        existing_types = table.data['TYPE']
        if kw.WAVEFRONT_CORRECTOR_TYPE_DM in existing_types:
            if kw.WAVEFRONT_CORRECTORS_DM_TABLE in hdus:
                self._check_bintable(hdus, kw.WAVEFRONT_CORRECTORS_DM_TABLE)
            else:
                raise ValueError(f"Missing table '{kw.WAVEFRONT_CORRECTORS_DM_TABLE}' must exist when "
                                 f"'{kw.WAVEFRONT_CORRECTOR_TYPE_DM}' type wavefront correctors exist")

        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.WAVEFRONT_CORRECTOR_FIELDS)

            uid = data[kw.REFERENCE_UID]
            t = data[kw.WAVEFRONT_CORRECTOR_TYPE]
            telescope = self._handle_reference(data[kw.TELESCOPE_REFERENCE], kw.TELESCOPES_TABLE)

            if t == kw.WAVEFRONT_CORRECTOR_TYPE_DM:
                # Try to find uid in secondary table
                f = next((x for x in hdus[kw.WAVEFRONT_CORRECTORS_DM_TABLE].data if x[kw.REFERENCE_UID] == uid), None)
                if f is None:
                    raise ValueError(f"Wavefront corrector '{uid}' not found in table "
                                     f"'{kw.WAVEFRONT_CORRECTORS_DM_TABLE}' even though it is of type '{t}'")

                other_data = _convert_row(hdus[kw.WAVEFRONT_CORRECTORS_DM_TABLE].columns, f,
                                          kw.WAVEFRONT_CORRECTOR_DM_FIELDS)
                cor = aotpy.DeformableMirror(
                    uid=uid,
                    telescope=telescope,
                    n_valid_actuators=data[kw.WAVEFRONT_CORRECTOR_N_VALID_ACTUATORS],
                    actuator_coordinates=[aotpy.Coordinates(x, y) for x, y in
                                          zip(other_data[kw.WAVEFRONT_CORRECTOR_DM_ACTUATORS_X],
                                              other_data[kw.WAVEFRONT_CORRECTOR_DM_ACTUATORS_Y])],
                    influence_function=self._handle_image(other_data[kw.WAVEFRONT_CORRECTOR_DM_INFLUENCE_FUNCTION]),
                    stroke=other_data[kw.WAVEFRONT_CORRECTOR_DM_STROKE]
                )
            elif t == kw.WAVEFRONT_CORRECTOR_TYPE_TTM:
                cor = aotpy.TipTiltMirror(
                    uid=uid,
                    telescope=telescope
                )
            elif t == kw.WAVEFRONT_CORRECTOR_TYPE_LS:
                cor = aotpy.LinearStage(
                    uid=uid,
                    telescope=telescope
                )
            else:
                warnings.warn(f"Skipped wavefront corrector '{uid}': unknown type '{t}'.")
                continue

            cor.pupil_mask = self._handle_image(data[kw.WAVEFRONT_CORRECTOR_PUPIL_MASK])
            cor.tfz_num = data[kw.WAVEFRONT_CORRECTOR_TFZ_NUM]
            cor.tfz_den = data[kw.WAVEFRONT_CORRECTOR_TFZ_DEN]
            cor.transformation_matrix = self._handle_image(data[kw.TRANSFORMATION_MATRIX])
            cor.aberration = self._handle_reference(data[kw.ABERRATION_REFERENCE], kw.ABERRATIONS_TABLE)

            self._wfcs[uid] = [cor, True]  # wavefront corectors are always referenced by AOSystem
            self._system.wavefront_correctors.append(cor)

    def _handle_loops(self, hdus: fits.HDUList):
        self._check_bintable(hdus, kw.LOOPS_TABLE)

        table = hdus[kw.LOOPS_TABLE]
        existing_types = table.data['TYPE']
        if kw.LOOPS_TYPE_CONTROL in existing_types:
            if kw.LOOPS_CONTROL_TABLE in hdus:
                self._check_bintable(hdus, kw.LOOPS_CONTROL_TABLE)
            else:
                raise ValueError(f"Missing table '{kw.LOOPS_CONTROL_TABLE}' must exist when "
                                 f"'{kw.LOOPS_TYPE_CONTROL}' type loops exist")
        if kw.LOOPS_TYPE_OFFLOAD in existing_types:
            if kw.LOOPS_OFFLOAD_TABLE in hdus:
                self._check_bintable(hdus, kw.LOOPS_OFFLOAD_TABLE)
            else:
                raise ValueError(f"Missing table '{kw.LOOPS_OFFLOAD_TABLE}' must exist when "
                                 f"'{kw.LOOPS_TYPE_OFFLOAD}' type loops exist")

        columns = table.columns
        for row in table.data:
            data = _convert_row(columns, row, kw.LOOPS_FIELDS)

            uid = data[kw.REFERENCE_UID]
            t = data[kw.WAVEFRONT_SENSOR_TYPE]
            commanded = self._handle_reference(data[kw.LOOPS_COMMANDED], kw.WAVEFRONT_CORRECTORS_TABLE)

            if t == kw.LOOPS_TYPE_CONTROL:
                # Try to find uid in secondary table
                f = next((x for x in hdus[kw.LOOPS_CONTROL_TABLE].data if x[kw.REFERENCE_UID] == uid), None)
                if f is None:
                    raise ValueError(f"Loop '{uid}' not found in table "
                                     f"'{kw.LOOPS_CONTROL_TABLE}' even though it is of type '{t}'")

                other_data = _convert_row(hdus[kw.LOOPS_CONTROL_TABLE].columns, f, kw.LOOPS_CONTROL_FIELDS)
                loop = aotpy.ControlLoop(
                    uid=uid,
                    commanded_corrector=commanded,
                    input_sensor=self._handle_reference(other_data[kw.LOOPS_CONTROL_INPUT_SENSOR],
                                                        kw.WAVEFRONT_SENSORS_TABLE),
                    modes=self._handle_image(other_data[kw.LOOPS_CONTROL_MODES]),
                    modal_coefficients=self._handle_image(other_data[kw.LOOPS_CONTROL_MODAL_COEFFICIENTS]),
                    control_matrix=self._handle_image(other_data[kw.LOOPS_CONTROL_CONTROL_MATRIX]),
                    measurements_to_modes=self._handle_image(other_data[kw.LOOPS_CONTROL_MEASUREMENTS_TO_MODES]),
                    modes_to_commands=self._handle_image(other_data[kw.LOOPS_CONTROL_MODES_TO_COMMANDS]),
                    interaction_matrix=self._handle_image(other_data[kw.LOOPS_CONTROL_INTERACTION_MATRIX]),
                    commands_to_modes=self._handle_image(other_data[kw.LOOPS_CONTROL_COMMANDS_TO_MODES]),
                    modes_to_measurements=self._handle_image(other_data[kw.LOOPS_CONTROL_MODES_TO_MEASUREMENTS]),
                    residual_commands=self._handle_image(other_data[kw.LOOPS_CONTROL_RESIDUAL_COMMANDS])
                )
            elif t == kw.LOOPS_TYPE_OFFLOAD:
                # Try to find uid in secondary table
                f = next((x for x in hdus[kw.LOOPS_OFFLOAD_TABLE].data if x[kw.REFERENCE_UID] == uid), None)
                if f is None:
                    raise ValueError(f"Loop '{uid}' not found in table "
                                     f"'{kw.LOOPS_OFFLOAD_TABLE}' even though it is of type '{t}'")

                other_data = _convert_row(hdus[kw.LOOPS_OFFLOAD_TABLE].columns, f, kw.LOOPS_OFFLOAD_FIELDS)
                loop = aotpy.OffloadLoop(
                    uid=uid,
                    commanded_corrector=commanded,
                    input_corrector=self._handle_reference(other_data[kw.LOOPS_OFFLOAD_INPUT_CORRECTOR],
                                                           kw.WAVEFRONT_CORRECTORS_TABLE),
                    offload_matrix=self._handle_image(other_data[kw.LOOPS_OFFLOAD_OFFLOAD_MATRIX])
                )
            else:
                warnings.warn(f"Skipped loop '{uid}': unknown type '{t}'.")
                continue

            loop.time = self._handle_reference(data[kw.TIME_REFERENCE], kw.TIME_TABLE)
            if (status := data[kw.LOOPS_STATUS]) is None:
                loop.closed = None
            elif status == kw.LOOPS_STATUS_CLOSED:
                loop.closed = True
            elif status == kw.LOOPS_STATUS_OPEN:
                loop.closed = False
            else:
                warnings.warn(f"Ignored unknown loop status '{status}'.")
                loop.closed = None
            loop.commands = self._handle_image(data[kw.LOOPS_COMMANDS])
            loop.ref_commands = self._handle_image(data[kw.LOOPS_REF_COMMANDS])
            loop.framerate = data[kw.LOOPS_FRAMERATE]
            loop.delay = data[kw.LOOPS_DELAY]
            loop.time_filter_num = self._handle_image(data[kw.LOOPS_TIME_FILTER_NUM])
            loop.time_filter_den = self._handle_image(data[kw.LOOPS_TIME_FILTER_DEN])

            self._system.loops.append(loop)
