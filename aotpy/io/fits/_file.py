import numbers
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

import numpy as np
from astropy.io import fits

from . import _strings as _s
from . import compat
from ._strings import *
from .utils import _keyword_is_relevant, _valid_filename, _datetime_to_iso

_image_reference_pattern = re.compile(r'([^<]+)<(.+)>(\d+)?')
_row_reference_pattern = re.compile(rf'{_s.ROW_REFERENCE}<(.+)>')

# NaN is defined here as a single precision float (32-bits), which is the lowest possible float precision in FITS.
# The goal is to ensure that low precision numpy arrays aren't unnecessarily upcasted just because of NaN.
_nan = np.single(np.nan)

# Given that all integer values in AOT are "counts" they should not be negative. Therefore, we represent null integers
# with the lowest integer in a signed 16-bit integer, in order to avoid unnecessary upcasting.
_int_min = np.iinfo(np.int16).min


class AOTVerificationWarning(UserWarning):
    pass


class AOTVerificationException(Exception):
    pass


class AOTFITSErrorLevel(IntEnum):
    PEDANTIC = 1
    """Goes against recommendations"""
    WARNING = 2
    """No functionality loss, but unexpected behavior"""
    SERIOUS = 3
    """Functionality and/or data integrity compromised"""
    CRITICAL = 4
    """Complete loss of functionality"""


class AOTFITSErrorManager:
    def __init__(self, *, exception_level: AOTFITSErrorLevel = AOTFITSErrorLevel.SERIOUS,
                 log_level: AOTFITSErrorLevel = AOTFITSErrorLevel.PEDANTIC):
        if not isinstance(exception_level, AOTFITSErrorLevel):
            raise ValueError("Invalid exception_level parameter")

        if not isinstance(log_level, AOTFITSErrorLevel):
            raise ValueError("Invalid exception_level parameter")

        self.exception_level = exception_level
        self.log_level = log_level

        self.error: dict[str, list[tuple[AOTFITSErrorLevel, str]]] = {}

    def add_error(self, level: AOTFITSErrorLevel, message: str, context: str):
        self.error.setdefault(context, []).append((level, message))
        if level >= self.exception_level:
            # TODO be more explicit on the error
            raise AOTVerificationException(f'[{context}] {message} (Error level: {level.name})')
        if level >= self.log_level:
            warnings.warn(f'[{context}] {message} (Error level: {level.name})', AOTVerificationWarning, stacklevel=2)

    def print_full_report(self):
        # TODO make this pretty
        if not self.error:
            print('No errors found.')
        else:
            print('Some errors were found:')
            for context in self.error:
                for level, message in self.error[context]:
                    print(f'[{context}] {message} (Error level: {level.name})')


def verify_file(path: str | os.PathLike, log_level=AOTFITSErrorLevel.CRITICAL,
                exception_level=AOTFITSErrorLevel.CRITICAL, **kwargs):
    f = AOTFITSFile.from_file(path, log_level=log_level, exception_level=exception_level, **kwargs)
    f.issue_manager.print_full_report()


@dataclass
class AOTField:
    name: str
    format: str
    unit: str
    mandatory: bool = False
    unique: bool = False
    reference: str = None
    description: str = None
    data: list = field(default_factory=list)
    allowed_list: list = None
    ignored: bool = False
    canonical_name: str = field(init=False, default=None)

    def __getitem__(self, item) -> str | int | float | list:
        return self.data[item]

    def append(self, item: str | int | float | list):
        self.data.append(item)

    def __post_init__(self):
        # 'name' might be modifed for reading older specifications, so we keep a canonical copy for writing
        self.canonical_name = self.name


class AOTFieldList(list):
    def __init__(self, *args: AOTField):
        super().__init__(args)
        self._dict = None

    def _get_dict(self) -> dict:
        if self._dict is None:
            self._dict = {fld.name: fld for fld in self}
        return self._dict

    def __getitem__(self, key) -> AOTField:
        if isinstance(key, str):
            return self._get_dict()[key]

        return super().__getitem__(key)

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._get_dict()

        return super().__contains__(key)


def rows_must_be_referenced(cls):
    """Decorator function for tables whose rows must be referenced."""

    def check_was_referenced(self: AOTFITSTable):
        for uid, was_referenced in self.uid_was_referenced.items():
            if not was_referenced:
                self.add_error(AOTFITSErrorLevel.WARNING, f"'{uid}' is never referenced.")

    cls.check_was_referenced = check_was_referenced
    return cls


# TODO check for compressed image
# fits.hdu.image.ExtensionHDU ?
def get_image_from_hdu(hdu: fits.ImageHDU) -> tuple[str, np.ndarray, str, str, list[fits.Card]]:
    unit = None
    time_ref = None
    metadata: list[fits.Card] = []
    for card in hdu.header.cards:
        if _keyword_is_relevant(card.keyword):
            if card.keyword == _s.IMAGE_UNIT:
                unit = card.value
            elif card.keyword == _s.TIME_REFERENCE:
                time_ref = card.value
            else:
                metadata.append(card)

    return hdu.name, hdu.data, unit, time_ref, metadata


@dataclass
class AOTFITSImage:
    extname: str
    data: np.ndarray
    unit: str
    time_ref: str
    metadata: list[fits.Card]
    time_uid: str = field(init=False, default=None)


@dataclass
class AOTFITSInternalImage(AOTFITSImage):
    was_referenced: bool = field(init=False, default=False)

    @classmethod
    def from_hdu(cls, hdu: fits.ImageHDU):
        return cls(*get_image_from_hdu(hdu))

    def to_hdu(self) -> fits.ImageHDU:
        hdr = fits.Header(self.metadata)
        if self.time_uid is not None:
            hdr[TIME_REFERENCE] = AOTFITSFile.create_row_reference(self.time_uid)
        if self.unit is not None:
            hdr[IMAGE_UNIT] = self.unit
        return fits.ImageHDU(name=self.extname, data=self.data, header=hdr)

    def __str__(self):
        return f'{INTERNAL_REFERENCE}<{self.extname}>'


@dataclass
class AOTFITSExternalImage(AOTFITSImage):
    is_url: bool
    path: str
    index: int

    @classmethod
    def from_hdu(cls, hdu: fits.ImageHDU, prefix: str, path: str, index: int):
        return cls(*get_image_from_hdu(hdu), prefix == URL_REFERENCE, path, index)

    def __str__(self):
        ref = f'{URL_REFERENCE if self.is_url else FILE_REFERENCE}<{self.path}>'
        if self.index is None:
            return ref
        return f'{ref}{self.index}'


# TODO check dimensions for images, warn if unexpected
class AOTFITSFile:
    def __init__(self, *, exception_level=AOTFITSErrorLevel.SERIOUS, log_level=AOTFITSErrorLevel.WARNING,
                 externals='manual', externals_dictionary=None, externals_directory=None, ignore_version=False,
                 **kwargs):
        self.primary_header = AOTFITSPrimaryHeader(self)
        self.time_table = AOTFITSTableTime(self)
        self.atmospheric_parameters_table = AOTFITSTableAtmosphericParameters(self)
        self.aberrations_table = AOTFITSTableAberrations(self)
        self.telescopes_table = AOTFITSTableTelescopes(self)
        self.sources_table = AOTFITSTableSources(self)
        self.sources_sodium_lgs_table = AOTFITSTableSourcesSodiumLGS(self, self.sources_table)
        self.sources_rayleigh_lgs_table = AOTFITSTableSourcesRayleighLGS(self, self.sources_table)
        self.detectors_table = AOTFITSTableDetectors(self)
        self.scoring_cameras_table = AOTFITSTableScoringCameras(self)
        self.wavefront_sensors_table = AOTFITSTableWavefrontSensors(self)
        self.wavefront_sensors_shack_hartmann_table = AOTFITSTableWavefrontSensorsShackHartmann(
            self, self.wavefront_sensors_table)
        self.wavefront_sensors_pyramid_table = AOTFITSTableWavefrontSensorsPyramid(self, self.wavefront_sensors_table)
        self.wavefront_correctors_table = AOTFITSTableWavefrontCorrectors(self)
        self.wavefront_correctors_dm_table = AOTFITSTableWavefrontCorrectorsDM(self, self.wavefront_correctors_table)
        self.loops_table = AOTFITSTableLoops(self)
        self.loops_control_table = AOTFITSTableLoopsControl(self, self.loops_table)
        self.loops_offload_table = AOTFITSTableLoopsOffload(self, self.loops_table)

        self.tables: dict[str, AOTFITSTable] = {
            self.time_table.extname: self.time_table,
            self.atmospheric_parameters_table.extname: self.atmospheric_parameters_table,
            self.aberrations_table.extname: self.aberrations_table,
            self.telescopes_table.extname: self.telescopes_table,
            self.sources_table.extname: self.sources_table,
            self.sources_sodium_lgs_table.extname: self.sources_sodium_lgs_table,
            self.sources_rayleigh_lgs_table.extname: self.sources_rayleigh_lgs_table,
            self.detectors_table.extname: self.detectors_table,
            self.scoring_cameras_table.extname: self.scoring_cameras_table,
            self.wavefront_sensors_table.extname: self.wavefront_sensors_table,
            self.wavefront_sensors_shack_hartmann_table.extname: self.wavefront_sensors_shack_hartmann_table,
            self.wavefront_sensors_pyramid_table.extname: self.wavefront_sensors_pyramid_table,
            self.wavefront_correctors_table.extname: self.wavefront_correctors_table,
            self.wavefront_correctors_dm_table.extname: self.wavefront_correctors_dm_table,
            self.loops_table.extname: self.loops_table,
            self.loops_control_table.extname: self.loops_control_table,
            self.loops_offload_table.extname: self.loops_offload_table
        }

        self.internal_images: dict[str, AOTFITSInternalImage] = {}
        self.external_images: list[AOTFITSExternalImage] = []
        self.extra_hdus: fits.HDUList = fits.HDUList()
        self.extra_images: fits.HDUList = fits.HDUList()

        self.issue_manager = AOTFITSErrorManager(exception_level=exception_level, log_level=log_level)
        if externals not in ['manual', 'enforce', 'ignore', 'disallow']:
            raise RuntimeError("Unknown value for 'externals' parameter")
        self.externals = externals
        self.externals_dictionary = externals_dictionary if externals_dictionary is not None else {}
        self.externals_directory = externals_directory
        self.ignore_version = ignore_version
        self.kwargs = kwargs

    @classmethod
    def from_file(cls, filename: str | os.PathLike, **kwargs):
        file = cls(**kwargs)
        file.read_from_file(filename)
        return file

    def get_extra_data(self) -> tuple[fits.HDUList, fits.HDUList, dict[str, list[fits.Column]]]:
        """ Return a tuple of extra data that may have been in AOTFITS file."""
        return self.extra_hdus, self.extra_images, {k: v.extra_columns for k, v in self.tables.items() if v}

    def to_hdulist(self, discard_empty_tables=True) -> fits.HDUList:
        primary_hdu = self.primary_header.to_hdu()
        bintable_hdus = [table.to_hdu() for table in self.tables.values()]
        if discard_empty_tables:
            bintable_hdus = [x for x in bintable_hdus if (x.size > 0 or
                                                          not isinstance(self.tables[x.name], AOTFITSSecondaryTable))]
        image_hdus = [image.to_hdu() for image in self.internal_images.values()]

        return fits.HDUList([primary_hdu, *bintable_hdus, *image_hdus])

    def to_file(self, filename: str | os.PathLike, discard_empty_tables=True, **kwargs) -> None:
        self.to_hdulist(discard_empty_tables).writeto(filename, **kwargs)

    @staticmethod
    def create_row_reference(uid: str) -> str:
        return f'{ROW_REFERENCE}<{uid}>'

    def add_error(self, level: AOTFITSErrorLevel, message: str):
        self.issue_manager.add_error(level, message, "AOTFITSFile")

    def verify_all_table_contents(self):
        for table in self.tables.values():
            table.verify_contents()

    def read_from_file(self, filename: str | os.PathLike):
        with fits.open(filename, **self.kwargs) as hdus:
            # TODO critical if can't be opened
            # TODO handle pedantic order
            self.primary_header.read_from_primary(hdus[0])

            # Skip PrimaryHDU already handled above
            for i, hdu in enumerate(hdus[1:], start=1):
                if hdu.name in self.tables:
                    table = self.tables[hdu.name]
                    if table.found:
                        # Table already found once
                        self.extra_hdus.append(hdu)
                        self.add_error(AOTFITSErrorLevel.WARNING,
                                       f"Table '{hdu.name}' repeated in file, ignoring further appearances.")
                        continue
                    table.found = True
                    table.read_from_bintable(hdu)
                else:
                    if hdu.is_image:
                        if not hdu.name:
                            self.extra_images.append(hdu)
                            self.add_error(AOTFITSErrorLevel.WARNING,
                                           f"Image in HDU index {i} has no name and thus cannot be referenced.")
                            continue
                        if hdu.name in self.internal_images:
                            self.extra_images.append(hdu)
                            self.add_error(AOTFITSErrorLevel.WARNING,
                                           f"Multiple images in the file share the name '{hdu.name}'."
                                           f"Only the first instance will be used for referencing.")
                            continue
                        if hdu.name == 'PRIMARY':
                            self.add_error(AOTFITSErrorLevel.PEDANTIC,
                                           "Image name 'PRIMARY' may be confused with the primary HDU.")
                        if hdu.size == 0:
                            self.add_error(AOTFITSErrorLevel.PEDANTIC, f"Image '{hdu.name}' has no data.")
                        self.internal_images[hdu.name] = AOTFITSInternalImage.from_hdu(hdu)
                    else:
                        self.add_error(AOTFITSErrorLevel.WARNING,
                                       f"HDU '{hdu.name}' is not an AOT binary table nor an image.")
                        self.extra_hdus.append(hdu)

        for table in self.tables.values():
            if isinstance(table, AOTFITSSecondaryTable):
                table.verify_from_main()
            elif not table.found:
                self.add_error(AOTFITSErrorLevel.SERIOUS, f"Missing mandatory table '{table.extname}'")

            table.verify_references()

        for image in self.internal_images.values():
            if not image.was_referenced:
                self.extra_images.append(image)
                self.add_error(AOTFITSErrorLevel.WARNING, f"Image '{image.extname}' is never referenced")
        self.internal_images = {k: v for k, v in self.internal_images.items() if v.was_referenced}

        for image in list(self.internal_images.values()) + self.external_images:
            image.time_uid = self.time_table.handle_reference(image.time_ref)

        for table in self.tables.values():
            table.check_was_referenced()

        self.verify_all_table_contents()

    def handle_reference(self, value: str):
        # Handles image references
        if value is None:
            return None

        fullmatch = _image_reference_pattern.fullmatch(value)
        if fullmatch is None:
            self.add_error(AOTFITSErrorLevel.SERIOUS, f"Image reference '{value}' was ignored: not properly formatted")
            return None

        prefix, name, index = fullmatch.groups()
        if index is not None:
            try:
                index = int(index)
            except ValueError:
                self.add_error(AOTFITSErrorLevel.WARNING,
                               f"Index in reference '{value}' was ignored: not properly formatted.")
                index = None
        match prefix:
            case _s.INTERNAL_REFERENCE:
                if index is not None:
                    self.add_error(AOTFITSErrorLevel.PEDANTIC,
                                   f"Internal image reference '{name}' should not have an index (got '{index}').")
                try:
                    im = self.internal_images[name]
                    im.was_referenced = True
                except KeyError:
                    self.add_error(AOTFITSErrorLevel.SERIOUS, f"Image '{name}' not found in file")
                    im = None

            case _s.FILE_REFERENCE:
                fullmatch = _valid_filename.fullmatch(name)
                if fullmatch is None:
                    self.add_error(AOTFITSErrorLevel.WARNING,
                                   f"File name '{name}' contains disallowed characters.")
                im = self.get_external_image(prefix, name, index)
                if im is not None:
                    self.external_images.append(im)
            case _s.URL_REFERENCE:
                im = self.get_external_image(prefix, name, index)
                if im is not None:
                    self.external_images.append(im)

            case _:
                self.add_error(AOTFITSErrorLevel.SERIOUS, f"Reference '{value}' was ignored: unknown reference type")
                return None
        return im

    def get_external_image(self, prefix: str, name: str, index: int) -> AOTFITSExternalImage | None:
        if self.externals == 'disallow':
            self.add_error(AOTFITSErrorLevel.CRITICAL, f"External image found while "
                                                       f"'externals_option' parameter is set to 'disallow'")
            return None
        if self.externals == 'ignore':
            self.add_error(AOTFITSErrorLevel.WARNING, f"External image ignored due to 'ignore' option in"
                                                      f"'externals_option' parameter.")
            return None
        if name in self.externals_dictionary:
            # Replace the name with a hardcoded replacement if it exists
            name = self.externals_dictionary[name]
        elif self.externals_directory is not None and prefix == FILE_REFERENCE:
            # If an externals directory is specified, prepend to file path
            name = os.path.join(self.externals_directory, name)
        try:
            with fits.open(name, **self.kwargs) as hdus:
                hdu = self.get_imagehdu_from_hdus(name, hdus, index)
                if hdu is None:
                    return None
                return AOTFITSExternalImage.from_hdu(hdu, prefix, name, index)
        except FileNotFoundError:
            if self.externals == 'enforce':
                self.add_error(AOTFITSErrorLevel.CRITICAL, f"Could not automatically find {name}"
                                                           f" while 'externals_option' parameter is set to 'enforce'.")

            # If it reaches this point, externals_option must be 'manual'
            from tkinter.filedialog import askopenfilename
            selected = askopenfilename(title=f"Please select '{name}'.",
                                       initialfile=name,
                                       filetypes=(('FITS files', '*.fits'),
                                                  ('Compressed FITS files', '*.fits.gz')))
            if not selected:
                self.add_error(AOTFITSErrorLevel.CRITICAL, f"Could not find '{name}' automatically."
                                                           f" Manual selection is required.")
                return None
            with fits.open(selected, **self.kwargs) as hdus:
                hdu = self.get_imagehdu_from_hdus(selected, hdus, index)
                if hdu is None:
                    return None
                return AOTFITSExternalImage.from_hdu(hdu, FILE_REFERENCE, selected, index)

    def get_imagehdu_from_hdus(self, filename: str | os.PathLike, hdus: fits.HDUList,
                               index: int) -> fits.ImageHDU | None:
        if index is None:
            for hdu in hdus:
                if hdu.is_image and hdu.size > 0:
                    break
            else:
                self.add_error(AOTFITSErrorLevel.SERIOUS, f"Could not find any image data in '{filename}'.")
                return None
        else:
            hdu = hdus[index]
            if not hdu.is_image:
                self.add_error(AOTFITSErrorLevel.SERIOUS,
                               f"HDU index {index} in '{filename}' is not an image.")
                return None
            elif hdu.size == 0:
                self.add_error(AOTFITSErrorLevel.PEDANTIC,
                               f"HDU index {index} in '{filename}' has no image data.")
        return hdu

    def handle_version(self):
        # TODO generalize
        if self.primary_header.version not in compat.known_versions:
            self.primary_header.add_error(AOTFITSErrorLevel.CRITICAL,
                                          f"Unknown version '{self.primary_header.version}'. "
                                          f"Use option 'ignore_version' to force reading.")
            self.primary_header.version = compat.latest_version
        if self.primary_header.version == compat.latest_version:
            return
        if self.primary_header.version == compat.AOTVersion(1, 0, 0):
            # Columns had different names
            self.atmospheric_parameters_table.seeing.name = compat.LEGACY_ATMOSPHERIC_PARAMETERS_FWHM
            self.atmospheric_parameters_table.layers_rel_weight.name = compat.LEGACY_ATMOSPHERIC_PARAMETERS_LAYERS_WEIGHT
            self.sources_table.fwhm.name = compat.LEGACY_SOURCE_WIDTH
            # Columns didn't exist
            self.detectors_table.field_centre_x.ignored = True
            self.detectors_table.field_centre_y.ignored = True
            # Dimensionless was represented differently
            for t in self.tables.values():
                for fld in t.fields:
                    if fld.unit == UNIT_DIMENSIONLESS:
                        fld.unit = compat.LEGACY_DIMENSIONLESS

        else:
            # This should never happen
            raise NotImplementedError


@dataclass
class AOTFITSPrimaryHeader:
    parent: AOTFITSFile
    version: compat.AOTVersion = None
    ao_mode: str = None
    date_beg: datetime = None
    date_end: datetime = None
    system_name: str = None
    strehl_ratio: float = None
    strehl_wavelength: float = None
    config: str = None
    metadata: list[fits.Card] = field(default_factory=list)

    def add_error(self, level: AOTFITSErrorLevel, message: str):
        self.parent.issue_manager.add_error(level, message, "Primary Header")

    def read_from_primary(self, hdu: fits.PrimaryHDU):
        if hdu.size != 0:
            self.add_error(AOTFITSErrorLevel.WARNING, f"Primary HDU should not contain data array.")

        hdr = hdu.header
        if self.parent.ignore_version:
            self.version = compat.latest_version
        else:
            try:
                self.version = compat.AOTVersion.from_string(hdr[AOT_VERSION])
            except ValueError:
                self.add_error(AOTFITSErrorLevel.SERIOUS, f"Keyword '{AOT_VERSION}' not formatted properly.")
                self.version = compat.latest_version
            except KeyError:
                self.add_error(AOTFITSErrorLevel.CRITICAL, f"Keyword '{AOT_VERSION}' not found, file is likely "
                                                           f"not AOT. Use option 'ignore_version' to force reading.")

        self.parent.handle_version()

        try:
            if (timesys := hdr[AOT_TIMESYS]) != AOT_TIMESYS_UTC:
                self.add_error(AOTFITSErrorLevel.WARNING, f"Keyword '{AOT_TIMESYS}' should have the value"
                                                          f" '{AOT_TIMESYS_UTC}'. Got {timesys} instead.")
        except KeyError:
            self.add_error(AOTFITSErrorLevel.SERIOUS, f"Mandatory keyword '{AOT_TIMESYS}' missing.")

        try:
            self.ao_mode = hdr[AOT_AO_MODE]
            if self.ao_mode not in AOT_AO_MODE_SET:
                self.add_error(AOTFITSErrorLevel.SERIOUS, f"Unknown value '{self.ao_mode}' for keyword '{AOT_AO_MODE}'."
                                                          f" Expected one of {AOT_AO_MODE_SET}.")
                self.ao_mode = 'SCAO'
        except KeyError:
            self.add_error(AOTFITSErrorLevel.SERIOUS, f"Mandatory keyword '{AOT_AO_MODE}' missing.")
            self.ao_mode = 'SCAO'

        if AOT_DATE_BEG in hdr and (date_str := hdr[AOT_DATE_BEG]):
            try:
                self.date_beg = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                self.add_error(AOTFITSErrorLevel.WARNING, f"Keyword '{AOT_DATE_BEG}' improperly formatted.")

        if AOT_DATE_END in hdr and (date_str := hdr[AOT_DATE_END]):
            try:
                self.date_end = datetime.fromisoformat(date_str)
            except (ValueError, TypeError):
                self.add_error(AOTFITSErrorLevel.WARNING, f"Keyword '{AOT_DATE_END}' improperly formatted.")

        if AOT_SYSTEM_NAME in hdr:
            self.system_name = hdr[AOT_SYSTEM_NAME]
            if not isinstance(self.system_name, str):
                self.add_error(AOTFITSErrorLevel.WARNING, f"Keyword '{AOT_SYSTEM_NAME}' should have a string value.")

        if AOT_STREHL_RATIO in hdr:
            self.strehl_ratio = hdr[AOT_STREHL_RATIO]
            if not isinstance(self.strehl_ratio, float):
                self.add_error(AOTFITSErrorLevel.WARNING,
                               f"Keyword '{AOT_STREHL_RATIO}' should have a floating-point value.")

        if AOT_STREHL_WAVELENGTH in hdr:
            self.strehl_wavelength = hdr[AOT_STREHL_WAVELENGTH]
            if not isinstance(self.strehl_wavelength, float):
                self.add_error(AOTFITSErrorLevel.WARNING,
                               f"Keyword '{AOT_STREHL_WAVELENGTH}' should have a floating-point value.")

        if AOT_CONFIG in hdr:
            self.config = hdr[AOT_CONFIG]
            if not isinstance(self.config, str):
                self.add_error(AOTFITSErrorLevel.WARNING, f"Keyword '{AOT_CONFIG}' should have a string value.")

        self.metadata = [card for card in hdr.cards
                         if card.keyword not in AOT_HEADER_SET and _keyword_is_relevant(card.keyword)]
        if self.metadata:
            self.add_error(AOTFITSErrorLevel.PEDANTIC, f"Found non-AOT keywords: {[x.keyword for x in self.metadata]}.")

    def to_hdu(self) -> fits.PrimaryHDU:
        # TODO check for types
        hdr = fits.Header()
        hdr[AOT_VERSION] = str(compat.latest_version)
        hdr[AOT_TIMESYS] = AOT_TIMESYS_UTC

        hdr[AOT_AO_MODE] = self.ao_mode
        if (date := _datetime_to_iso(self.date_beg)) is not None:
            hdr[AOT_DATE_BEG] = date
        if (date := _datetime_to_iso(self.date_end)) is not None:
            hdr[AOT_DATE_END] = date

        if self.system_name is not None:
            hdr[AOT_SYSTEM_NAME] = self.system_name
        if self.strehl_ratio is not None:
            hdr[AOT_STREHL_RATIO] = self.strehl_ratio
        if self.strehl_wavelength is not None:
            hdr[AOT_STREHL_RATIO] = self.strehl_wavelength
        if self.config is not None:
            hdr[AOT_CONFIG] = self.config

        hdr.extend(self.metadata)
        return fits.PrimaryHDU(header=hdr)


@dataclass
class AOTFITSTable:
    extname: str
    fields: AOTFieldList[AOTField]
    parent: AOTFITSFile
    uid: AOTField = field(init=False)
    uid_dict: dict[str, int] = field(init=False, default_factory=dict)
    uid_was_referenced: dict[str, bool] = field(init=False, default_factory=dict)

    found: bool = field(init=False, default=False)
    extra_columns: list[fits.Column] = field(init=False, default_factory=list)

    def add_error(self, level: AOTFITSErrorLevel, message: str):
        self.parent.issue_manager.add_error(level, message, self.extname)

    def read_from_bintable(self, hdu: fits.BinTableHDU):
        cols: fits.ColDefs = hdu.columns
        data: fits.FITS_rec = hdu.data
        n_rows = data.size

        found_columns = {}
        for col in cols:
            if col.name in self.fields:
                # Standard column
                if col.name in found_columns:
                    # TODO AstroPy should have been able to detect this
                    self.add_error(AOTFITSErrorLevel.WARNING,
                                   f"Column '{col.name}' repeated in table, ignoring further appearances.")
                    continue
                else:
                    found_columns[col.name] = col
                fld = self.fields[col.name]
                datacolumn = data[fld.name]

                fld.data = [_convert_null_to_none(x, col) for x in datacolumn]

                if fld.mandatory and None in fld.data:
                    # This won't trigger on null lists, but there are no mandatory lists
                    self.add_error(AOTFITSErrorLevel.SERIOUS,
                                   f"Mandatory column '{col.name}' contains null entries.")

                if _fits_type_to_aot(col.format) != fld.format:
                    # Column doesn't match the expected format
                    if fld.mandatory:
                        # Very dangerous!
                        self.add_error(AOTFITSErrorLevel.SERIOUS,
                                       f"Mandatory column '{col.name}' does not match the expected format.")
                    elif all(v is None for v in fld.data):
                        # It has no data, so this isn't really dangerous
                        self.add_error(AOTFITSErrorLevel.PEDANTIC,
                                       f"Empty column '{col.name}' does not match the expected format.")
                    else:
                        self.add_error(AOTFITSErrorLevel.WARNING,
                                       f"Column '{col.name}' does not match the expected format.")

                if fld.allowed_list is not None:
                    for i, x in enumerate(fld.data):
                        if x is not None and x not in fld.allowed_list:
                            if fld.mandatory:
                                self.add_error(AOTFITSErrorLevel.SERIOUS,
                                               f"Mandatory column '{col.name}' contains unrecognized entry {x}. "
                                               f"Expected one of {fld.allowed_list}.")
                            else:
                                self.add_error(AOTFITSErrorLevel.WARNING,
                                               f"Column '{col.name}' contains unrecognized entry {x}. "
                                               f"Expected one of {fld.allowed_list}.")
                            fld.data[i] = fld.allowed_list[0]

                if fld.unit:
                    # Not dimensionless
                    if col.unit != fld.unit:
                        self.add_error(AOTFITSErrorLevel.WARNING,
                                       f"Column '{col.name}' does not match the expected unit.")
                else:
                    # Dimensionless
                    if col.unit is not None:
                        self.add_error(AOTFITSErrorLevel.PEDANTIC,
                                       f"Dimensionless column '{col.name}' specifies a unit.")

                if fld.unique:
                    # These are the UID fields
                    if len(np.unique(datacolumn)) != len(datacolumn):
                        self.add_error(AOTFITSErrorLevel.SERIOUS,
                                       f"Unique column '{col.name}' contains repeated entries.")
            else:
                self.extra_columns.append(col)
                self.add_error(AOTFITSErrorLevel.WARNING,
                               f"Table contains non-AOT column '{col.name}'.")

        if list(found_columns.keys()) != [fld.name for fld in self.fields if fld.name in found_columns]:
            self.add_error(AOTFITSErrorLevel.PEDANTIC,
                           f"Columns not in recommended order.")
        # Verify if all fields were found
        if len(found_columns) != len(self.fields):
            # Find which fields are missing
            for fld in self.fields:
                if fld.name not in found_columns:
                    # Field might be ignored due to version compatibility reasons
                    if not fld.ignored:
                        if fld.mandatory:
                            self.add_error(AOTFITSErrorLevel.SERIOUS,
                                           f"Mandatory column '{fld.name}' is missing.")
                        else:
                            self.add_error(AOTFITSErrorLevel.WARNING,
                                           f"Column '{fld.name}' is missing.")

                    # Column not found, so we create null data in its place
                    if fld.format == LIST_FORMAT:
                        fld.data = [[] for _ in range(n_rows)]
                    else:
                        fld.data = [None for _ in range(n_rows)]

        for i, value in enumerate(self.uid.data):
            self.uid_dict[value] = i
            self.uid_was_referenced[value] = False

    def verify_references(self):
        for fld in self.fields:
            if fld.reference is None:
                # Not a reference field
                continue

            match fld.reference:
                case _s.IMAGE_REF:
                    referenced_table = self.parent
                case _s.TIME_TABLE:
                    referenced_table = self.parent.time_table
                case _s.ABERRATIONS_TABLE:
                    referenced_table = self.parent.aberrations_table
                case _s.TELESCOPES_TABLE:
                    referenced_table = self.parent.telescopes_table
                case _s.DETECTORS_TABLE:
                    referenced_table = self.parent.detectors_table
                case _s.SOURCES_TABLE:
                    referenced_table = self.parent.sources_table
                case _s.WAVEFRONT_CORRECTORS_TABLE:
                    referenced_table = self.parent.wavefront_correctors_table
                case _s.WAVEFRONT_SENSORS_TABLE:
                    referenced_table = self.parent.wavefront_sensors_table
                case _:
                    # This should never happen
                    raise RuntimeError

            fld.data = [referenced_table.handle_reference(x) for x in fld.data]
            if fld.mandatory and None in fld.data:
                # This won't trigger on null lists, but there are no mandatory lists
                self.add_error(AOTFITSErrorLevel.SERIOUS,
                               f"Mandatory column '{fld.name}' contains references that could not be resolved.")

    def verify_contents(self):
        pass

    def _ensure_has_data(self):
        if not self.uid.data:
            self.add_error(AOTFITSErrorLevel.WARNING, f"'{self.extname}' contains no data (at least one row expected).")
    def _ensure_same_length(self, a: AOTField, b: AOTField, independent=False):
        for i, (x, y) in enumerate(zip(a.data, b.data)):
            if not x and not y:
                # Both are empty lists
                if independent:
                    # This is just the case for AOT_TIME, one must be not-null
                    self.add_error(AOTFITSErrorLevel.SERIOUS, f"In table '{self.extname}', for UID '{self.uid[i]}', "
                                                              f"both entries for '{a.name}' and '{b.name}' are null.")
                continue
            if (x and not y) or (y and not x):
                # One is emtpy and the other isn't
                if not independent:
                    self.add_error(AOTFITSErrorLevel.SERIOUS, f"In table '{self.extname}', for UID '{self.uid[i]}', one"
                                                              f" of '{a.name}' and '{b.name}' is null,"
                                                              f" while the other is not.")
                continue
            if len(x) != len(y):
                self.add_error(AOTFITSErrorLevel.SERIOUS, f"In table '{self.extname}', for UID '{self.uid[i]}', the "
                                                          f"entries for '{a.name}' and '{b.name}' do not have the same"
                                                          f" length.")

    def handle_reference(self, ref):
        # Handles row references
        # TODO if reference is missing we should create a fake placeholder
        if ref is None:
            return None
        fullmatch = _row_reference_pattern.fullmatch(ref)
        if fullmatch is None:
            self.add_error(AOTFITSErrorLevel.SERIOUS, f"Row reference '{ref}' was ignored: not properly formatted")
            return None
        uid = fullmatch.group(1)
        if uid not in self.uid_dict:
            self.add_error(AOTFITSErrorLevel.SERIOUS, f"Referenced UID '{uid}' not found in table {self.extname}")
            return None

        self.uid_was_referenced[uid] = True
        return uid

    def check_was_referenced(self):
        # We consider that by default rows in tables don't need to be referenced. The tables that need to have their
        # rows refereced use the 'rows_must_be_referenced' decorator.
        return

    def to_hdu(self):
        cols = []
        for fld in self.fields:
            if fld.reference is not None:
                if fld.reference == _s.IMAGE_REF:
                    # AOTFITSImage subclasses have a __str__ that creates the reference string automatically
                    data = ['' if x is None else str(x) for x in fld.data]
                else:
                    data = ['' if x is None else AOTFITSFile.create_row_reference(x) for x in fld.data]
            else:
                data = [self.convert_none_to_null(x, fld) for x in fld.data]
            if fld.format == LIST_FORMAT:
                array = np.empty(len(data), dtype=np.object_)

                # Convert every entry to numpy array, if any 64-bit floats are detected we need to use 'D' format.
                flag = False
                for i, l in enumerate(data):
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
                col = fits.Column(name=fld.canonical_name,
                                  format=f"Q{'D' if flag else 'E'}", unit=fld.unit, array=array)
            else:
                # Convert to numpy array and try as much as possible to keep the resulting dtype
                array = np.array(data)
                if fld.format == STRING_FORMAT:
                    col = fits.Column(name=fld.canonical_name,
                                      format=f'{np.char._get_num_chars(array)}A', unit=fld.unit, array=array)
                elif fld.format == _s.INTEGER_FORMAT:
                    if (t := array.dtype) == np.int16 or t == np.int8:
                        f = '1I'
                    elif t == np.int32:
                        f = '1J'
                    else:
                        if t != np.int64 and array.size > 0:
                            # If not int16, int32 or int64 make one last-ditch effort to convert to int64 by default
                            array = array.astype(np.int64, casting='safe')
                        f = '1K'

                    col = fits.Column(name=fld.canonical_name, format=f, null=_int_min, unit=fld.unit, array=array)
                elif fld.format == _s.FLOAT_FORMAT:
                    if (t := array.dtype) == np.float32:
                        f = '1E'
                    else:
                        if t != np.float64 and array.size > 0:
                            # If not float32 or float64 make one last-ditch effort to convert to float64 by default
                            array = array.astype(np.float64, casting='safe')
                        f = '1D'
                    col = fits.Column(name=fld.canonical_name, format=f, unit=fld.unit, array=array)
                else:
                    # This should never happen
                    raise RuntimeError
            cols.append(col)
        return fits.BinTableHDU.from_columns(name=self.extname, columns=cols)

    def convert_none_to_null(self, value, fld: AOTField):
        if fld.mandatory and value is None:
            raise ValueError(f"Got 'None' value for mandatory field '{fld.name} on table '{self.extname}")
        match fld.format:
            case _s.STRING_FORMAT:
                if value is None:
                    return ''
                if not isinstance(value, str) or (isinstance(value, np.ndarray)
                                                  and not np.issubdtype(value.dtype, np.character)):
                    raise ValueError(f"Unxpected value in table '{self.extname}' column '{fld.name}'. "
                                     f"Expected '{fld.format}' format, got: {value}.")
            case _s.FLOAT_FORMAT:
                if value is None:
                    return _nan
                if not isinstance(value, numbers.Real):
                    raise ValueError(f"Unxpected value in table '{self.extname}' column '{fld.name}'. "
                                     f"Expected '{fld.format}' format, got: {value}.")
            case _s.INTEGER_FORMAT:
                if value is None:
                    return _int_min
                if not isinstance(value, numbers.Integral):
                    raise ValueError(f"Unxpected value in table '{self.extname}' column '{fld.name}'. "
                                     f"Expected '{fld.format}' format, got: {value}.")
            case _s.LIST_FORMAT:
                if value is None:
                    return []
                try:
                    value = [_nan if v is None else v for v in value]
                except TypeError:
                    # Not iterable
                    raise ValueError(f"Unxpected value in table '{self.extname}' column '{fld.name}'. "
                                     f"Expected '{fld.format}' format, got: {value}.") from None
            case _:
                # This will trigger a warning later
                pass
        return value


@dataclass
class AOTFITSSecondaryTable(AOTFITSTable):
    main: AOTFITSTable
    type_name: str

    def verify_references(self):
        for uid in self.uid.data:
            if uid not in self.main.uid_dict:
                self.add_error(AOTFITSErrorLevel.WARNING,
                               f"UID '{uid}' in {self.extname} does not exist in {self.main.extname}.")

        super().verify_references()

    def verify_from_main(self):
        # If the main table has a row with the correct type, we need to have that UID in the secondary table
        expected_uids = [uid for uid, index in self.main.uid_dict.items() if
                         self.main.type.data[index] == self.type_name]

        if expected_uids and not self.found:
            self.add_error(AOTFITSErrorLevel.SERIOUS, f"Table {self.extname} does not exist even though there"
                                                      f" are rows of type {self.type_name} in {self.main.extname}.")
        for uid in expected_uids:
            if uid not in self.uid_dict:
                if self.found:
                    # Only report here if table is found, otherwise we'll be double reporting
                    self.add_error(AOTFITSErrorLevel.SERIOUS, f"UID '{uid}' in {self.main.extname} has type "
                                                              f"{self.type_name} but does not exist in table "
                                                              f"{self.extname}.")
                self.uid_dict[uid] = len(self.uid.data)
                self.uid.data.append(uid)
                for fld in self.fields:
                    if fld.name != REFERENCE_UID:
                        if fld.format == LIST_FORMAT:
                            fld.data.append([])
                        else:
                            fld.data.append(None)


@rows_must_be_referenced
class AOTFITSTableTime(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.timestamps = AOTField(TIME_TIMESTAMPS, LIST_FORMAT, UNIT_SECONDS)
        self.frame_numbers = AOTField(TIME_FRAME_NUMBERS, LIST_FORMAT, UNIT_COUNT)

        fields = AOTFieldList(self.uid, self.timestamps, self.frame_numbers)

        super().__init__(TIME_TABLE, fields, parent)

    def verify_contents(self):
        self._ensure_same_length(self.timestamps, self.frame_numbers, True)


class AOTFITSTableAtmosphericParameters(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.wavelength = AOTField(ATMOSPHERIC_PARAMETERS_WAVELENGTH, FLOAT_FORMAT, UNIT_METERS)
        self.time_uid = AOTField(TIME_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=TIME_TABLE)
        self.r0 = AOTField(ATMOSPHERIC_PARAMETERS_R0, LIST_FORMAT, UNIT_METERS)
        self.seeing = AOTField(ATMOSPHERIC_PARAMETERS_SEEING, LIST_FORMAT, UNIT_ARCSEC)
        self.tau0 = AOTField(ATMOSPHERIC_PARAMETERS_TAU0, LIST_FORMAT, UNIT_SECONDS)
        self.theta0 = AOTField(ATMOSPHERIC_PARAMETERS_THETA0, LIST_FORMAT, UNIT_RADIANS)
        self.layers_rel_weight = AOTField(ATMOSPHERIC_PARAMETERS_LAYERS_REL_WEIGHT, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                          reference=IMAGE_REF)
        self.layers_height = AOTField(ATMOSPHERIC_PARAMETERS_LAYERS_HEIGHT, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                      reference=IMAGE_REF)
        self.layers_l0 = AOTField(ATMOSPHERIC_PARAMETERS_LAYERS_L0, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                  reference=IMAGE_REF)
        self.layers_wind_speed = AOTField(ATMOSPHERIC_PARAMETERS_LAYERS_WIND_SPEED, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                          reference=IMAGE_REF)
        self.layers_wind_direction = AOTField(ATMOSPHERIC_PARAMETERS_LAYERS_WIND_DIRECTION, STRING_FORMAT,
                                              UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.transformation_matrix = AOTField(TRANSFORMATION_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)

        fields = AOTFieldList(self.uid, self.wavelength, self.time_uid, self.r0, self.seeing, self.tau0,
                              self.theta0, self.layers_rel_weight, self.layers_height, self.layers_l0,
                              self.layers_wind_speed, self.layers_wind_direction, self.transformation_matrix)

        super().__init__(ATMOSPHERIC_PARAMETERS_TABLE, fields, parent)


@rows_must_be_referenced
class AOTFITSTableAberrations(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.modes = AOTField(ABERRATION_MODES, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, reference=IMAGE_REF)
        self.coefficients = AOTField(ABERRATION_COEFFICIENTS, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True,
                                     reference=IMAGE_REF)
        self.x_offsets = AOTField(ABERRATION_X_OFFSETS, LIST_FORMAT, UNIT_RADIANS)
        self.y_offsets = AOTField(ABERRATION_Y_OFFSETS, LIST_FORMAT, UNIT_RADIANS)

        fields = AOTFieldList(self.uid, self.modes, self.coefficients, self.x_offsets, self.y_offsets)

        super().__init__(ABERRATIONS_TABLE, fields, parent)

    def verify_contents(self):
        self._ensure_same_length(self.x_offsets, self.y_offsets)


@rows_must_be_referenced
class AOTFITSTableTelescopes(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.type = AOTField(TELESCOPE_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                             mandatory=True, allowed_list=TELESCOPE_TYPE_LIST)
        self.latitude = AOTField(TELESCOPE_LATITUDE, FLOAT_FORMAT, UNIT_DEGREES)
        self.longitude = AOTField(TELESCOPE_LONGITUDE, FLOAT_FORMAT, UNIT_DEGREES)
        self.elevation = AOTField(TELESCOPE_ELEVATION, FLOAT_FORMAT, UNIT_DEGREES)
        self.azimuth = AOTField(TELESCOPE_AZIMUTH, FLOAT_FORMAT, UNIT_DEGREES)
        self.parallactic = AOTField(TELESCOPE_PARALLACTIC, FLOAT_FORMAT, UNIT_DEGREES)
        self.pupil_mask = AOTField(TELESCOPE_PUPIL_MASK, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.pupil_angle = AOTField(TELESCOPE_PUPIL_ANGLE, FLOAT_FORMAT, UNIT_RADIANS)
        self.enclosing_d = AOTField(TELESCOPE_ENCLOSING_D, FLOAT_FORMAT, UNIT_METERS)
        self.inscribed_d = AOTField(TELESCOPE_INSCRIBED_D, FLOAT_FORMAT, UNIT_METERS)
        self.obstruction_d = AOTField(TELESCOPE_OBSTRUCTION_D, FLOAT_FORMAT, UNIT_METERS)
        self.segment_type = AOTField(TELESCOPE_SEGMENTS_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                     mandatory=True, allowed_list=TELESCOPE_SEGMENT_LIST)
        self.segment_size = AOTField(TELESCOPE_SEGMENTS_SIZE, FLOAT_FORMAT, UNIT_METERS)
        self.segments_x = AOTField(TELESCOPE_SEGMENTS_X, LIST_FORMAT, UNIT_METERS)
        self.segments_y = AOTField(TELESCOPE_SEGMENTS_Y, LIST_FORMAT, UNIT_METERS)
        self.transformation_matrix = AOTField(TRANSFORMATION_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)
        self.aberration_uid = AOTField(ABERRATION_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                       reference=ABERRATIONS_TABLE)

        fields = AOTFieldList(self.uid, self.type, self.latitude, self.longitude, self.elevation, self.azimuth,
                              self.parallactic, self.pupil_mask, self.pupil_angle, self.enclosing_d, self.inscribed_d,
                              self.obstruction_d, self.segment_type, self.segment_size, self.segments_x,
                              self.segments_y, self.transformation_matrix, self.aberration_uid)

        self.main_telescope_uid = None

        super().__init__(TELESCOPES_TABLE, fields, parent)

    def read_from_bintable(self, hdu: fits.BinTableHDU):
        super().read_from_bintable(hdu)

        main_telescopes_list = [uid for uid, t in zip(self.uid.data, self.type.data) if t == _s.TELESCOPE_TYPE_MAIN]
        if (n := len(main_telescopes_list)) == 0:
            self.add_error(AOTFITSErrorLevel.SERIOUS, f"Table {self.extname} does not contain a main telescope.")
            # If we get to this point, try to create a placeholder telescope just for functionality
            rows = len(self.uid.data)
            # Filling the mandatory fields
            self.main_telescope_uid = 'FAKE PLACEHOLDER MAIN TELESCOPE'
            self.uid.data.append(self.main_telescope_uid)
            self.type.data.append(_s.TELESCOPE_TYPE_MAIN)
            self.segment_type.data.append(_s.TELESCOPE_SEGMENT_TYPE_MONOLITHIC)
            for fld in self.fields:
                if not fld.mandatory:
                    # Mandatory fields are handled above
                    if fld.format == _s.LIST_FORMAT:
                        fld.data.append([])
                    else:
                        fld.data.append(None)
            self.uid_dict[self.main_telescope_uid] = rows
            self.uid_was_referenced[self.main_telescope_uid] = True
        else:
            if n > 1:
                self.add_error(AOTFITSErrorLevel.SERIOUS,
                               f"Table {self.extname} contains more than one main telescope.")
            self.main_telescope_uid = main_telescopes_list[0]
            self.uid_was_referenced[self.main_telescope_uid] = True

    def verify_contents(self):
        # TODO if Monolithic we should reject segment coordinates
        self._ensure_same_length(self.segments_x, self.segments_y)


class AOTFITSTableSources(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.type = AOTField(SOURCE_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                             mandatory=True, allowed_list=SOURCE_TYPE_LIST)
        self.right_ascension = AOTField(SOURCE_RIGHT_ASCENSION, FLOAT_FORMAT, UNIT_DEGREES)
        self.declination = AOTField(SOURCE_DECLINATION, FLOAT_FORMAT, UNIT_DEGREES)
        self.elevation_offset = AOTField(SOURCE_ELEVATION_OFFSET, FLOAT_FORMAT, UNIT_DEGREES)
        self.azimuth_offset = AOTField(SOURCE_AZIMUTH_OFFSET, FLOAT_FORMAT, UNIT_DEGREES)
        self.fwhm = AOTField(SOURCE_FWHM, FLOAT_FORMAT, UNIT_RADIANS)

        fields = AOTFieldList(self.uid, self.type, self.right_ascension, self.declination, self.elevation_offset,
                              self.azimuth_offset, self.fwhm)

        super().__init__(SOURCES_TABLE, fields, parent)

    def verify_contents(self):
        self._ensure_has_data()


class AOTFITSTableSourcesSodiumLGS(AOTFITSSecondaryTable):
    # TODO verify that llt_uid is not a MainTelescope
    def __init__(self, parent: AOTFITSFile, main: AOTFITSTableSources):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.height = AOTField(SOURCE_SODIUM_LGS_HEIGHT, FLOAT_FORMAT, UNIT_METERS)
        self.profile = AOTField(SOURCE_SODIUM_LGS_PROFILE, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.altitudes = AOTField(SOURCE_SODIUM_LGS_ALTITUDES, LIST_FORMAT, UNIT_METERS)
        self.llt_uid = AOTField(LASER_LAUNCH_TELESCOPE_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                reference=TELESCOPES_TABLE)

        fields = AOTFieldList(self.uid, self.height, self.profile, self.altitudes, self.llt_uid)

        super().__init__(SOURCES_SODIUM_LGS_TABLE, fields, parent, main, _s.SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR)


class AOTFITSTableSourcesRayleighLGS(AOTFITSSecondaryTable):
    # TODO verify that llt_uid is not a MainTelescope
    def __init__(self, parent: AOTFITSFile, main: AOTFITSTableSources):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.distance = AOTField(SOURCE_RAYLEIGH_LGS_DISTANCE, FLOAT_FORMAT, UNIT_METERS)
        self.depth = AOTField(SOURCE_RAYLEIGH_LGS_DEPTH, FLOAT_FORMAT, UNIT_METERS)
        self.llt_uid = AOTField(LASER_LAUNCH_TELESCOPE_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                reference=TELESCOPES_TABLE)

        fields = AOTFieldList(self.uid, self.distance, self.depth, self.llt_uid)

        super().__init__(SOURCES_RAYLEIGH_LGS_TABLE, fields, parent, main, _s.SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR)


@rows_must_be_referenced
class AOTFITSTableDetectors(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.type = AOTField(DETECTOR_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS)
        self.sampling_technique = AOTField(DETECTOR_SAMPLING_TECHNIQUE, STRING_FORMAT, UNIT_DIMENSIONLESS)
        self.shutter_type = AOTField(DETECTOR_SHUTTER_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS)
        self.flat_field = AOTField(DETECTOR_FLAT_FIELD, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.readout_noise = AOTField(DETECTOR_READOUT_NOISE, FLOAT_FORMAT,
                                      f'{UNIT_ELECTRONS}*{UNIT_SECONDS}^-1*{UNIT_PIXELS}^-1')
        self.pixel_intensities = AOTField(DETECTOR_PIXEL_INTENSITIES, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                          reference=IMAGE_REF)
        self.field_centre_x = AOTField(DETECTOR_FIELD_CENTRE_X, FLOAT_FORMAT, UNIT_PIXELS)
        self.field_centre_y = AOTField(DETECTOR_FIELD_CENTRE_Y, FLOAT_FORMAT, UNIT_PIXELS)
        self.integration_time = AOTField(DETECTOR_INTEGRATION_TIME, FLOAT_FORMAT, UNIT_SECONDS)
        self.coadds = AOTField(DETECTOR_COADDS, INTEGER_FORMAT, UNIT_COUNT)
        self.dark = AOTField(DETECTOR_DARK, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.weight_map = AOTField(DETECTOR_WEIGHT_MAP, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.quantum_efficiency = AOTField(DETECTOR_QUANTUM_EFFICIENCY, FLOAT_FORMAT, UNIT_DIMENSIONLESS)
        self.pixel_scale = AOTField(DETECTOR_PIXEL_SCALE, FLOAT_FORMAT, f'{UNIT_RADIANS}*{UNIT_PIXELS}^-1')
        self.binning = AOTField(DETECTOR_BINNING, INTEGER_FORMAT, UNIT_COUNT)
        self.bandwidth = AOTField(DETECTOR_BANDWIDTH, FLOAT_FORMAT, UNIT_METERS)
        self.transmission_wavelength = AOTField(DETECTOR_TRANSMISSION_WAVELENGTH, LIST_FORMAT, UNIT_METERS)
        self.transmission = AOTField(DETECTOR_TRANSMISSION, LIST_FORMAT, UNIT_DIMENSIONLESS)
        self.sky_background = AOTField(DETECTOR_SKY_BACKGROUND, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.gain = AOTField(DETECTOR_GAIN, FLOAT_FORMAT, UNIT_ELECTRONS)
        self.excess_noise = AOTField(DETECTOR_EXCESS_NOISE, FLOAT_FORMAT, UNIT_ELECTRONS)
        self.filter = AOTField(DETECTOR_FILTER, STRING_FORMAT, UNIT_DIMENSIONLESS)
        self.bad_pixel_map = AOTField(DETECTOR_BAD_PIXEL_MAP, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.dynamic_range = AOTField(DETECTOR_DYNAMIC_RANGE, FLOAT_FORMAT, UNIT_DECIBELS)
        self.readout_rate = AOTField(DETECTOR_READOUT_RATE, FLOAT_FORMAT, f'{UNIT_PIXELS}*{UNIT_SECONDS}^-1')
        self.frame_rate = AOTField(DETECTOR_FRAME_RATE, FLOAT_FORMAT, f'{UNIT_FRAME}*{UNIT_SECONDS}^-1')
        self.transformation_matrix = AOTField(TRANSFORMATION_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)

        fields = AOTFieldList(self.uid, self.type, self.sampling_technique, self.shutter_type, self.flat_field,
                              self.readout_noise, self.pixel_intensities, self.field_centre_x, self.field_centre_y,
                              self.integration_time, self.coadds, self.dark, self.weight_map, self.quantum_efficiency,
                              self.pixel_scale, self.binning, self.bandwidth, self.transmission_wavelength,
                              self.transmission, self.sky_background, self.gain, self.excess_noise, self.filter,
                              self.bad_pixel_map, self.dynamic_range, self.readout_rate, self.frame_rate,
                              self.transformation_matrix)

        super().__init__(DETECTORS_TABLE, fields, parent)

    def verify_contents(self):
        self._ensure_same_length(self.transmission_wavelength, self.transmission)


class AOTFITSTableScoringCameras(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.pupil_mask = AOTField(SCORING_CAMERA_PUPIL_MASK, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.wavelength = AOTField(SCORING_CAMERA_WAVELENGTH, FLOAT_FORMAT, UNIT_METERS)
        self.transformation_matrix = AOTField(TRANSFORMATION_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)
        self.detector_uid = AOTField(DETECTOR_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=DETECTORS_TABLE)
        self.aberration_uid = AOTField(ABERRATION_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                       reference=ABERRATIONS_TABLE)

        fields = AOTFieldList(self.uid, self.pupil_mask, self.wavelength, self.transformation_matrix,
                              self.detector_uid, self.aberration_uid)

        super().__init__(SCORING_CAMERAS_TABLE, fields, parent)


class AOTFITSTableWavefrontSensors(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.type = AOTField(WAVEFRONT_SENSOR_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                             mandatory=True, allowed_list=WAVEFRONT_SENSOR_TYPE_LIST)
        self.source_uid = AOTField(SOURCE_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                   mandatory=True, reference=SOURCES_TABLE)
        self.dimensions = AOTField(WAVEFRONT_SENSOR_DIMENSIONS, INTEGER_FORMAT, UNIT_COUNT, mandatory=True)
        self.n_valid_subapertures = AOTField(WAVEFRONT_SENSOR_N_VALID_SUBAPERTURES, INTEGER_FORMAT, UNIT_COUNT,
                                             mandatory=True)
        self.measurements = AOTField(WAVEFRONT_SENSOR_MEASUREMENTS, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                     reference=IMAGE_REF)
        self.ref_measurements = AOTField(WAVEFRONT_SENSOR_REF_MEASUREMENTS, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                         reference=IMAGE_REF)
        self.subaperture_mask = AOTField(WAVEFRONT_SENSOR_SUBAPERTURE_MASK, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                         reference=IMAGE_REF)
        self.mask_x_offsets = AOTField(WAVEFRONT_SENSOR_MASK_X_OFFSETS, LIST_FORMAT, UNIT_PIXELS)
        self.mask_y_offsets = AOTField(WAVEFRONT_SENSOR_MASK_Y_OFFSETS, LIST_FORMAT, UNIT_PIXELS)
        self.subaperture_size = AOTField(WAVEFRONT_SENSOR_SUBAPERTURE_SIZE, FLOAT_FORMAT, UNIT_PIXELS)
        self.subaperture_intensities = AOTField(WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES, STRING_FORMAT,
                                                UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.wavelength = AOTField(WAVEFRONT_SENSOR_WAVELENGTH, FLOAT_FORMAT, UNIT_METERS)
        self.optical_gain = AOTField(WAVEFRONT_SENSOR_OPTICAL_GAIN, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                     reference=IMAGE_REF)
        self.transformation_matrix = AOTField(TRANSFORMATION_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)
        self.detector_uid = AOTField(DETECTOR_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=DETECTORS_TABLE)
        self.aberration_uid = AOTField(ABERRATION_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                       reference=ABERRATIONS_TABLE)
        self.ncpa_uid = AOTField(NCPA_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=ABERRATIONS_TABLE)

        fields = AOTFieldList(self.uid, self.type, self.source_uid, self.dimensions, self.n_valid_subapertures,
                              self.measurements, self.ref_measurements, self.subaperture_mask, self.mask_x_offsets,
                              self.mask_y_offsets, self.subaperture_size, self.subaperture_intensities, self.wavelength,
                              self.optical_gain, self.transformation_matrix, self.detector_uid, self.aberration_uid,
                              self.ncpa_uid)

        super().__init__(WAVEFRONT_SENSORS_TABLE, fields, parent)

    def verify_contents(self):
        # TODO "source" should not be a science star
        self._ensure_has_data()
        self._ensure_same_length(self.mask_x_offsets, self.mask_y_offsets)



class AOTFITSTableWavefrontSensorsShackHartmann(AOTFITSSecondaryTable):
    # TODO
    # if dimensions != 2:
    #   warnings.warn(f"Unexpected value for '{kw.WAVEFRONT_SENSOR_DIMENSIONS}' in wavefront sensor '{uid}'"
    #                 f" of type '{t}'. Expected 2, got {dimensions}.")
    def __init__(self, parent: AOTFITSFile, main: AOTFITSTableWavefrontSensors):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.centroiding_algorithm = AOTField(WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROIDING_ALGORITHM, STRING_FORMAT,
                                              UNIT_DIMENSIONLESS)
        self.centroid_gains = AOTField(WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROID_GAINS, STRING_FORMAT,
                                       UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.spot_fwhm = AOTField(WAVEFRONT_SENSOR_SHACK_HARTMANN_SPOT_FWHM, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                  reference=IMAGE_REF)

        fields = AOTFieldList(self.uid, self.centroiding_algorithm, self.centroid_gains, self.spot_fwhm)

        super().__init__(WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE, fields, parent, main,
                         _s.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN)


class AOTFITSTableWavefrontSensorsPyramid(AOTFITSSecondaryTable):
    def __init__(self, parent: AOTFITSFile, main: AOTFITSTableWavefrontSensors):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.n_sides = AOTField(WAVEFRONT_SENSOR_PYRAMID_N_SIDES, INTEGER_FORMAT, UNIT_COUNT, mandatory=True)
        self.modulation = AOTField(WAVEFRONT_SENSOR_PYRAMID_MODULATION, FLOAT_FORMAT, UNIT_METERS)

        fields = AOTFieldList(self.uid, self.n_sides, self.modulation)
        super().__init__(WAVEFRONT_SENSORS_PYRAMID_TABLE, fields, parent, main, _s.WAVEFRONT_SENSOR_TYPE_PYRAMID)


class AOTFITSTableWavefrontCorrectors(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.type = AOTField(WAVEFRONT_CORRECTOR_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                             mandatory=True, allowed_list=WAVEFRONT_CORRECTOR_TYPE_LIST)
        self.telescope_uid = AOTField(TELESCOPE_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True,
                                      reference=TELESCOPES_TABLE)
        self.n_valid_actuators = AOTField(WAVEFRONT_CORRECTOR_N_VALID_ACTUATORS, INTEGER_FORMAT, UNIT_COUNT)
        self.pupil_mask = AOTField(WAVEFRONT_CORRECTOR_PUPIL_MASK, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                   reference=IMAGE_REF)
        self.tfz_num = AOTField(WAVEFRONT_CORRECTOR_TFZ_NUM, LIST_FORMAT, UNIT_DIMENSIONLESS)
        self.tfz_den = AOTField(WAVEFRONT_CORRECTOR_TFZ_DEN, LIST_FORMAT, UNIT_DIMENSIONLESS)
        self.transformation_matrix = AOTField(TRANSFORMATION_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)
        self.aberration_uid = AOTField(ABERRATION_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                       reference=ABERRATIONS_TABLE)

        fields = AOTFieldList(self.uid, self.type, self.telescope_uid, self.n_valid_actuators, self.pupil_mask,
                              self.tfz_num, self.tfz_den, self.transformation_matrix, self.aberration_uid)
        super().__init__(WAVEFRONT_CORRECTORS_TABLE, fields, parent)

    def verify_contents(self):
        # TODO ensure that n_valid_actuators is 2 for TT and 1 for LS
        self._ensure_has_data()


class AOTFITSTableWavefrontCorrectorsDM(AOTFITSSecondaryTable):
    def __init__(self, parent: AOTFITSFile, main: AOTFITSTableWavefrontCorrectors):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.actuators_x = AOTField(WAVEFRONT_CORRECTOR_DM_ACTUATORS_X, LIST_FORMAT, UNIT_METERS)
        self.actuators_y = AOTField(WAVEFRONT_CORRECTOR_DM_ACTUATORS_Y, LIST_FORMAT, UNIT_METERS)
        self.influence_function = AOTField(WAVEFRONT_CORRECTOR_DM_INFLUENCE_FUNCTION, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                           reference=IMAGE_REF)
        self.stroke = AOTField(WAVEFRONT_CORRECTOR_DM_STROKE, FLOAT_FORMAT, UNIT_METERS)

        fields = AOTFieldList(self.uid, self.actuators_x, self.actuators_y, self.influence_function, self.stroke)
        super().__init__(WAVEFRONT_CORRECTORS_DM_TABLE, fields, parent, main, _s.WAVEFRONT_CORRECTOR_TYPE_DM)

    def verify_contents(self):
        # TOOD check that the length is equal to N_VALID_ACTUATORS
        self._ensure_same_length(self.actuators_x, self.actuators_y)


class AOTFITSTableLoops(AOTFITSTable):
    def __init__(self, parent: AOTFITSFile):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.type = AOTField(LOOPS_TYPE, STRING_FORMAT, UNIT_DIMENSIONLESS,
                             mandatory=True, allowed_list=LOOPS_TYPE_LIST)
        self.commanded_uid = AOTField(LOOPS_COMMANDED, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True,
                                      reference=WAVEFRONT_CORRECTORS_TABLE)
        self.time_uid = AOTField(TIME_REFERENCE, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=TIME_TABLE)
        self.status = AOTField(LOOPS_STATUS, STRING_FORMAT, UNIT_DIMENSIONLESS, allowed_list=LOOPS_STATUS_LIST)
        self.commands = AOTField(LOOPS_COMMANDS, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.ref_commands = AOTField(LOOPS_REF_COMMANDS, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.framerate = AOTField(LOOPS_FRAMERATE, FLOAT_FORMAT, UNIT_HERTZ)
        self.delay = AOTField(LOOPS_DELAY, FLOAT_FORMAT, UNIT_FRAME)
        self.time_filter_num = AOTField(LOOPS_TIME_FILTER_NUM, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.time_filter_den = AOTField(LOOPS_TIME_FILTER_DEN, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)

        fields = AOTFieldList(self.uid, self.type, self.commanded_uid, self.time_uid, self.status, self.commands,
                              self.ref_commands, self.framerate, self.delay, self.time_filter_num, self.time_filter_den)
        super().__init__(LOOPS_TABLE, fields, parent)

    def verify_contents(self):
        self._ensure_has_data()


class AOTFITSTableLoopsControl(AOTFITSSecondaryTable):
    def __init__(self, parent: AOTFITSFile, main: AOTFITSTableLoops):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.input_sensor_uid = AOTField(LOOPS_CONTROL_INPUT_SENSOR, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True,
                                         reference=WAVEFRONT_SENSORS_TABLE)
        self.modes = AOTField(LOOPS_CONTROL_MODES, STRING_FORMAT, UNIT_DIMENSIONLESS, reference=IMAGE_REF)
        self.modal_coefficients = AOTField(LOOPS_CONTROL_MODAL_COEFFICIENTS, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                           reference=IMAGE_REF)
        self.control_matrix = AOTField(LOOPS_CONTROL_CONTROL_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                       reference=IMAGE_REF)
        self.measurements_to_modes = AOTField(LOOPS_CONTROL_MEASUREMENTS_TO_MODES, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)
        self.modes_to_commands = AOTField(LOOPS_CONTROL_MODES_TO_COMMANDS, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                          reference=IMAGE_REF)
        self.interaction_matrix = AOTField(LOOPS_CONTROL_INTERACTION_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                           reference=IMAGE_REF)
        self.commands_to_modes = AOTField(LOOPS_CONTROL_COMMANDS_TO_MODES, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                          reference=IMAGE_REF)
        self.modes_to_measurements = AOTField(LOOPS_CONTROL_MODES_TO_MEASUREMENTS, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                              reference=IMAGE_REF)
        self.residual_commands = AOTField(LOOPS_CONTROL_RESIDUAL_COMMANDS, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                          reference=IMAGE_REF)

        fields = AOTFieldList(self.uid, self.input_sensor_uid, self.modes, self.modal_coefficients,
                              self.control_matrix, self.measurements_to_modes, self.modes_to_commands,
                              self.interaction_matrix, self.commands_to_modes, self.modes_to_measurements,
                              self.residual_commands)
        super().__init__(LOOPS_CONTROL_TABLE, fields, parent, main, _s.LOOPS_TYPE_CONTROL)


class AOTFITSTableLoopsOffload(AOTFITSSecondaryTable):
    def __init__(self, parent: AOTFITSFile, main: AOTFITSTableLoops):
        self.uid = AOTField(REFERENCE_UID, STRING_FORMAT, UNIT_DIMENSIONLESS, mandatory=True, unique=True)
        self.input_corrector_uid = AOTField(LOOPS_OFFLOAD_INPUT_CORRECTOR, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                            mandatory=True, reference=WAVEFRONT_CORRECTORS_TABLE)
        self.offload_matrix = AOTField(LOOPS_OFFLOAD_OFFLOAD_MATRIX, STRING_FORMAT, UNIT_DIMENSIONLESS,
                                       reference=IMAGE_REF)

        fields = AOTFieldList(self.uid, self.input_corrector_uid, self.offload_matrix)

        super().__init__(LOOPS_OFFLOAD_TABLE, fields, parent, main, _s.LOOPS_TYPE_OFFLOAD)


def _fits_type_to_aot(fits_type: str) -> str:
    if fits_type in ['E', 'D']:
        return FLOAT_FORMAT
    if fits_type in ['B', 'I', 'J', 'K']:
        return INTEGER_FORMAT
    if re.fullmatch(r'\d*A', fits_type):
        return STRING_FORMAT
    if re.fullmatch(r'[QP][DE]\(\d*\)', fits_type):
        return LIST_FORMAT
    return ''


def _convert_null_to_none(value, col: fits.Column):
    match _fits_type_to_aot(col.format):
        case _s.STRING_FORMAT:
            if value == '':
                value = None
        case _s.FLOAT_FORMAT:
            if np.isnan(value):
                value = None
        case _s.INTEGER_FORMAT:
            if col.null and value == col.null:
                value = None
        case _s.LIST_FORMAT:
            if value is None:
                value = []
            value = [None if np.isnan(v) else v for v in value]
        case _:
            # This will trigger a warning later
            pass
    return value
