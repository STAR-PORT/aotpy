"""
This module contains classes and functions that aid in handling AOT FITS files.
"""

import datetime
import os
import re

import numpy as np
from astropy.io import fits

import aotpy
from . import _keywords as kw

__all__ = ['FITSFileImage', 'FITSURLImage', 'image_from_file', 'image_from_hdus', 'image_from_hdu',
           'metadatum_from_card', 'metadata_from_hdu', 'datetime_to_iso', 'keyword_is_relevant']


def keyword_is_relevant(keyword):
    """Check if keyword is relevant. Keywords are considered "irrelevant" if they are already reflected elsewhere in the
     object produced by Astropy."""
    _standard_keywords = {'SIMPLE', 'EXTEND', 'BSCALE', 'BZERO', 'XTENSION', 'BITPIX',
                          'PCOUNT', 'GCOUNT', 'EXTNAME', 'CHECKSUM', 'DATASUM'}
    _standard_patterns = re.compile(r'NAXIS\d*')

    return keyword not in _standard_keywords and not _standard_patterns.fullmatch(keyword)


class _FITSExternalImage(aotpy.Image):
    """Describes an external FITS file containing multidimensional data and metadata related to it."""

    def to_internal(self) -> aotpy.Image:
        """
        Convert external image into an internal image.
        """
        return aotpy.Image(name=self.name,
                           data=self.data,
                           unit=self.unit,
                           time=self.time,
                           metadata=self.metadata)


class FITSFileImage(_FITSExternalImage):
    """Describes an external FITS file present locally, containing multidimensional data and metadata related to it.

    `FITSFileImage` remembers the path from which the data came from, enabling future referencing of that path.

    Parameters
    ----------
    path
        Path to FITS file that contains multidimensional data.
    index: int, optional
        Index of the HDU that contains the image data. If omitted, the first HDU containing image data is assumed.
    **kwargs
        Keyword arguments passed on as options to the file handling function.
    """

    def __init__(self, path: str | os.PathLike, index: int = None, **kwargs):
        self.filename = os.path.basename(path)
        self.index = index
        self.name, self.data, self.unit, self._time, self.metadata = _get_image_fields_from_file(path, index, **kwargs)

    def __eq__(self, other):
        return self.filename == other.filename and self.index == other.index and self.time == other.time


class FITSURLImage(_FITSExternalImage):
    """Describes an external FITS file available via URL, containing multidimensional data and metadata related to it.

    `FITSURLImage` remembers the URL from which the data came from, enabling future referencing of that URL.

    Parameters
    ----------
    url
        URL to FITS file that contains multidimensional data.
    index: int, optional
        Index of the HDU that contains the image data. If omitted, the first HDU containing image data is assumed.
    **kwargs
        Keyword arguments passed on as options to the file handling function.
    """

    def __init__(self, url: str, index: int = None, **kwargs):
        self.url = url
        self.index = index
        self.name, self.data, self.unit, self._time, self.metadata = _get_image_fields_from_file(url, index, **kwargs)

    def __eq__(self, other):
        return self.url == other.url and self.index == other.index and self.time == other.time


def image_from_file(path: str | os.PathLike, index: int = None, *, name: str = None, **kwargs) -> aotpy.Image:
    """
    Get `Image` from specified path or URL.

    Parameters
    ----------
    path
        Path/URL to FITS file that contains multidimensional data.
    index: int, optional
        Index of the HDU that contains the image data. If omitted, the first HDU containing image data is assumed.
    name: str, optional
        Name of the Image. If None, the name is the same as in the file.
    **kwargs
        Keyword arguments passed on as options to the file handling function.
    """
    _name, data, unit, _time, metadata = _get_image_fields_from_file(path, index, **kwargs)
    if name is None:
        name = _name
    image = aotpy.Image(name=name, data=data, unit=unit, metadata=metadata)
    image._time = _time
    return image


def image_from_hdus(hdus: fits.HDUList, index: int = None, *, name: str = None) -> aotpy.Image:
    """
    Get `Image` from specified path or URL.

    Parameters
    ----------
    hdus
        List of HDUs from which `Image` is to be extracted.
    index: int, optional
        Index of the HDU that contains the image data. If omitted, the first HDU containing image data is assumed.
    name: str, optional
        Name of the Image. If None, the name is the same as in the HDU.
    """
    _name, data, unit, _time, metadata = _get_image_fields_from_hdus(hdus, index)
    if name is None:
        name = _name
    image = aotpy.Image(name=name, data=data, unit=unit, metadata=metadata)
    image._time = _time
    return image


def image_from_hdu(hdu: fits.ImageHDU, *, name: str = None) -> aotpy.Image:
    """
    Get `Image` from specified HDU.

    Parameters
    ----------
    hdu
        HDU from which `Image` is to be extracted.
    name: str, optional
        Name of the Image. If None, the name is the same as in the HDU.
    """
    _name, data, unit, _time, metadata = _get_image_fields_from_hdu(hdu)
    if name is None:
        name = _name
    image = aotpy.Image(name=name, data=data, unit=unit, metadata=metadata)
    image._time = _time
    return image


def _get_image_fields_from_file(path: str | os.PathLike, index: int = None, **kwargs) -> \
        tuple[str, np.ndarray, str, str, list[aotpy.Metadatum]]:
    try:
        with fits.open(path, **kwargs) as hdus:
            return _get_image_fields_from_hdus(hdus, index)
    except FileNotFoundError:
        from tkinter.filedialog import askopenfilename
        filename = os.path.basename(path)
        selected = askopenfilename(title=f"Please select '{filename}'.",
                                   initialfile=filename, initialdir=os.path.dirname(path),
                                   filetypes=(('FITS files', '*.fits'),
                                              ('Compressed FITS files', '*.fits.gz')))
        if not selected:
            raise ValueError(f"Could not find '{filename}' automatically. Please select it manually")
        with fits.open(selected, **kwargs) as hdus:
            return _get_image_fields_from_hdus(hdus, index)


def _get_image_fields_from_hdus(hdus: fits.HDUList, index: int = None) -> \
        tuple[str, np.ndarray, str, str, list[aotpy.Metadatum]]:
    if index is None:
        for hdu in hdus:
            if hdu.is_image and hdu.data is not None:
                break
        else:
            raise ValueError('Could not find any image data in FITS file.')
    else:
        hdu = hdus[index]
        if not hdu.is_image or hdu.data is None:
            raise ValueError(f'Referenced HDU {hdu.name} does not contain image data.')

    return _get_image_fields_from_hdu(hdu)


def _get_image_fields_from_hdu(hdu) -> tuple[str, np.ndarray, str, str, list[aotpy.Metadatum]]:
    metadata = metadata_from_hdu(hdu)
    unit = None
    if (md := next((x for x in metadata if x.key == kw.IMAGE_UNIT), None)) is not None:
        unit = md.value

    _time = None
    if (md := next((x for x in metadata if x.key == kw.TIME_REFERENCE), None)) is not None:
        _time = md.value
        metadata.remove(md)
    return hdu.name, hdu.data, unit, _time, metadata


def metadatum_from_card(card: fits.Card):
    """
    Get `Metadatum` from `Card`.

    Parameters
    ----------
    card
        `Card` to convert to `Metadatum`.
    """
    return aotpy.Metadatum(card.keyword, card.value, card.comment if card.comment != '' else None)


def metadata_from_hdu(hdu: fits.ImageHDU) -> list[aotpy.Metadatum]:
    """
    Get metadata (list of `Metadatum`) from HDU. Only relevant keywords are extracted.

    Parameters
    ----------
    hdu
        HDU from which to extract metadata.
    """
    # If the keywords are irrelevant they don't need to be part of the image metadata, as that information is already
    # implied in the numpy data.
    return [metadatum_from_card(card) for card in hdu.header.cards if keyword_is_relevant(card.keyword)]


def datetime_to_iso(d: datetime.datetime):
    """
    Convert datetime to ISO format string. If timezone information is present, convert time to UTC and remove
    information.

    Parameters
    ----------
    d
        `datetime` to be converted to an ISO string.
    """
    if d is None:
        return ''
    if d.tzinfo is not None:
        # If datetime has timezone data, convert the datetime to UTC and then remove the timezone data.
        # This is done to ensure dt.isoformat() doesn't print UTC offsets.
        # If datetime doesn't have timezone data, we assume it is already in UTC.
        d = d.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return d.isoformat()
