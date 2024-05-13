"""
A package for handling FITS files using the AOT format.
"""

from ._file import AOTFITSErrorLevel, verify_file
from .reader import AOTFITSReader
from .utils import FITSFileImage, FITSURLImage, image_from_fits_file, image_from_hdus, image_from_hdu
from .writer import AOTFITSWriter
