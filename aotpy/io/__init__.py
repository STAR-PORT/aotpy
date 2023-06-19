"""
This subpackage contains modules and packages for handling AOT files using data storage formats supported by aotpy.
Currently only FITS is supported.
"""

from .base import SystemReader, SystemWriter
from .fits import *
from .. import _AVAILABLE_WRITERS, _AVAILABLE_READERS

# Add available writers/readers to their respective dictionaries
_AVAILABLE_WRITERS['fits'] = FITSWriter
_AVAILABLE_READERS['fits'] = FITSReader
