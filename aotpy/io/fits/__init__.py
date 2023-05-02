"""
A package for handling FITS files using the AOT format.
"""

from .reader import read_system_from_fits, FITSReader
from .utils import *
from .writer import write_system_to_fits, FITSWriter
