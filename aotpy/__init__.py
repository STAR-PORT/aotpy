"""
aotpy is a package that supports the Adaptive Optics Telemetry (AOT) data exchange standard.
"""
from typing import Type

# Initialize dictionaries of available writers/readers in order to enable their usage in AOSystem
_AVAILABLE_WRITERS: dict[str, Type['SystemWriter']] = {}
_AVAILABLE_READERS: dict[str, Type['SystemReader']] = {}

# First import core and then import io to force dictionaries to be updated
from .core import *
from .io import *

__all__ = [s for s in dir() if not s.startswith('_')]
