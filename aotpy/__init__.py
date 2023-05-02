"""
aotpy is a package that supports the Adaptive Optics Telemetry (AOT) data exchange standard.
"""

# Initialize dictionaries of available writers/readers in order to enable their usage in AOSystem
_AVAILABLE_WRITERS = {}
_AVAILABLE_READERS = {}

# First import core and then import io to force dictionaries to be updated
from .core import *
from .io import *

__all__ = [s for s in dir() if not s.startswith('_')]
