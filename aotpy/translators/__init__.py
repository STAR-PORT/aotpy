"""
This subpackage contains modules with translators, which are able to convert non-standard AO telemetry files into aotpy
objects.

The aotpy objects created by these translators can be handled as any other aotpy object regardless of origin.
"""

from .ciao import CIAOTranslator
from .eris import ERISTranslator
from .galacsi import GALACSITranslator
from .naomi import NAOMITranslator
from .papyrus import PAPYRUSTranslator
