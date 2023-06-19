"""
This module contains a base class for translating non-standard AO telemetry data.
"""

from abc import ABC, abstractmethod

import aotpy


class BaseTranslator(ABC):
    """Abstract class for translators.

    Translators are able to convert non-standard AO telemetry data files into an `AOSystem` object.

    Parameters
    ----------
    *args
        Arguments used to initialize the translator class.

    """

    @abstractmethod
    def __init__(self, *args) -> None:
        self.system: aotpy.AOSystem = aotpy.AOSystem()

    @classmethod
    def translate(cls, *args) -> aotpy.AOSystem:
        """
        Initialize class with `args`, return translated `AOSystem`.

        Parameters
        ----------
        *args
            Arguments used to initialize the translator class.

        Returns
        -------
            `AOSystem` containing translated data.
        """
        t = cls(*args)
        return t.system
