"""
This module contains base classes for defining writers and readers in aotpy.
"""

import os
from abc import ABC, abstractmethod

import aotpy


class SystemWriter(ABC):
    """Abstract class for system writers.

    Writers are able to write an `AOSystem` into a file using some format.

    Parameters
    ----------
    system
        `AOSystem` to be written into a file.

    """

    @abstractmethod
    def __init__(self, system: aotpy.AOSystem) -> None:
        pass

    @abstractmethod
    def write(self, filename: str | os.PathLike, **kwargs) -> None:
        """
        Write the initialized `system` into the specified `filename`.

        Parameters
        ----------
        filename
            Path to the file that will be written.
        **kwargs
            Keyword arguments passed on as options to the file handling function.
        """
        pass


class SystemReader(ABC):
    """Abstract class for system readers.

    Readers are able to create an `AOSystem` based on a file of a certain format.

    Parameters
    ----------
    filename
        Path to file to be read into an `AOSystem`.
    **kwargs
        Keyword arguments passed on as options to the file handling function.
    """

    @abstractmethod
    def __init__(self, filename: str | os.PathLike, **kwargs) -> None:
        pass

    @abstractmethod
    def get_system(self) -> aotpy.AOSystem:
        """
        Return `AOSystem` that has been read.
        """
        pass
