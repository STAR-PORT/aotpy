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
        self._system = system

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
    extra_data : default = False
        Whether it is expected that the file contains some data that does not fit the AOT standard. If `extra_data` is
        not `True`, user will be warned if extra data is detected.
    **kwargs
        Keyword arguments passed on as options to the file handling function.
    """
    def __init__(self, filename: str | os.PathLike, *, extra_data: bool = False, **kwargs) -> None:
        self._filename = filename
        self._extra_data_flag = extra_data

        self._initialize_data()
        self._system, self._extra_data = self._read(**kwargs)

    def get_system(self) -> aotpy.AOSystem:
        """
        Return `AOSystem` that has been read.
        """
        return self._system

    def get_extra_data(self) -> list | None:
        """
        Return a list of extra data that may have been in file. If no extra data exists, `None` is returned.
        """
        return self._extra_data

    @abstractmethod
    def _initialize_data(self) -> None:
        """
        Initialize data structures necessary for reading the file.
        """
        pass

    @abstractmethod
    def _read(self, **kwargs) -> tuple[aotpy.AOSystem, list]:
        """
        Read file and build `AOSystem` that contains the data in the initialized `filename`.
        Return the system that was built along with any extra data that may be present.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed on as options to the file handling function.
        """
        pass
