"""
This module contains a class for describing the passage of time related to data in the system.
"""


from dataclasses import dataclass, field

from .base import Referenceable

__all__ = ['Time']


@dataclass(kw_only=True)
class Time(Referenceable):
    """Contains data that describes the passage of time. All time in a system must be synchronous.
    Can be associated with time-varying data."""

    timestamps: list[float] = field(default_factory=list)
    'List of Unix timestamps at which the respective data applies. (in s units)'

    frame_numbers: list[float] = field(default_factory=list)
    'List of frame numbers at which the respective data applies. (in count units)'
