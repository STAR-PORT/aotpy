"""
This module contains classes that describe conceptual loops in an adaptive optics system.
"""

from dataclasses import dataclass

from .base import Referenceable
from .image import Image
from .optical_sensor import WavefrontSensor
from .time import Time
from .wavefront_corrector import WavefrontCorrector

__all__ = ['Loop', 'ControlLoop', 'OffloadLoop']


@dataclass(kw_only=True)
class Loop(Referenceable):
    """Base class that contains data regarding one system loop."""

    commanded_corrector: WavefrontCorrector
    """Wavefront corrector that is being commanded by this loop."""

    time: Time = None

    closed: bool = True
    "Indicates whether the loop was opened or closed for the duration of the data collection."

    commands: Image = None
    """Sequence of commands sent to the associated wavefront corrector. Each of the :math:`t` frames contains the values
    sent to each of the :math:`a_v` valid actuators of a certain wavefront corrector.
    (Dimensions :math:`t \\times a_v`, in m units, using data type flt)"""

    ref_commands: Image = None
    """Reference offset commands for each of :math:`a_v` actuators.
    (Dimensions :math:`a_v`, in m units, using data type flt)"""

    framerate: float = None
    'Frequency at which the loop operates. (in Hz units)'

    delay: float = None
    """Full AO loop delay, measured from the mid-point of the integration to the mid-point of the command applied.
    Includes the RTC latency and other communication delays. (in frame units)"""

    time_filter_num: Image = None
    """One set of time filter numerators per mode being described. If :math:`m=1`, it is assumed to be applicable to all
    modes. The first numerator is the loop gain.
    (Dimensions :math:`m \\times i`, dimensionless quantity, using data type flt)"""

    time_filter_den: Image = None
    r"""One set of time filter denominators per mode being described. If :math:`m=1`, it is assumed to be applicable to
    all modes. Uses standard transfer function :math:`sys(z) = \sum_{i=0}^{N-1} b_i z^i/\sum_{j=0}^N a_j z^j`.
    (Dimensions :math:`m \times j`, dimensionless quantity, using data type flt)"""


@dataclass(kw_only=True)
class ControlLoop(Loop):
    """Contains data relevant to a control loop (relation between one wavefront sensor and one wavefront corrector)."""

    input_sensor: WavefrontSensor

    modes: Image = None
    """Set of :math:`m` different :math:`h \\times w` arrays, each representing the orthonormal basis of the
    corresponding mode. (Dimensions :math:`m \\times h \\times w`, dimensionless quantity, using data type flt)"""

    modal_coefficients: Image = None
    """Sequence of coefficients of the modes to be corrected on the wavefront corrector. Each of the :math:`t` frames
    contains the coefficients respective to each of the :math:`m` modes being corrected.
    (Dimensions :math:`t \\times m`, in user defined units, using data type flt)"""

    control_matrix: Image = None
    """Linear relationship between the wavefront sensor measurements (:math:`d \\times s_v`) and the corrector commands 
    (:math:`a_v`). (Dimensions :math:`a_v \\times d \\times s_v`, in user defined units, using data type flt)"""

    measurements_to_modes: Image = None
    """Linear relationship between the wavefront sensor measurements (:math:`d \\times s_v`) and the modes to be
    corrected (:math:`m`). (Dimensions :math:`m \\times d \\times s_v`, in user defined units, using data type flt)"""

    modes_to_commands: Image = None
    """Linear relationship between the modes (:math:`m`) to be corrected and the corrector commands (:math:`a_v`).
    (Dimensions :math:`a_v \\times m`, in user defined units, using data type flt)"""

    interaction_matrix: Image = None
    """Represents the measurements of the wavefront sensor (:math:`s_v \\times d`) in response to each actuator of the
    wavefront corrector (:math:`a_v`).
    (Dimensions :math:`s_v \\times d \\times a_v`, in user defined units, using data type flt)"""

    commands_to_modes: Image = None
    """Represents the modal response (:math:`m`) to each actuator of the wavefront corrector (:math:`a_v`).
    (Dimensions :math:`m \\times a_v`, in user defined units, using data type flt)"""

    modes_to_measurements: Image = None
    """Linear relationship between the modal response (:math:`m`) and the wavefront sensor measurements 
    (:math:`s_v \\times d`). (Dimensions :math:`s_v \\times d \\times m`, in user defined units, using data type flt)"""

    residual_commands: Image = None
    """Reconstructed corrector commands before time filtering
    (Dimensions :math:`t \\times a_v`, in user defined units, using data type flt)"""


@dataclass(kw_only=True)
class OffloadLoop(Loop):
    """Contains data relevant to an offload loop (relation between two wavefront correctors)."""

    input_corrector: WavefrontCorrector

    offload_matrix: Image = None
    """Linear relationship between the commands from one actuator (:math:`a1`) and the corresponding commands that get
    offloaded to another actuator (:math:`a2`).
    (Dimensions :math:`a2_v \\times a1_v`, in user defined units, using data type flt)"""
