# aotpy
Helper Python package for handling Adaptive Optics Telemetry (AOT) standard files.

## How to install
Support is offered for Python 3.9 or later.

### From PyPi (recommended)
    python -m pip install aotpy

### From the repository
Clone the repository and then install with:

    python -m pip install path/to/aotpy
    
where 'path/to/aotpy' is the path to the root of the cloned repository. 



## Code example
```python
import numpy as np
from aotpy import AOSystem, RTC, MainTelescope, NaturalGuideStar, LaserGuideStar,\
    ShackHartmann, ControlLoop, DeformableMirror, Image
from aotpy.fits import write_to_fits
tel = MainTelescope(name="Example telescope")
cor = DeformableMirror(name="Example DM", n_actuators=32, telescope=tel)
ngs = NaturalGuideStar(name="Example NGS")
ngs_wfs = ShackHartmann(name="WFS1", source=ngs,
                        slopes=Image(name="NGS slopes", data=np.ones((10000, 8))))
lgs = LaserGuideStar(name="Example LGS")
lgs_wfs = ShackHartmann(name="WFS2", source=lgs,
                        slopes=Image(name="LGS slopes", data=np.ones((10000, 8))))
lo_loop = ControlLoop(name="LO loop", input_wfs=ngs_wfs, commanded_corrector=cor,
                      commands=Image(name="LO commands", data=np.ones((10000, 32))),
                      control_matrix=Image(name='LO control matrix', data=np.ones((8, 32))))
ho_loop = ControlLoop(name="HO loop", input_wfs=lgs_wfs, commanded_corrector=cor,
                      commands=Image(name="HO commands", data=np.ones((10000, 32))),
                      control_matrix=Image(name='HO control matrix', data=np.ones((8, 32))))
rtc = RTC(loops=[lo_loop, ho_loop])
system = AOSystem(sources=[ngs, lgs],
                  telescope=tel,
                  wavefront_sensors=[ngs_wfs, lgs_wfs],
                  rtc=rtc,
                  wavefront_correctors=[cor])
write_to_fits(system, "example.fits")

```
