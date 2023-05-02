# aotpy
Helper Python package for handling Adaptive Optics Telemetry (AOT) standard files.
Basic [documentation available](https://aotpy.readthedocs.io/en/latest/).

## How to install
Support is offered for Python 3.10 or later.

### From PyPi (recommended)
    python -m pip install aotpy

### From the repository
Clone the repository and then install with:

    python -m pip install path/to/aotpy
    
where 'path/to/aotpy' is the path to the root of the cloned repository. 



## Code example
```python
import numpy as np
import aotpy

system = aotpy.AOSystem(ao_mode='GLAO')
system.main_telescope = aotpy.MainTelescope("Example telescope")

dm = aotpy.DeformableMirror("Example DM", n_valid_actuators=32, telescope=system.main_telescope)
system.wavefront_correctors.append(dm)

ngs = aotpy.NaturalGuideStar("Example NGS")
lgs = aotpy.SodiumLaserGuideStar("Example LGS")
system.sources = [ngs, lgs]

ngs_wfs = aotpy.ShackHartmann("WFS1", source=ngs, n_valid_subapertures=4,
                              measurements=aotpy.Image("NGS slopes", data=np.ones((10000, 2, 4))))
lgs_wfs = aotpy.ShackHartmann("WFS2", source=lgs, n_valid_subapertures=8,
                              measurements=aotpy.Image("LGS slopes", data=np.ones((10000, 2, 8))))
system.wavefront_sensors = [ngs_wfs, lgs_wfs]

lo_loop = aotpy.ControlLoop("LO loop", input_sensor=ngs_wfs, commanded_corrector=dm,
                            commands=aotpy.Image("LO commands", data=np.ones((10000, 32))),
                            control_matrix=aotpy.Image('LO control matrix', data=np.ones((32, 2, 4))))
ho_loop = aotpy.ControlLoop("HO loop", input_sensor=lgs_wfs, commanded_corrector=dm,
                            commands=aotpy.Image("HO commands", data=np.ones((10000, 32))),
                            control_matrix=aotpy.Image('HO control matrix', data=np.ones((32, 2, 8))))
system.loops = [lo_loop, ho_loop]

system.write_to_file("example.fits", overwrite=True)

```

## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 101004719 (OPTICON–RadioNet Pilot).
