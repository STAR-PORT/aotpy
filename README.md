[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8187229.svg)](https://doi.org/10.5281/zenodo.8187229)


# aotpy
Helper Python package for handling Adaptive Optics Telemetry (AOT) standard files.
Basic [documentation available](https://aotpy.readthedocs.io/en/latest/).

## How to install
Support is offered for Python 3.10 or later.

### From PyPI (recommended)
    python -m pip install aotpy

### From the repository
Clone the repository and then install with:

    python -m pip install path/to/aotpy
    
where 'path/to/aotpy' is the path to the root of the cloned repository. 



## Code example
### Creating an AOT FITS file
Here we exemplify of how to create an AOT FITS file from scratch. In general terms, we create an AOSystem object,
fill the relevant data fields and then write that object into a FITS file which follows the AOT specification.

```python
import numpy as np

import aotpy

# Create the system object and its main telescope
system = aotpy.AOSystem(ao_mode='GLAO')
system.main_telescope = aotpy.MainTelescope("Example telescope")

# Create the DM, which is installed on the main telescope.
dm = aotpy.DeformableMirror("Example DM", n_valid_actuators=32, telescope=system.main_telescope)
system.wavefront_correctors.append(dm)

# Create the sources being sensed.
ngs = aotpy.NaturalGuideStar("Example NGS")
lgs = aotpy.SodiumLaserGuideStar("Example LGS")
system.sources = [ngs, lgs]

# Create a 4 subaperture Shack-Hartmann WFS sensing the NGS source. It contains 10000 frames of slopes.
ngs_wfs = aotpy.ShackHartmann("WFS1", source=ngs, n_valid_subapertures=4,
                              measurements=aotpy.Image("NGS slopes", data=np.ones((10000, 2, 4))))
# Also create a detector for this WFS, containing 100 frames of pixel data (20x20).
ngs_wfs.detector = aotpy.Detector("DET1", pixel_intensities=aotpy.Image("Pixels", data=np.random.random((100, 20, 20))))

# Create a 8 subaperture Shack-Hartmann WFS sensing the LGS source. It contains 10000 frames of slopes.
lgs_wfs = aotpy.ShackHartmann("WFS2", source=lgs, n_valid_subapertures=8,
                              measurements=aotpy.Image("LGS slopes", data=np.ones((10000, 2, 8))))
system.wavefront_sensors = [ngs_wfs, lgs_wfs]

# Create the loops in the system.
# Both loops command the DM, but one handles LO modes using the NGS, while the other handles HO modes using the LGS.
lo_loop = aotpy.ControlLoop("LO loop", input_sensor=ngs_wfs, commanded_corrector=dm,
                            commands=aotpy.Image("LO commands", data=np.ones((10000, 32))),
                            control_matrix=aotpy.Image('LO control matrix', data=np.ones((32, 2, 4))))
ho_loop = aotpy.ControlLoop("HO loop", input_sensor=lgs_wfs, commanded_corrector=dm,
                            commands=aotpy.Image("HO commands", data=np.ones((10000, 32))),
                            control_matrix=aotpy.Image('HO control matrix', data=np.ones((32, 2, 8))))
system.loops = [lo_loop, ho_loop]

# Write the system to an AOT file. This file can then be read, recovering the exact same system.
system.write_to_file("example.fits")
```
### Reading an existing AOT file
In this example we assume we already have an AOT file. We can easily read it and explore its data. For demonstration 
purposes, we assume matplotlib is installed, although it is not a requirement for this package.

```python
import matplotlib.pyplot as plt

import aotpy

# If we open the file that was created in the example above, the resulting "system" object will contain the exact same
# data as the "system" object that was built in the example above.
system = aotpy.AOSystem.read_from_file("example.fits")

# We can then display a frame of pixel data. For the data above, we know the first WFS has pixel data.
pixel_data = system.wavefront_sensors[0].detector.pixel_intensities.data
# Then we display the first frame of data.
plt.imshow(pixel_data[0])
plt.show()
```

## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant 
agreement No. 101004719 (OPTICON–RadioNet Pilot).
