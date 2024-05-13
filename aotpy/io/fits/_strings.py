"""
This module contains all the 'special strings' used in the AOT FITS format itself.
Not meant to be directly imported by users.
"""

STRING_FORMAT = 'str'  # e.g. 64A
INTEGER_FORMAT = 'int'  # e.g. K
FLOAT_FORMAT = 'flt'  # e.g. D
LIST_FORMAT = 'lst'  # e.g. QD

IMAGE_REF = 'IMAGE'  # Only used internally
ROW_REFERENCE = 'ROWREF'
INTERNAL_REFERENCE = 'INTREF'
FILE_REFERENCE = 'FILEREF'
URL_REFERENCE = 'URLREF'
IMAGE_UNIT = 'BUNIT'

UNIT_DIMENSIONLESS = ''
UNIT_COUNT = 'count'
UNIT_METERS = 'm'
UNIT_SECONDS = 's'
UNIT_RADIANS = 'rad'
UNIT_DEGREES = 'deg'
UNIT_ELECTRONS = 'electron'
UNIT_PIXELS = 'pix'
UNIT_DECIBELS = 'dB'
UNIT_FRAME = 'frame'
UNIT_HERTZ = 'Hz'
UNIT_ARCSEC = 'arcsec'

TIME_TABLE = 'AOT_TIME'
ATMOSPHERIC_PARAMETERS_TABLE = 'AOT_ATMOSPHERIC_PARAMETERS'
ABERRATIONS_TABLE = 'AOT_ABERRATIONS'
TELESCOPES_TABLE = 'AOT_TELESCOPES'
SOURCES_TABLE = 'AOT_SOURCES'
SOURCES_SODIUM_LGS_TABLE = 'AOT_SOURCES_SODIUM_LGS'
SOURCES_RAYLEIGH_LGS_TABLE = 'AOT_SOURCES_RAYLEIGH_LGS'
DETECTORS_TABLE = 'AOT_DETECTORS'
SCORING_CAMERAS_TABLE = 'AOT_SCORING_CAMERAS'
WAVEFRONT_SENSORS_TABLE = 'AOT_WAVEFRONT_SENSORS'
WAVEFRONT_SENSORS_SHACK_HARTMANN_TABLE = 'AOT_WAVEFRONT_SENSORS_SHACK_HARTMANN'
WAVEFRONT_SENSORS_PYRAMID_TABLE = 'AOT_WAVEFRONT_SENSORS_PYRAMID'
WAVEFRONT_CORRECTORS_TABLE = 'AOT_WAVEFRONT_CORRECTORS'
WAVEFRONT_CORRECTORS_DM_TABLE = 'AOT_WAVEFRONT_CORRECTORS_DM'
LOOPS_TABLE = 'AOT_LOOPS'
LOOPS_CONTROL_TABLE = 'AOT_LOOPS_CONTROL'
LOOPS_OFFLOAD_TABLE = 'AOT_LOOPS_OFFLOAD'

# AOSystem keywords
AOT_VERSION = 'AOT-VERS'
AOT_AO_MODE = 'AO-MODE'
AOT_AO_MODE_SET = {'SCAO', 'SLAO', 'GLAO', 'MOAO', 'LTAO', 'MCAO'}
AOT_TIMESYS = 'TIMESYS'
AOT_TIMESYS_UTC = 'UTC'
AOT_DATE_BEG = 'DATE-BEG'
AOT_DATE_END = 'DATE-END'
AOT_STREHL_RATIO = 'STREHL-R'
AOT_STREHL_WAVELENGTH = 'STREHL-L'
AOT_SYSTEM_NAME = 'SYS-NAME'
AOT_CONFIG = 'CONFIG'
AOT_HEADER_SET = {AOT_VERSION, AOT_AO_MODE, AOT_TIMESYS, AOT_DATE_BEG, AOT_DATE_END, AOT_STREHL_RATIO,
                  AOT_STREHL_WAVELENGTH, AOT_SYSTEM_NAME, AOT_CONFIG}

REFERENCE_UID = 'UID'
TIME_REFERENCE = 'TIME_UID'
ABERRATION_REFERENCE = 'ABERRATION_UID'
LASER_LAUNCH_TELESCOPE_REFERENCE = 'LLT_UID'
DETECTOR_REFERENCE = 'DETECTOR_UID'
SOURCE_REFERENCE = 'SOURCE_UID'
NCPA_REFERENCE = 'NCPA_UID'
TELESCOPE_REFERENCE = 'TELESCOPE_UID'
TRANSFORMATION_MATRIX = 'TRANSFORMATION_MATRIX'

# AOT_TIME fields
TIME_TIMESTAMPS = 'TIMESTAMPS'
TIME_FRAME_NUMBERS = 'FRAME_NUMBERS'

# AOT_ATMOSPHERIC_PARAMETERS fields
ATMOSPHERIC_PARAMETERS_WAVELENGTH = 'WAVELENGTH'
ATMOSPHERIC_PARAMETERS_R0 = 'R0'
ATMOSPHERIC_PARAMETERS_SEEING = 'SEEING'
ATMOSPHERIC_PARAMETERS_TAU0 = 'TAU0'
ATMOSPHERIC_PARAMETERS_THETA0 = 'THETA0'
ATMOSPHERIC_PARAMETERS_LAYERS_REL_WEIGHT = 'LAYERS_REL_WEIGHT'
ATMOSPHERIC_PARAMETERS_LAYERS_HEIGHT = 'LAYERS_HEIGHT'
ATMOSPHERIC_PARAMETERS_LAYERS_L0 = 'LAYERS_LO'
ATMOSPHERIC_PARAMETERS_LAYERS_WIND_SPEED = 'LAYERS_WIND_SPEED'
ATMOSPHERIC_PARAMETERS_LAYERS_WIND_DIRECTION = 'LAYERS_WIND_DIRECTION'

# AOT_ABERRATIONS fields
ABERRATION_MODES = 'MODES'
ABERRATION_COEFFICIENTS = 'COEFFICIENTS'
ABERRATION_X_OFFSETS = 'X_OFFSETS'
ABERRATION_Y_OFFSETS = 'Y_OFFSETS'

# AOT_TELESCOPES fields
TELESCOPE_TYPE = 'TYPE'
TELESCOPE_LATITUDE = 'LATITUDE'
TELESCOPE_LONGITUDE = 'LONGITUDE'
TELESCOPE_ELEVATION = 'ELEVATION'
TELESCOPE_AZIMUTH = 'AZIMUTH'
TELESCOPE_PARALLACTIC = 'PARALLACTIC'
TELESCOPE_PUPIL_MASK = 'PUPIL_MASK'
TELESCOPE_PUPIL_ANGLE = 'PUPIL_ANGLE'
TELESCOPE_ENCLOSING_D = 'ENCLOSING_D'
TELESCOPE_INSCRIBED_D = 'INSCRIBED_D'
TELESCOPE_OBSTRUCTION_D = 'OBSTRUCTION_D'
TELESCOPE_SEGMENTS_TYPE = 'SEGMENT_TYPE'
TELESCOPE_SEGMENTS_SIZE = 'SEGMENT_SIZE'
TELESCOPE_SEGMENTS_X = 'SEGMENTS_X'
TELESCOPE_SEGMENTS_Y = 'SEGMENTS_Y'

TELESCOPE_TYPE_LLT = 'Laser Launch Telescope'
TELESCOPE_TYPE_MAIN = 'Main Telescope'
TELESCOPE_TYPE_LIST = [TELESCOPE_TYPE_LLT, TELESCOPE_TYPE_MAIN]
TELESCOPE_SEGMENT_TYPE_MONOLITHIC = 'Monolithic'
TELESCOPE_SEGMENT_TYPE_HEXAGON = 'Hexagon'
TELESCOPE_SEGMENT_TYPE_CIRCLE = 'Circle'
TELESCOPE_SEGMENT_LIST = [TELESCOPE_SEGMENT_TYPE_MONOLITHIC, TELESCOPE_SEGMENT_TYPE_HEXAGON,
                          TELESCOPE_SEGMENT_TYPE_CIRCLE]

# AOT_SOURCES fields
SOURCE_TYPE = 'TYPE'
SOURCE_RIGHT_ASCENSION = 'RIGHT_ASCENSION'
SOURCE_DECLINATION = 'DECLINATION'
SOURCE_ELEVATION_OFFSET = 'ELEVATION_OFFSET'
SOURCE_AZIMUTH_OFFSET = 'AZIMUTH_OFFSET'
SOURCE_FWHM = 'FWHM'

SOURCE_TYPE_NATURAL_GUIDE_STAR = 'Natural Guide Star'
SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR = 'Sodium Laser Guide Star'
SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR = 'Rayleigh Laser Guide Star'
SOURCE_TYPE_SCIENCE_STAR = 'Science Star'
SOURCE_TYPE_LIST = [SOURCE_TYPE_NATURAL_GUIDE_STAR, SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR,
                    SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR, SOURCE_TYPE_SCIENCE_STAR]

# AOT_SOURCES_SODIUM_LGS fields
SOURCE_SODIUM_LGS_HEIGHT = 'HEIGHT'
SOURCE_SODIUM_LGS_PROFILE = 'PROFILE'
SOURCE_SODIUM_LGS_ALTITUDES = 'ALTITUDES'

# AOT_SOURCES_RAYLEIGH_LGS fields
SOURCE_RAYLEIGH_LGS_DISTANCE = 'DISTANCE'
SOURCE_RAYLEIGH_LGS_DEPTH = 'DEPTH'

# AOT_DETECTORS fields
DETECTOR_TYPE = 'TYPE'
DETECTOR_SAMPLING_TECHNIQUE = 'SAMPLING_TECHNIQUE'
DETECTOR_SHUTTER_TYPE = 'SHUTTER_TYPE'
DETECTOR_FLAT_FIELD = 'FLAT_FIELD'
DETECTOR_READOUT_NOISE = 'READOUT_NOISE'
DETECTOR_PIXEL_INTENSITIES = 'PIXEL_INTENSITIES'
DETECTOR_FIELD_CENTRE_X = 'FIELD_CENTRE_X'
DETECTOR_FIELD_CENTRE_Y = 'FIELD_CENTRE_Y'
DETECTOR_INTEGRATION_TIME = 'INTEGRATION_TIME'
DETECTOR_COADDS = 'COADDS'
DETECTOR_DARK = 'DARK'
DETECTOR_WEIGHT_MAP = 'WEIGHT_MAP'
DETECTOR_QUANTUM_EFFICIENCY = 'QUANTUM_EFFICIENCY'
DETECTOR_PIXEL_SCALE = 'PIXEL_SCALE'
DETECTOR_BINNING = 'BINNING'
DETECTOR_BANDWIDTH = 'BANDWIDTH'
DETECTOR_TRANSMISSION_WAVELENGTH = 'TRANSMISSION_WAVELENGTH'
DETECTOR_TRANSMISSION = 'TRANSMISSION'
DETECTOR_SKY_BACKGROUND = 'SKY_BACKGROUND'
DETECTOR_GAIN = 'GAIN'
DETECTOR_EXCESS_NOISE = 'EXCESS_NOISE'
DETECTOR_FILTER = 'FILTER'
DETECTOR_BAD_PIXEL_MAP = 'BAD_PIXEL_MAP'
DETECTOR_DYNAMIC_RANGE = 'DYNAMIC_RANGE'
DETECTOR_READOUT_RATE = 'READOUT_RATE'
DETECTOR_FRAME_RATE = 'FRAME_RATE'

# AOT_SCORING_CAMERAS fields
SCORING_CAMERA_PUPIL_MASK = 'PUPIL_MASK'
SCORING_CAMERA_WAVELENGTH = 'WAVELENGTH'

# AOT_WAVEFRONT_SENSORS
WAVEFRONT_SENSOR_TYPE = 'TYPE'
WAVEFRONT_SENSOR_DIMENSIONS = 'DIMENSIONS'
WAVEFRONT_SENSOR_N_VALID_SUBAPERTURES = 'N_VALID_SUBAPERTURES'
WAVEFRONT_SENSOR_MEASUREMENTS = 'MEASUREMENTS'
WAVEFRONT_SENSOR_REF_MEASUREMENTS = 'REF_MEASUREMENTS'
WAVEFRONT_SENSOR_SUBAPERTURE_MASK = 'SUBAPERTURE_MASK'
WAVEFRONT_SENSOR_MASK_X_OFFSETS = 'MASK_X_OFFSETS'
WAVEFRONT_SENSOR_MASK_Y_OFFSETS = 'MASK_Y_OFFSETS'
WAVEFRONT_SENSOR_SUBAPERTURE_SIZE = 'SUBAPERTURE_SIZE'
WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES = 'SUBAPERTURE_INTENSITIES'
WAVEFRONT_SENSOR_WAVELENGTH = 'WAVELENGTH'
WAVEFRONT_SENSOR_OPTICAL_GAIN = 'OPTICAL_GAIN'

WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN = 'Shack-Hartmann'
WAVEFRONT_SENSOR_TYPE_PYRAMID = 'Pyramid'
WAVEFRONT_SENSOR_TYPE_LIST = [WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN, WAVEFRONT_SENSOR_TYPE_PYRAMID]

# AOT_WAVEFRONT_SENSORS_SHACK_HARTMANN
WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROIDING_ALGORITHM = 'CENTROIDING_ALGORITHM'
WAVEFRONT_SENSOR_SHACK_HARTMANN_CENTROID_GAINS = 'CENTROID_GAINS'
WAVEFRONT_SENSOR_SHACK_HARTMANN_SPOT_FWHM = 'SPOT_FWHM'

# AOT_WAVEFRONT_SENSORS_PYRAMID
WAVEFRONT_SENSOR_PYRAMID_N_SIDES = 'N_SIDES'
WAVEFRONT_SENSOR_PYRAMID_MODULATION = 'MODULATION'

# AOT_WAVEFRONT_CORRECTORS fields
WAVEFRONT_CORRECTOR_TYPE = 'TYPE'
WAVEFRONT_CORRECTOR_N_VALID_ACTUATORS = 'N_VALID_ACTUATORS'
WAVEFRONT_CORRECTOR_PUPIL_MASK = 'PUPIL_MASK'
WAVEFRONT_CORRECTOR_TFZ_NUM = 'TFZ_NUM'
WAVEFRONT_CORRECTOR_TFZ_DEN = 'TFZ_DEN'

WAVEFRONT_CORRECTOR_TYPE_DM = 'Deformable Mirror'
WAVEFRONT_CORRECTOR_TYPE_TTM = 'Tip-Tilt Mirror'
WAVEFRONT_CORRECTOR_TYPE_LS = 'Linear Stage'
WAVEFRONT_CORRECTOR_TYPE_LIST = [WAVEFRONT_CORRECTOR_TYPE_DM, WAVEFRONT_CORRECTOR_TYPE_TTM, WAVEFRONT_CORRECTOR_TYPE_LS]

# AOT_WAVEFRONT_CORRECTORS_DM fields
WAVEFRONT_CORRECTOR_DM_ACTUATORS_X = 'ACTUATORS_X'
WAVEFRONT_CORRECTOR_DM_ACTUATORS_Y = 'ACTUATORS_Y'
WAVEFRONT_CORRECTOR_DM_INFLUENCE_FUNCTION = 'INFLUENCE_FUNCTION'
WAVEFRONT_CORRECTOR_DM_STROKE = 'STROKE'

# AOT_LOOPS fields
LOOPS_TYPE = 'TYPE'
LOOPS_COMMANDED = 'COMMANDED_UID'
LOOPS_STATUS = 'STATUS'
LOOPS_COMMANDS = 'COMMANDS'
LOOPS_REF_COMMANDS = 'REF_COMMANDS'
LOOPS_FRAMERATE = 'FRAMERATE'
LOOPS_DELAY = 'DELAY'
LOOPS_TIME_FILTER_NUM = 'TIME_FILTER_NUM'
LOOPS_TIME_FILTER_DEN = 'TIME_FILTER_DEN'

LOOPS_TYPE_CONTROL = 'Control Loop'
LOOPS_TYPE_OFFLOAD = 'Offload Loop'
LOOPS_TYPE_LIST = [LOOPS_TYPE_CONTROL, LOOPS_TYPE_OFFLOAD]
LOOPS_STATUS_CLOSED = 'Closed'
LOOPS_STATUS_OPEN = 'Open'
LOOPS_STATUS_LIST = [LOOPS_STATUS_CLOSED, LOOPS_STATUS_OPEN]

# AOT_LOOPS_CONTROL fields
LOOPS_CONTROL_INPUT_SENSOR = 'INPUT_SENSOR_UID'
LOOPS_CONTROL_MODES = 'MODES'
LOOPS_CONTROL_MODAL_COEFFICIENTS = 'MODAL_COEFFICIENTS'
LOOPS_CONTROL_CONTROL_MATRIX = 'CONTROL_MATRIX'
LOOPS_CONTROL_MEASUREMENTS_TO_MODES = 'MEASUREMENTS_TO_MODES'
LOOPS_CONTROL_MODES_TO_COMMANDS = 'MODES_TO_COMMANDS'
LOOPS_CONTROL_INTERACTION_MATRIX = 'INTERACTION_MATRIX'
LOOPS_CONTROL_COMMANDS_TO_MODES = 'COMMANDS_TO_MODES'
LOOPS_CONTROL_MODES_TO_MEASUREMENTS = 'MODES_TO_MEASUREMENTS'
LOOPS_CONTROL_RESIDUAL_COMMANDS = 'RESIDUAL_COMMANDS'

# AOT_LOOPS_OFFLOAD fields
LOOPS_OFFLOAD_INPUT_CORRECTOR = 'INPUT_CORRECTOR_UID'
LOOPS_OFFLOAD_OFFLOAD_MATRIX = 'OFFLOAD_MATRIX'