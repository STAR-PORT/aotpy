ATMOSPHERIC_PARAMETERS_TABLE = 'AOT_ATMOSPHERIC_PARAMETERS'
DETECTORS_TABLE = 'AOT_DETECTORS'
OPTICAL_RELAYS_TABLE = 'AOT_OPTICAL_RELAYS'
OPTICAL_ABERRATIONS_TABLE = 'AOT_OPTICAL_ABERRATIONS'
SCORING_CAMERAS_TABLE = 'AOT_SCORING_CAMERAS'
WAVEFRONT_SENSORS_TABLE = 'AOT_WAVEFRONT_SENSORS'
RTC_TABLE = 'AOT_RTC'
SOURCES_TABLE = 'AOT_SOURCES'
TELESCOPES_TABLE = 'AOT_TELESCOPES'
WAVEFRONT_CORRECTORS_TABLE = 'AOT_WAVEFRONT_CORRECTORS'
TABLE_SET = {'PRIMARY', ATMOSPHERIC_PARAMETERS_TABLE, DETECTORS_TABLE, OPTICAL_RELAYS_TABLE,
             OPTICAL_ABERRATIONS_TABLE, SCORING_CAMERAS_TABLE, WAVEFRONT_SENSORS_TABLE,
             RTC_TABLE, SOURCES_TABLE, TELESCOPES_TABLE, WAVEFRONT_CORRECTORS_TABLE}

# AOSystem keywords
AOT_VERSION = 'AOT-VERS'
AOT_LEVEL = 'TELEMLVL'
AOT_DATE = 'OBS-DATE'
AOT_EXPOSURE_START = 'EXPSTART'
AOT_EXPOSURE_END = 'EXPSTOP'
AOT_AO_MODE = 'AO-MODE'

# AtmosphericParameters keywords
ATMOSPHERIC_PARAMETERS_DATA_SOURCE = 'DATA_SOURCE'
ATMOSPHERIC_PARAMETERS_TIMESTAMP = 'TIMESTAMP'
ATMOSPHERIC_PARAMETERS_WAVELENGTH = 'WAVELENGTH'
ATMOSPHERIC_PARAMETERS_R0 = 'R0'
ATMOSPHERIC_PARAMETERS_L0 = 'L0'
ATMOSPHERIC_PARAMETERS_LAYER_WEIGHT = 'LAYER_WEIGHT'
ATMOSPHERIC_PARAMETERS_LAYER_HEIGHT = 'LAYER_HEIGHT'
ATMOSPHERIC_PARAMETERS_LAYER_WIND_SPEED = 'LAYER_WIND_SPEED'
ATMOSPHERIC_PARAMETERS_LAYER_WIND_DIRECTION = 'LAYER_WIND_DIRECTION'

# Detector keywords
DETECTOR_NAME = 'NAME'
DETECTOR_FLAT_FIELD = 'FLAT_FIELD'
DETECTOR_READOUT_NOISE = 'READOUT_NOISE'
DETECTOR_PIXEL_INTENSITIES = 'PIXEL_INTENSITIES'
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

# OpticalRelay keywords
OPTICAL_RELAY_NAME = 'NAME'
OPTICAL_RELAY_FIELD_OF_VIEW = 'FIELD_OF_VIEW'
OPTICAL_RELAY_FOCAL_LENGTH = 'FOCAL_LENGTH'

# OpticalAberration keywords
OPTICAL_ABERRATION_NAME = 'NAME'
OPTICAL_ABERRATION_COEFFICIENTS = 'COEFFICIENTS'
OPTICAL_ABERRATION_MODES = 'MODES'
OPTICAL_ABERRATION_PUPIL = 'PUPIL'

# ScoringCamera keywords
SCORING_CAMERA_NAME = 'NAME'
SCORING_CAMERA_PUPIL = 'PUPIL'
SCORING_CAMERA_THETA = 'THETA'
SCORING_CAMERA_WAVELENGTH = 'WAVELENGTH'
SCORING_CAMERA_FRAME = 'FRAME'
SCORING_CAMERA_FIELD_STATIC_MAP = 'FIELD_STATIC_MAP'
SCORING_CAMERA_X_STAT = 'X_STAT'
SCORING_CAMERA_Y_STAT = 'Y_STAT'
SCORING_CAMERA_DETECTOR_NAME = 'DETECTOR_NAME'
SCORING_CAMERA_OPTICAL_RELAY_NAME = 'OPTICAL_RELAY_NAME'
SCORING_CAMERA_OPTICAL_ABERRATION_NAME = 'OPTICAL_ABERRATION_NAME'

# WavefrontSensor keywords
WAVEFRONT_SENSOR_NAME = 'NAME'
WAVEFRONT_SENSOR_TYPE = 'TYPE'
WAVEFRONT_SENSOR_SOURCE_NAME = 'SOURCE_NAME'
WAVEFRONT_SENSOR_SLOPES = 'SLOPES'
WAVEFRONT_SENSOR_REF_SLOPES = 'REF_SLOPES'
WAVEFRONT_SENSOR_THETA = 'THETA'
WAVEFRONT_SENSOR_WAVELENGTH = 'WAVELENGTH'
WAVEFRONT_SENSOR_D_WAVELENGTH = 'D_WAVELENGTH'
WAVEFRONT_SENSOR_N_SUBAPERTURES = 'N_SUBAPERTURES'
WAVEFRONT_SENSOR_VALID_SUBAPERTURES = 'VALID_SUBAPERTURES'
WAVEFRONT_SENSOR_SUBAPERTURE_SIZE = 'SUBAPERTURE_SIZE'
WAVEFRONT_SENSOR_SUBAPERTURE_INTENSITIES = 'SUBAPERTURE_INTENSITIES'
WAVEFRONT_SENSOR_PUPIL_ANGLE = 'PUPIL_ANGLE'
WAVEFRONT_SENSOR_ALGORITHM = 'ALGORITHM'
WAVEFRONT_SENSOR_OPTICAL_GAIN = 'OPTICAL_GAIN'
WAVEFRONT_SENSOR_CENTROID_GAINS = 'CENTROID_GAINS'
WAVEFRONT_SENSOR_DETECTOR_NAME = 'DETECTOR_NAME'
WAVEFRONT_SENSOR_OPTICAL_RELAY_NAME = 'OPTICAL_RELAY_NAME'
WAVEFRONT_SENSOR_OPTICAL_ABERRATION_NAME = 'OPTICAL_ABERRATION_NAME'
WAVEFRONT_SENSOR_SPOT_FWHM = 'SPOT_FWHM'
WAVEFRONT_SENSOR_MODULATION = 'MODULATION'

WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN = 'Shack-Hartmann'
WAVEFRONT_SENSOR_TYPE_PYRAMID = 'Pyramid'

# RTC keywords
RTC_N_LOOKUP_TABLES = 'NLUT'
RTC_LOOKUP_TABLE = 'LUT'
RTC_N_NON_COMMON_PATH_ABERRATIONS = 'NNCPA'
RTC_NON_COMMON_PATH_ABERRATION= 'NCPA'

RTC_LOOPS_NAME = 'NAME'
RTC_LOOPS_TYPE = 'TYPE'
RTC_LOOPS_INPUT = 'INPUT_NAME'
RTC_LOOPS_COMMANDED_COR_NAME = 'COMMANDED_CORRECTOR_NAME'
RTC_LOOPS_COMMANDS = 'COMMANDS'
RTC_LOOPS_REF_COMMANDS = 'REF_COMMANDS'
RTC_LOOPS_TIMESTAMPS = 'TIMESTAMPS'
RTC_LOOPS_FRAMERATE = 'FRAMERATE'
RTC_LOOPS_DELAY = 'DELAY'
RTC_LOOPS_TIME_FILTER_NUM = 'TIME_FILTER_NUM'
RTC_LOOPS_TIME_FILTER_DEN = 'TIME_FILTER_DEN'
RTC_LOOPS_RESIDUAL_WAVEFRONT = 'RESIDUAL_WAVEFRONT'
RTC_LOOPS_CONTROL_MATRIX = 'CONTROL_MATRIX'
RTC_LOOPS_INTERACTION_MATRIX = 'INTERACTION_MATRIX'
RTC_LOOPS_OFFLOAD_MATRIX = 'OFFLOAD_MATRIX'

RTC_LOOPS_TYPE_CONTROL = 'Control Loop'
RTC_LOOPS_TYPE_OFFLOAD = 'Offload Loop'

# Source keywords
SOURCE_NAME = 'NAME'
SOURCE_TYPE = 'TYPE'
SOURCE_RIGHT_ASCENSION = 'RIGHT_ASCENSION'
SOURCE_DECLINATION = 'DECLINATION'
SOURCE_ZENITH_ANGLE = 'ZENITH_ANGLE'
SOURCE_AZIMUTH = 'AZIMUTH'
SOURCE_LASER_LAUNCH_TELESCOPE_NAME = 'LASER_LAUNCH_TELESCOPE_NAME'
SOURCE_SODIUM_HEIGHT = 'SODIUM_HEIGHT'
SOURCE_SODIUM_WIDTH = 'SODIUM_WIDTH'
SOURCE_SODIUM_PROFILE = 'SODIUM_PROFILE'
SOURCE_SODIUM_ALTITUDES = 'SODIUM_ALTITUDES'

SOURCE_TYPE_NGS = 'NGS'
SOURCE_TYPE_LGS = 'LGS'

# Telescope keywords
TELESCOPE_NAME = 'NAME'
TELESCOPE_TYPE = 'TYPE'
TELESCOPE_D_HEX = 'D_HEX'
TELESCOPE_D_CIRCLE = 'D_CIRCLE'
TELESCOPE_D_EQ = 'D_EQ'
TELESCOPE_COBS = 'COBS'
TELESCOPE_PUPIL = 'PUPIL'
TELESCOPE_ELEVATION = 'ELEVATION'
TELESCOPE_AZIMUTH = 'AZIMUTH'
TELESCOPE_PUPIL_ANGLE = 'PUPIL_ANGLE'
TELESCOPE_STATIC_MAP = 'STATIC_MAP'

TELESCOPE_TYPE_MAIN = 'MainTelescope'
TELESCOPE_TYPE_LLT = 'LLT'

# WavefrontCorrector keywords
WAVEFRONT_CORRECTOR_NAME = 'NAME'
WAVEFRONT_CORRECTOR_TYPE = 'TYPE'
WAVEFRONT_CORRECTOR_TELESCOPE_NAME = 'TELESCOPE_NAME'
WAVEFRONT_CORRECTOR_PITCH = 'PITCH'
WAVEFRONT_CORRECTOR_N_ACTUATORS = 'N_ACTUATORS'
WAVEFRONT_CORRECTOR_TFZ_NUM = 'TFZ_NUM'
WAVEFRONT_CORRECTOR_TFZ_DEN = 'TFZ_DEN'
WAVEFRONT_CORRECTOR_OPTICAL_ABERRATION_NAME = 'OPTICAL_ABERRATION_NAME'
WAVEFRONT_CORRECTOR_VALID_ACTUATORS = 'VALID_ACTUATORS'
WAVEFRONT_CORRECTOR_INFLUENCE_FUNCTION = 'INFLUENCE_FUNCTION'

WAVEFRONT_CORRECTOR_TYPE_TT = 'Tip-Tilt Mirror'
WAVEFRONT_CORRECTOR_TYPE_DM = 'Deformable Mirror'
WAVEFRONT_CORRECTOR_TYPE_LS = 'Linear Stage'
