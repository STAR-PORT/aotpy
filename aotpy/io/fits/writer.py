"""
This module contains classes and functions that enable writing AOT FITS files.
"""

import aotpy
from ._file import AOTFITSFile, AOTFITSInternalImage, AOTFITSExternalImage
from ._strings import TELESCOPE_TYPE_MAIN, TELESCOPE_TYPE_LLT, TELESCOPE_SEGMENT_TYPE_MONOLITHIC, \
    TELESCOPE_SEGMENT_TYPE_HEXAGON, TELESCOPE_SEGMENT_TYPE_CIRCLE, SOURCE_TYPE_SCIENCE_STAR, \
    SOURCE_TYPE_NATURAL_GUIDE_STAR, SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR, SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR, \
    WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN, WAVEFRONT_SENSOR_TYPE_PYRAMID, WAVEFRONT_CORRECTOR_TYPE_DM, \
    WAVEFRONT_CORRECTOR_TYPE_TTM, WAVEFRONT_CORRECTOR_TYPE_LS, LOOPS_TYPE_CONTROL, LOOPS_TYPE_OFFLOAD, \
    LOOPS_STATUS_CLOSED, LOOPS_STATUS_OPEN
from .compat import latest_version
from .utils import card_from_metadatum, FITSURLImage, FITSFileImage
from ..base import SystemWriter


# TODO all checks need to happen here
# check for columns of matching length
# check for image dimensions
class AOTFITSWriter(SystemWriter):
    def __init__(self, system: aotpy.AOSystem) -> None:
        self._file = AOTFITSFile()
        self._internalimages: dict[str, aotpy.Image] = {}
        self._times: dict[str, aotpy.Time] = {}
        self._aberrations: dict[str, aotpy.Aberration] = {}
        self._telescopes: dict[str, aotpy.Telescope] = {}
        self._sources: dict[str, aotpy.Source] = {}
        self._detectors: dict[str, aotpy.Detector] = {}
        self._wavefrontsensors: dict[str, aotpy.WavefrontSensor] = {}
        self._wavefrontcorrectors: dict[str, aotpy.WavefrontCorrector] = {}
        self._scoring_cameras: dict[str, aotpy.ScoringCamera] = {}
        self._loops: dict[str, aotpy.Loop] = {}
        self._atmosphericparameters: dict[str, aotpy.AtmosphericParameters] = {}
        self._sys: aotpy.AOSystem = system

        self._convert_header()
        for atm in self._sys.atmosphere_params:
            self._convert_atmospheric_parameters(atm)

        self._convert_telescope(self._sys.main_telescope, True)

        for src in self._sys.sources:
            self._convert_source(src, True)

        for cam in self._sys.scoring_cameras:
            self._convert_scoring_camera(cam)

        for wfs in self._sys.wavefront_sensors:
            self._convert_wavefront_sensor(wfs, True)

        for cor in self._sys.wavefront_correctors:
            self._convert_wavefront_corrector(cor, True)

        for loop in self._sys.loops:
            self._convert_loop(loop)

        self._file.verify_all_table_contents()

    def write(self, filename, **kwargs) -> None:
        self._file.to_file(filename, **kwargs)

    def get_hdulist(self):
        """
        Get the HDUList that produces the AOT FITS file for the initialized `system`.
        """
        return self._file.to_hdulist()

    def _handle_reference(self, name: str, d: dict, obj: aotpy.Referenceable) -> bool:
        if obj.uid in d:
            if d[obj.uid] is obj:
                # Same object, already added
                return True
            raise ValueError(f"Different {name} objects share the same UID '{obj.uid}'.")
        d[obj.uid] = obj
        return False

    def _convert_header(self):
        hdr = self._file.primary_header
        hdr.version = latest_version
        hdr.ao_mode = self._sys.ao_mode

        hdr.date_beg = self._sys.date_beginning
        hdr.date_end = self._sys.date_end
        hdr.system_name = self._sys.name
        hdr.strehl_ratio = self._sys.strehl_ratio
        hdr.strehl_wavelength = self._sys.strehl_wavelength
        hdr.config = self._sys.config

        hdr.metadata = [card_from_metadatum(md) for md in self._sys.metadata]

    def _convert_time(self, time: aotpy.Time) -> str | None:
        if time is None:
            return None
        if self._handle_reference('Time', self._times, time):
            return time.uid

        if time.timestamps and time.frame_numbers and len(time.timestamps) != len(time.frame_numbers):
            raise ValueError(f"Error in Time '{time.uid}': If both 'timestamps' and 'frame_numbers' are non-null, they "
                             f"must have the same length.")

        tbl = self._file.time_table
        tbl.uid.append(time.uid)
        tbl.timestamps.append(time.timestamps)
        tbl.frame_numbers.append(time.frame_numbers)
        return time.uid

    def _convert_atmospheric_parameters(self, atm: aotpy.AtmosphericParameters) -> None:
        if atm is None:
            raise ValueError("'AOSystem.atmosphere_params' list cannot contain 'None' items.")
        if self._handle_reference('AtmpshericParameters', self._atmosphericparameters, atm):
            raise ValueError("'AOSystem.atmosphere_params' list cannot contain repeated items.")
        tbl = self._file.atmospheric_parameters_table
        tbl.uid.append(atm.uid)
        tbl.wavelength.append(atm.wavelength)
        tbl.time_uid.append(self._convert_time(atm.time))
        tbl.r0.append(atm.r0)
        tbl.seeing.append(atm.seeing)
        tbl.tau0.append(atm.tau0)
        tbl.theta0.append(atm.theta0)
        tbl.layers_rel_weight.append(self._convert_image(atm.layers_relative_weight))
        tbl.layers_height.append(self._convert_image(atm.layers_height))
        tbl.layers_l0.append(self._convert_image(atm.layers_l0))
        tbl.layers_wind_speed.append(self._convert_image(atm.layers_wind_speed))
        tbl.layers_wind_direction.append(self._convert_image(atm.layers_wind_direction))
        tbl.transformation_matrix.append(self._convert_image(atm.transformation_matrix))

    def _convert_aberration(self, abr: aotpy.Aberration) -> str | None:
        if abr is None:
            return None
        if self._handle_reference('Aberration', self._aberrations, abr):
            return abr.uid

        tbl = self._file.aberrations_table
        tbl.uid.append(abr.uid)
        tbl.modes.append(self._convert_image(abr.modes))
        tbl.coefficients.append(self._convert_image(abr.coefficients))
        tbl.x_offsets.append([off.x for off in abr.offsets])
        tbl.y_offsets.append([off.y for off in abr.offsets])
        return abr.uid

    def _convert_telescope(self, tel: aotpy.Telescope, enforce: bool = False, llt: bool = False) -> str | None:
        if tel is None:
            if enforce:
                raise ValueError("'AOSystem.main_telescope' must not be 'None'.")
            return None
        if self._handle_reference('Telescope', self._telescopes, tel):
            return tel.uid

        if isinstance(tel, aotpy.MainTelescope):
            tel_type = TELESCOPE_TYPE_MAIN
            if llt:
                raise ValueError("Referenced laser launch telescope must be of type 'aotpy.LaserLaunchTelescope'.")
            elif tel is not self._sys.main_telescope:
                raise ValueError("Telescope references cannot reference a 'MainTelescope' object that"
                                 " is not AOSystem.main_telescope.")
        elif isinstance(tel, aotpy.LaserLaunchTelescope):
            tel_type = TELESCOPE_TYPE_LLT
            if enforce:
                raise ValueError("'AOSystem.main_telescope' must be of type 'aotpy.MainTelescope'.")
        else:
            raise ValueError(f"Unexpected type '{type(tel)}' for Telescope object.")

        if tel.segments is None:
            raise ValueError("'aotpy.Telescope.segments' cannot be 'None'.")

        if isinstance(tel.segments, aotpy.Monolithic):
            segments_type = TELESCOPE_SEGMENT_TYPE_MONOLITHIC
        elif isinstance(tel.segments, aotpy.HexagonalSegments):
            segments_type = TELESCOPE_SEGMENT_TYPE_HEXAGON
        elif isinstance(tel.segments, aotpy.CircularSegments):
            segments_type = TELESCOPE_SEGMENT_TYPE_CIRCLE
        else:
            raise ValueError(f"Unexpected type '{type(tel.segments)}' for Segments object.")

        tbl = self._file.telescopes_table
        tbl.uid.append(tel.uid)
        tbl.type.append(tel_type)
        tbl.latitude.append(tel.latitude)
        tbl.longitude.append(tel.longitude)
        tbl.elevation.append(tel.elevation)
        tbl.azimuth.append(tel.azimuth)
        tbl.parallactic.append(tel.parallactic)
        tbl.pupil_mask.append(self._convert_image(tel.pupil_mask))
        tbl.pupil_angle.append(tel.pupil_angle)
        tbl.enclosing_d.append(tel.enclosing_diameter)
        tbl.inscribed_d.append(tel.inscribed_diameter)
        tbl.obstruction_d.append(tel.obstruction_diameter)
        tbl.segment_type.append(segments_type)
        tbl.segment_size.append(tel.segments.size)
        tbl.segments_x.append([coord.x for coord in tel.segments.coordinates])
        tbl.segments_y.append([coord.y for coord in tel.segments.coordinates])
        tbl.transformation_matrix.append(self._convert_image(tel.transformation_matrix))
        return tel.uid

    def _convert_source(self, src: aotpy.Source, enforce: bool = False) -> str | None:
        if src is None:
            if enforce:
                raise ValueError("'AOSystem.sources' list cannot contain 'None' items.")
            return None
        if self._handle_reference('Source', self._sources, src):
            if enforce:
                raise ValueError("'AOSystem.sources' list cannot contain repeated items.")
            return src.uid
        elif not enforce:
            raise ValueError(f"Source '{src.uid}' was referenced but is not on the 'AOSystem.sources' list.")

        if isinstance(src, aotpy.ScienceStar):
            src_type = SOURCE_TYPE_SCIENCE_STAR
        elif isinstance(src, aotpy.NaturalGuideStar):
            src_type = SOURCE_TYPE_NATURAL_GUIDE_STAR
        elif isinstance(src, aotpy.SodiumLaserGuideStar):
            src_type = SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR
            sec_tbl = self._file.sources_sodium_lgs_table
            sec_tbl.uid.append(src.uid)
            sec_tbl.height.append(src.height)
            sec_tbl.profile.append(self._convert_image(src.profile))
            sec_tbl.altitudes.append(src.altitudes)
            sec_tbl.llt_uid.append(self._convert_telescope(src.laser_launch_telescope, llt=True))
        elif isinstance(src, aotpy.RayleighLaserGuideStar):
            src_type = SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR
            sec_tbl = self._file.sources_rayleigh_lgs_table
            sec_tbl.uid.append(src.uid)
            sec_tbl.distance.append(src.distance)
            sec_tbl.depth.append(src.depth)
            sec_tbl.llt_uid.append(self._convert_telescope(src.laser_launch_telescope, llt=True))
        else:
            raise ValueError(f"Unexpected type '{type(src)}' for Source object.")

        tbl = self._file.sources_table
        tbl.uid.append(src.uid)
        tbl.type.append(src_type)
        tbl.right_ascension.append(src.right_ascension)
        tbl.declination.append(src.declination)
        tbl.elevation_offset.append(src.elevation_offset)
        tbl.azimuth_offset.append(src.azimuth_offset)
        tbl.fwhm.append(src.fwhm)
        return src.uid

    def _convert_detector(self, det: aotpy.Detector) -> str | None:
        if det is None:
            return None
        if self._handle_reference('Detector', self._detectors, det):
            return det.uid

        tbl = self._file.detectors_table
        tbl.uid.append(det.uid)
        tbl.type.append(det.type)
        tbl.sampling_technique.append(det.sampling_technique)
        tbl.shutter_type.append(det.shutter_type)
        tbl.flat_field.append(self._convert_image(det.flat_field))
        tbl.readout_noise.append(det.readout_noise)
        tbl.pixel_intensities.append(self._convert_image(det.pixel_intensities))
        tbl.field_centre_x.append(det.field_centre.x)
        tbl.field_centre_y.append(det.field_centre.y)
        tbl.integration_time.append(det.integration_time)
        tbl.coadds.append(det.coadds)
        tbl.dark.append(self._convert_image(det.dark))
        tbl.weight_map.append(self._convert_image(det.weight_map))
        tbl.quantum_efficiency.append(det.quantum_efficiency)
        tbl.pixel_scale.append(det.pixel_scale)
        tbl.binning.append(det.binning)
        tbl.bandwidth.append(det.bandwidth)
        tbl.transmission_wavelength.append(det.transmission_wavelength)
        tbl.transmission.append(det.transmission)
        tbl.sky_background.append(self._convert_image(det.sky_background))
        tbl.gain.append(det.gain)
        tbl.excess_noise.append(det.excess_noise)
        tbl.filter.append(det.filter)
        tbl.bad_pixel_map.append(self._convert_image(det.bad_pixel_map))
        tbl.dynamic_range.append(det.dynamic_range)
        tbl.readout_rate.append(det.readout_rate)
        tbl.frame_rate.append(det.frame_rate)
        tbl.transformation_matrix.append(self._convert_image(det.transformation_matrix))
        return det.uid

    def _convert_scoring_camera(self, cam: aotpy.ScoringCamera) -> None:
        if cam is None:
            raise ValueError("'AOSystem.scoring_cameras' list cannot contain 'None' items.")
        if self._handle_reference('ScoringCamera', self._scoring_cameras, cam):
            raise ValueError("'AOSystem.scoring_cameras' list cannot contain repeated items.")

        tbl = self._file.scoring_cameras_table
        tbl.uid.append(cam.uid)
        tbl.pupil_mask.append(self._convert_image(cam.pupil_mask))
        tbl.wavelength.append(cam.wavelength)
        tbl.transformation_matrix.append(self._convert_image(cam.transformation_matrix))
        tbl.detector_uid.append(self._convert_detector(cam.detector))
        tbl.aberration_uid.append(self._convert_aberration(cam.aberration))

    def _convert_wavefront_sensor(self, wfs: aotpy.WavefrontSensor, enforce: bool = False):
        if wfs is None:
            if enforce:
                raise ValueError("'AOSystem.wavefront_sensors' list cannot contain 'None' items.")
            return None
        if self._handle_reference('WavefrontSensor', self._wavefrontsensors, wfs):
            if enforce:
                raise ValueError("'AOSystem.wavefront_sensors' list cannot contain repeated items.")
            return wfs.uid
        elif not enforce:
            raise ValueError(f"WavefrontSensor '{wfs.uid}' was referenced but is not on the "
                             f"'AOSystem.wavefront_sensors' list.")

        if isinstance(wfs, aotpy.ShackHartmann):
            wfs_type = WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN
            sec_tbl = self._file.wavefront_sensors_shack_hartmann_table
            sec_tbl.uid.append(wfs.uid)
            sec_tbl.centroiding_algorithm.append(wfs.centroiding_algorithm)
            sec_tbl.centroid_gains.append(self._convert_image(wfs.centroid_gains))
            sec_tbl.spot_fwhm.append(self._convert_image(wfs.spot_fwhm))
        elif isinstance(wfs, aotpy.Pyramid):
            wfs_type = WAVEFRONT_SENSOR_TYPE_PYRAMID
            sec_tbl = self._file.wavefront_sensors_pyramid_table
            sec_tbl.uid.append(wfs.uid)
            sec_tbl.n_sides.append(wfs.n_sides)
            sec_tbl.modulation.append(wfs.modulation)
        else:
            raise ValueError(f"Unexpected type '{type(wfs)}' for WavefrontSensor object.")

        tbl = self._file.wavefront_sensors_table
        tbl.uid.append(wfs.uid)
        tbl.type.append(wfs_type)
        tbl.source_uid.append(self._convert_source(wfs.source))
        tbl.dimensions.append(wfs.dimensions)
        tbl.n_valid_subapertures.append(wfs.n_valid_subapertures)
        tbl.measurements.append(self._convert_image(wfs.measurements))
        tbl.ref_measurements.append(self._convert_image(wfs.ref_measurements))
        tbl.subaperture_mask.append(self._convert_image(wfs.subaperture_mask))
        tbl.mask_x_offsets.append([off.x for off in wfs.mask_offsets])
        tbl.mask_y_offsets.append([off.y for off in wfs.mask_offsets])
        tbl.subaperture_size.append(wfs.subaperture_size)
        tbl.subaperture_intensities.append(self._convert_image(wfs.subaperture_intensities))
        tbl.wavelength.append(wfs.wavelength)
        tbl.optical_gain.append(self._convert_image(wfs.optical_gain))
        tbl.transformation_matrix.append(self._convert_image(wfs.transformation_matrix))
        tbl.detector_uid.append(self._convert_detector(wfs.detector))
        tbl.aberration_uid.append(self._convert_aberration(wfs.aberration))
        tbl.ncpa_uid.append(self._convert_aberration(wfs.non_common_path_aberration))
        return wfs.uid

    def _convert_wavefront_corrector(self, wfc: aotpy.WavefrontCorrector, enforce: bool = False):
        if wfc is None:
            if enforce:
                raise ValueError("'AOSystem.wavefront_correctors' list cannot contain 'None' items.")
            return None
        if self._handle_reference('WavefrontCorrector', self._wavefrontcorrectors, wfc):
            if enforce:
                raise ValueError("'AOSystem.wavefront_correctors' list cannot contain repeated items.")
            return wfc.uid
        elif not enforce:
            raise ValueError(f"WavefrontCorrector '{wfc.uid}' was referenced but is not on the "
                             f"'AOSystem.wavefront_correctors' list.")

        if isinstance(wfc, aotpy.DeformableMirror):
            wfc_type = WAVEFRONT_CORRECTOR_TYPE_DM
            sec_tbl = self._file.wavefront_correctors_dm_table
            sec_tbl.uid.append(wfc.uid)
            sec_tbl.actuators_x.append([coord.x for coord in wfc.actuator_coordinates])
            sec_tbl.actuators_y.append([coord.y for coord in wfc.actuator_coordinates])
            sec_tbl.influence_function.append(self._convert_image(wfc.influence_function))
            sec_tbl.stroke.append(wfc.stroke)
        elif isinstance(wfc, aotpy.TipTiltMirror):
            wfc_type = WAVEFRONT_CORRECTOR_TYPE_TTM
        elif isinstance(wfc, aotpy.LinearStage):
            wfc_type = WAVEFRONT_CORRECTOR_TYPE_LS
        else:
            raise ValueError(f"Unexpected type '{type(wfc)}' for WavefrontCorrector object.")

        tbl = self._file.wavefront_correctors_table
        tbl.uid.append(wfc.uid)
        tbl.type.append(wfc_type)
        tbl.telescope_uid.append(self._convert_telescope(wfc.telescope))
        tbl.n_valid_actuators.append(wfc.n_valid_actuators)
        tbl.pupil_mask.append(self._convert_image(wfc.pupil_mask))
        tbl.tfz_num.append(wfc.tfz_num)
        tbl.tfz_den.append(wfc.tfz_den)
        tbl.transformation_matrix.append(self._convert_image(wfc.transformation_matrix))
        tbl.aberration_uid.append(self._convert_aberration(wfc.aberration))
        return wfc.uid

    def _convert_loop(self, loop: aotpy.Loop) -> None:
        if loop is None:
            raise ValueError("'AOSystem.loops' list cannot contain 'None' items.")
        if self._handle_reference('Loop', self._loops, loop):
            raise ValueError("'AOSystem.loops' list cannot contain repeated items.")

        if isinstance(loop, aotpy.ControlLoop):
            loop_type = LOOPS_TYPE_CONTROL
            sec_tbl = self._file.loops_control_table
            sec_tbl.uid.append(loop.uid)
            sec_tbl.input_sensor_uid.append(self._convert_wavefront_sensor(loop.input_sensor))
            sec_tbl.modes.append(self._convert_image(loop.modes))
            sec_tbl.modal_coefficients.append(self._convert_image(loop.modal_coefficients))
            sec_tbl.control_matrix.append(self._convert_image(loop.control_matrix))
            sec_tbl.measurements_to_modes.append(self._convert_image(loop.measurements_to_modes))
            sec_tbl.modes_to_commands.append(self._convert_image(loop.modes_to_commands))
            sec_tbl.interaction_matrix.append(self._convert_image(loop.interaction_matrix))
            sec_tbl.commands_to_modes.append(self._convert_image(loop.commands_to_modes))
            sec_tbl.modes_to_measurements.append(self._convert_image(loop.modes_to_measurements))
            sec_tbl.residual_commands.append(self._convert_image(loop.residual_commands))
        elif isinstance(loop, aotpy.OffloadLoop):
            loop_type = LOOPS_TYPE_OFFLOAD
            sec_tbl = self._file.loops_offload_table
            sec_tbl.uid.append(loop.uid)
            sec_tbl.input_corrector_uid.append(self._convert_wavefront_corrector(loop.input_corrector))
            sec_tbl.offload_matrix.append(self._convert_image(loop.offload_matrix))
        else:
            raise ValueError(f"Unexpected type '{type(loop)}' for Loop object.")

        tbl = self._file.loops_table
        tbl.uid.append(loop.uid)
        tbl.type.append(loop_type)
        tbl.commanded_uid.append(self._convert_wavefront_corrector(loop.commanded_corrector))
        tbl.time_uid.append(self._convert_time(loop.time))
        tbl.status.append(LOOPS_STATUS_CLOSED if loop.closed else LOOPS_STATUS_OPEN)
        tbl.commands.append(self._convert_image(loop.commands))
        tbl.ref_commands.append(self._convert_image(loop.ref_commands))
        tbl.framerate.append(loop.framerate)
        tbl.delay.append(loop.delay)
        tbl.time_filter_num.append(self._convert_image(loop.time_filter_num))
        tbl.time_filter_den.append(self._convert_image(loop.time_filter_den))

    def _convert_image(self, img: aotpy.Image) -> AOTFITSInternalImage | AOTFITSExternalImage | None:
        if img is None:
            return None
        time_uid = self._convert_time(img.time)
        if isinstance(img, FITSURLImage):
            im = AOTFITSExternalImage(img.name, img.data, img.unit, None,
                                      [card_from_metadatum(md) for md in img.metadata], True, img.url, img.index)
        elif isinstance(img, FITSFileImage):
            im = AOTFITSExternalImage(img.name, img.data, img.unit, None,
                                      [card_from_metadatum(md) for md in img.metadata], False, img.filename, img.index)
        elif isinstance(img, aotpy.Image):
            if img.name in self._internalimages:
                if self._internalimages[img.name] is img:
                    # Same object, already added
                    return self._file.internal_images[img.name]
                raise ValueError(f"Different Image objects share the same name '{img.name}'.")
            self._internalimages[img.name] = img
            im = AOTFITSInternalImage(img.name, img.data, img.unit, None,
                                      [card_from_metadatum(md) for md in img.metadata])
            self._file.internal_images[img.name] = im
        else:
            raise ValueError(f"Unexpected type '{type(img)}' for Image object.")
        im.time_uid = time_uid
        return im
