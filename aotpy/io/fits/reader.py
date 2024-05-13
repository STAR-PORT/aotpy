import aotpy
from . import _strings as kw
from ._file import AOTFITSFile, AOTFITSImage, AOTFITSInternalImage, AOTFITSExternalImage
from .utils import metadatum_from_card, FITSURLImage, FITSFileImage
from ..base import SystemReader


# TODO preferably there would be no dependency on FITS keywords
class AOTFITSReader(SystemReader):
    def __init__(self, filename, **kwargs):
        self._internalimages: dict[str, aotpy.Image] = {}
        self._times: dict[str, aotpy.Time] = {}
        self._aberrations: dict[str, aotpy.Aberration] = {}
        self._telescopes: dict[str, aotpy.Telescope] = {}
        self._sources: dict[str, aotpy.Source] = {}
        self._detectors: dict[str, aotpy.Detector] = {}
        self._wavefrontsensors: dict[str, aotpy.WavefrontSensor] = {}
        self._wavefrontcorrectors: dict[str, aotpy.WavefrontCorrector] = {}

        self._file: AOTFITSFile = AOTFITSFile.from_file(filename, **kwargs)
        self._system = self._system_from_file()

    def get_system(self) -> aotpy.AOSystem:
        return self._system

    def get_extra_data(self):
        """
         Return a tuple of extra data that may have been in AOTFITS file.
         """
        return self._file.get_extra_data()

    def _system_from_file(self):
        hdr = self._file.primary_header
        sys = aotpy.AOSystem(
            ao_mode=hdr.ao_mode,
            date_beginning=hdr.date_beg,
            date_end=hdr.date_end,
            name=hdr.system_name,
            strehl_ratio=hdr.strehl_ratio,
            strehl_wavelength=hdr.strehl_wavelength,
            config=hdr.config,
            metadata=[metadatum_from_card(card) for card in hdr.metadata]
        )

        sys.atmosphere_params = [self._atmospheric_parameters_from_uid(uid)
                                 for uid in self._file.atmospheric_parameters_table.uid_dict]
        sys.main_telescope = self._telescope_from_uid(self._file.telescopes_table.main_telescope_uid)
        sys.sources = [self._source_from_uid(uid) for uid in self._file.sources_table.uid_dict]
        sys.scoring_cameras = [self._scoring_camera_from_uid(uid) for uid in self._file.scoring_cameras_table.uid_dict]
        sys.wavefront_sensors = [self._wavefront_sensor_from_uid(uid)
                                 for uid in self._file.wavefront_sensors_table.uid_dict]
        sys.wavefront_correctors = [self._wavefront_corrector_from_uid(uid)
                                    for uid in self._file.wavefront_correctors_table.uid_dict]
        sys.loops = [self._loop_from_uid(uid) for uid in self._file.loops_table.uid_dict]

        return sys

    def _atmospheric_parameters_from_uid(self, uid) -> aotpy.AtmosphericParameters:
        tbl = self._file.atmospheric_parameters_table
        i = tbl.uid_dict[uid]

        return aotpy.AtmosphericParameters(
            uid=uid,
            wavelength=tbl.wavelength[i],
            time=self._time_from_uid(tbl.time_uid[i]),
            r0=tbl.r0[i],
            seeing=tbl.seeing[i],
            tau0=tbl.tau0[i],
            theta0=tbl.theta0[i],
            layers_relative_weight=self._image_from_reference(tbl.layers_rel_weight[i]),
            layers_height=self._image_from_reference(tbl.layers_height[i]),
            layers_l0=self._image_from_reference(tbl.layers_l0[i]),
            layers_wind_speed=self._image_from_reference(tbl.layers_wind_speed[i]),
            layers_wind_direction=self._image_from_reference(tbl.layers_wind_direction[i]),
            transformation_matrix=self._image_from_reference(tbl.transformation_matrix[i])
        )

    def _telescope_from_uid(self, uid: str) -> aotpy.Telescope | None:
        if uid is None:
            return None
        if uid in self._telescopes:
            return self._telescopes[uid]

        tbl = self._file.telescopes_table
        i = tbl.uid_dict[uid]
        t = tbl.type[i]

        if t == kw.TELESCOPE_TYPE_MAIN:
            tel = aotpy.MainTelescope(uid)
        elif t == kw.TELESCOPE_TYPE_LLT:
            tel = aotpy.LaserLaunchTelescope(uid)
        else:
            # This should never happen
            raise NotImplementedError

        tel.latitude = tbl.latitude[i]
        tel.longitude = tbl.longitude[i]
        tel.elevation = tbl.elevation[i]
        tel.azimuth = tbl.azimuth[i]
        tel.parallactic = tbl.parallactic[i]
        tel.pupil_mask = self._image_from_reference(tbl.pupil_mask[i])
        tel.pupil_angle = tbl.pupil_angle[i]
        tel.enclosing_diameter = tbl.enclosing_d[i]
        tel.inscribed_diameter = tbl.inscribed_d[i]
        tel.obstruction_diameter = tbl.obstruction_d[i]

        seg_type = tbl.segment_type[i]
        if seg_type == kw.TELESCOPE_SEGMENT_TYPE_MONOLITHIC:
            seg = aotpy.Monolithic()
        else:
            if t == kw.TELESCOPE_SEGMENT_TYPE_CIRCLE:
                seg = aotpy.CircularSegments()
            elif t == kw.TELESCOPE_SEGMENT_TYPE_HEXAGON:
                seg = aotpy.HexagonalSegments()
            else:
                # This should never happen
                raise NotImplementedError
            seg.size = tbl.segment_size[i]
            seg.coordinates = [aotpy.Coordinates(x, y) for x, y in zip(tbl.segments_x[i], tbl.segments_y[i])]
        tel.segments = seg
        tel.transformation_matrix = self._image_from_reference(tbl.transformation_matrix[i])
        tel.aberration = self._aberration_from_uid(tbl.aberration_uid[i])

        self._telescopes[uid] = tel
        return tel

    def _source_from_uid(self, uid) -> aotpy.Source | None:
        if uid is None:
            return None

        if uid in self._sources:
            return self._sources[uid]

        tbl = self._file.sources_table
        i = tbl.uid_dict[uid]
        t = tbl.type[i]

        if t == kw.SOURCE_TYPE_SCIENCE_STAR:
            src = aotpy.ScienceStar(uid)
        elif t == kw.SOURCE_TYPE_NATURAL_GUIDE_STAR:
            src = aotpy.NaturalGuideStar(uid)
        elif t == kw.SOURCE_TYPE_SODIUM_LASER_GUIDE_STAR:
            sec_tbl = self._file.sources_sodium_lgs_table
            sec_i = sec_tbl.uid_dict[uid]
            src = aotpy.SodiumLaserGuideStar(
                uid=uid,
                height=sec_tbl.height[sec_i],
                profile=self._image_from_reference(sec_tbl.profile[sec_i]),
                altitudes=sec_tbl.altitudes[sec_i],
                laser_launch_telescope=self._telescope_from_uid(sec_tbl.llt_uid[sec_i])
            )
        elif t == kw.SOURCE_TYPE_RAYLEIGH_LASER_GUIDE_STAR:
            sec_tbl = self._file.sources_rayleigh_lgs_table
            sec_i = sec_tbl.uid_dict[uid]

            src = aotpy.RayleighLaserGuideStar(
                uid=uid,
                distance=sec_tbl.distance[sec_i],
                depth=sec_tbl.depth[sec_i],
                laser_launch_telescope=self._telescope_from_uid(sec_tbl.llt_uid[sec_i])
            )
        else:
            # This should never happen
            raise NotImplementedError

        src.right_ascension = tbl.right_ascension[i]
        src.declination = tbl.declination[i]
        src.elevation_offset = tbl.elevation_offset[i]
        src.azimuth_offset = tbl.azimuth_offset[i]
        src.fwhm = tbl.fwhm[i]

        self._sources[uid] = src
        return src

    def _scoring_camera_from_uid(self, uid) -> aotpy.ScoringCamera:
        tbl = self._file.scoring_cameras_table
        i = tbl.uid_dict[uid]

        return aotpy.ScoringCamera(
            uid=uid,
            pupil_mask=self._image_from_reference(tbl.pupil_mask[i]),
            wavelength=tbl.wavelength[i],
            transformation_matrix=self._image_from_reference(tbl.transformation_matrix[i]),
            detector=self._detector_from_uid(tbl.detector_uid[i]),
            aberration=self._aberration_from_uid(tbl.aberration_uid[i])
        )

    def _wavefront_sensor_from_uid(self, uid) -> aotpy.WavefrontSensor | None:
        if uid is None:
            return None

        if uid in self._wavefrontsensors:
            return self._wavefrontsensors[uid]

        tbl = self._file.wavefront_sensors_table
        i = tbl.uid_dict[uid]
        t = tbl.type[i]

        source = self._source_from_uid(tbl.source_uid[i])
        dimensions = tbl.dimensions[i]
        n_valid_subapertures = tbl.n_valid_subapertures[i]

        if t == kw.WAVEFRONT_SENSOR_TYPE_SHACK_HARTMANN:
            sec_tbl = self._file.wavefront_sensors_shack_hartmann_table
            sec_i = sec_tbl.uid_dict[uid]
            wfs = aotpy.ShackHartmann(
                uid=uid,
                source=source,
                n_valid_subapertures=n_valid_subapertures,
                centroiding_algorithm=sec_tbl.centroiding_algorithm[sec_i],
                centroid_gains=self._image_from_reference(sec_tbl.centroid_gains[sec_i]),
                spot_fwhm=self._image_from_reference(sec_tbl.spot_fwhm[sec_i])
            )
        elif t == kw.WAVEFRONT_SENSOR_TYPE_PYRAMID:
            sec_tbl = self._file.wavefront_sensors_pyramid_table
            sec_i = sec_tbl.uid_dict[uid]
            wfs = aotpy.Pyramid(
                uid=uid,
                source=source,
                dimensions=dimensions,
                n_valid_subapertures=n_valid_subapertures,
                n_sides=sec_tbl.n_sides[sec_i],
                modulation=sec_tbl.modulation[sec_i]
            )
        else:
            # This should never happen
            raise NotImplementedError

        wfs.measurements = self._image_from_reference(tbl.measurements[i])
        wfs.ref_measurements = self._image_from_reference(tbl.ref_measurements[i])
        wfs.subaperture_mask = self._image_from_reference(tbl.subaperture_mask[i])
        wfs.mask_offsets = [aotpy.Coordinates(x, y) for x, y in zip(tbl.mask_x_offsets[i], tbl.mask_y_offsets[i])]
        wfs.subaperture_size = tbl.subaperture_size[i]
        wfs.subaperture_intensities = self._image_from_reference(tbl.subaperture_intensities[i])
        wfs.wavelength = tbl.wavelength[i]
        wfs.optical_gain = self._image_from_reference(tbl.optical_gain[i])
        wfs.transformation_matrix = self._image_from_reference(tbl.transformation_matrix[i])
        wfs.detector = self._detector_from_uid(tbl.detector_uid[i])
        wfs.aberration = self._aberration_from_uid(tbl.aberration_uid[i])
        wfs.non_common_path_aberration = self._aberration_from_uid(tbl.ncpa_uid[i])

        self._wavefrontsensors[uid] = wfs
        return wfs

    def _wavefront_corrector_from_uid(self, uid) -> aotpy.WavefrontCorrector | None:
        if uid is None:
            return None

        if uid in self._wavefrontcorrectors:
            return self._wavefrontcorrectors[uid]

        tbl = self._file.wavefront_correctors_table
        i = tbl.uid_dict[uid]
        t = tbl.type[i]
        telescope = self._telescope_from_uid(tbl.telescope_uid[i])

        if t == kw.WAVEFRONT_CORRECTOR_TYPE_DM:
            sec_tbl = self._file.wavefront_correctors_dm_table
            sec_i = sec_tbl.uid_dict[uid]
            cor = aotpy.DeformableMirror(
                uid=uid,
                telescope=telescope,
                n_valid_actuators=tbl.n_valid_actuators[i],
                actuator_coordinates=[aotpy.Coordinates(x, y) for x, y in zip(sec_tbl.actuators_x[sec_i],
                                                                              sec_tbl.actuators_y[sec_i])],
                influence_function=self._image_from_reference(sec_tbl.influence_function[sec_i]),
                stroke=sec_tbl.stroke[sec_i]
            )
        elif t == kw.WAVEFRONT_CORRECTOR_TYPE_TTM:
            cor = aotpy.TipTiltMirror(uid=uid, telescope=telescope)
        elif t == kw.WAVEFRONT_CORRECTOR_TYPE_LS:
            cor = aotpy.LinearStage(uid=uid, telescope=telescope)
        else:
            # This should never happen
            raise NotImplementedError

        cor.pupil_mask = self._image_from_reference(tbl.pupil_mask[i])
        cor.tfz_num = tbl.tfz_num[i]
        cor.tfz_den = tbl.tfz_den[i]
        cor.transformation_matrix = self._image_from_reference(tbl.transformation_matrix[i])
        cor.aberration = self._aberration_from_uid(tbl.aberration_uid[i])

        self._wavefrontcorrectors[uid] = cor
        return cor

    def _loop_from_uid(self, uid) -> aotpy.Loop:
        tbl = self._file.loops_table
        i = tbl.uid_dict[uid]

        t = tbl.type[i]
        commanded = self._wavefront_corrector_from_uid(tbl.commanded_uid[i])

        if t == kw.LOOPS_TYPE_CONTROL:
            sec_tbl = self._file.loops_control_table
            sec_i = sec_tbl.uid_dict[uid]
            loop = aotpy.ControlLoop(
                uid=uid,
                commanded_corrector=commanded,
                input_sensor=self._wavefront_sensor_from_uid(sec_tbl.input_sensor_uid[sec_i]),
                modes=self._image_from_reference(sec_tbl.modes[sec_i]),
                modal_coefficients=self._image_from_reference(sec_tbl.modal_coefficients[sec_i]),
                control_matrix=self._image_from_reference(sec_tbl.control_matrix[sec_i]),
                measurements_to_modes=self._image_from_reference(sec_tbl.measurements_to_modes[sec_i]),
                modes_to_commands=self._image_from_reference(sec_tbl.modes_to_commands[sec_i]),
                interaction_matrix=self._image_from_reference(sec_tbl.interaction_matrix[sec_i]),
                commands_to_modes=self._image_from_reference(sec_tbl.commands_to_modes[sec_i]),
                modes_to_measurements=self._image_from_reference(sec_tbl.modes_to_measurements[sec_i]),
                residual_commands=self._image_from_reference(sec_tbl.residual_commands[sec_i]),
            )
        elif t == kw.LOOPS_TYPE_OFFLOAD:
            sec_tbl = self._file.loops_offload_table
            sec_i = sec_tbl.uid_dict[uid]
            loop = aotpy.OffloadLoop(
                uid=uid,
                commanded_corrector=commanded,
                input_corrector=self._wavefront_corrector_from_uid(sec_tbl.input_corrector_uid[sec_i]),
                offload_matrix=self._image_from_reference(sec_tbl.offload_matrix[sec_i])
            )
        else:
            # This should never happen
            raise NotImplementedError

        loop.time = self._time_from_uid(tbl.time_uid[i])

        if (status := tbl.status[i]) is None:
            loop.closed = None
        elif status == kw.LOOPS_STATUS_CLOSED:
            loop.closed = True
        elif status == kw.LOOPS_STATUS_OPEN:
            loop.closed = False
        else:
            # This should never happen
            raise NotImplementedError

        loop.commands = self._image_from_reference(tbl.commands[i])
        loop.ref_commands = self._image_from_reference(tbl.ref_commands[i])
        loop.framerate = tbl.framerate[i]
        loop.delay = tbl.delay[i]
        loop.time_filter_num = self._image_from_reference(tbl.time_filter_num[i])
        loop.time_filter_den = self._image_from_reference(tbl.time_filter_den[i])

        return loop

    def _time_from_uid(self, uid: str) -> aotpy.Time | None:
        if uid is None:
            return None

        if uid in self._times:
            return self._times[uid]

        tbl = self._file.time_table
        i = tbl.uid_dict[uid]
        t = aotpy.Time(uid=uid, timestamps=tbl.timestamps[i], frame_numbers=tbl.frame_numbers[i])
        self._times[uid] = t
        return t

    def _aberration_from_uid(self, uid: str) -> aotpy.Aberration | None:
        if uid is None:
            return None

        if uid in self._aberrations:
            return self._aberrations[uid]

        tbl = self._file.aberrations_table
        i = tbl.uid_dict[uid]

        ab = aotpy.Aberration(
            uid=uid,
            modes=self._image_from_reference(tbl.modes[i]),
            coefficients=self._image_from_reference(tbl.coefficients[i]),
            offsets=[aotpy.Coordinates(x, y) for x, y in zip(tbl.x_offsets[i], tbl.y_offsets[i])]
        )
        self._aberrations[uid] = ab
        return ab

    def _detector_from_uid(self, uid: str) -> aotpy.Detector | None:
        if uid is None:
            return None

        if uid in self._detectors:
            return self._detectors[uid]

        tbl = self._file.detectors_table
        i = tbl.uid_dict[uid]

        det = aotpy.Detector(
            uid=uid,
            type=tbl.type[i],
            sampling_technique=tbl.sampling_technique[i],
            shutter_type=tbl.shutter_type[i],
            flat_field=self._image_from_reference(tbl.flat_field[i]),
            readout_noise=tbl.readout_noise[i],
            pixel_intensities=self._image_from_reference(tbl.pixel_intensities[i]),
            integration_time=tbl.integration_time[i],
            field_centre=aotpy.Coordinates(tbl.field_centre_x[i], tbl.field_centre_y[i]),
            coadds=tbl.coadds[i],
            dark=self._image_from_reference(tbl.dark[i]),
            weight_map=self._image_from_reference(tbl.weight_map[i]),
            quantum_efficiency=tbl.quantum_efficiency[i],
            pixel_scale=tbl.pixel_scale[i],
            binning=tbl.binning[i],
            bandwidth=tbl.bandwidth[i],
            transmission_wavelength=tbl.transmission_wavelength[i],
            transmission=tbl.transmission[i],
            sky_background=self._image_from_reference(tbl.sky_background[i]),
            gain=tbl.gain[i],
            excess_noise=tbl.excess_noise[i],
            filter=tbl.filter[i],
            bad_pixel_map=self._image_from_reference(tbl.bad_pixel_map[i]),
            dynamic_range=tbl.dynamic_range[i],
            readout_rate=tbl.readout_rate[i],
            frame_rate=tbl.frame_rate[i],
            transformation_matrix=self._image_from_reference(tbl.transformation_matrix[i])
        )

        self._detectors[uid] = det
        return det

    def _image_from_reference(self, ref: AOTFITSImage) -> aotpy.Image | None:
        if ref is None:
            return None
        if isinstance(ref, AOTFITSInternalImage):
            if ref.extname in self._internalimages:
                return self._internalimages[ref.extname]
            img = aotpy.Image(
                name=ref.extname,
                data=ref.data,
                unit=ref.unit,
                time=self._time_from_uid(ref.time_uid),
                metadata=[metadatum_from_card(card) for card in ref.metadata]
            )

            self._internalimages[ref.extname] = img
        elif isinstance(ref, AOTFITSExternalImage):
            if ref.is_url:
                img = FITSURLImage(ref.path, ref.index, read_data=False)
            else:
                img = FITSFileImage(ref.path, ref.index, read_data=False)
            img.name = ref.extname
            img.data = ref.data
            img.unit = ref.unit
            img.time = self._time_from_uid(ref.time_uid)
            img.metadata = ref.metadata
        else:
            # This should never happen
            raise RuntimeError(f"Unknown image type {type(ref)}")

        return img
