import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np
from astropy.utils.data import download_file
from astropy.io import fits
from scipy.io import readsav

from aotpy import *

__all__ = ['load_from_keck']


def load_from_keck(telemetry_filename, science_filename=None, ncpa_path=None, pupil_folder=None) -> AOSystem:
    # TODO: we only support NIRC2
    telemetry = readsav(telemetry_filename)
    if science_filename is None:
        hdr = fits.Header.fromstring(b''.join(telemetry['header']).decode())
    else:
        hdr = fits.getheader(science_filename)

    system = AOSystem()

    system.start_datetime = datetime.fromisoformat(f"{hdr['DATE-OBS']} {hdr['EXPSTART']}").replace(tzinfo=timezone.utc)
    system.end_datetime = datetime.fromisoformat(f"{hdr['DATE-OBS']} {hdr['EXPSTOP']}").replace(tzinfo=timezone.utc)
    assert system.end_datetime > system.start_datetime

    atm_median = AtmosphericParameters(
        data_source='Median Conditions',
        wavelength=500e-9,
        r0=3600 * 180 / np.pi * 0.98 * 500e-9 / 0.16,
        l0=25,
        layers=[
            AtmosphereLayer(weight=0.517, height=0.0, wind_speed=6.8, wind_direction=0),
            AtmosphereLayer(weight=0.119, height=500.0, wind_speed=6.9, wind_direction=0),
            AtmosphereLayer(weight=0.063, height=1000.0, wind_speed=7.1, wind_direction=0),
            AtmosphereLayer(weight=0.061, height=2000.0, wind_speed=7.5, wind_direction=0),
            AtmosphereLayer(weight=0.105, height=4000.0, wind_speed=10.0, wind_direction=0),
            AtmosphereLayer(weight=0.081, height=8000.0, wind_speed=26.9, wind_direction=0),
            AtmosphereLayer(weight=0.054, height=16000.0, wind_speed=18.5, wind_direction=0)
        ]
    )
    system.atmosphere_params.append(atm_median)

    system.atmosphere_params.extend(_get_atmosphere_conditions('dimm', system.start_datetime, system.end_datetime))
    system.atmosphere_params.extend(_get_atmosphere_conditions('mass', system.start_datetime, system.end_datetime))
    system.atmosphere_params.extend(_get_atmosphere_conditions('masspro', system.start_datetime, system.end_datetime))

    if hdr['LSPROP'].upper() == 'YES':
        system.ao_mode = 'LGS'
    else:
        system.ao_mode = 'NGS'
    assert (system.ao_mode == 'LGS') == ('b' in telemetry)

    cam = ScoringCamera(name=hdr['CURRINST'].upper())
    if cam.name == 'NIRC2':
        # Setup NIRC2 plate scales
        scales = {"NARROW": 9.942,
                  "MEDIUM": 19.829,
                  "WIDE": 39.686}
        cam.detector = Detector('CamDetector', pixel_scale=scales[hdr['CAMNAME'].upper()])
        cam.optical_relay = OpticalRelay('NIRC2 relay', field_of_view=hdr['NAXIS1'] * cam.detector.pixel_scale)

    elif cam.name == 'OSIRIS':
        cam.detector.pixel_scale = 9.95
        raise NotImplementedError
    else:
        raise NotImplementedError

    system.scoring_cameras = [cam]
    # cam.detector.transmission_wavelength, cam.detector.transmission = _get_transmission(hdr)
    # TODO: we have sourced transmission data,but we are working on how to best package it
    cam.wavelength = float(hdr['CENWAVE']) * 1e-6
    cam.detector.integration_time = hdr['ITIME']
    cam.detector.coadds = hdr['COADDS']
    if hdr['SAMPMODE'] == 2:
        cam.detector.readout_noise = 60
    else:
        cam.detector.readout_noise = 15.0 * (16.0 / hdr['MULTISAM']) ** 0.5
    cam.detector.gain = hdr['GAIN']

    system.telescope = MainTelescope(
        name=hdr['TELESCOP'],
        d_eq=10.5,
        d_hex=10.949,
        # TODO d_circle
        cobs=0.2311,
        elevation=float(hdr['EL']),
        pupil_angle=float(hdr['ROTPOSN']) - float(hdr['INSTANGL'])
    )
    if cam.name == 'OSIRIS':
        system.telescope.pupil_angle += 42.5

    if pupil_folder:
        pmsname = hdr['PMSNAME']
        if pmsname.upper() == 'LARGEHEX':
            system.telescope.pupil = Image(name=f'{pmsname} telescope pupil',
                                           data=fits.getdata(pupil_folder + 'keck_pupil_largeHex_272px.fits'))
        else:
            system.telescope.pupil = Image(name=f'{pmsname} telescope pupil',
                                           data=fits.getdata(pupil_folder + 'keck_pupil_open2_240px.fits'))
        system.telescope.pupil.data = system.telescope.pupil.data.astype(np.uint8, copy=False)  # save space

    # TODO: RA, DEC, EL, AL
    ngs = NaturalGuideStar(name='NGS')
    system.sources.append(ngs)

    fast_wfs = ShackHartmann(
        name='Fast Wavefront Sensor',
        source=ngs,
        wavelength=float(hdr['GUIDWAVE']) * 1e-6,
        slopes=Image('Fast WFS slopes', telemetry.A['OFFSETCENTROID'][0]),
        n_subapertures=400,
        subaperture_size=0.5625,
        algorithm='cog',
        theta=90,
        optical_relay=OpticalRelay('Fast WFS relay', field_of_view=3200),
        detector=Detector(
            name='FastWFS Detector',
            pixel_scale=800,
            binning=1,
            readout_noise=3,
            gain=1,
            excess_noise=1
        )
    )

    fast_wfs.valid_subapertures = Image(name='Fast WFS valid subapertures', data=np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8))
    # TODO: convert from z_position to object_plane
    # if system.ao_mode == 'LGS':
    #     fast_wfs.optical_relay.z_position = abs(float(hdr['OBWF']) - float(hdr['AOFCLGFO']) * 1e3)
    # else:
    #     fast_wfs.optical_relay.z_position = abs(float(hdr['OBWF']) - float(hdr['AOFCNGFO']) * 1e3)

    intensity = np.zeros((fast_wfs.slopes.data.shape[0], fast_wfs.n_subapertures))
    intensity[:, fast_wfs.valid_subapertures.data.reshape(-1).astype(bool)] = telemetry.A['SUBAPINTENSITY'][0]
    intensity = intensity.reshape((fast_wfs.slopes.data.shape[0],
                                   fast_wfs.valid_subapertures.data.shape[0],
                                   fast_wfs.valid_subapertures.data.shape[1]))
    fast_wfs.subaperture_intensities = Image('Fast WFS subaperture intensities', intensity)

    fast_wfs.centroid_gains = Image('Fast WFS centroid gains', telemetry['CENT_G'])
    if isinstance(fast_wfs.centroid_gains.data, np.recarray):
        fast_wfs.centroid_gains.data = fast_wfs.centroid_gains.data[0][0][0]  # Sometimes this comes nested
    system.wavefront_sensors.append(fast_wfs)

    if system.ao_mode == 'LGS':
        # TODO: RA, DEC, EL, AL
        lgs = LaserGuideStar(name='LGS', sodium_height=90e3)
        # TODO: should sodium layer height be hdr['AOFCSALT']?
        system.sources.append(lgs)

        fast_wfs.source = lgs
        fast_wfs.wavelength = 589e-9

        lb_wfs = ShackHartmann(
            name='Low Bandwidth Wavefront Sensor (LBWFS)',
            source=ngs,
            wavelength=float(hdr['GUIDWAVE']) * 1e-6,
            n_subapertures=400,  # source: https://www2.keck.hawaii.edu/optics/aodocs/PASP06_MvD.pdf
            subaperture_size=0.5625
        )

        # TODO strap centroid_gains, theta?
        # TODO: Check which values are placeholders and which are known
        strap = ShackHartmann(
            name='System for Tip-tilt Removal with Avalanche Photodiodes (STRAP)',
            source=ngs,
            wavelength=float(hdr['GUIDWAVE']) * 1e-6,
            slopes=Image('STRAP slopes', telemetry.B['DTTCENTROIDS'][0]),
            n_subapertures=1,
            subaperture_size=10.5,
            algorithm="cog",
            subaperture_intensities=Image('STRAP subaperture intensisties', telemetry.B['APDCOUNTS'][0]), # TODO probably not the same units as fast_wfs?
            optical_relay=OpticalRelay('STRAP relay', field_of_view=3200),
            detector=Detector(
                name='STRAP Detector',
                pixel_scale=800,
                binning=1,
                readout_noise=3,
                gain=1,
                excess_noise=1
            )
        )
        system.wavefront_sensors.extend([lb_wfs, strap])

    volt_to_meter = 0.4095e-6
    dm = DeformableMirror(
        name='DM',
        telescope=system.telescope,
        n_actuators=349,
        pitch=0.5625,
    )
    dm.valid_actuators = Image('DM valid actuators', np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8))

    ttm = TipTiltMirror(name='TTM', telescope=system.telescope)
    fcs = LinearStage(name='Focus Stage', telescope=system.telescope)
    system.wavefront_correctors = [dm, ttm, fcs]

    system.rtc = RTC()
    ho_delay, tt_delay = _get_delays(int(hdr['WSSMPRG']))  # WFS_SciMeas program

    ho_loop = ControlLoop(name="High Order", input_wfs=fast_wfs, commanded_corrector=dm)
    ho_loop.delay = ho_delay
    timestamps = telemetry.A['TIMESTAMP'][0]
    ho_loop.timestamps = (timestamps - timestamps[0]).tolist()
    ho_loop.framerate = float(hdr['WSFRRT'])
    assert np.isclose(telemetry['DM_SERVO'][0], float(hdr['DMGAIN']))
    ho_loop.time_filter_num = Image('HO Loop time filter numerators', telemetry['DM_SERVO'][0:4])
    ho_loop.time_filter_den = Image('HO Loop time filter denominators', telemetry['DM_SERVO'][4:])
    ho_loop.commands = Image('HO DM commands', telemetry.A['DMCOMMAND'][0])
    ho_loop.residual_wavefront = Image(name='HO residual wavefront',
                                       data=telemetry.A['RESIDUALWAVEFRONT'][0][:, 0:dm.n_actuators] * volt_to_meter)

    # Get DM commands reconstructors from slopes
    # TODO: understand NREC better
    mc = np.reshape(telemetry['rx'], (dm.n_actuators + 3, fast_wfs.slopes.data.shape[1], telemetry['NREC']))
    ho_loop.control_matrix = Image('HO Loop Control Matrix', mc[:dm.n_actuators, :, :] * volt_to_meter)

    if system.ao_mode == 'NGS':
        tt_loop = ControlLoop(name="Tip-Tilt", input_wfs=fast_wfs, commanded_corrector=ttm)
        tt_loop.timestamps = ho_loop.timestamps
        tt_loop.framerate = ho_loop.framerate

        tt_loop.delay = tt_delay
        tilt_to_meter = 12.68e-6  # should be np.pi*tel.D/4/3600/180
        tt_loop.commands = Image('TT commands', telemetry.A['TTCOMMANDS'][0])

        fc_loop = ControlLoop(name="Focus", input_wfs=fast_wfs, commanded_corrector=dm)
    else:
        tt_loop = ControlLoop(name="Tip-Tilt", input_wfs=strap, commanded_corrector=ttm)
        timestamps = telemetry.B['TIMESTAMP'][0]
        tt_loop.timestamps = (timestamps - timestamps[0]).tolist()
        tt_loop.framerate = 1 / (100e-9 * np.mean(np.diff(tt_loop.timestamps)))
        tt_loop.delay = 1e-3  # STRAP
        tilt_to_meter = 3.2 * 1.268e-05  # to be verified
        tt_loop.commands = Image('TT commands', telemetry.B['DTTCOMMANDS'][0])

        fc_loop = ControlLoop(name="Focus", input_wfs=lb_wfs, commanded_corrector=fcs)

    assert np.isclose(telemetry['DT_SERVO'][0], float(hdr['DTGAIN']))
    tt_loop.time_filter_num = Image('TT Loop time filter numerators', telemetry['DT_SERVO'][0:4])
    tt_loop.time_filter_den = Image('TT Loop time filter denominators', telemetry['DT_SERVO'][4:])
    tt_loop.residual_wavefront = Image(name='TT residual wavefront',
                                       data=telemetry.A['RESIDUALWAVEFRONT'][0][:, dm.n_actuators:dm.n_actuators + 2])
    tt_loop.control_matrix = Image('TT Loop Control Matrix', mc[dm.n_actuators:dm.n_actuators + 2, :, :] * tilt_to_meter)

    fc_loop.residual_wavefront = Image('Focus residual wavefront', telemetry.A['RESIDUALWAVEFRONT'][0][:, -1])
    fc_loop.control_matrix = Image('Focus Loop Control Matrix', mc[-1:, :, :] * 1e-6)  # focus comes in microns

    system.rtc.loops = [ho_loop, tt_loop, fc_loop]
    if ncpa_path:
        system.rtc.non_common_path_aberrations = [Image('System NCPA', fits.getdata(ncpa_path))]

    return system


def _get_transmission(hdr: fits.Header):
    # TODO: a set of files with transmission data will be provided
    inst_name = hdr['CURRINST']  # cam.name
    if inst_name == 'NIRC2':
        filter1 = hdr['FWINAME']
        filter2 = hdr['FWONAME']
        filter_name = filter1
        if filter1.startswith('PK'):
            filter_name = filter2
        filters = ['J', 'H', 'K', 'Kcont', 'Kp', 'Ks', 'Lp', 'Ms',
                   'Hcont', 'Brgamma', 'FeII']
        filters_upper = [x.upper() for x in filters]

        if filter_name.upper() not in filters_upper:
            print('Could not find profile for filter %s.' % filter_name)
            print('Choices are: ', filters)
            raise NotImplementedError

        path = 'path/to/folder' + filter_name + '.dat'  # TODO: WIP
        filter_name = filters[filters_upper.index(filter_name.upper())]
        table = np.loadtxt(path)
        return table.T[0].tolist(), table.T[1].tolist()
    elif inst_name == 'OSIRIS':
        f = hdr['IFILTER']
        filter_name = f.split('-')[0]
        # TODO
        raise NotImplementedError
    else:
        raise NotImplementedError


def _get_delays(program: int) -> tuple[float, float]:
    t0_dict = {0: 8000e-6,
               1: 2600e-6,
               2: 850e-6,
               3: 700e-6,
               4: 28000e-6,
               5: 9500e-6,
               6: 2000e-6,
               7: 1350e-6}
    t0 = t0_dict[program]
    # Total delays
    t8 = 40e-6
    t9 = 1e-6
    t10 = 2e-6
    textra_tt = 200e-6  # retrieved from rejection transfer function fit

    ho_delay = t0 + t8
    tt_delay = t0 + t9 + t10 + textra_tt
    return ho_delay, tt_delay


def _get_atmosphere_conditions(data_source: str, start_date: datetime, end_date: datetime):
    assert data_source in ('dimm', 'mass', 'masspro')
    url_root = 'http://mkwc.ifa.hawaii.edu/current/seeing/'
    params = []

    date_str1 = start_date.strftime('%Y%m%d')
    date_str2 = (end_date + timedelta(days=1)).strftime('%Y%m%d')
    try:
        paths = [
            download_file(f"{url_root}{data_source}/{date_str1}.{data_source}.dat", cache=True, timeout=2),
            download_file(f"{url_root}{data_source}/{date_str2}.{data_source}.dat", cache=True, timeout=2)
        ]
    except urllib.error.URLError:
        print(f'{data_source.upper()} not available during the acquisition')
        return params
    lines = []
    for path in paths:
        with open(path) as f:
            lines.extend(f.readlines())

    relevant_data = []
    split = lines[0].split()
    last_date = datetime(*map(int, split[:6])).replace(tzinfo=timezone(timedelta(hours=-10), 'HST'))
    last_seeing = tuple(map(float, split[6:]))
    for line in lines[1:]:
        split = line.split()
        date = datetime(*map(int, split[:6])).replace(tzinfo=timezone(timedelta(hours=-10), 'HST'))
        seeing = tuple(map(float, split[6:]))
        if date > start_date:
            relevant_data.append((last_date, last_seeing))
        if date > end_date:
            relevant_data.append((date, seeing))
            break
        last_date = date
        last_seeing = seeing

    for data in relevant_data:
        if data_source == 'masspro':
            cn2dh = np.array(data[1][:6])
            cn2dh /= cn2dh.sum()
            params.append(
                AtmosphericParameters(
                    data_source=data_source.upper(),
                    wavelength=500e-9,
                    timestamp=data[0].astimezone(timezone.utc),
                    layers=[
                        AtmosphereLayer(height=500, weight=cn2dh[0]),
                        AtmosphereLayer(height=1000, weight=cn2dh[1]),
                        AtmosphereLayer(height=2000, weight=cn2dh[2]),
                        AtmosphereLayer(height=4000, weight=cn2dh[3]),
                        AtmosphereLayer(height=8000, weight=cn2dh[4]),
                        AtmosphereLayer(height=16000, weight=cn2dh[5])
                    ],
                    r0=data[1][6]
                )
            )
        else:
            params.append(
                AtmosphericParameters(
                    data_source=data_source.upper(),
                    wavelength=500e-9,
                    timestamp=data[0].astimezone(timezone.utc),
                    r0=data[1][0]
                )
            )
    return params
