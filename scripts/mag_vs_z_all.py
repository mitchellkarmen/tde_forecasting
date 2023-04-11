import kcorrect.kcorrect
import numpy as np
import sncosmo
from tqdm import tqdm
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants as const
import sys
import json
import pandas as pd

tde_utils_location = '../'
sys.path.append(tde_utils_location)
from scripts import tde_utils

with open('../survey_master_list.json', 'r') as fp:
    survey_limits = json.load(fp)
    
roman_wide_limits = survey_limits['roman_wide_limits']
roman_deep_limits = survey_limits['roman_deep_limits']
roman_hour_limits = survey_limits['roman_1hour_limits']
roman_minute_limits = survey_limits['roman_1min_limits']
proposal_limits = survey_limits['roman_proposal_limits']

jwst_nircam_limits = survey_limits['jwst_nircam_10000s_limits']

at2018hyz = {'u_mag': 19.195, 'err_u': 0.034,
           'g_mag': 17.605, 'err_g': 0.005,
           'r_mag': 17.040, 'err_r': 0.005,
           'i_mag': 16.755, 'err_i': 0.005,
           'z_mag': 16.546, 'err_z': 0.0014,
            'z':  0.0457}

mag_dict = {'sdss_u0': (at2018hyz['u_mag'], at2018hyz['err_u']),
           'sdss_g0': (at2018hyz['g_mag'], at2018hyz['err_g']),
           'sdss_r0': (at2018hyz['r_mag'], at2018hyz['err_r']),
           'sdss_i0': (at2018hyz['i_mag'], at2018hyz['err_i']),
           'sdss_z0': (at2018hyz['z_mag'], at2018hyz['err_z'])}

gal = tde_utils.GalaxySource(mag_dict = mag_dict, obs_z = at2018hyz['z'])

zs = np.linspace(0.1, 15.5, 50)

tde_mags = {}
gal_mags = {}
snia_mags = {}
qso_mags = {}

for passband in jwst_nircam_limits.keys():
    tde_mags[passband] = np.zeros(len(zs))
    gal_mags[passband] = np.zeros(len(zs))
    snia_mags[passband] = np.zeros(len(zs))
    qso_mags[passband] = np.zeros(len(zs))

qso_source = tde_utils.QSOSource(r_mag = -25.)

filter_loop = tqdm(list(tde_mags.keys()), total = len(list(tde_mags.keys()))*len(zs))

for filt in filter_loop:
    print(filt)
    for i, z in enumerate(zs):
    
        luminosity_distance = cosmo.luminosity_distance(z)
        
        # TDE Magnitudes
        source = tde_utils.BlackBodySource(30000)
        bb_model = sncosmo.Model(source=source)
        bb_model.set(z=z)

        filter_zstretched_mag = bb_model.bandmag(filt, 'ab', 0)
        app_mag = filter_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
        tde_mags[filt][i] = app_mag

        # Galaxy Magnitudes

        gal_model = sncosmo.Model(source=gal)
        gal_model.set(z=z)
        try:
            gal_zstretched_mag = gal_model.bandmag(filt, 'ab', 0)
            app_mag = gal_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
            gal_mags[filt][i] = app_mag
        except:
            gal_mags[filt][i] = np.nan
        
        # SNe Ia Magnitudes

        snia_model = sncosmo.Model(source='salt2-extended')
        snia_model.set(z=z)
        snia_model.set(x1=0, c=0, x0= 1051627384124.3574, t0=0)
        try:
            snia_zstretched_mag = snia_model.bandmag(filt, 'ab', 0)
            snia_app_mag = snia_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
            snia_mags[filt][i] = snia_app_mag
        except Exception as e:
            #snia_mags[filt].append(np.nan)
            print('simulating sn as blackbody')
            snia_scale = 1.718e-16
            snia_source = tde_utils.BlackBodySource(temperature=5500., scale=snia_scale)
            snia_model = sncosmo.Model(source=snia_source)
            snia_model.set(z=z)
            
            snia_zstretched_mag = snia_model.bandmag(filt, 'ab', 0)
            snia_app_mag = snia_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
            snia_mags[filt][i] = snia_app_mag
            
        # QSO Magnitudes
        
        qso_model = sncosmo.Model(source=qso_source)
        qso_model.set(z=z)
        try:
            qso_zstretched_mag = qso_model.bandmag(filt, 'ab', 0)
            app_mag = qso_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
            qso_mags[filt][i] = app_mag
        except:
            qso_mags[filt][i] = np.nan
            
        filter_loop.update()


for mag_dict in [tde_mags, gal_mags, snia_mags, qso_mags]:
	mag_dict['z'] = zs

tde_df = pd.DataFrame(tde_mags)
tde_df.to_csv('../data/tde_mag_z.csv')

gal_df = pd.DataFrame(gal_mags)
gal_df.to_csv('../data/galaxy_mag_z.csv')

snia_df = pd.DataFrame(snia_mags)
snia_df.to_csv('../data/snia_mag_z.csv')

qso_df = pd.DataFrame(qso_mags)
qso_df.to_csv('../data/qso_mag_z.csv')
