import kcorrect.kcorrect
import numpy as np
import json
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sncosmo
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants as const
from tqdm import tqdm


import tde_utils
sns.set_context('paper')

#hack
import sys
sys.path.append('../')
from survey_master_list import roman_wide_limits, roman_deep_limits

l_alpha = 1215.67 #angstrom

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

# with open('../survey_master_list.json') as json_file:
#   survey_lims = json.load(json_file)

gal = tde_utils.GalaxySource(mag_dict = mag_dict, obs_z = at2018hyz['z'])

#hack
wide_limits = roman_wide_limits #survey_lims['roman_wide_limits']
deep_limits = roman_deep_limits #survey_lims['roman_deep_limits']

zs = np.linspace(0.1, 7.5, 24)

result_dict = {'f062': np.zeros(len(zs)), 
               'f087': np.zeros(len(zs)), 
               'f106': np.zeros(len(zs)),
               'f129': np.zeros(len(zs)),
               'f158': np.zeros(len(zs)),
               'f184': np.zeros(len(zs)),
               'f213': np.zeros(len(zs)),
               'f146': np.zeros(len(zs))
              }
tde_mags = {'f062': np.zeros(len(zs)), 
               'f087': np.zeros(len(zs)), 
               'f106': np.zeros(len(zs)),
               'f129': np.zeros(len(zs)),
               'f158': np.zeros(len(zs)),
               'f184': np.zeros(len(zs)),
               'f213': np.zeros(len(zs)),
               'f146': np.zeros(len(zs))
              }
gal_mags = {'f062': np.zeros(len(zs)), 
               'f087': np.zeros(len(zs)), 
               'f106': np.zeros(len(zs)),
               'f129': np.zeros(len(zs)),
               'f158': np.zeros(len(zs)),
               'f184': np.zeros(len(zs)),
               'f213': np.zeros(len(zs)),
               'f146': np.zeros(len(zs))
              }
snia_mags = {'f062': np.zeros(len(zs)), 
               'f087': np.zeros(len(zs)), 
               'f106': np.zeros(len(zs)),
               'f129': np.zeros(len(zs)),
               'f158': np.zeros(len(zs)),
               'f184': np.zeros(len(zs)),
               'f213': np.zeros(len(zs)),
               'f146': np.zeros(len(zs))
              }

qso_mags = {'f062': np.zeros(len(zs)), 
               'f087': np.zeros(len(zs)), 
               'f106': np.zeros(len(zs)),
               'f129': np.zeros(len(zs)),
               'f158': np.zeros(len(zs)),
               'f184': np.zeros(len(zs)),
               'f213': np.zeros(len(zs)),
               'f146': np.zeros(len(zs))
              }

# abs_r = kc_sdss.absmag(redshift = redshift, maggies = maggies, 
#                        ivar=ivar, coeffs = coeffs)[2]

filter_loop = tqdm(list(result_dict.keys()), total = len(list(result_dict.keys()))*len(zs))
qso_source = tde_utils.QSOSource(r_mag = -25.)

for roman_filter in filter_loop:
    print(roman_filter)
    for i, z in enumerate(zs):
        luminosity_distance = cosmo.luminosity_distance(z)
        
        # TDE Magnitudes
        source = tde_utils.BlackBodySource(30000)
        bb_model = sncosmo.Model(source=source)
        bb_model.set(z=z)

        filter_zstretched_mag = bb_model.bandmag(roman_filter, 'ab', 0)
        app_mag = filter_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
        tde_mags[roman_filter][i] = app_mag

        # Galaxy Magnitudes
        gal_model = sncosmo.Model(source=gal)
        gal_model.set(z=z)
        try:
            gal_zstretched_mag = gal_model.bandmag(roman_filter, 'ab', 0)
            app_mag = gal_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
            gal_mags[roman_filter][i] = app_mag
        except:
            gal_mags[roman_filter][i] = np.nan
        
        # SNe Ia Magnitudes
        snia_model = sncosmo.Model(source='salt2-extended')
        snia_model.set(z=z)
        snia_model.set(x1=0, c=0, x0= 1051627384124.3574, t0=0)
        try:
            snia_zstretched_mag = snia_model.bandmag(roman_filter, 'ab', 0)
            snia_app_mag = snia_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
            snia_mags[roman_filter][i] = snia_app_mag
        except Exception as e:
            #snia_mags[roman_filter].append(np.nan)
            print('simulating sn as blackbody')
            snia_scale = 1.718e-16
            snia_source = tde_utils.BlackBodySource(temperature=5500., scale=snia_scale)
            snia_model = sncosmo.Model(source=snia_source)
            snia_model.set(z=z)
            
            snia_zstretched_mag = snia_model.bandmag(roman_filter, 'ab', 0)
            snia_app_mag = snia_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
            snia_mags[roman_filter][i] = snia_app_mag
            
        # QSO Magnitudes
        # qso_model = sncosmo.Model(source=qso_source)
        # qso_model.set(z=z)
        # try:
        #     qso_zstretched_mag = qso_model.bandmag(roman_filter, 'ab', 0)
        #     app_mag = qso_zstretched_mag + 5*np.log10(luminosity_distance/(10 * u.parsec))
        #     qso_mags[roman_filter][i] = app_mag
        # except:
        #     qso_mags[roman_filter][i] = np.nan
            
        filter_loop.update()


# fig, axes = plt.subplots(nrows=len(result_dict.keys()), dpi = 200, 
#                          figsize = [6, 4*len(result_dict.keys())])

fig, ax = plt.subplots(dpi=300)

# lyman alpha
roman_filter='f184'
band = sncosmo.get_bandpass(roman_filter)
z_contam = (band.minwave() / l_alpha) - 1

ax.plot(zs, tde_mags[roman_filter], label = 'TDEs')
ax.plot(zs, gal_mags[roman_filter], label = 'Host Galaxies')
ax.plot(zs, snia_mags[roman_filter], label = 'SNe Ia')
#ax.plot(zs, qso_mags[roman_filter], label = 'Quasars')

xmin, xmax = ax.get_xlim()
ax.axvspan(z_contam, xmax, zorder = -1, alpha = 0.3, color = 'grey')

if roman_filter in wide_limits.keys():
    ax.axhline(wide_limits[roman_filter], c = 'k', ls = '--', label = 'HLTDS Wide Limit')
if roman_filter in deep_limits.keys():
    ax.axhline(deep_limits[roman_filter], c = 'k', ls = '-.', label = 'HLTDS Deep Limit')

ax.axhline(27.4, c = 'k', ls = ':', label = 'One Hour Point Source Limit')
ax.axhline(24.4, c = 'k', ls = 'solid', label = 'One Minute Point Source Limit')

ax.set_title(roman_filter)
ax.set_xlabel('z')
ax.set_ylabel('Magnitude')
ax.legend()
ymin, ymax = ax.get_ylim()
ax.set_ylim(np.min([ymax, 32]), ymin)
ax.set_xlim(min(zs), max(zs))

fig.suptitle('Roman Magnitudes vs Redshift', y = 0.995)
plt.tight_layout()
#plt.savefig(f'../figures/f184_mag_vs_z.pdf', dpi = 300, bbox_inches='tight')
plt.show()