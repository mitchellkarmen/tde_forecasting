import numpy as np
import sncosmo
from astropy import constants as const
from astropy import units as u
import kcorrect.kcorrect
import yaml
import pysynphot as S

c = const.c.cgs
h = const.h.cgs
k_B = const.k_B.cgs
with open ('../scripts/CONFIG.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

base_dir = config['base_dir']

class StellarSource(sncosmo.Source):
    _param_names = ['temperature', 'log_g', 'metallicity']
    param_names_latex = ['Temperature (K)', 'log(g)', 'metallicity']

    def __init__(self, t=10000, logg=5, metallicity=-1.5):

        self.name = "Star"
        self.version = 0.1
        self.phase = 0

        spec = S.Icat('ck04models', Teff = t, log_g= logg, 
            metallicity = metallicity)

        self._spec = np.ravel(spec.flux[:1212]) #end early to avoid binning issue
        self._wave = np.ravel(spec.wave[:1212])
        self._parameters = [t, logg, metallicity]

    def _flux(self, phase, wave):
        default_wave = self._wave
        fill_errs = np.full(len(default_wave), 0.001)
        default_flux = self._spec
        spectrum = sncosmo.Spectrum(default_wave, default_flux, fill_errs)
        binned_spectrum = spectrum.rebin(wave)
        flux = binned_spectrum.flux
        return flux.reshape(1, len(flux))



class GalaxySource(sncosmo.Source):
    _param_names = ['c1', 'c2', 'c3', 'c4', 'c5']
    param_names_latex = ['C1', 'C2', 'C3', 'C4', 'C5']
    
    def __init__(self, mag_dict, obs_z):
                
        """
        mag_dict: keys are input passbands, values are tuples of magnitudes and uncertainty
        """
        
        self.name = 'Galaxy'
        self.version = 0.1
        self._phase = 0
        
        bands, responses = list(mag_dict.keys()), list(mag_dict.values())
        kc = kcorrect.kcorrect.Kcorrect(responses = bands)
        mags = [r[0] for r in responses]
        errs = [r[1] for r in responses]
        maggies, ivar = kcorrect.utils.sdss_asinh_to_maggies(mags, errs)
        coeffs = kc.fit_coeffs(redshift=obs_z, maggies=maggies, ivar = ivar)
        rf_mags = kc.absmag(redshift=obs_z, maggies=maggies, ivar=ivar, coeffs=coeffs)
        maggies, ivar = kcorrect.utils.sdss_asinh_to_maggies(rf_mags, errs)
        coeffs = kc.fit_coeffs(redshift=0, maggies=maggies, ivar = ivar)        
        
        templates = kc.templates
        default_flux = coeffs.dot(templates.flux)
        #templates.set_redshift(obs_z)
        self._wave = templates.restframe_wave # * (1. + obs_z)
        self.obs_z = obs_z
        self.kc = kc
        self.coeffs = coeffs
        self._parameters = coeffs
        self.templates = templates
        self.default_flux = default_flux
        
    def _flux(self, phase, wave):
        default_wave = self._wave
        rf_flux = self.default_flux 
        fill_errs = np.full(len(default_wave), 0.001)
        spectrum = sncosmo.Spectrum(default_wave, rf_flux, fill_errs)
        binned_spectrum = spectrum.rebin(wave)
        flux = binned_spectrum.flux
        return flux.reshape(1, len(flux))

class BlackBodySource(sncosmo.Source):
    _param_names = ['temperature', 'scale']
    param_names_latex = ['T (K)', 'Scale']
    
    def __init__(self, temperature, scale = 1.0477889428400054e-17):
        self.temperature = temperature
        self.name = 'BlackBody'
        self._wave = np.linspace(1, 1500000, 200)
        self._parameters = [self.temperature, scale]
        self.scale = scale
        
    def _flux(self, phase, wave):
        temperature = self.temperature * u.Kelvin
        #nu = c / (wave * u.angstrom).cgs
        wave = (wave* u.angstrom).cgs
        scale = self.scale #scale abs sdss g mag to -19.5
        #print(f'generating spectrum for temperature {temperature} and scale {scale}')
        
        coeff = 2 * h * (c ** 2) / (wave **5)
        denominator = np.exp(h* c / (wave * k_B * temperature)) - 1
        spectrum = (coeff / denominator) #.to((u.erg) * (u.s**-1) * (u.cm**-2) * (u.Angstrom**-1))
        spectrum = scale * spectrum.value
        
        return spectrum.reshape(1, len(spectrum))

class CalzettiDust(PropagationEffect):
    """Cardelli, Clayton, Mathis (1989) extinction model dust."""
    _param_names = ['a_v', 'r_v']
    param_names_latex = ['A_V', 'R_V']
    _minwave = 1200.
    _maxwave = 22000.

    def __init__(self):
        self._parameters = np.array([0., 4.2])

    def propagate(self, wave, flux, phase=None):
        """Propagate the flux."""
        ebv, r_v = self._parameters
        return extinction.apply(extinction.ccm89(wave, ebv * r_v, r_v), flux)

class QSOSource(sncosmo.Source):
    _param_names = ['r_mag']
    param_names_latex = ['r-band magnitude']
    
    def __init__(self, r_mag):
                
        """
        r-mag is rest-frame sdss r-band magnitude
        """
        
        self.name = 'Quasar'
        self.version = 0.1
        self._phase = 0
        
        qso_spectrum = np.loadtxt(base_dir + 'data/median_composite_quasar_spectrum.txt', skiprows=23) 
        wave, relative_flux, relative_flux_err = qso_spectrum.T
        
        default_r_mag = -21.332976980967537
        scale_factor = 10**((r_mag - default_r_mag) / -2.5)
        scaled_flux = relative_flux * scale_factor
        scale_flux_err = relative_flux_err * scale_factor
        
        
        self._wave = wave
        self._scaled_flux = scaled_flux
        self._scaled_fluxerr = scale_flux_err
        self._parameters = [r_mag]
        
    def _flux(self, phase, wave):
        default_wave = self._wave
        default_flux = self._scaled_flux
        default_flux_err = self._scaled_fluxerr
        spectrum = sncosmo.Spectrum(default_wave, default_flux, default_flux_err)
        binned_spectrum = spectrum.rebin(wave)
        flux = binned_spectrum.flux
        return flux.reshape(1, len(flux))

class DustEchoSource(BlackBodySource):

    def _flux(self, phase, wave):
        temperature = self.temperature * u.Kelvin
        nu = c / (wave * u.angstrom).cgs
        scale = 144.90744350236645 / 31 #scale abs sdss g mag to -19.5
        
        coeff = 2 * h * (nu**3) / (c ** 2)
        denominator = np.exp(h* nu / (k_B * temperature)) - 1
        spectrum = scale * (coeff / denominator).value
        
        return spectrum.reshape(1, len(spectrum))
