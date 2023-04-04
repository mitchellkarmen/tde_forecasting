import fsps
import dynesty
import sedpy
import h5py, astropy
import numpy as np
from astropy.io import fits
from astropy.table import Table
from sedpy.observate import load_filters
from prospect.utils.obsutils import fix_obs
from prospect.models.templates import TemplateLibrary
from prospect.models import SpecModel
from prospect.sources import CSPSpecBasis
from prospect.fitting import lnprobfn, fit_model
from prospect.io import write_results as writer

import os
os.environ['SPS_HOME'] = "/home/mkarmen1/offcenter_blackholes/misc/fsps"

# read the data
file = '../data/spec-at2018bsi-host-sdss.fits'
t_spec = Table.read(file, hdu=1)
t_phot = Table.read(file, hdu=2)
mags = t_phot['CMODELMAG'].value[0]
mag_errs = t_phot['CMODELMAGERR'].value[0]
bands = "ugriz"

filters = load_filters([f"sdss_{b}0" for b in bands])
maggies = np.array([10**(-0.4 * m) for m in mags])
magerr = np.array([m for m in mag_errs])
magerr = np.clip(magerr, 0.05, np.inf)

obs = dict(wavelength=None, spectrum=None, unc=None, redshift=t_phot["Z"].value,
           maggies=maggies, maggies_unc=magerr * maggies / 1.086, filters=filters)
obs = fix_obs(obs)

model_params = TemplateLibrary["parametric_sfh"]
model_params.update(TemplateLibrary["nebular"])
model_params["zred"]["init"] = obs["redshift"]

model = SpecModel(model_params)
assert len(model.free_params) == 5
noise_model = (None, None)
sps = CSPSpecBasis(zcontinuous=1)

current_parameters = ",".join([f"{p}={v}" for p, v in zip(model.free_params, model.theta)])
print(current_parameters)
spec, phot, mfrac = model.predict(model.theta, obs=obs, sps=sps)
print(phot / obs["maggies"])

#do the fit 
fitting_kwargs = dict(nlive_init=400, 
                      nested_method="rwalk", 
                      nested_target_n_effective=1000, 
                      nested_dlogz_init=0.05)

output = fit_model(obs, model, sps, 
                   optimize=False, dynesty=True, 
                   lnprobfn=lnprobfn, noise=noise_model, 
                   **fitting_kwargs)

result, duration = output["sampling"]

hfile = "./data/at2018bsi-host-fit.h5"
writer.write_hdf5(hfile, {}, model, obs,
                 output["sampling"][0], None,
                 sps=sps,
                 tsample=output["sampling"][1],
                 toptimize=0.0)
print("DONE")