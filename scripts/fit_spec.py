import os
os.environ['SPS_HOME'] = "/home/mkarmen1/offcenter_blackholes/misc/fsps" #change on different machine
import numpy as np
import time, sys

from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer


def build_model(object_redshift=None):
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel
    from prospect.models import SedModel

    model_params = TemplateLibrary["parametric_sfh"]
    # Turn on nebular emission and add associated parameters
    model_params.update(TemplateLibrary["nebular"])

    model_params["gas_logu"]["isfree"] = True
    model_params["zred"]["init"] = object_redshift
    model_params['zred']['isfree'] = False
    model_params["mass"]["init"] = 1e10
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.3
    model_params["tage"]["init"] = 5.

    model = SedModel(model_params)

    return model


def build_obs(tde_name, photometry = True, spectrum = False):
    from prospect.utils.obsutils import fix_obs
    import pandas as pd

    data = np.genfromtxt('/home/mkarmen1/tde_forecasting/data/host_photo_table.txt', names = True, 
                     dtype = None, encoding = 'ascii')
    host_mags = pd.DataFrame(data)
    host_mags = host_mags.set_index('name')
    tde_host = host_mags.loc[tde_name]
    
    bands = "ugriz"
    filters = load_filters([f"sdss_{b}0" for b in bands])
    
    wavelength = None
    spectrum=None
    unc = None
    maggies = None
    maggies_unc = None
    if (photometry is False) and (spectrum is False):
        raise ValueError('You need to provide data!')
    if spectrum:
        # do something with t_spec
        wavelength = ...
        spectrum = ...
    elif photometry:
        maggies = np.array([10**(-0.4 * tde_host[f"mag_{b}"]) for b in bands])
        magerr = np.array([tde_host[f"errmag_{b}"] for b in bands])
        magerr = np.clip(magerr, 0.05, np.inf)
        
        maggies_unc = magerr * maggies / 1.086
    

    obs = dict(wavelength=None, spectrum=None, unc=None, redshift=tde_host['z'],
            maggies=maggies, maggies_unc=maggies_unc, filters=filters)
    obs = fix_obs(obs)
    
    return obs


def build_sps(zcontinuous=1):
    from prospect.sources import CSPSpecBasis

    sps = CSPSpecBasis(zcontinuous)
    return sps


if __name__ == '__main__':

    parser = prospect_args.get_parser()
    parser.add_argument('--name', type=str, default='',
                        help=("Name of TDE to fit host of.  Must be in host mag file."))


    args = parser.parse_args()
    run_params = vars(args)

    tde_name = run_params.pop('name')
    hfile = tde_name + '_host_result.h5' 
    print('tde is', tde_name)
    obs = build_obs(tde_name)
    model = build_model(object_redshift = obs['redshift'])
    sps = build_sps()

    output = fit_model(obs, model, sps)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass
