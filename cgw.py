import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import glob
import json
import ephem
import pprint
import corner
import argparse 
import matplotlib
import numpy as np
matplotlib.use("Agg")
import astropy.units as u
from functools import partial
import matplotlib.pyplot as plt
from astropy.time import TimeDelta
from enterprise.pulsar import Pulsar
from scipy.special import ndtr, ndtri
from functools import partial, lru_cache
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord
from enterprise import constants as const
from enterprise.signals import gp_signals, utils
from concurrent.futures import ProcessPoolExecutor
from enterprise.signals.gp_signals import TimingModel
script_dir = os.path.dirname(os.path.abspath(__file__))
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise.signals.parameter import TruncNormalSampler
from enterprise_extensions.blocks import common_red_noise_block
from enterprise.signals.selections import Selection, no_selection
from enterprise_extensions import deterministic as ee_deterministic
from enterprise.signals import parameter, white_signals, signal_base
from inject_signals import make_ideal_parallel, inject_white_noise, inject_red_noise, inject_curn, inject_cw


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 17,           
    "axes.titlesize": 17,       
    "axes.labelsize": 17,      
    
    "xtick.labelsize": 17,     
    "ytick.labelsize": 17,      

    "legend.fontsize": 17,     
    "figure.titlesize": 17, 
    "font.family": "serif",       
    "legend.framealpha": 0.8,     
    "legend.edgecolor": "black", 
    "axes.linewidth": 1,        
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
})

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
def parse_args():
    parser = argparse.ArgumentParser(description='Run a PTA analysis for continuous GWs.')
    parser.add_argument('--suffix',type=str,default='',help='suffix for dir.')
    parser.add_argument('-o', '--output',type=str,default='__test__',help='out dir.')
    parser.add_argument('--de_whiten', '-dw', action='store_true', help='If specified, run the plotting script.')
    parser.add_argument('--corner', action='store_true', help='If specified, run the plotting script.')
    parser.add_argument('--run_mcmc', action='store_true', help='If specified, run the full MCMC simulation.')
    parser.add_argument('--add_inj', action='store_true', help='If specified, add the Injection to the data.')
    parser.add_argument('--hist', action='store_true', help='If specified, plot individual parameter histograms.')
    parser.add_argument('--corner_white', action='store_true', help='If specified, plot corner in whitened space.')
    parser.add_argument('--no_curn',dest='curn',action='store_false',help='Disable the CURN component from the analysis.')
    parser.add_argument('--no_wn',dest='wn',action='store_false',help='Disable the White Noise component from the analysis.')
    parser.add_argument('-p', '--num_psrs',type=int,default=10,help='Number of pulsars to use in the analysis (default: 10).')
    parser.add_argument('-b', '--num_bina',type=int,default=1,help='Number of Binaries to model the analysis for (default: 1).')
    parser.add_argument('-c', '--comment', type=str, default='', help='Comment to save to comment.txt in the output directory.')
    parser.add_argument('--data_dir',type=str,default='./future_ipta_data/',help='Directory containing input data (default: ).')
    parser.add_argument('--no_prn',dest='prn',action='store_false',help='Disable the Pulsar Red Noise component from the analysis.')
    parser.add_argument('-n', '--n_samples',type=int,default=10000000,help='Number of MCMC samples to generate (default: 10,000,000).')
    parser.add_argument('--no_cw',dest='cw',action='store_false',help='Disable the Continuous Wave signal components from the analysis.')

    return parser.parse_args()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
phase_approx_val = True
evolve_val = False
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

args = parse_args()

cw = args.cw               
wn = args.wn               
prn = args.prn             
curn = args.curn           
suffix = args.suffix
add_inj = args.add_inj
comment = args.comment       
output_dir = args.output
data_dir = args.data_dir
num_psrs = args.num_psrs   
dewhiten = args.de_whiten
make_hist_plot = args.hist
n_samples = args.n_samples 
num_binaries = args.num_bina
make_corner_plot = args.corner           
run_mcmc_analysis = args.run_mcmc  
make_corner_white = args.corner_white

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

params_list = []
if wn: params_list.append("wn_")
if prn: params_list.append("prn_")
if curn: params_list.append("curn_")
if cw: params_list.append(f"{num_binaries}cw_")

def make_unique_dir(path):
    base = path
    i = 1
    while os.path.exists(path):
        path = f"{base}_{i}"
        i += 1
    os.makedirs(path)
    return path

data_dir = args.data_dir
suffix = args.suffix
output_dir = args.output
params = "".join(params_list)
chains_outdir = os.path.join(script_dir, f'{output_dir}/{num_psrs}psrs_{params}{int(n_samples/1000)}k_samples_{suffix}')
if run_mcmc_analysis: chains_outdir = make_unique_dir(chains_outdir)
plt_out_filename = os.path.join(chains_outdir, f'_plot_{num_psrs}psr_{params}_{int(n_samples/1000)}k_samples_{suffix}')
skymap_filename = os.path.join(chains_outdir, 'skymap')
if comment:
    with open(os.path.join(chains_outdir, 'comment.txt'), 'w') as f:
        f.write(comment + '\n')

def radec_to_gwtheta_phi(ra_hms, dec_dms):
    coord = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg))
    gwphi = coord.ra.to(u.rad).value
    gwtheta = (np.pi / 2) - coord.dec.to(u.rad).value
    return gwtheta, gwphi

@signal_base.function
def cw_shape_helper(toas, pos, pdist, cos_gwtheta, gwphi, cos_inc,
                log10_mc, log10_fgw, phase0, psi, psrTerm, p_dist,
                p_phase, evolve, phase_approx, check, tref):
    
    # Generate waveform at reference distance of 1 Mpc (log10_dist = 0)
    return ee_deterministic.cw_delay(toas, pos, pdist, 
                                     cos_gwtheta=cos_gwtheta, gwphi=gwphi,
                                     cos_inc=cos_inc, log10_mc=log10_mc, log10_fgw=log10_fgw,
                                     log10_dist=0.0, # FIXED at 1 Mpc
                                     log10_h=None, phase0=phase0,
                                     psi=psi, psrTerm=psrTerm, p_dist=p_dist, p_phase=p_phase,
                                     evolve=evolve, phase_approx=phase_approx, check=check, tref=tref)

@lru_cache(maxsize=128)
def get_log10_dist_ref(z, Om0, Ode0):
    """Calculates the reference distance at H0=100 exactly once."""
    cosmo = LambdaCDM(H0=100.0, Om0=Om0, Ode0=Ode0)
    return np.log10(cosmo.luminosity_distance(z).value)

def get_log10_dist_cached(z, log10_H0, Om0, Ode0):
    """Analytically scales distance with log10(H0), completely bypassing Astropy during MCMC."""
    return get_log10_dist_ref(z, Om0, Ode0) + 2.0 - log10_H0


@signal_base.function
def cw_delay_H0(toas, pos, pdist, z, Om0, Ode0, log10_H0, cos_gwtheta, gwphi, cos_inc, log10_mc, log10_fgw, 
                phase0, psi, psrTerm, p_dist, p_phase, evolve, phase_approx, check, tref):
                

    log10_dist = get_log10_dist_cached(z, log10_H0, Om0, Ode0)

    shape = cw_shape_helper(toas, pos, pdist, cos_gwtheta, gwphi, cos_inc,
                            log10_mc, log10_fgw, phase0, psi, psrTerm, p_dist,
                            p_phase, evolve, phase_approx, check, tref)

    return shape * (10.0**(-log10_dist))



def cw_block_H0(z, log10_H0, pdist, p_dist=None, p_phase=None, log10_Mc=None, log10_fgw=None, cosinc=None, 
                phase0=None, psi=None, costh=None, phi=None, psrTerm=True, Om0=0.315, Ode0=0.685, tref=0.0, 
                evolve=evolve_val, phase_approx=phase_approx_val, check=evolve_val, name="cw"):
    """
    A CW block that takes a GLOBAL H0 parameter as an argument.
    It also creates uniquely named pulsar term parameters.
    """
    if log10_fgw is None: log10_fgw = parameter.Uniform(-10, -8)("{}_log10_fgw".format(name))
    if phase0 is None: phase0 = parameter.Uniform(0.0, 2 * np.pi)("{}_phase0".format(name))
    if log10_Mc is None: log10_Mc = parameter.Uniform(8, 11)("{}_log10_Mc".format(name))
    if cosinc is None: cosinc = parameter.Uniform(-1.0, 1.0)("{}_cosinc".format(name))
    if costh is None: costh = parameter.Uniform(-1, 1)("{}_costheta".format(name))
    if phi is None: phi = parameter.Uniform(0, 2 * np.pi)("{}_phi".format(name))
    if psi is None: psi = parameter.Uniform(0, np.pi)("{}_psi".format(name))

    if psrTerm is None:
        p_phase = None
        p_dist = 0

    wf = cw_delay_H0(
        z=z, log10_H0=log10_H0, pdist=pdist, Om0=Om0, Ode0=Ode0, cos_gwtheta=costh, gwphi=phi, cos_inc=cosinc,
        log10_mc=log10_Mc, log10_fgw=log10_fgw, phase0=phase0, psi=psi, psrTerm=psrTerm, p_dist=p_dist, 
        p_phase=p_phase, evolve=evolve, phase_approx=phase_approx, check=check, tref=tref
    )
    
    cw = ee_deterministic.CWSignal(wf, ecc=False, psrTerm=psrTerm, name=name)
    return cw

# ──────────────────────────────────────── LOAD CONFIG FROM JSON FILES ──────────────────────────────────────── #
config_dir = "./cw_config/"
with open(os.path.join(config_dir, 'binary_params_Bpop.json'), 'r') as f:
    config = json.load(f)

with open(os.path.join(config_dir, 'cpta_pulsar_distances.json'), 'r') as f:
    pdist_dict = {k: tuple(v) for k, v in json.load(f).items()}

# Extract cosmology and GWB parameters
cosmo_config = config['cosmology']
gwb_config = config['gwb']

binary_params = {}
for bname, bdata in config['binaries'].items():
    gwtheta, gwphi = radec_to_gwtheta_phi(bdata['ra'], bdata['dec'])
    binary_params[bname] = {
        'name': bdata['name'],
        'ra': bdata['ra'],
        'dec': bdata['dec'],
        'gwtheta': gwtheta,
        'gwphi': gwphi,
        'mc': bdata['mc'],
        'sig': bdata['mc_sigma_log10'],
        'lum_dist': bdata['lum_dist'],
        'fgw': bdata['fgw'],
        'phase0': bdata['phase0'],
        'psi': bdata['psi'],
        'inc': np.deg2rad(bdata['inc_deg']),
        'z': bdata['z'],
    }

B1_ra, B1_dec = binary_params['B1']['ra'], binary_params['B1']['dec']
B1_gwtheta, B1_gwphi = binary_params['B1']['gwtheta'], binary_params['B1']['gwphi']
B1_z = binary_params['B1']['z']
B1_mc = binary_params['B1']['mc'] 
B1_sig = binary_params['B1']['sig']
B1_lum_dist = binary_params['B1']['lum_dist']
B1_fgw = binary_params['B1']['fgw'] 
B1_phase0, B1_psi = binary_params['B1']['phase0'], binary_params['B1']['psi']
B1_inc = binary_params['B1']['inc']

B2_ra, B2_dec = binary_params['B2']['ra'], binary_params['B2']['dec']
B2_gwtheta, B2_gwphi = binary_params['B2']['gwtheta'], binary_params['B2']['gwphi']
B2_z = binary_params['B2']['z']
B2_mc = binary_params['B2']['mc'] 
B2_sig = binary_params['B2']['sig']
B2_lum_dist = binary_params['B2']['lum_dist']
B2_fgw = binary_params['B2']['fgw'] 
B2_phase0, B2_psi = binary_params['B2']['phase0'], binary_params['B2']['psi']
B2_inc = binary_params['B2']['inc']

B3_ra, B3_dec = binary_params['B3']['ra'], binary_params['B3']['dec']
B3_gwtheta, B3_gwphi = binary_params['B3']['gwtheta'], binary_params['B3']['gwphi']
B3_z = binary_params['B3']['z']
B3_mc = binary_params['B3']['mc'] 
B3_sig = binary_params['B3']['sig']
B3_lum_dist = binary_params['B3']['lum_dist']
B3_fgw = binary_params['B3']['fgw'] 
B3_phase0, B3_psi = binary_params['B3']['phase0'], binary_params['B3']['psi']
B3_inc = binary_params['B3']['inc']

truth_dict = {
    "log10_H0": cosmo_config['log10_H0_truth'],
    "H0": 10**cosmo_config['log10_H0_truth'],
    "Omega_m": cosmo_config['Om0'],
    "Omega_de": cosmo_config['Ode0'],
    'gw_gamma': gwb_config['gw_gamma'],
    'gw_log10_A': gwb_config['gw_log10_A'],
}

mcmc_start_dict = truth_dict.copy()

for bname, bp in binary_params.items():
    truth_dict[f"{bname}_log10_Mc"] = np.log10(bp['mc'])
    truth_dict[f"{bname}_cosinc"] = np.cos(bp['inc'])
    truth_dict[f"{bname}_phase0"] = bp['phase0']
    truth_dict[f"{bname}_psi"] = bp['psi']
    truth_dict[f"{bname}_log10_fgw"] = np.log10(bp['fgw'])
    truth_dict[f"{bname}_costheta"] = np.cos(bp['gwtheta'])
    truth_dict[f"{bname}_phi"] = bp['gwphi']

    mcmc_start_dict[f"{bname}_log10_Mc"] = np.log10(bp['mc']) 
    mcmc_start_dict[f"{bname}_cosinc"] = np.cos(bp['inc'])
    mcmc_start_dict[f"{bname}_phase0"] = bp['phase0']
    mcmc_start_dict[f"{bname}_psi"] = bp['psi']
    mcmc_start_dict[f"{bname}_log10_fgw"] = np.log10(bp['fgw']) 
    mcmc_start_dict[f"{bname}_costheta"] = np.cos(bp['gwtheta'])
    mcmc_start_dict[f"{bname}_phi"] = bp['gwphi']
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

if run_mcmc_analysis:
    print("Starting with: ", num_psrs)
    print("Paramaers: ", params)
    print("N_samples: ", n_samples)
    print("Output Directory: ", chains_outdir)
    print("filename: ", plt_out_filename)

    parfiles = [os.path.abspath(p) for p in sorted(glob.glob(data_dir + '*.par'))][:num_psrs]
    timfiles = [os.path.abspath(f) for f in sorted(glob.glob(data_dir + '*.tim'))][:num_psrs]

    def _load(args):
        par, tim = args
        return Pulsar(par, tim)

    with ProcessPoolExecutor(max_workers=len(parfiles)) as ex:
        enterprise_psrs = list(ex.map(_load, zip(parfiles, timfiles)))
# ──────────────────────────────────────── INJECTION ──────────────────────────────────────── #
    np.random.seed(7777)
    injected_prn_val = {}

    if add_inj:
        make_ideal_parallel(enterprise_psrs, parfiles, timfiles)
        import gc; gc.collect()

        max_T_all = max([psr.toas.max() - psr.toas.min() for psr in enterprise_psrs])
        for psr in enterprise_psrs:

            if wn: inject_white_noise(psr)

            if prn: 
                log10_A_sample = TruncNormalSampler(mu=-16.6, sigma=1.8, pmin=-19.0, pmax=-10.0)
                gamma_sample = TruncNormalSampler(mu=2.5, sigma=1.1, pmin=1.0, pmax=7.0)                
                inject_red_noise(psr, log10_A=log10_A_sample, gamma=gamma_sample)
                injected_prn_val[f"{psr.name}"] = log10_A_sample, gamma_sample

            if curn: inject_curn(psr, log10_A=truth_dict["gw_log10_A"], gamma=truth_dict["gw_gamma"], max_T_all=max_T_all)

            if cw:
                tref_inj = 53000 * 86400.0

                binaries = [
                    dict(cos_gwtheta=np.cos(B1_gwtheta), gwphi=B1_gwphi, cos_inc=np.cos(B1_inc),
                        log10_mc=np.log10(B1_mc), log10_fgw=np.log10(B1_fgw),
                        log10_dist=np.log10(B1_lum_dist), phase0=B1_phase0, psi=B1_psi),
                    dict(cos_gwtheta=np.cos(B2_gwtheta), gwphi=B2_gwphi, cos_inc=np.cos(B2_inc),
                        log10_mc=np.log10(B2_mc), log10_fgw=np.log10(B2_fgw),
                        log10_dist=np.log10(B2_lum_dist), phase0=B2_phase0, psi=B2_psi),
                    dict(cos_gwtheta=np.cos(B3_gwtheta), gwphi=B3_gwphi, cos_inc=np.cos(B3_inc),
                        log10_mc=np.log10(B3_mc), log10_fgw=np.log10(B3_fgw),
                        log10_dist=np.log10(B3_lum_dist), phase0=B3_phase0, psi=B3_psi),
                ]

                common_kwargs = dict(
                    pdist=(pdist_dict.get(psr.name)[0], 0.0),
                    psrTerm=True, p_dist=0, p_phase=None,
                    evolve=evolve_val, phase_approx=phase_approx_val,
                    check=evolve_val, tref=tref_inj,
                )

                for binary in binaries[:num_binaries]:
                    inject_cw(psr, **common_kwargs, **binary)
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
    tnm = TimingModel(use_svd=True, normed=True, coefficients=False,)
    wnm = white_signals.MeasurementNoise(efac=parameter.Constant(1), selection=Selection(no_selection),)
    log10_H0_glob = parameter.Uniform(1.5, 2.0)("log10_H0")

    # B1 Globals
    nameB1 = "B1"
    B1_log10_Mc_prior = parameter.Normal(mu=np.log10(B1_mc), sigma=B1_sig)("{}_log10_Mc".format(nameB1))
    B1_phase0_prior = parameter.Uniform(0.0, 2 * np.pi)("{}_phase0".format(nameB1))
    B1_costh = parameter.Constant(np.cos(B1_gwtheta))("{}_costheta".format(nameB1))
    B1_cosinc_prior = parameter.Uniform(-1.0, 1.0)("{}_cosinc".format(nameB1))
    B1_psi_prior = parameter.Uniform(0, np.pi)("{}_psi".format(nameB1))
    B1_phi = parameter.Constant(B1_gwphi)("{}_phi".format(nameB1))

    # B2 Globals
    nameB2 = "B2"
    B2_log10_Mc_prior = parameter.Normal(mu=np.log10(B2_mc), sigma=B2_sig)("{}_log10_Mc".format(nameB2))
    B2_phase0_prior = parameter.Uniform(0.0, 2 * np.pi)("{}_phase0".format(nameB2))
    B2_costh = parameter.Constant(np.cos(B2_gwtheta))("{}_costheta".format(nameB2))
    B2_cosinc_prior = parameter.Uniform(-1.0, 1.0)("{}_cosinc".format(nameB2))
    B2_psi_prior = parameter.Uniform(0, np.pi)("{}_psi".format(nameB2))
    B2_phi = parameter.Constant(B2_gwphi)("{}_phi".format(nameB2))

    # B3 Globals
    nameB3 = "B3"
    B3_log10_Mc_prior = parameter.Normal(mu=np.log10(B3_mc), sigma=B3_sig)("{}_log10_Mc".format(nameB3))
    B3_phase0_prior = parameter.Uniform(0.0, 2 * np.pi)("{}_phase0".format(nameB3))
    B3_costh = parameter.Constant(np.cos(B3_gwtheta))("{}_costheta".format(nameB3))
    B3_cosinc_prior = parameter.Uniform(-1.0, 1.0)("{}_cosinc".format(nameB3))
    B3_psi_prior = parameter.Uniform(0, np.pi)("{}_psi".format(nameB3))
    B3_phi = parameter.Constant(B3_gwphi)("{}_phi".format(nameB3))

    tref_mod = 53000*86400.0

    curnm = common_red_noise_block(psd="powerlaw", prior="log-uniform", logmin=-18.0, logmax=-10.0, gammamin=1,
    gammamax=7, components=30,)

    models = []
    for psr in enterprise_psrs:
        log10_A = parameter.TruncNormal(mu=-16.6, sigma=1.8, pmin=-19.0, pmax=-10.0)(f'{psr.name}_red_noise_log10_A')
        gamma = parameter.TruncNormal(mu=2.5, sigma=1.1, pmin=1.0, pmax=7.0)(f'{psr.name}_red_noise_gamma') 
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)   
        prnm = gp_signals.FourierBasisGP(pl, components=30)

        p_dist_prior = parameter.Normal(0, 1)(f'{psr.name}_p_dist')
        cw_gondorM = cw_block_H0(
                        log10_H0=log10_H0_glob, pdist=pdist_dict.get(psr.name), name=f'B1', costh=B1_costh, phi=B1_phi,
                        log10_Mc = B1_log10_Mc_prior, cosinc=B1_cosinc_prior, phase0=B1_phase0_prior, psi=B1_psi_prior,
                        log10_fgw=np.log10(B1_fgw), z=B1_z, psrTerm=True, evolve=evolve_val, phase_approx=phase_approx_val, tref=tref_mod,
                        check=evolve_val, p_dist=p_dist_prior, #p_phase=p_phase_prior,
                        )
        cw_rohanM = cw_block_H0(
                        log10_H0=log10_H0_glob, pdist=pdist_dict.get(psr.name), name=f'B2', costh=B2_costh, phi=B2_phi,
                        log10_Mc = B2_log10_Mc_prior, cosinc=B2_cosinc_prior, phase0=B2_phase0_prior, psi=B2_psi_prior,
                        log10_fgw=np.log10(B2_fgw), z=B2_z, psrTerm=True, evolve=evolve_val, phase_approx=phase_approx_val, tref=tref_mod,
                        check=evolve_val, p_dist=p_dist_prior, #p_phase=p_phase_prior,
                        )
        cw_sdssM = cw_block_H0(
                        log10_H0=log10_H0_glob, pdist=pdist_dict.get(psr.name), name=f'B3', costh=B3_costh, phi=B3_phi,
                        log10_Mc = B3_log10_Mc_prior, cosinc=B3_cosinc_prior, phase0=B3_phase0_prior, psi=B3_psi_prior,
                        log10_fgw=np.log10(B3_fgw), z=B3_z, psrTerm=True, evolve=evolve_val, phase_approx=phase_approx_val, tref=tref_mod,
                        check=evolve_val, p_dist=p_dist_prior, #p_phase=p_phase_prior,
                        )
        
        model = tnm
        if wn: model += wnm
        if prn: model += prnm
        if curn: model += curnm
        if cw: 
            if num_binaries==1: model += cw_gondorM
            if num_binaries==2: model += cw_gondorM + cw_rohanM 
            if num_binaries==3: model += cw_gondorM + cw_rohanM + cw_sdssM
        models.append(model(psr))
    
    pta = signal_base.PTA(models)
    param_names = pta.param_names
    print(param_names)

    # ──────────────────────────────────────── AITOFF PLOT ──────────────────────────────────────── #
    if add_inj:
        # ── Pulsar coordinates ──
        ra_list, dec_list, name_list = zip(*[
            (psr._raj, psr._decj, psr.name) for psr in enterprise_psrs
        ])
        coords_psrs = SkyCoord(ra=np.array(ra_list)*u.rad, dec=np.array(dec_list)*u.rad, frame='icrs')
        ra1  = coords_psrs.ra.wrap_at(180*u.deg).radian
        dec1 = coords_psrs.dec.radian

        # ── Sky sensitivity grid ──
        pulsar_positions = [
            np.array([np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)])
            for r, d in zip(ra1, dec1)
        ]
        n_ra, n_dec = 120, 60                          # 2× resolution
        ra_range  = np.linspace(-np.pi, np.pi, n_ra)
        dec_range = np.linspace(-np.pi/2, np.pi/2, n_dec)
        gwphi_grid, gwdec_grid = np.meshgrid(ra_range, dec_range)
        gwtheta_grid = np.pi/2 - gwdec_grid

        sky_sensitivity = np.array([
            [sum(
                (lambda fp, fc, _: fp**2 + fc**2)(
                    *utils.create_gw_antenna_pattern(pos, gwtheta_grid[i,j], gwphi_grid[i,j])
                )
                for pos in pulsar_positions
            ) for j in range(n_ra)]
            for i in range(n_dec)
        ])

        # ── Binary coordinates ──
        ra_dec_list = [(globals()[f"B{i}_ra"], globals()[f"B{i}_dec"]) for i in range(1, num_binaries+1)]
        coords_bin  = SkyCoord([r for r,d in ra_dec_list], [d for r,d in ra_dec_list],
                            unit=(u.hourangle, u.deg), frame='icrs')
        ra2  = coords_bin.ra.wrap_at(180*u.deg).radian
        dec2 = coords_bin.dec.radian

        # ──────────────────────────── Figure ────────────────────────────
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 8), facecolor='#0d0d0d')
        ax  = fig.add_subplot(111, projection='aitoff')
        ax.set_facecolor('#0d0d0d')
        ax.grid(True, color='white', alpha=0.12, linewidth=0.5, linestyle='--')

        # ── Sensitivity heatmap ──
        im = ax.pcolormesh(gwphi_grid, gwdec_grid, np.log10(sky_sensitivity),
                        cmap='inferno', shading='gouraud', alpha=0.92)
        cbar = fig.colorbar(im, orientation='horizontal', pad=0.06, fraction=0.03,
                            shrink=0.6, aspect=40)
        cbar.set_label('log10 (Relative Sensitivity)', fontsize=11, color='white', labelpad=8)
        cbar.ax.xaxis.set_tick_params(color='white', labelsize=9)
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color='white')

        # ── Pulsars ──
        ax.scatter(ra1, dec1, s=25, color='#00e5ff', marker='o',
                zorder=3, alpha=0.85, linewidths=0, label='Pulsars')
        for i, name in enumerate(name_list):
            ax.text(ra1[i] + 0.02, dec1[i] + 0.02, name,
                    fontsize=6, alpha=0.75, color='#80f0ff',
                    ha='left', va='bottom',
                    fontfamily='monospace')

        glow_sizes   = [600, 300, 120]
        glow_alphas  = [0.06, 0.12, 0.25]
        glow_color   = '#ffdd00'
        for gs, ga in zip(glow_sizes, glow_alphas):
            ax.scatter(ra2, dec2, s=gs, color=glow_color, marker='*',
                    zorder=4, alpha=ga, linewidths=0)
        ax.scatter(ra2, dec2, s=90, color=glow_color, marker='*',
                zorder=5, linewidths=0.4, label='Injected Binaries')
        for i, (r, d) in enumerate(zip(ra2, dec2), start=1):
            ax.text(r + 0.03, d + 0.03, chr(64+i),
                    fontsize=11, fontweight='bold',
                    ha='left', va='bottom', color=glow_color,
                    fontfamily='monospace', zorder=6)

        # ── Legend & title ──
        leg = ax.legend(loc='lower right', fontsize=10, framealpha=0.25,
                        facecolor='#1a1a1a', edgecolor='white',
                        markerscale=1.4, labelcolor='white')
        ax.set_title('PTA Sky Sensitivity  ·  Pulsars and Injected Binaries',
                    fontsize=15, color='white', pad=22, fontweight='bold',
                    fontfamily='monospace')

        # ── Axis tick colours ──
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        plt.tight_layout()
        plt.savefig(f"{skymap_filename}.png", bbox_inches='tight', dpi=300,
                    facecolor=fig.get_facecolor())
        plt.close()
        plt.style.use('default')

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #

    pulsar_names = [psr.name for psr in enterprise_psrs]
    for name in pulsar_names:
        for _dict in (truth_dict, mcmc_start_dict):
            _dict[f"{name}_p_dist"] = 0.0
            _dict[f"{name}_p_phase"] = np.pi/2
            if prn:
                _dict[f"{name}_red_noise_log10_A"] = injected_prn_val[f"{name}"][0]
                _dict[f"{name}_red_noise_gamma"] = injected_prn_val[f"{name}"][1]


    truth_dict_filename = os.path.join(chains_outdir, 'truth.json')
    clean_dict = {k: float(v) for k, v in truth_dict.items()}
    with open(truth_dict_filename, "w") as f:
        json.dump(clean_dict, f, indent=4)
        
    mcmc_start_dict_filename = os.path.join(chains_outdir, 'mcmc_start.json')
    clean_mcmc_dict = {k: float(v) for k, v in mcmc_start_dict.items()}
    with open(mcmc_start_dict_filename, "w") as f:
        json.dump(clean_mcmc_dict, f, indent=4)

    
    x0 = np.array([mcmc_start_dict[p] for p in param_names])
    ndim = len(x0)

    # ── Parameter whitening ────────────────────────────────────────────────────
    from whiten import WhitenedPTA
    wpta = WhitenedPTA(pta) 
    u0 = wpta.transform(x0) 

    cov = np.eye(ndim) * 0.01**2

    sampler = ptmcmc(
        ndim,
        wpta.get_lnlikelihood,        
        wpta.get_lnprior,            
        cov,
        outDir=chains_outdir,
        resume=True,
        seed=7777,
    )
    np.savetxt(os.path.join(chains_outdir, 'pars.txt'), pta.param_names, fmt='%s')

    np.save(os.path.join(chains_outdir, 'whiten_a.npy'), wpta._a)
    np.save(os.path.join(chains_outdir, 'whiten_b.npy'), wpta._b)
    np.save(os.path.join(chains_outdir, 'whiten_ttype.npy'),
            np.array(wpta._ttype, dtype=str))
    np.save(os.path.join(chains_outdir, 'whiten_tn_lo.npy'), wpta._tn_lo)
    np.save(os.path.join(chains_outdir, 'whiten_tn_range.npy'), wpta._tn_range)

    print("\nStarting sampler (whitened parameter space)...")
    sampler.sample(u0, n_samples, DEweight=50, AMweight=15, SCAMweight=30)
    print(f"\n<=== MCMC run complete! Chains saved in '{chains_outdir}' ===>")

if dewhiten:
    print('Reading whitened chain file...')
    
    chain_bin_path = os.path.join(chains_outdir, 'chain_1.npy')
    chain_txt_path = os.path.join(chains_outdir, 'chain_1.txt')
    
    import pandas as pd
    chain_raw = pd.read_csv(chain_txt_path, sep=r'\s+', header=None, dtype=np.float64).values    
    print('Read!')

    whiten_ttype_file = os.path.join(chains_outdir, 'whiten_ttype.npy')
    if os.path.exists(whiten_ttype_file):
        _a      = np.load(os.path.join(chains_outdir, 'whiten_a.npy'))
        _b      = np.load(os.path.join(chains_outdir, 'whiten_b.npy'))
        _ttype  = np.load(whiten_ttype_file, allow_pickle=True)
        _tn_lo  = np.load(os.path.join(chains_outdir, 'whiten_tn_lo.npy'))
        _tn_range = np.load(os.path.join(chains_outdir, 'whiten_tn_range.npy'))

        ndim_w = len(_a)
        u = chain_raw[:, :ndim_w]

        mask_uniform   = np.array([t == 'uniform'   for t in _ttype])
        mask_truncnorm = np.array([t == 'truncnorm' for t in _ttype])
        mask_linear    = ~mask_uniform & ~mask_truncnorm

        x = np.empty_like(u)

        if mask_uniform.any():
            phi = ndtr(u[:, mask_uniform])
            x[:, mask_uniform] = _a[mask_uniform] + (_b[mask_uniform] - _a[mask_uniform]) * phi

        if mask_truncnorm.any():
            phi = ndtr(u[:, mask_truncnorm])
            p = np.clip(
                _tn_lo[mask_truncnorm] + phi * _tn_range[mask_truncnorm],
                1e-10, 1 - 1e-10
            )
            x[:, mask_truncnorm] = _a[mask_truncnorm] + _b[mask_truncnorm] * ndtri(p)

        if mask_linear.any():
            x[:, mask_linear] = _a[mask_linear] + _b[mask_linear] * u[:, mask_linear]

        chain = chain_raw.copy()
        chain[:, :ndim_w] = x
        print("[INFO] Inverse-transformed whitened chain to physical space.")
    else:
        chain = chain_raw
        print("[INFO] No whitening detected — using chain as-is.")

    print('Saving chain (physical)...')
    np.save(os.path.join(chains_outdir, "chain_physical.npy"), chain)
    print('Saved!')

if make_corner_plot:
    parfiles = [p for p in sorted(glob.glob(data_dir + '*.par')) if '.t2' not in p][:num_psrs]
    timfiles = [f for f in sorted(glob.glob(data_dir + '*.tim')) if '.t2' not in f][:num_psrs]
    pulsar_names = [os.path.basename(f).split('.')[0] for f in parfiles]


    print('Reading chain_physical.npy')
    if not dewhiten: chain = np.load(os.path.join(chains_outdir, "chain_physical.npy"))
    print('Read chain_physical.npy!')
    
    params_full = np.loadtxt(os.path.join(chains_outdir, 'pars.txt'), dtype=str)

    burn_in_fraction = 0.25
    chain_burned = chain[int(burn_in_fraction * len(chain)):]

    print(f"Loaded {len(chain_burned)} posterior samples after discarding {burn_in_fraction*100}% for burn-in.")

    params_to_plot = []
    params_to_plot_names = []
    if cw:
        for i in range (1, num_binaries+1):
            params_to_plot.append(f"B{i}_log10_Mc")
            if i == 1: params_to_plot_names.append(r"B1 $\ \log_{10}M_c$")
            if i == 2: params_to_plot_names.append(r"B2 $\ \log_{10}M_c$")
            if i == 3: params_to_plot_names.append(r"B3 $\ \log_{10}M_c$")


        params_to_plot.append("log10_H0")
        params_to_plot_names.append(r"$H_0~[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
    
    if curn:
        params_to_plot.append('gw_log10_A')
        params_to_plot.append('gw_gamma')
        params_to_plot_names.append(r"$\mathrm{log_{10}A_{gw}}$")
        params_to_plot_names.append(r"$\mathrm{\gamma_{gw}}$")
    
    param_indices = [list(params_full).index(p) for p in params_to_plot]
    posterior_samples_filtered = chain_burned[:, param_indices]
    
    if 'log10_H0' in params_to_plot:
        h0_idx = params_to_plot.index('log10_H0')
        posterior_samples_filtered[:, h0_idx] = 10 ** posterior_samples_filtered[:, h0_idx]
        params_to_plot[h0_idx] = 'H0'

    print("\nPlotting the corner plot for the following parameters:")
    print(params_to_plot)

    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    truth_path = os.path.join(chains_outdir, "truth.json")

    with open(truth_path, "r") as f:
        true_values = json.load(f)

    expected_vals_ordered = []
    for p in params_to_plot:
        if p == "H0" and "H0" in true_values:
            expected_vals_ordered.append(true_values["H0"])
        else:
            expected_vals_ordered.append(true_values.get(p))

    print("──────────────────────────────────────── Plotting ────────────────────────────────────────")
    print("\nGenerating corner plot...")
    fig = corner.corner(posterior_samples_filtered,
                        labels=params_to_plot_names,
                        show_titles=True,
                        quantiles=[0.16, 0.5, 0.84],
                        title_fmt=".5f",
                        bins=30,
                        smooth=False,
                        color="#5da5da",
                        plot_datapoints=False,
                        fill_contours=True,
                        plot_density=False,
                        levels=[0.393, 0.865, 0.989],
                        truths=expected_vals_ordered,
                        
                        title_kwargs={"fontsize": 17},
                        label_kwargs={"fontsize": 17},

                        truth_color="orange",
                        hist_kwargs={"linewidth": 2}     
                        )
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=14)

    plt.savefig(f"{plt_out_filename}.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{plt_out_filename}.pdf", bbox_inches='tight')
    print(f"Saved to {plt_out_filename}.png")
    print(f"Saved to {plt_out_filename}.pdf")
    plt.close()


if make_hist_plot:
    print("\nGenerating individual histogram plots...")

    if 'chain_burned' not in dir():
        parfiles = [p for p in sorted(glob.glob(data_dir + '*.par')) if '.t2' not in p][:num_psrs]
        pulsar_names = [os.path.basename(f).split('.')[0] for f in parfiles]

        print('Reading chain_physical.npy')
        if not make_corner_plot: chain = np.load(os.path.join(chains_outdir, "chain_physical.npy"))
        print('Read chain_physical.npy!')

        params_full = np.loadtxt(os.path.join(chains_outdir, 'pars.txt'), dtype=str)
        burn_in_fraction = 0.25
        chain_burned = chain[int(burn_in_fraction * len(chain)):]
        print(f"Loaded {len(chain_burned)} posterior samples after discarding {burn_in_fraction*100}% for burn-in.")

    params_full = np.loadtxt(os.path.join(chains_outdir, 'pars.txt'), dtype=str)
    all_params = list(params_full)
    posterior_all = chain_burned[:, :len(all_params)].copy()

    if 'log10_H0' in all_params:
        h0_idx = all_params.index('log10_H0')
        posterior_all[:, h0_idx] = 10 ** posterior_all[:, h0_idx]
        all_params[h0_idx] = 'H0'

    # ── Load truths ──
    truth_path = os.path.join(chains_outdir, "truth.json")
    with open(truth_path, "r") as f:
        true_values = json.load(f)

    truth_vals_all = []
    for p in all_params:
        if p == "H0" and "H0" in true_values:
            truth_vals_all.append(true_values["H0"])
        else:
            truth_vals_all.append(true_values.get(p, None))

    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    n_params = len(all_params)
    n_cols = 4
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, param in enumerate(all_params):
        ax = axes[i]
        samples = posterior_all[:, i]

        ax.hist(samples, bins=50, density=True, color="#5da5da", alpha=0.7,
                linewidth=1.5, edgecolor='white', label='Posterior')

        q16, q50, q84 = np.quantile(samples, [0.16, 0.50, 0.84])
        ax.axvline(q50, color="#5da5da", linewidth=2,   linestyle='-',  label='Median')
        ax.axvline(q16, color="#5da5da", linewidth=1.5, linestyle='--')
        ax.axvline(q84, color="#5da5da", linewidth=1.5, linestyle='--')

        truth_val = truth_vals_all[i]
        if truth_val is not None:
            ax.axvline(truth_val, color='orange', linewidth=2, linestyle='-', label='Truth')

        err_lo = q50 - q16
        err_hi = q84 - q50
        ax.set_title(f"${q50:.4f}^{{+{err_hi:.4f}}}_{{-{err_lo:.4f}}}$", fontsize=11)
        ax.set_xlabel(param, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.tick_params(axis='both', labelsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Posterior Histograms — {num_psrs} pulsars, {params}",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    hist_filename = os.path.join(chains_outdir, f'_hist_{num_psrs}psr_{params}_{int(n_samples/1000)}k_samples_{suffix}')
    plt.savefig(f"{hist_filename}.pdf", bbox_inches='tight')
    print(f"Saved to {hist_filename}.pdf")
    plt.close()

if make_corner_white:
    print("\nGenerating whitened-space corner plot...")

    chain_txt_path = os.path.join(chains_outdir, 'chain_1.txt')
    import pandas as pd
    print('Reading chain_1.txt (whitened)...')
    chain_white_raw = pd.read_csv(chain_txt_path, sep=r'\s+', header=None, dtype=np.float64).values
    print('Read!')

    params_full = np.loadtxt(os.path.join(chains_outdir, 'pars.txt'), dtype=str)
    all_params   = list(params_full)
    ndim_w       = len(all_params)

    burn_in_fraction = 0.25
    chain_white_burned = chain_white_raw[int(burn_in_fraction * len(chain_white_raw)):]
    chain_white_burned = chain_white_burned[::10]
    posterior_white = chain_white_burned[:, :ndim_w]

    print(f"Whitened chain shape after burn-in + thinning: {posterior_white.shape}")
    print(f"Per-parameter std in whitened space (should be ~1 if well mixed):")
    for i, p in enumerate(all_params):
        print(f"  {p:45s}  std = {posterior_white[:, i].std():.4f}  mean = {posterior_white[:, i].mean():.4f}")