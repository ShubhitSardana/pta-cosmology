import os
import tempfile
import numpy as np
from pint.residuals import Residuals
from astropy.time import TimeDelta
import pint.toa as toa
import pint.models as models
import gc
from enterprise.pulsar import Pulsar
from enterprise import constants as const
from concurrent.futures import ThreadPoolExecutor
from enterprise.signals.gp_bases import createfourierdesignmatrix_red
from enterprise_extensions import deterministic as ee_deterministic
from concurrent.futures import ProcessPoolExecutor


def _make_ideal_single(args):
    parfile, timfile, ephem, iterations = args
    
    m = models.get_model(parfile)
    t = toa.get_TOAs(timfile, ephem=ephem, planets=False)
    
    for _ in range(iterations):
        res = Residuals(t, m)
        t.adjust_TOAs(TimeDelta(-1.0 * res.time_resids))
    
    corrected_stoas = np.ascontiguousarray(t.get_mjds().value * 86400.0, dtype=np.float64)
    final_res = np.ascontiguousarray(Residuals(t, m).time_resids.to_value('s'), dtype=np.float64)
    
    del m, t, res
    return corrected_stoas, final_res


from concurrent.futures import ProcessPoolExecutor

def make_ideal_parallel(enterprise_psrs, parfiles, timfiles, iterations=2, ephem='DE440'):
    args = [(par, tim, ephem, iterations) for par, tim in zip(parfiles, timfiles)]
    
    with ProcessPoolExecutor(max_workers=min(len(parfiles), os.cpu_count()//2 or 1)) as ex:
        results = list(ex.map(_make_ideal_single, args))
    
    for psr, (corrected_stoas, final_res) in zip(enterprise_psrs, results):
        psr._stoas = corrected_stoas
        psr._residuals = final_res
    
    gc.collect()


def _make_ideal_older(psr):
    psr._stoas -= psr.residuals
    psr._residuals = np.zeros(len(psr.residuals))


def inject_white_noise(psr):
    white_noise = np.random.normal(0, np.ascontiguousarray(psr.toaerrs, dtype=np.float64))
    psr._stoas += white_noise
    psr._residuals += white_noise


def inject_red_noise(psr, log10_A, gamma, components=30):
    T = psr.toas.max() - psr.toas.min()
    F, Ffreqs = createfourierdesignmatrix_red(psr.toas, nmodes=components, Tspan=T)
    df = np.diff(np.concatenate((np.array([0]), Ffreqs[::2])))
    psd = (10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma - 3) * Ffreqs**(-gamma) * np.repeat(df, 2)
    y = np.sqrt(psd) * np.random.randn(len(Ffreqs))
    red_noise = np.ascontiguousarray(F @ y, dtype=np.float64)
    psr._stoas += red_noise
    psr._residuals += red_noise


def inject_curn(psr, log10_A, gamma, max_T_all, components=30):
    F, Ffreqs = createfourierdesignmatrix_red(psr.toas, nmodes=components, Tspan=max_T_all)
    df = np.diff(np.concatenate((np.array([0]), Ffreqs[::2])))
    psd = (10**log10_A)**2 / 12.0 / np.pi**2 * const.fyr**(gamma - 3) * Ffreqs**(-gamma) * np.repeat(df, 2)
    y = np.sqrt(psd) * np.random.randn(len(Ffreqs))
    curn_noise = np.ascontiguousarray(F @ y, dtype=np.float64)
    psr._stoas += curn_noise
    psr._residuals += curn_noise


def inject_cw(psr, pdist, cos_gwtheta, gwphi, cos_inc, log10_mc, log10_fgw,
              log10_dist, phase0, psi, tref, p_dist=0, p_phase=None,
              psrTerm=True, evolve=False, phase_approx=True, check=False):
    delay = np.ascontiguousarray(ee_deterministic.cw_delay(
        toas=psr.toas, pos=psr.pos, pdist=pdist,
        cos_gwtheta=cos_gwtheta, gwphi=gwphi, cos_inc=cos_inc,
        log10_mc=log10_mc, log10_fgw=log10_fgw, log10_dist=log10_dist,
        phase0=phase0, psi=psi, psrTerm=psrTerm, p_dist=p_dist,
        p_phase=p_phase, evolve=evolve, phase_approx=phase_approx,
        check=check, tref=tref,
    ), dtype=np.float64)
    psr._stoas += delay
    psr._residuals += delay