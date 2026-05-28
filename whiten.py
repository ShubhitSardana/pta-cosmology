"""
whiten.py — Parameter whitening for PTMCMCSampler (optimised)

Transforms all PTA parameters to unit-Gaussian space so the sampler
sees every dimension with the same scale. This eliminates AM cold-start
and DE overshooting in high-dimensional runs.

Uses scipy.special (bare C) instead of scipy.stats to avoid per-call overhead.
"""
import re
import numpy as np
from scipy.special import ndtr, ndtri   # raw C: Φ(x) and Φ⁻¹(p)


def _parse_prior(param):
    """Parse an enterprise Parameter's _typename to extract prior info."""
    tn = param._typename
    m = re.match(r'(\w+)\((.+)\)', tn)
    if not m:
        raise ValueError(f"Cannot parse prior from _typename: {tn}")
    cls = m.group(1)
    kwargs = {}
    for kv in m.group(2).split(', '):
        k, v = kv.split('=')
        kwargs[k] = float(v)
    return cls, kwargs


_EPS = 1e-10


class WhitenedPTA:
    """Wraps an enterprise PTA to transform parameters to unit-Gaussian space.

    Supported prior types:
        Uniform(pmin, pmax)                → probit transform
        Normal(mu, sigma)                  → standardise
        LinearExp(pmin, pmax)              → probit
        TruncNormal(mu, sigma, pmin, pmax) → truncated normal CDF → probit
    """

    def __init__(self, pta):
        self.pta = pta
        self.param_names = pta.param_names
        self.ndim = len(pta.param_names)

        self._ttype = []
        self._a = np.zeros(self.ndim)
        self._b = np.ones(self.ndim)
        # For TruncNormal: standardised clip bounds + precomputed CDF range
        self._tn_lo = np.zeros(self.ndim)   # Φ((pmin-μ)/σ)
        self._tn_range = np.ones(self.ndim) # Φ((pmax-μ)/σ) - Φ((pmin-μ)/σ)

        for i, p in enumerate(pta.params):
            cls, kw = _parse_prior(p)
            if cls in ('Uniform', 'LinearExp'):
                self._ttype.append('uniform')
                self._a[i] = kw['pmin']
                self._b[i] = kw['pmax']
            elif cls == 'Normal':
                self._ttype.append('normal')
                self._a[i] = kw['mu']
                self._b[i] = kw['sigma']
            elif cls == 'TruncNormal':
                self._ttype.append('truncnorm')
                mu, sig = kw['mu'], kw['sigma']
                self._a[i] = mu
                self._b[i] = sig
                lo = ndtr((kw['pmin'] - mu) / sig)
                hi = ndtr((kw['pmax'] - mu) / sig)
                self._tn_lo[i] = lo
                self._tn_range[i] = hi - lo
            else:
                raise ValueError(f"Unsupported prior type '{cls}' for param '{p.name}'")

        self._uni_mask = np.array([t == 'uniform' for t in self._ttype])
        self._norm_mask = np.array([t == 'normal' for t in self._ttype])
        self._tn_mask = np.array([t == 'truncnorm' for t in self._ttype])

        # Precompute index arrays (avoid np.where every call)
        self._tn_idxs = np.where(self._tn_mask)[0]

        n_uni = int(np.sum(self._uni_mask))
        n_norm = int(np.sum(self._norm_mask))
        n_tn = len(self._tn_idxs)
        print(f"[WhitenedPTA] {n_uni} Uniform/LinearExp + {n_norm} Normal + "
              f"{n_tn} TruncNormal → all mapped to N(0,1)")

    def transform(self, x):
        """Map physical parameters x → whitened parameters u."""
        x = np.asarray(x, dtype=float)
        u = np.empty_like(x)

        # Uniform: u = Φ⁻¹( (x - a) / (b - a) )
        m = self._uni_mask
        t = (x[m] - self._a[m]) / (self._b[m] - self._a[m])
        np.clip(t, _EPS, 1 - _EPS, out=t)
        u[m] = ndtri(t)

        # Normal: u = (x - μ) / σ
        m = self._norm_mask
        u[m] = (x[m] - self._a[m]) / self._b[m]

        # TruncNormal: u = Φ⁻¹( (Φ((x-μ)/σ) - Φ_lo) / Φ_range )
        for j in self._tn_idxs:
            z = (x[j] - self._a[j]) / self._b[j]
            p = (ndtr(z) - self._tn_lo[j]) / self._tn_range[j]
            p = np.clip(p, _EPS, 1 - _EPS)
            u[j] = ndtri(p)

        return u

    def inverse(self, u):
        """Map whitened parameters u → physical parameters x."""
        u = np.asarray(u, dtype=float)
        x = np.empty_like(u)

        # Uniform: x = a + (b - a) * Φ(u)
        m = self._uni_mask
        x[m] = self._a[m] + (self._b[m] - self._a[m]) * ndtr(u[m])

        # Normal: x = μ + σ * u
        m = self._norm_mask
        x[m] = self._a[m] + self._b[m] * u[m]

        # TruncNormal: x = μ + σ * Φ⁻¹( Φ_lo + Φ(u) * Φ_range )
        for j in self._tn_idxs:
            p = self._tn_lo[j] + ndtr(u[j]) * self._tn_range[j]
            p = np.clip(p, _EPS, 1 - _EPS)
            x[j] = self._a[j] + self._b[j] * ndtri(p)

        return x

    def inverse_chain(self, chain):
        """Inverse-transform an entire chain array (N x ndim or N x ncols)."""
        out = chain.copy()
        for i in range(len(chain)):
            out[i, :self.ndim] = self.inverse(chain[i, :self.ndim])
        return out

    def get_lnlikelihood(self, u):
        """Evaluate log-likelihood in whitened space."""
        x = self.inverse(u)
        return self.pta.get_lnlikelihood(x) #+ self._log_jacobian(u)

    def get_lnprior(self, u):
        """Prior in whitened space = standard Gaussian N(0, I)."""
        return -0.5 * np.sum(u**2)

    def make_cov_from_stds(self, param_stds, x0=None):
        """Build a diagonal covariance matrix in whitened space from known
        posterior standard deviations in physical space (e.g. from a corner plot).

        param_stds : dict mapping param substring → physical std.
                     e.g. {'log10_Mc': 0.003, 'red_noise_gamma': 0.92}
                     Substring matching: 'red_noise_gamma' matches all pulsars.
        x0         : starting point in physical space (for Jacobian evaluation).
                     If None, uses the prior midpoint (u=0 mapped back).

        Returns a (ndim x ndim) diagonal covariance matrix in whitened space.
        """
        # Build physical sigma vector via substring matching
        sigma_phys = np.full(self.ndim, 0.1)   # fallback: 0.1 in physical space
        for i, pname in enumerate(self.param_names):
            for key, std in param_stds.items():
                if key.lstrip('_') in pname:
                    sigma_phys[i] = std
                    break

        # Evaluate the Jacobian |du/dx| at x0 (or prior midpoint)
        if x0 is None:
            x0 = self.inverse(np.zeros(self.ndim))
        u0 = self.transform(np.asarray(x0, dtype=float))

        jacobian = np.ones(self.ndim)

        # Uniform/LinearExp: du/dx = 1 / [(b-a) * φ(u)]
        m = self._uni_mask
        phi_u = np.exp(-0.5 * u0[m]**2) / np.sqrt(2 * np.pi)
        jacobian[m] = 1.0 / ((self._b[m] - self._a[m]) * phi_u)

        # Normal: du/dx = 1/σ
        m = self._norm_mask
        jacobian[m] = 1.0 / self._b[m]

        # TruncNormal: du/dx = φ((x-μ)/σ) / (σ · φ(u) · Φ_range)
        for j in self._tn_idxs:
            xj = x0[j]
            z = (xj - self._a[j]) / self._b[j]
            phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            phi_u_j = np.exp(-0.5 * u0[j]**2) / np.sqrt(2 * np.pi)
            jacobian[j] = phi_z / (self._b[j] * phi_u_j * self._tn_range[j])

        # σ_whitened = σ_physical × |du/dx|
        sigma_white = sigma_phys * jacobian
        sigma_white = np.clip(sigma_white, 1e-4, 10.0)  # sanity bounds

        print("[WhitenedPTA] Whitened σ range: "
              f"[{sigma_white.min():.4f}, {sigma_white.max():.4f}]")
        return np.diag(sigma_white**2)

