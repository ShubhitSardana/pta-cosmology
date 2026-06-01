# The Targeted Standard Siren Cosmology with Pulsar Timing Arrays

[![arXiv](https://img.shields.io/badge/arXiv-2603.12168v1-B31B1B.svg)](https://arxiv.org/abs/2603.12168v1)

This repository contains the Bayesian inference code used in the paper **“The Targeted Standard Siren Cosmology with Pulsar Timing Arrays”** ([arXiv:2603.12168v1](https://arxiv.org/abs/2603.12168v1)).

The framework performs joint parameter estimation of:

* Multiple Continuous Gravitational Wave (CGW) sources
* Pulsar Timing Array (PTA) noise processes
* Cosmological parameters such as:

  * Hubble constant $H_0$
  * Matter density parameter $\Omega_m$

using PTA timing residual data and MCMC-based Bayesian inference.

---

# Repository Structure

```text
.
├── cgw_cosmo.py      # Joint CGW + cosmology MCMC analysis
├── cgw.py            # Standard multi-binary CGW MCMC analysis
├── inject_signals.py # Signal and noise injection utilities
├── whiten.py         # Whitening transforms for PTMCMCSampler
├── data/             # Simulated CPTA .par and .tim files
└── cw_config/        # JSON configuration files
└── output_chains/    # Output chain files
```

## Main Scripts

### `cgw_cosmo.py`

Primary execution script for the combined Continuous Gravitational Wave and cosmology inference.

### `cgw.py`

Runs standard CGW parameter estimation without Matter density parameter $\Omega_m$ inference.

### `inject_signals.py`

Utilities for injecting:

* Continuous gravitational wave signals
* White noise
* Pulsar red noise
* Common red noise processes

into PTA datasets.

### `whiten.py`

Implements optimized parameter whitening transformations for `PTMCMCSampler` to improve convergence and sampling efficiency in high-dimensional parameter spaces.

---

# Features

* Bayesian inference for CGW sources in PTA datasets
* Joint cosmological parameter estimation
* Multi-binary signal modeling
* Synthetic signal injection
* Noise modeling:

  * White Noise
  * Pulsar Red Noise
  * Common Uncorrelated Red Noise
* Whitening support for faster MCMC convergence
* Posterior visualization:

  * Corner plots
  * 1D marginalized histograms
---

# Dependencies

The codebase relies on the standard PTA analysis ecosystem and scientific Python libraries.

## Core PTA Packages

* `enterprise`
* `enterprise_extensions`
* `PTMCMCSampler`
* `pint-pulsar`

## Scientific Python Stack

* `numpy`
* `scipy`
* `pandas`
* `astropy`

## Visualization

* `matplotlib`
* `corner`

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ShubhitSardana/pta-cosmology.git
cd pta-cosmology
```
---

# Usage

Both `cgw_cosmo.py` and `cgw.py` expose a similar command-line interface via `argparse`.

## Example

Run an MCMC analysis with injected signals for 10 pulsars and generate posterior corner plots:

```bash
python -u cgw_local_whiten.py \
    --run_mcmc \
    --add_inj \
    -p 10 \
    -b 2 \
    -n 10_000_000 \
    --suffix H0+Om \
    --data_dir ./data/CPTA/cpta-b40-40yr/ \
    -o ./output_local_whiten/b40-40yr/
```


---

# Command-Line Arguments

| Argument             | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| `--run_mcmc`         | Execute MCMC sampling                                      |
| `--add_inj`          | Inject synthetic GW signals and noise                      |
| `-p`, `--num_psrs`   | Number of pulsars to include                               |
| `-b`, `--num_bina`   | Number of GW binaries to model                             |
| `-n`, `--n_samples`  | Number of MCMC samples                                     |
| `--de_whiten`, `-dw` | Transform whitened chains back to physical parameter space |
| `--corner`           | Generate posterior corner plots                            |
| `--hist`             | Generate marginalized 1D histograms                        |
| `--no_cw`            | Disable continuous wave modeling                           |
| `--no_wn`            | Disable white noise                                        |
| `--no_prn`           | Disable pulsar red noise                                   |
| `--no_curn`          | Disable common uncorrelated red noise                      |

---

# Data and Configuration

## `data/`

Contains PTA timing data:

* `.par` files
* `.tim` files

## `cw_config/`

Contains JSON configuration files for:

* Injected binaries
* Cosmological parameters
* Pulsar distances

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{cgw_cosmo_2026,
  title={Continuous Gravitational Waves and Cosmology with PTAs},
  author={Author Names},
  journal={arXiv preprint arXiv:2603.12168},
  year={2026}
}
```

---

# License

Specify your project license here.

Example:

```text
MIT License
```
