# Paris Monte Carlo Sampler

An advanced Monte Carlo sampler with adaptive covariance and clustering capabilities for Bayesian inference and optimization problems.

## Features

* **Auto-balanced Exploration**: Samples adaptively shift between discovering new regions and refining known peaks, without manual tuning.
* **Evidence Estimation Built-in**: Accurate Bayesian evidence comes directly from the same weighted samples used for posterior reconstruction.
* **Adaptive Proposals**: Mixture proposals evolve around high-posterior regions while self-correcting overemphasis, ensuring stable coverage.
* **Parallel Mode Discovery**: Independent processes explore in parallel, merging when they converge on the same mode to reduce redundancy.
* **Intuitive Hyperparameters**: Configuration maps naturally to prior knowledge (e.g., expected mode number, scale of prior volume), making it easy for users to fine-tune.
* **Efficient Large-scale Performance**: Scales smoothly in high-dimensional, multi-modal spaces with substantially fewer likelihood calls.
* **Robust Boundary Handling**: Parameters defined on the unit hypercube are automatically respected.
* **Multiprocessing Ready**: Straightforward support for parallel execution on modern hardware.

## Installation

### From PyPI (when available)
```bash
pip install parismc
```

### From Source
```bash
git clone https://github.com/yourusername/parismc.git
cd parismc
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/yourusername/parismc.git
cd parismc
pip install -e .[dev]
```

## Quick Start

```python
import numpy as np
from parismc import Sampler, SamplerConfig

# Define your log-likelihood function
def log_likelihood(x):
    """Example: multivariate Gaussian log-likelihood"""
    return -0.5 * np.sum(x**2, axis=1)

# Create sampler configuration
config = SamplerConfig(
    alpha=1000,
    latest_prob_index=1000,
    boundary_limiting=True,
    use_pool=False  # Set to True for multiprocessing
)

# Initialize sampler
ndim = 2
n_walkers = 5
init_cov_list = [np.eye(ndim) * 0.1] * n_walkers

sampler = Sampler(
    ndim=ndim,
    n_seed=n_walkers,
    log_reward_func=log_likelihood,
    init_cov_list=init_cov_list,
    config=config
)

# Prepare initial samples
sampler.prepare_lhs_samples(lhs_num=1000, batch_size=100)

# Run sampling
sampler.run_sampling(num_iterations=500, savepath='./results')

# Get results
samples, weights = sampler.get_samples_with_weights(flatten=True)
```

## Advanced Usage

### Custom Prior Transform

```python
def uniform_to_normal(x):
    """Transform from [0,1]^d to unbounded space"""
    from scipy.stats import norm
    return norm.ppf(x)

sampler = Sampler(
    ndim=ndim,
    n_seed=n_walkers,
    log_reward_func=log_likelihood,
    init_cov_list=init_cov_list,
    prior_transform=uniform_to_normal
)
```

### Configuration Options

```python
config = SamplerConfig(
    proc_merge_prob=0.9,        # Probability threshold for merging clusters
    alpha=1000,                 # Importance sampling parameter
    latest_prob_index=1000,     # Number of recent samples for weighting
    trail_size=1000,           # Maximum trial samples per iteration
    boundary_limiting=True,     # Enable boundary constraint handling
    use_beta=True,             # Use beta correction for boundaries
    integral_num=100000,       # Monte Carlo samples for beta estimation
    gamma=100,                 # Covariance update frequency
    use_pool=True,             # Enable multiprocessing
    n_pool=4                   # Number of processes
)
```

## API Reference

### Main Classes

- `Sampler`: Main sampling class
- `SamplerConfig`: Configuration dataclass

### Key Methods

- `prepare_lhs_samples()`: Initialize with Latin Hypercube Sampling
- `run_sampling()`: Execute the sampling process
- `get_samples_with_weights()`: Retrieve samples and importance weights
- `save_state()` / `load_state()`: State persistence

### Utility Functions

- `find_sigma_level()`: Compute confidence level thresholds
- `oracle_approximating_shrinkage()`: Covariance regularization
- Various weighting and clustering utilities

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- smt >= 2.0.0
- tqdm >= 4.62.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{parismc,
  title={Parallel Adaptive Reweighting Importance Sampling (PARIS)},
  author={Miaoxin Liu, Alvin J. K. Chua},
  year={2025},
  url={https://github.com/mx-Liu123/parismc}
}

```

