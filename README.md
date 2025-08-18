# Paris Monte Carlo Sampler

An advanced Monte Carlo sampler with adaptive covariance and clustering capabilities for Bayesian inference and optimization problems.

## Features

- **Adaptive Covariance**: Automatically adjusts proposal covariance matrices based on sample history
- **Intelligent Clustering**: Merges similar sampling chains to improve efficiency
- **Boundary Handling**: Smart boundary constraints for parameters in [0,1]^d space
- **Multiprocessing Support**: Optional parallel processing for large-scale problems
- **Flexible Configuration**: Extensive customization options through SamplerConfig

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
  title={Paris Monte Carlo Sampler},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/parismc}
}
```