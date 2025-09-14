"""
Paris Monte Carlo Sampler Package

An advanced Monte Carlo sampler with adaptive covariance and clustering capabilities.
"""

from .sampler import Sampler, SamplerConfig
from .optimization import find_max_beta, oracle_approximating_shrinkage, negative_BN_log_like
from .utils import (
    weighting_seeds_manypoint,
    weighting_seeds_manycov, 
    weighting_seeds_onepoint_with_onemean,
    find_sigma_level
)
from .clustering import (
    get_cluster_indices_cov,
    get_cluster_indices,
    merge_arrays,
    merge_max_list,
    merge_element_num_list
)

__version__ = "0.1.0"
__author__ = "Miaoxin Liu, Alvin J. K. Chua"
__email__ = "miaoxin.liu@u.nus.edu"

# Make main classes easily accessible
__all__ = [
    'Sampler',
    'SamplerConfig',
    'find_max_beta',
    'oracle_approximating_shrinkage',
    'weighting_seeds_manypoint',
    'weighting_seeds_manycov',
    'weighting_seeds_onepoint_with_onemean',
    'find_sigma_level',
    'get_cluster_indices_cov',
    'get_cluster_indices',
]
