import numpy as np
from smt.sampling_methods import LHS
from multiprocessing import Pool
import pickle
import os
from scipy.stats import multivariate_normal
import logging
from typing import List, Optional, Callable, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from .utils import (
    weighting_seeds_manypoint,
    weighting_seeds_manycov,
    weighting_seeds_onepoint_with_onemean,
    find_sigma_level,
)
from .clustering import (
    get_cluster_indices_cov,
    merge_arrays,
    merge_max_list,
    merge_element_num_list,
    find_points_within_threshold_cov
)
from .optimization import find_max_beta, oracle_approximating_shrinkage

try:
    # Try to determine if we're running in Jupyter
    from IPython import get_ipython
    if get_ipython() is not None and 'IPKernelApp' in get_ipython().config:
        # We're in Jupyter
        from tqdm.notebook import tqdm
    else:
        # We're not in Jupyter
        from tqdm import tqdm
except ImportError:
    # IPython is not available, so we're not in Jupyter
    from tqdm import tqdm

@dataclass
class SamplerConfig:
    """Configuration parameters for the Sampler."""
    proc_merge_prob: float = 0.9
    alpha: int = 1000
    latest_prob_index: int = 1000
    trail_size: int = int(1e3)
    boundary_limiting: bool = True
    use_beta: bool = True
    integral_num: int = int(1e5)
    gamma: int = 100
    exclude_scale_z: float = np.inf
    use_pool: bool = False
    n_pool: int = 10

class Sampler:
    """
    Advanced Monte Carlo sampler with adaptive covariance and clustering.
    
    This sampler implements an importance sampling algorithm with:
    - Adaptive proposal covariance matrices
    - Automatic cluster merging
    - Boundary-aware sampling
    - Optional multiprocessing support
    
    Parameters
    ----------
    ndim : int
        Dimensionality of the parameter space
    n_seed : int
        Number of initial seed points (walkers)
    log_reward_func : callable
        Function that computes log rewards for sample points
    init_cov_list : list of array-like
        Initial covariance matrices for each walker
    prior_transform : callable, optional
        Function to transform from unit cube to parameter space
    config : SamplerConfig, optional
        Configuration object with sampling parameters
        
    Examples
    --------
    >>> def log_reward(x):
    ...     return -0.5 * np.sum(x**2, axis=1)
    >>> sampler = Sampler(ndim=2, n_seed=3, log_reward_func=log_reward,
    ...                   init_cov_list=[np.eye(2)] * 3)
    >>> sampler.prepare_lhs_samples(1000, 100)
    >>> sampler.run_sampling(500, './results')
    """
    
    # Class constants
    MIN_LOG_DET_COV = -500  # Minimum acceptable log determinant
    USE_BETA_THRESHOLD = 0.1  # Threshold for enabling beta boundary correction (fraction of out-of-bounds samples) 
    LOOKBACK_WINDOW = 100          # Lookback window size for n_guess calculation
    GUESS_SIZE_DIVISOR = 2         # Divisor for guess size calculation
    MIN_GUESS_SIZE = 1             # Minimum guess size  
    EVIDENCE_ESTIMATION_FRACTION = 0.5  # Fraction of samples used for evidence estimation
    
    def __init__(self, 
                 ndim: int, 
                 n_seed: int, 
                 log_reward_func: Callable[[np.ndarray], np.ndarray],
                 init_cov_list: List[np.ndarray], 
                 prior_transform: Optional[Callable] = None,
                 config: Optional[SamplerConfig] = None) -> None:
        """Initialize the Sampler with given parameters."""
        
        # Use default config if none provided
        if config is None:
            config = SamplerConfig()
        
        # Input validation
        if ndim <= 0:
            raise ValueError("ndim must be positive")
        if n_seed <= 0:
            raise ValueError("n_seed must be positive")
        if len(init_cov_list) != n_seed:
            raise ValueError("init_cov_list length must equal n_seed")
        if not callable(log_reward_func):
            raise TypeError("log_reward_func must be callable")
        if any(cov.shape != (ndim, ndim) for cov in init_cov_list):
            raise ValueError("All covariance matrices must be ndim x ndim")            
        
        self.ndim = ndim
        self.n_seed = n_seed
        
        self.log_reward_func_original = log_reward_func
        if prior_transform is not None:
            self.prior_transform = prior_transform
            self.log_reward_func = self.transformed_log_reward_func
        else:
            self.prior_transform = None
            self.log_reward_func = log_reward_func
            
        self.init_cov_list = init_cov_list
        
        # Set configuration parameters
        self.proc_merge_prob = config.proc_merge_prob
        self.alpha = config.alpha
        self.latest_prob_index = config.latest_prob_index
        self.trail_size = config.trail_size
        self.boundary_limiting = config.boundary_limiting
        self.use_beta = config.use_beta
        self.integral_num = config.integral_num
        self.gamma = config.gamma
        self.exclude_scale_z = config.exclude_scale_z
        self.use_pool = config.use_pool
        self.n_pool = config.n_pool
        
        self.batch_point_num = 1
        self.cov_update_count = self.batch_point_num * self.gamma
        self.merge_dist = find_sigma_level(self.ndim, self.proc_merge_prob)
        self.current_iter = 0
        self.loglike_normalization = None
        self.n_walker = None
        
        # Initialize multiprocessing pool if needed
        if self.use_pool:
            self.pool = Pool(self.n_pool)
        else:
            self.pool = None

        # Initialize state variables
        self.searched_log_rewards_list: List[np.ndarray] = []
        self.searched_points_list: List[np.ndarray] = []
        self.means_list: List[np.ndarray] = []
        self.inv_covariances_list: List[np.ndarray] = []
        self.gaussian_normterm_list: List[np.ndarray] = []
        self.call_num_list: List[np.ndarray] = []
        self.rej_num_list: List[np.ndarray] = []
        self.wcoeff_list: List[np.ndarray] = []
        self.wdeno_list: List[np.ndarray] = []
        self.proposalcoeff_list: List[np.ndarray] = []
        self.max_loglike_list: List[float] = []
        self.element_num_list: List[int] = []
        self.last_gaussian_points: List[np.ndarray] = []
        self.now_covariances: List[np.ndarray] = []
        self.now_normterms: List[float] = []
        self.now_means: List[np.ndarray] = []

    def __del__(self) -> None:
        """Cleanup resources when object is destroyed."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()

    def apply_prior_transform(self, points: np.ndarray, prior_transform: Optional[Callable]) -> np.ndarray:
        """Apply prior transformation to points in unit hypercube [0,1]^ndim"""
        if prior_transform is None:
            return points
        return prior_transform(points)   
    
    def transformed_log_reward_func(self, x: np.ndarray) -> np.ndarray:
        """Apply prior transform before calling the original log reward function."""
        transformed_x = self.apply_prior_transform(x, self.prior_transform)
        return self.log_reward_func_original(transformed_x)

    def prepare_lhs_samples(self, lhs_num: int, batch_size: int) -> None:
        """Prepare LHS samples and initialize the sampler state."""
        if lhs_num <= 0:
            raise ValueError("lhs_num must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        xlimits = np.array([[0, 1]] * self.ndim, dtype=np.float32)
        sampling = LHS(xlimits=xlimits)
        x = sampling(lhs_num).astype(np.float32)
        lhs_log_rewards = np.zeros(lhs_num)
        
        for i in tqdm(range(0, lhs_num, batch_size), desc="Computing LHS rewards"):
            end = min(i + batch_size, lhs_num)
            lhs_log_rewards[i:end] = self.log_reward_func(x[i:end])
            
        self.lhs_points_initial = x
        self.lhs_log_rewards = lhs_log_rewards
        logger.info(f"Prepared {lhs_num} LHS samples")
            
    def initialize_first_iteration(self, num_iterations: int,
                             external_lhs_points: Optional[np.ndarray] = None,
                             external_lhs_log_rewards: Optional[np.ndarray] = None) -> None:
        """Initialize the first iteration with LHS samples.
        
        Parameters
        ----------
        external_lhs_points : np.ndarray, optional
            External LHS points to use instead of internal ones
        external_lhs_log_rewards : np.ndarray, optional
            External LHS rewards corresponding to external_lhs_points
        """
        # Use external LHS data if provided, otherwise use internal data
        if external_lhs_points is not None and external_lhs_log_rewards is not None:
            if len(external_lhs_log_rewards.shape) != 1:
                raise RuntimeError("External LHS log rewards must be 1D") 
            if len(external_lhs_log_rewards) < self.n_seed:
                raise RuntimeError("External LHS data must larger than n_seed")            
            lhs_points = external_lhs_points
            lhs_log_rewards = external_lhs_log_rewards
            logger.info(f"Using external LHS samples: {len(lhs_points)} points")
        else:
            if not hasattr(self, 'lhs_log_rewards'):
                raise RuntimeError("Must call prepare_lhs_samples first or provide external LHS data")
            lhs_points = self.lhs_points_initial
            lhs_log_rewards = self.lhs_log_rewards
            logger.info(f"Using internal LHS samples: {len(lhs_points)} points")
         
            
        indices = np.argsort(-lhs_log_rewards)
        selected_lhs_log_rewards = lhs_log_rewards[indices][:self.n_seed]
        selected_lhs_points_initial = lhs_points[indices][:self.n_seed]
        
        self.loglike_normalization = selected_lhs_log_rewards[0].copy()
        self.n_walker = self.n_seed
        self.maximum_array_size = self.batch_point_num * num_iterations

        # Initialize lists
        for i in range(self.n_seed):
            self.searched_log_rewards_list.append(np.empty((self.maximum_array_size,)))
            self.searched_points_list.append(np.empty((self.maximum_array_size, self.ndim)))
            self.means_list.append(np.empty((self.maximum_array_size, self.ndim)))
            self.inv_covariances_list.append(np.empty((int(self.maximum_array_size / self.cov_update_count), self.ndim, self.ndim)))
            self.gaussian_normterm_list.append(np.empty((int(self.maximum_array_size / self.cov_update_count),)))
            self.call_num_list.append(np.zeros((self.maximum_array_size,), dtype=np.float64))
            self.rej_num_list.append(np.zeros((self.maximum_array_size,), dtype=np.float64))
            self.wcoeff_list.append(np.ones((self.maximum_array_size,), dtype=np.float64))
            self.wdeno_list.append(np.zeros((self.maximum_array_size,), dtype=np.float64))
            self.proposalcoeff_list.append(np.ones((self.maximum_array_size,)))

            self.searched_points_list[i][:self.batch_point_num] = selected_lhs_points_initial[i].reshape(-1, self.ndim)
            self.searched_log_rewards_list[i][:self.batch_point_num] = selected_lhs_log_rewards[i]
            self.means_list[i][:self.batch_point_num] = selected_lhs_points_initial[i].reshape(-1, self.ndim)
            self.inv_covariances_list[i][:1] = np.linalg.inv(self.init_cov_list[i]).reshape(-1, self.ndim, self.ndim)
            self.max_loglike_list.append(-np.inf)
            self.element_num_list.append(self.batch_point_num)
            self.call_num_list[i][:self.batch_point_num] += 1
            self.rej_num_list[i][:self.batch_point_num] += 1
            det_covs = np.linalg.det(self.init_cov_list[i])
            self.gaussian_normterm_list[i][:1] = 1 / np.sqrt((2 * np.pi) ** self.ndim * det_covs)
            self.wdeno_list[i][:self.batch_point_num] = self.gaussian_normterm_list[i][:1] * np.exp(-self.ndim / 2) # initialize regularized weighting

            self.now_means.append(np.average(self.searched_points_list[i], axis=0))
            self.now_covariances.append(self.init_cov_list[i].copy())
            sign, log_det_cov = np.linalg.slogdet(self.init_cov_list[i])
            if sign <= 0 or log_det_cov < self.MIN_LOG_DET_COV:
                log_det_cov = self.gaussian_normterm_list[i][0]
                logger.warning('Negative or close to zero determinant covariance matrix')
            self.now_normterms.append(np.exp(-0.5 * log_det_cov) / np.sqrt((2 * np.pi) ** self.ndim))
        self.current_iter = 0

    def _extend_arrays_if_needed(self, num_iterations: int) -> None:
        """Extend arrays if more iterations are needed than originally allocated."""
        required_size = self.batch_point_num * (self.current_iter + num_iterations)
        if required_size > self.maximum_array_size:
            extension_size = required_size - self.maximum_array_size
            cov_extension = int(extension_size / self.cov_update_count) + 1
            
            for j in range(self.n_walker):
                # Extend each array
                self.searched_log_rewards_list[j] = np.append(self.searched_log_rewards_list[j], 
                                                             np.empty((extension_size,)), axis=0)
                self.searched_points_list[j] = np.append(self.searched_points_list[j], 
                                                        np.empty((extension_size, self.ndim)), axis=0)
                self.means_list[j] = np.append(self.means_list[j], 
                                              np.empty((extension_size, self.ndim)), axis=0)
                
                # For arrays related to covariance updates
                self.inv_covariances_list[j] = np.append(self.inv_covariances_list[j], 
                                                        np.empty((cov_extension, self.ndim, self.ndim)), axis=0)
                self.gaussian_normterm_list[j] = np.append(self.gaussian_normterm_list[j], 
                                                          np.empty((cov_extension,)), axis=0)
                
                # Other arrays
                self.call_num_list[j] = np.append(self.call_num_list[j], 
                                                 np.zeros((extension_size,), dtype=np.float64), axis=0)
                self.rej_num_list[j] = np.append(self.rej_num_list[j], 
                                                np.zeros((extension_size,), dtype=np.float64), axis=0)
                self.wcoeff_list[j] = np.append(self.wcoeff_list[j], 
                                               np.ones((extension_size,), dtype=np.float64), axis=0)
                self.wdeno_list[j] = np.append(self.wdeno_list[j], 
                                              np.zeros((extension_size,), dtype=np.float64), axis=0)
                self.proposalcoeff_list[j] = np.append(self.proposalcoeff_list[j], 
                                                      np.ones((extension_size,)), axis=0)
            
            self.maximum_array_size = required_size

    def run_sampling(self, num_iterations: int, savepath: str, print_iter: int = 1,
                 external_lhs_points: Optional[np.ndarray] = None,
                 external_lhs_log_rewards: Optional[np.ndarray] = None) -> None:
        """Run the sampling process for a specified number of iterations."""
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        # Validate external inputs if provided
        if external_lhs_points is not None or external_lhs_log_rewards is not None:
            if external_lhs_points is None or external_lhs_log_rewards is None:
                raise ValueError("Both external_lhs_points and external_lhs_log_rewards must be provided together")
            if len(external_lhs_points) != len(external_lhs_log_rewards):
                raise ValueError("external_lhs_points and external_lhs_log_rewards must have same length")
            if external_lhs_points.shape[1] != self.ndim:
                raise ValueError("external_lhs_points must have shape (n_samples, ndim)")
            
        self.savepath = savepath
        self.print_iter = print_iter

        # If starting from scratch, initialize everything
        if self.current_iter == 0:
            self.initialize_first_iteration(num_iterations, external_lhs_points, external_lhs_log_rewards)  # self.initialize_first_iteration(num_iterations)
            # Initialize additional variables used in the loop
            self.keep_trial_seeds = np.full(self.n_walker, True)
            self.eff_calls = 0        
            num_iterations -= 1            
        else:
            # If resuming from a previous run, check if arrays need to be extended
            self._extend_arrays_if_needed(num_iterations)
        
        try:
            pbar = tqdm(total=num_iterations, initial=0, desc="Sampling")
            
            for i in range(self.current_iter, self.current_iter + num_iterations):
                points_list = []
                probabilities_list = []
    
                # Weighting calculations
                if i > 0:
                    for j in range(self.n_walker):
                        ind1 = max(-self.latest_prob_index + self.element_num_list[j], 0)
                        ind2 = self.element_num_list[j]
                        points_list.append(self.searched_points_list[j][ind1:ind2])
                        probabilities_list.append(np.exp(self.searched_log_rewards_list[j][ind1:ind2] - self.loglike_normalization))
    
                        # Weighting Part 1: New proposals
                        ind1_newproposals = max(ind2 - self.batch_point_num, 0)
                        means_cache = self.means_list[j][ind1_newproposals:ind2]
                        proposalcoeff_cache = self.proposalcoeff_list[j][ind1_newproposals:ind2]
                        inv_covariances_cache = self.inv_covariances_list[j][int((ind2 - 1) / self.cov_update_count)]
                        norm_terms_cache = self.gaussian_normterm_list[j][int((ind2 - 1) / self.cov_update_count)]
                        
                        if self.use_pool and self.pool is not None:
                            # Multiprocessing logic
                            points_j_list = [points_list[j]] * self.n_pool
                            means_cache_list = np.array_split(means_cache, self.n_pool)
                            inv_covariances_cache_list = [inv_covariances_cache] * self.n_pool
                            norm_terms_cache_list = [norm_terms_cache] * self.n_pool
                            proposalcoeff_cache_list = np.array_split(proposalcoeff_cache, self.n_pool)
                            results = self.pool.starmap(weighting_seeds_manypoint, zip(points_j_list, means_cache_list, inv_covariances_cache_list, norm_terms_cache_list, proposalcoeff_cache_list))
                            self.wdeno_list[j][ind1:ind2] += np.concatenate(results)
                        else:
                            addon_weights = weighting_seeds_manypoint(points_list[j], means_cache, inv_covariances_cache, norm_terms_cache, proposalcoeff_cache)
                            self.wdeno_list[j][ind1:ind2] += addon_weights.copy()
    
                        # Weighting Part 2: New points with old proposals
                        ind2_oldproposals = self.element_num_list[j] - self.batch_point_num
                        means_cache = self.means_list[j][ind1:ind2_oldproposals]
                        proposalcoeff_cache = self.proposalcoeff_list[j][ind1:ind2_oldproposals]
                        index_array = np.arange(ind1, ind2_oldproposals) // self.cov_update_count
                        inv_covariances_cache = self.inv_covariances_list[j][index_array]
                        norm_terms_cache = self.gaussian_normterm_list[j][index_array]
                        points_cache = self.searched_points_list[j][ind2_oldproposals:ind2]
                        
                        if self.use_pool and self.pool is not None:
                            points_j_list = [points_cache] * self.n_pool
                            means_cache_list = np.array_split(means_cache, self.n_pool)
                            inv_covariances_cache_list = np.array_split(inv_covariances_cache, self.n_pool)
                            norm_terms_cache_list = np.array_split(norm_terms_cache, self.n_pool)
                            proposalcoeff_cache_list = np.array_split(proposalcoeff_cache, self.n_pool)
                            results = self.pool.starmap(weighting_seeds_manycov, zip(points_j_list, means_cache_list, inv_covariances_cache_list, norm_terms_cache_list, proposalcoeff_cache_list))
                            self.wdeno_list[j][ind2_oldproposals:ind2] += np.concatenate(results)
                        else:
                            addon_weights = weighting_seeds_manycov(points_cache, means_cache, inv_covariances_cache, norm_terms_cache, proposalcoeff_cache)
                            self.wdeno_list[j][ind2_oldproposals:ind2] += addon_weights.copy()
    
                        # Weighting Part 3: Subtracting for old proposals
                        if ind1 > 0:
                            ind1_oldproposals = max(ind1 - self.batch_point_num, 0)
                            means_cache = self.means_list[j][ind1_oldproposals:ind1]
                            proposalcoeff_cache = self.proposalcoeff_list[j][ind1_oldproposals:ind1]
                            inv_covariances_cache = self.inv_covariances_list[j][int((ind1 - 1) / self.cov_update_count)]
                            norm_terms_cache = self.gaussian_normterm_list[j][int((ind1 - 1) / self.cov_update_count)]
                            points_cache = self.searched_points_list[j][ind1:ind2_oldproposals]
                            
                            if self.use_pool and self.pool is not None:
                                points_j_list = [points_cache] * self.n_pool
                                means_cache_list = np.array_split(means_cache, self.n_pool)
                                inv_covariances_cache_list = [inv_covariances_cache] * self.n_pool
                                norm_terms_cache_list = [norm_terms_cache] * self.n_pool
                                proposalcoeff_cache_list = np.array_split(proposalcoeff_cache, self.n_pool)
                                results = self.pool.starmap(weighting_seeds_manypoint, zip(points_j_list, means_cache_list, inv_covariances_cache_list, norm_terms_cache_list, proposalcoeff_cache_list))
                                self.wdeno_list[j][ind1:ind2_oldproposals] -= np.concatenate(results)
                            else:
                                addon_weights = weighting_seeds_manypoint(points_cache, means_cache, inv_covariances_cache, norm_terms_cache, proposalcoeff_cache)
                                self.wdeno_list[j][ind1:ind2_oldproposals] -= addon_weights.copy()
    
                # Merging clusters
                if i > 0:
                    combined = sorted(zip(self.max_loglike_list, self.last_gaussian_points, self.searched_points_list, self.searched_log_rewards_list, self.means_list, self.inv_covariances_list, self.gaussian_normterm_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.wdeno_list, self.element_num_list, self.now_covariances, self.now_normterms, self.proposalcoeff_list), reverse=True, key=lambda x: x[0])
                    
                    self.max_loglike_list, self.last_gaussian_points, self.searched_points_list, self.searched_log_rewards_list, self.means_list, self.inv_covariances_list, self.gaussian_normterm_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.wdeno_list, self.element_num_list, self.now_covariances, self.now_normterms, self.proposalcoeff_list = zip(*combined)
                    self.last_gaussian_points = list(np.array(self.last_gaussian_points))
                    cluster_indices = get_cluster_indices_cov(np.array(self.last_gaussian_points), self.now_covariances, dist=self.merge_dist)
                    cluster_indices = [sorted(sublist) for sublist in cluster_indices]
    
                    lists_to_merge = [self.wdeno_list, self.searched_points_list, self.inv_covariances_list, self.gaussian_normterm_list, self.means_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.proposalcoeff_list]
                    merged_lists, self.searched_log_rewards_list = merge_arrays(lists_to_merge, cluster_indices, self.element_num_list, self.searched_log_rewards_list, self.latest_prob_index, self.cov_update_count)
                    (self.wdeno_list, self.searched_points_list, self.inv_covariances_list, self.gaussian_normterm_list, self.means_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.proposalcoeff_list) = merged_lists
                    self.max_loglike_list = merge_max_list(self.max_loglike_list, cluster_indices)
                    self.element_num_list = merge_element_num_list(self.element_num_list, cluster_indices)
                    self.now_covariances = merge_max_list(self.now_covariances, cluster_indices)
                    self.now_normterms = merge_max_list(self.now_normterms, cluster_indices)
                    self.n_walker = len(self.searched_log_rewards_list)
                    last_gaussian_points_cache = [self.last_gaussian_points[cluster_indices[j][0]] for j in range(self.n_walker)]
                    self.last_gaussian_points = last_gaussian_points_cache
    
                # Update means and covariances
                points_list = []
                probabilities_list = []
                self.now_means = []
                if (i + 1) % self.gamma == 0:
                    self.now_covariances = []
                    self.now_normterms = []
                    self.keep_trial_seeds = np.full(self.n_walker, True)
    
                for j in range(self.n_walker):
                    ind1 = 0
                    ind2 = self.element_num_list[j]
                    points_list.append(self.searched_points_list[j][ind1:ind2])
                    probabilities_list.append(np.exp(self.searched_log_rewards_list[j][ind1:ind2] - self.loglike_normalization))
                    if np.any(self.wdeno_list[j][ind1:ind2] <= 0.):
                        logger.warning(f"Weights <= 0, seed ind {j}, iter {i}")
                    else:
                        probabilities_list[j] /= self.wdeno_list[j][ind1:ind2]
                    probabilities_list[j][probabilities_list[j] < 0] = 0
                    weights_sum = np.sum(probabilities_list[j])
                    probabilities_list[j] /= weights_sum
                    mean = np.average(points_list[j], weights=probabilities_list[j], axis=0)
                    self.now_means.append(mean)
                    
                    if (i + 1) % self.gamma == 0:
                        covariance = np.cov(points_list[j], aweights=probabilities_list[j], rowvar=False, ddof=0)
                        n_samples = len(probabilities_list[j])
                        mvn = multivariate_normal(mean=np.zeros(self.ndim), cov=covariance, allow_singular=True)
                        original_zeromean_samples = mvn.rvs(size=self.integral_num)
                        is_out_of_bounds = np.any((original_zeromean_samples < (0 - mean)) | (original_zeromean_samples > (1 - mean)), axis=1)
                        if is_out_of_bounds.sum() / self.integral_num < self.USE_BETA_THRESHOLD:
                            self.use_beta = False
                        else:
                            self.use_beta = True
                        if self.boundary_limiting and self.use_beta:
                            cov_inv = np.linalg.inv(covariance)
                            diff = points_list[j] - mean
                            Adiff = np.einsum('jk,ik->ji', cov_inv, diff)
                            W = probabilities_list[j]
                            W_sum = 1
                            sign, original_log_det_cov = np.linalg.slogdet(covariance)
                            if sign > 0:
                                beta = find_max_beta(diff, Adiff, W, W_sum, original_log_det_cov, original_zeromean_samples, mean, self.ndim, self.integral_num)
                                covariance = covariance / beta[:, None] / beta[None, :]
                            else:
                                beta = np.ones((self.ndim))
                            covariance, shrinkage = oracle_approximating_shrinkage(covariance, n_samples)
                        sign, log_det_cov = np.linalg.slogdet(covariance)
                        if sign <= 0 or log_det_cov < self.MIN_LOG_DET_COV:
                            self.now_normterms.append(self.gaussian_normterm_list[j][0])
                            self.now_covariances.append(self.init_cov_list[j])
                            logger.warning(f'Negative or close zero determinant covariance matrix, seed {j}, sign {sign}, log_det_cov {log_det_cov}')
                        else:
                            self.now_normterms.append(np.exp(-0.5 * log_det_cov) / np.sqrt((2 * np.pi) ** self.ndim))
                            self.now_covariances.append(covariance)
                        self.inv_covariances_list[j][int(ind2 / self.cov_update_count)] = np.linalg.inv(self.now_covariances[j])
                        self.gaussian_normterm_list[j][int(ind2 / self.cov_update_count)] = self.now_normterms[j]
    
                # Generate new points
                self.last_gaussian_points = []
                for j in range(self.n_walker):
                    points_all = points_list[j]
                    probabilities_all = probabilities_list[j]
                    mean = self.now_means[j]
                    covariance = self.now_covariances[j]
                    ind1 = self.element_num_list[j]
                    ind2 = self.element_num_list[j] + self.batch_point_num
                    n_guess = int(min(
                        self.trail_size, 
                        max(
                            self.call_num_list[j][ind1 - self.LOOKBACK_WINDOW:ind1].sum() / self.LOOKBACK_WINDOW / self.GUESS_SIZE_DIVISOR, 
                            self.MIN_GUESS_SIZE
                        )
                    ))
                    mvn = multivariate_normal(mean=np.zeros(self.ndim), cov=covariance, allow_singular=True)
                    self.last_gaussian_points.append(mean.copy())
                    gaussian_log_rewards = -np.inf * np.ones((n_guess,))
                    single_weight_deno = np.ones((n_guess))
    
                    if self.boundary_limiting:
                        out_of_bound_indices = np.full(n_guess, True)
                        zeromean_samples = mvn.rvs(size=self.trail_size)
                        bulky_mean_inds = np.random.choice(len(probabilities_all), self.trail_size, p=probabilities_all, replace=True)
                        bulky_ind1 = 0
                        while_call_count = 0
                        while True:
                            if bulky_ind1 + n_guess > self.trail_size:
                                bulky_ind1 = 0
                                zeromean_samples = mvn.rvs(size=self.trail_size)
                                bulky_mean_inds = np.random.choice(len(probabilities_all), self.trail_size, p=probabilities_all, replace=True)
                            bulky_ind2 = bulky_ind1 + n_guess
                            indices_here = bulky_mean_inds[bulky_ind1:bulky_ind2]
                            gaussian_means = points_all[indices_here]
                            gaussian_points = zeromean_samples[bulky_ind1:bulky_ind2] + gaussian_means
                            gaussian_log_rewards[:] = -np.inf
                            single_weight_deno[:] = 1
                            out_of_bound_indices = np.any((gaussian_points < 0) | (gaussian_points > 1), axis=1)
                            is_within_bounds = ~out_of_bound_indices
                            
                            if self.use_pool and self.pool is not None and is_within_bounds.sum() > 0:
                                gaussian_points_list = [gaussian_points[k, :] for k in range(gaussian_points.shape[0]) if is_within_bounds[k]]
                                results = self.pool.map(self.log_reward_func, gaussian_points_list)
                                gaussian_log_rewards[is_within_bounds] = np.array(results).flatten()
                            elif is_within_bounds.sum() > 0:
                                gaussian_log_rewards[is_within_bounds] = self.log_reward_func(gaussian_points[is_within_bounds])
                            #print('gaussian_log_rewards[is_within_bounds]',repr(gaussian_points[is_within_bounds]),gaussian_log_rewards[is_within_bounds])
                                
                            single_weight_deno[is_within_bounds] = weighting_seeds_onepoint_with_onemean(gaussian_points[is_within_bounds], gaussian_means[is_within_bounds], self.inv_covariances_list[j][int((i + 1) * self.batch_point_num / self.cov_update_count)], self.gaussian_normterm_list[j][int((i + 1) * self.batch_point_num / self.cov_update_count)])
                            if_weights_big = np.exp(gaussian_log_rewards - self.loglike_normalization) / single_weight_deno > weights_sum / self.exclude_scale_z
                            if_weights_big = np.full(if_weights_big.shape, True, dtype=bool)
                            bulky_ind1 = bulky_ind2
                            while_call_count += n_guess
                            if not if_weights_big.any():
                                self.call_num_list[j][ind1:ind2] += n_guess
                                self.eff_calls += is_within_bounds.sum()
                            elif if_weights_big[0]:
                                valid_index = 0
                                self.call_num_list[j][ind1:ind2] += 1
                                self.eff_calls += 1
                                gaussian_log_rewards = gaussian_log_rewards[valid_index:valid_index + 1]
                                gaussian_points = gaussian_points[valid_index:valid_index + 1]
                                gaussian_means = gaussian_means[valid_index:valid_index + 1]
                                break
                            else:
                                valid_index = np.argmax(if_weights_big)
                                self.call_num_list[j][ind1:ind2] += valid_index + 1
                                true_indices = np.flatnonzero(is_within_bounds)
                                pos = np.where(true_indices == valid_index)[0][0] + 1
                                self.eff_calls += pos
                                gaussian_log_rewards = gaussian_log_rewards[valid_index:valid_index + 1]
                                gaussian_points = gaussian_points[valid_index:valid_index + 1]
                                gaussian_means = gaussian_means[valid_index:valid_index + 1]
                                break
                            if while_call_count > int(self.trail_size) or not self.keep_trial_seeds[j]:
                                self.keep_trial_seeds[j] = False
                                valid_index = 0
                                self.call_num_list[j][ind1:ind2] += n_guess
                                self.eff_calls += is_within_bounds.sum()
                                gaussian_log_rewards = gaussian_log_rewards[valid_index:valid_index + 1]
                                gaussian_points = gaussian_points[valid_index:valid_index + 1]
                                gaussian_means = gaussian_means[valid_index:valid_index + 1]
                                break
    
                    proposalcoeffs = np.ones((len(gaussian_means)))
                    self.searched_log_rewards_list[j][ind1:ind2] = gaussian_log_rewards.copy()
                    self.searched_points_list[j][ind1:ind2] = gaussian_points.copy()
                    self.means_list[j][ind1:ind2] = gaussian_means.copy()
                    self.proposalcoeff_list[j][ind1:ind2] = proposalcoeffs.copy()
                    self.element_num_list[j] += self.batch_point_num
                    self.max_loglike_list[j] = max(self.max_loglike_list[j], gaussian_log_rewards.max())
    
                # Update diagnostics
                if i % self.print_iter == 0:
                    c_term = self.loglike_normalization
                    calls = sum(self.element_num_list)
                    ind1 = max(int(self.element_num_list[0] * (1 - self.EVIDENCE_ESTIMATION_FRACTION)), 0)
                    ind2 = self.element_num_list[0] - self.batch_point_num
                    wsum = sum(np.sum(np.exp(self.searched_log_rewards_list[j][ind1:ind2] - c_term) / self.wdeno_list[j][ind1:ind2]) for j in range(self.n_walker)) * self.n_walker * (self.alpha * self.batch_point_num)
                    Nsum = sum(self.call_num_list[j][ind1:ind2].sum() for j in range(self.n_walker))
                    logZ = c_term - np.log(Nsum) + np.log(wsum)
                    
                    status = f"samples: {Nsum}, evals: {calls}, walkers: {self.n_walker}, cov: {self.now_covariances[0][0, 0]:.5e}, logZ: {logZ:.5f}, max_ll: {self.max_loglike_list[0]:.5f}"
                    pbar.set_description(status)
                    pbar.update(self.print_iter)
    
                self.current_iter += 1
                
                # Clean up temporary variables to save memory
                del points_list, probabilities_list
        
            self.save_state()  
            pbar.close()
            
            
        except KeyboardInterrupt:
            logger.warning("Sampling interrupted by user")
            self.save_state()
        except MemoryError:
            logger.error("Out of memory during sampling")
            self.save_state()
            raise
        except ValueError as e:
            logger.error(f"Value error in sampling: {e}")
            self.save_state()
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.save_state()
            raise            
    
    def imp_weights_list(self) -> List[np.ndarray]:
        """
        Calculate and return importance weights for all samples across all walkers,
        handling the special case of the latest batch of samples.
        
        Returns:
        -------
        list of numpy.ndarray:
            A list where each element is an array of importance weights for a walker.
        """
        weights_list = []
        for j in range(self.n_walker):
            element_num = self.element_num_list[j]
            
            # Calculate importance weights: exp(log_reward - normalization) / denominator
            weights = np.exp(self.searched_log_rewards_list[j][:element_num] - self.loglike_normalization)
            
            # Handle the latest batch of points (which might have zero denominators)
            latest_batch_start = element_num - self.batch_point_num
            
            # For all points except the latest batch
            if latest_batch_start > 0:
                weights[:latest_batch_start] = weights[:latest_batch_start] / self.wdeno_list[j][:latest_batch_start]
            
            # For the latest batch, calculate the denominator on-the-fly
            if latest_batch_start < element_num:
                # Get the latest points
                latest_points = self.searched_points_list[j][latest_batch_start:element_num]
                
                # Calculate their denominators using the same approach as in the main loop
                # This replicates the weighting calculation that would happen in the next iteration
                
                # Use the latest covariance and means for weighting
                latest_cov_idx = int((element_num - 1) / self.cov_update_count)
                inv_cov = self.inv_covariances_list[j][latest_cov_idx]
                norm_term = self.gaussian_normterm_list[j][latest_cov_idx]
                
                # Get all means that contribute to the weight denominator
                all_means = self.means_list[j][:latest_batch_start]
                all_proposalcoeffs = self.proposalcoeff_list[j][:latest_batch_start]
                
                # Calculate weights for the latest batch
                latest_denos = weighting_seeds_manypoint(latest_points, all_means, inv_cov, norm_term, all_proposalcoeffs)
                
                # Apply the calculated denominators
                weights[latest_batch_start:element_num] = weights[latest_batch_start:element_num] / latest_denos
            
            weights_list.append(weights)
        return weights_list      
    
    def get_samples_with_weights(self, flatten: bool = False) -> Union[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
        """
        Get samples and their weights in the parameter space.
        
        Parameters:
        ----------
        flatten : bool, optional
            If True, returns concatenated arrays of all samples and weights.
            If False, returns lists of arrays for each walker. Default is False.
        
        Returns:
        -------
        If flatten=False:
            tuple: (transformed_samples_list, weights_list) where each is a list of arrays
        If flatten=True:
            tuple: (all_samples, all_weights) where each is a single concatenated array
        """
        # Get weights
        weights_list = self.imp_weights_list()
        
        # Get transformed samples
        if self.prior_transform is not None:
            transformed_samples_list = []
            for j in range(self.n_walker):
                element_num = self.element_num_list[j]
                samples = self.searched_points_list[j][:element_num]
                transformed_samples = self.apply_prior_transform(samples, self.prior_transform)
                transformed_samples_list.append(transformed_samples)
        else:
            transformed_samples_list = []
            for j in range(self.n_walker):
                element_num = self.element_num_list[j]
                transformed_samples_list.append(self.searched_points_list[j][:element_num])
        
        if flatten:
            # Concatenate all samples and weights
            all_samples = np.concatenate(transformed_samples_list)
            all_weights = np.concatenate(weights_list)
            return all_samples, all_weights
        else:
            return transformed_samples_list, weights_list 

    def save_state(self, filename: Optional[str] = None) -> None:
        """Save the current state of the sampler to a file.
        
        Parameters:
        ----------
        filename : str, optional
            The filename to save the state to. If None, uses self.savepath/sampler_state.pkl
        """
        if filename is None:
            if hasattr(self, 'savepath'):
                filename = os.path.join(self.savepath, 'sampler_state.pkl')
            else:
                filename = 'sampler_state.pkl'
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save the entire class instance using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Sampler state saved to {filename}")
    
    @staticmethod
    def load_state(filename: str) -> 'Sampler':
        """Load a sampler state from a file.
        
        Parameters:
        ----------
        filename : str
            The filename to load the state from.
            
        Returns:
        -------
        Sampler
            The loaded Sampler instance.
        """
        with open(filename, 'rb') as f:
            sampler = pickle.load(f)
        
        logger.info(f"Sampler state loaded from {filename}")
        return sampler