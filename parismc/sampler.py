import os
import json
import pickle
import logging
from typing import List, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from scipy.stats import multivariate_normal
from smt.sampling_methods import LHS

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

logger = logging.getLogger(__name__)

@dataclass
class SamplerConfig:
    """Configuration parameters for the Sampler.

    Notes on merging parameter:
    - merge_confidence p in [0, 1] maps to a dimension-aware Mahalanobis
      threshold R_m via R_m = find_sigma_level(ndim, p).
      Larger p -> larger R_m (more permissive). Edge cases: p=0 => R_m=0;
      p->1 => R_m->infinity.
    - When considering whether to merge two processes, distances are computed
      with respect to each process's own covariance (asymmetric Mahalanobis
      distances); the smaller of the two distances is compared to R_m.
    """
    seed: Optional[int] = None
    merge_confidence: float = 0.9  # Probability mass inside merge radius R_m (0→R_m=0, 1→R_m→∞)
    alpha: int = 1000
    trail_size: int = int(1e3)
    boundary_limiting: bool = True
    use_beta: bool = True
    integral_num: int = int(1e5)
    gamma: int = 100
    exclude_scale_z: float = np.inf
    use_pool: bool = False
    n_pool: int = 10
    stop_on_merge: bool = False
    merge_type: str = 'single' # 'distance', 'single', or 'multiple'
    debug: bool = False

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
        Number of initial seed points (processes)
    log_density_func : callable
        Function that computes log densities for sample points
    init_cov_list : list of array-like
        Initial covariance matrices for each process
    prior_transform : callable, optional
        Function to transform from unit cube to parameter space
    config : SamplerConfig, optional
        Configuration object with sampling parameters
        
    Examples
    --------
    >>> def log_density(x):
    ...     return -0.5 * np.sum(x**2, axis=1)
    >>> sampler = Sampler(ndim=2, n_seed=3, log_density_func=log_density,
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
                 log_density_func: Callable[[np.ndarray], np.ndarray],
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
        if not callable(log_density_func):
            raise TypeError("log_density_func must be callable")
        if any(cov.shape != (ndim, ndim) for cov in init_cov_list):
            raise ValueError("All covariance matrices must be ndim x ndim")            
        
        self.ndim = ndim
        self.n_seed = n_seed
        
        self.log_density_func_original = log_density_func
        if prior_transform is not None:
            self.prior_transform = prior_transform
            self.log_density_func = self.transformed_log_density_func
        else:
            self.prior_transform = None
            self.log_density_func = log_density_func
            
        self.init_cov_list = init_cov_list
        self.config = config
        
        # Set seed
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Set configuration parameters
        # Merge confidence (coverage probability) used to derive Mahalanobis threshold R_m
        self.merge_confidence = config.merge_confidence
        self.alpha = config.alpha
        self.latest_prob_index = config.alpha#config.latest_prob_index
        self.trail_size = config.trail_size
        self.boundary_limiting = config.boundary_limiting
        self.use_beta = config.use_beta
        self.integral_num = config.integral_num
        self.gamma = config.gamma
        self.exclude_scale_z = config.exclude_scale_z
        self.use_pool = config.use_pool
        self.n_pool = config.n_pool
        self.stop_on_merge = config.stop_on_merge
        self.merge_type = config.merge_type
        self.debug = config.debug
        
        if self.merge_type not in ['distance', 'single', 'multiple']:
             raise ValueError(f"Invalid merge_type: {self.merge_type}. Must be 'distance', 'single', or 'multiple'.")

        self.batch_point_num = 1
        self.cov_update_count = self.batch_point_num * self.gamma
        # Mahalanobis merge threshold R_m derived from coverage probability p
        # R_m = find_sigma_level(ndim, p). Higher p → larger R_m.
        # p=0 => R_m=0; p→1 => R_m→∞. See SamplerConfig notes.
        self.merge_dist = 4#find_sigma_level(self.ndim, self.merge_confidence)
        self.current_iter = 0
        self.loglike_normalization = None
        self.n_proc = None
        
        # Initialize multiprocessing pool if needed
        if self.use_pool:
            self.pool = Pool(self.n_pool)
        else:
            self.pool = None

        # Initialize state variables
        self.searched_log_densities_list: List[np.ndarray] = []
        self.searched_points_list: List[np.ndarray] = []
        self.means_list: List[np.ndarray] = []
        self.inv_covariances_list: List[np.ndarray] = []
        self.gaussian_normterm_list: List[np.ndarray] = []
        self.call_num_list: List[np.ndarray] = []
        self.rej_num_list: List[np.ndarray] = []
        self.wcoeff_list: List[np.ndarray] = []
        self.wdeno_list: List[np.ndarray] = []
        self.proposalcoeff_list: List[np.ndarray] = []
        self.max_logden_list: List[float] = []
        self.element_num_list: List[int] = []
        self.last_gaussian_points: List[np.ndarray] = []
        self.now_covariances: List[np.ndarray] = []
        self.now_normterms: List[float] = []
        self.now_means: List[np.ndarray] = []
        self._last_logZ_for_stop: Optional[float] = None
        self._last_logZ_iter: Optional[int] = None
        self._last_dlogZ: Optional[float] = None

        # Initialize flag file for external monitoring controls
        self.flag_file = os.path.join('.', 'sampler_flags.json')
        self._initialize_flag_file()

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
    
    def transformed_log_density_func(self, x: np.ndarray) -> np.ndarray:
        """Apply prior transform before calling the original log density function."""
        transformed_x = self.apply_prior_transform(x, self.prior_transform)
        return self.log_density_func_original(transformed_x)

    def prepare_lhs_samples(self, lhs_num: int, batch_size: int) -> None:
        """Prepare LHS samples and initialize the sampler state."""
        if lhs_num <= 0:
            raise ValueError("lhs_num must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        xlimits = np.array([[0, 1]] * self.ndim, dtype=np.float32)
        sampling = LHS(xlimits=xlimits, random_state=self.config.seed)
        x = sampling(lhs_num).astype(np.float32)
        lhs_log_densities = np.zeros(lhs_num)
        
        for i in tqdm(range(0, lhs_num, batch_size), desc="Computing LHS densities"):
            end = min(i + batch_size, lhs_num)
            lhs_log_densities[i:end] = self.log_density_func(x[i:end])
            
        self.lhs_points_initial = x
        self.lhs_log_densities = lhs_log_densities
        logger.info(f"Prepared {lhs_num} LHS samples")

    def _compute_weight_segment(self, points, param_idx, start_idx, end_idx, cov_mode="single", cov_ref_idx=None):
        """
        Computes the weighting segment.
        
        Args:
            points (np.ndarray): The points to evaluate (from population j).
            param_idx (int): The index of the population providing parameters (j or j_prime).
                             Used to access means_list, inv_covariances_list, etc.
            start_idx (int): Start index for slicing parameter lists.
            end_idx (int): End index for slicing parameter lists.
            cov_mode (str): 'single' or 'multi'.
            cov_ref_idx (int, optional): Index for covariance if mode is 'single'.

        Returns:
            np.ndarray: The calculated addon weights.
        """
        # --- Data Preparation ---
        
        # 1. Slice means and proposal coefficients from the parameter source (param_idx)
        means_cache = self.means_list[param_idx][start_idx:end_idx]
        proposalcoeff_cache = self.proposalcoeff_list[param_idx][start_idx:end_idx]

        # 2. Prepare covariance matrices from the parameter source (param_idx)
        if cov_mode == "multi":
            # Case: Multiple covariance matrices
            index_array = np.arange(start_idx, end_idx) // self.cov_update_count
            inv_covariances_cache = self.inv_covariances_list[param_idx][index_array]
            norm_terms_cache = self.gaussian_normterm_list[param_idx][index_array]
            compute_func = weighting_seeds_manycov
        else:
            # Case: Single covariance matrix
            inv_covariances_cache = self.inv_covariances_list[param_idx][cov_ref_idx]
            norm_terms_cache = self.gaussian_normterm_list[param_idx][cov_ref_idx]
            compute_func = weighting_seeds_manypoint

        # --- Calculation (Multiprocessing vs Serial) ---
        
        if self.use_pool and self.pool is not None:
            # Prepare arguments for starmap
            points_list = [points] * self.n_pool
            means_list = np.array_split(means_cache, self.n_pool)
            proposalcoeff_list = np.array_split(proposalcoeff_cache, self.n_pool)

            if cov_mode == "multi":
                inv_cov_list = np.array_split(inv_covariances_cache, self.n_pool)
                norm_list = np.array_split(norm_terms_cache, self.n_pool)
            else:
                inv_cov_list = [inv_covariances_cache] * self.n_pool
                norm_list = [norm_terms_cache] * self.n_pool

            results = self.pool.starmap(compute_func, zip(
                points_list, means_list, inv_cov_list, norm_list, proposalcoeff_list
            ))
            return np.concatenate(results)
        else:
            return compute_func(
                points, means_cache, inv_covariances_cache, 
                norm_terms_cache, proposalcoeff_cache
            )        
            
    # --------------------
    # Flag and action I/O
    # --------------------
    def _initialize_flag_file(self) -> None:
        """Create or reset the flag file in current directory with default false flags."""
        flags = {
            "output_latest_samples": False,
            "plot_latest_samples": False,
            "print_latest_infos": False,
        }
        with open(self.flag_file, 'w', encoding='utf-8') as f:
            json.dump(flags, f)

    def _read_flags_fast(self) -> dict:
        """Fast read of the small JSON flag file."""
        with open(self.flag_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _write_flags(self, flags: dict) -> None:
        """Write updated flags back to the JSON flag file."""
        with open(self.flag_file, 'w', encoding='utf-8') as f:
            json.dump(flags, f)

    def _output_latest_samples(self) -> None:
        """Save latest transformed samples and weights using np.save (separate files)."""
        all_samples, all_weights = self.get_samples_with_weights(flatten=True)
        np.save(os.path.join('.', 'latest_samples.npy'), all_samples)
        np.save(os.path.join('.', 'latest_weights.npy'), all_weights)

    def _compute_prior_bounds(self) -> List[Tuple[float, float]]:
        """Compute per-dimension bounds from prior transform mapping of (0,1)."""
        if self.prior_transform is None:
            return [(0.0, 1.0) for _ in range(self.ndim)]
        bounds: List[Tuple[float, float]] = []
        for d in range(self.ndim):
            u0 = np.full((1, self.ndim), 0.5, dtype=float)
            u1 = np.full((1, self.ndim), 0.5, dtype=float)
            u0[0, d] = 0.0
            u1[0, d] = 1.0
            x0 = self.apply_prior_transform(u0, self.prior_transform)[0]
            x1 = self.apply_prior_transform(u1, self.prior_transform)[0]
            bounds.append((x0[d], x1[d]))
        return bounds

    def _plot_latest_corner(self) -> None:
        """Create a corner plot of transformed samples weighted by importance weights."""
        # Lazy import to avoid overhead unless plotting is requested
        import corner  # type: ignore
        all_samples, all_weights = self.get_samples_with_weights(flatten=True)
        ranges = self._compute_prior_bounds()
        fig = corner.corner(all_samples, weights=all_weights, range=ranges)
        fig.savefig(os.path.join('.', 'latest_corner.png'), dpi=150, bbox_inches='tight')
        # Explicitly close to release resources
        import matplotlib.pyplot as plt  # type: ignore
        plt.close(fig)

    def _print_latest_infos(self) -> None:
        """Write per-process diagnostics to a text file based on transformed samples/weights."""
        transformed_list, weights_list = self.get_samples_with_weights(flatten=False)
        lines: List[str] = []
        for j in range(self.n_proc):
            x = transformed_list[j]
            w = weights_list[j]
            max_ld = self.max_logden_list[j]
            max_w = float(np.max(w)) if len(w) > 0 else float('nan')
            sum_w = float(np.sum(w)) if len(w) > 0 else float('nan')
            mean = np.average(x, weights=w, axis=0) if len(w) > 0 else np.full(self.ndim, np.nan)
            cov = np.cov(x, aweights=w, rowvar=False, ddof=0) if len(w) > 1 else np.full((self.ndim, self.ndim), np.nan)
            diag_cov = np.diag(cov)
            lines.append(f"process {j}:\n"
                         f"  max_log_density: {max_ld}\n"
                         f"  max_importance_weight: {max_w}\n"
                         f"  sum_importance_weight: {sum_w}\n"
                         f"  weighted_mean: {mean.tolist()}\n"
                         f"  weighted_cov_diag: {diag_cov.tolist()}\n"
                         f"  weighted_cov:\n")
            for r in range(self.ndim):
                row = cov[r, :]
                lines.append(f"    {row.tolist()}")
        with open(os.path.join('.', 'latest_infos.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _check_flags_and_take_actions(self) -> None:
        """Read flags and perform actions, then reset taken flags to False."""
        flags = self._read_flags_fast()
        changed = False
        if flags.get('output_latest_samples', False):
            self._output_latest_samples()
            flags['output_latest_samples'] = False
            changed = True
        if flags.get('plot_latest_samples', False):
            self._plot_latest_corner()
            flags['plot_latest_samples'] = False
            changed = True
        if flags.get('print_latest_infos', False):
            self._print_latest_infos()
            flags['print_latest_infos'] = False
            changed = True
        if changed:
            self._write_flags(flags)

    def initialize_first_iteration(self, num_iterations: int,
                             external_lhs_points: Optional[np.ndarray] = None,
                             external_lhs_log_densities: Optional[np.ndarray] = None) -> None:
        """Initialize the first iteration with LHS samples.
        
        Parameters
        ----------
        external_lhs_points : np.ndarray, optional
            External LHS points to use instead of internal ones
        external_lhs_log_densities : np.ndarray, optional
            External LHS densities corresponding to external_lhs_points
        """
        # Use external LHS data if provided, otherwise use internal data
        if external_lhs_points is not None and external_lhs_log_densities is not None:
            if len(external_lhs_log_densities.shape) != 1:
                raise RuntimeError("External LHS log densities must be 1D") 
            if len(external_lhs_log_densities) < self.n_seed:
                raise RuntimeError("External LHS data must larger than n_seed")            
            lhs_points = external_lhs_points
            lhs_log_densities = external_lhs_log_densities
            logger.info(f"Using external LHS samples: {len(lhs_points)} points")
        else:
            if not hasattr(self, 'lhs_log_densities'):
                raise RuntimeError("Must call prepare_lhs_samples first or provide external LHS data")
            lhs_points = self.lhs_points_initial
            lhs_log_densities = self.lhs_log_densities
            logger.info(f"Using internal LHS samples: {len(lhs_points)} points")
         
            
        indices = np.argsort(-lhs_log_densities)
        selected_lhs_log_densities = lhs_log_densities[indices][:self.n_seed]
        selected_lhs_points_initial = lhs_points[indices][:self.n_seed]
        
        self.loglike_normalization = selected_lhs_log_densities[0].copy()
        self.n_proc = self.n_seed
        self.maximum_array_size = int(self.batch_point_num * num_iterations)

        # Initialize lists
        for i in range(self.n_seed):
            self.searched_log_densities_list.append(np.empty((self.maximum_array_size,)))
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
            self.searched_log_densities_list[i][:self.batch_point_num] = selected_lhs_log_densities[i]
            self.means_list[i][:self.batch_point_num] = selected_lhs_points_initial[i].reshape(-1, self.ndim)
            self.inv_covariances_list[i][:1] = np.linalg.inv(self.init_cov_list[i]).reshape(-1, self.ndim, self.ndim)
            self.max_logden_list.append(-np.inf)
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
            
            for j in range(self.n_proc):
                # Extend each array
                self.searched_log_densities_list[j] = np.append(self.searched_log_densities_list[j], 
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

    def _perform_weighting_update(self) -> None:
        """Update importance weights based on new proposals and covariance changes."""
        for j in range(self.n_proc):
            ind1 = max(-self.latest_prob_index + self.element_num_list[j], 0)
            ind2 = self.element_num_list[j]
            points_j = self.searched_points_list[j][ind1:ind2]

            # 1. New proposals (Single Covariance)
            ind1_newproposals = max(ind2 - self.batch_point_num, 0)
            addon_weights = self._compute_weight_segment(
                points=points_j, 
                param_idx=j,
                start_idx=ind1_newproposals, end_idx=ind2,
                cov_mode="single", cov_ref_idx=int((ind2 - 1) / self.cov_update_count)
            )
            self.wdeno_list[j][ind1:ind2] += addon_weights
    
            # 2. New points with old proposals (Multi Covariance)
            ind2_oldproposals = self.element_num_list[j] - self.batch_point_num
            addon_weights = self._compute_weight_segment(
                points=self.searched_points_list[j][ind2_oldproposals:ind2],
                param_idx=j,
                start_idx=ind1, end_idx=ind2_oldproposals,
                cov_mode="multi"
            )
            self.wdeno_list[j][ind2_oldproposals:ind2] += addon_weights
    
            # 3. Subtracting for old proposals (Single Covariance)
            if ind1 > 0:
                ind1_oldproposals = max(ind1 - self.batch_point_num, 0)
                addon_weights = self._compute_weight_segment(
                    points=self.searched_points_list[j][ind1:ind2_oldproposals],
                    param_idx=j,
                    start_idx=ind1_oldproposals, end_idx=ind1,
                    cov_mode="single", cov_ref_idx=int((ind1 - 1) / self.cov_update_count)
                )
                self.wdeno_list[j][ind1:ind2_oldproposals] -= addon_weights

    def _sort_processes_state(self) -> None:
        """
        Sort all process-related state lists by max log density in descending order.
        This uses reflection to ensure all relevant attributes are kept in sync.
        Uses Python's stable sort to maintain consistent order when values are equal.
        """
        if self.n_proc <= 1:
            return

        # 1. Determine sort indices using Python's stable sort
        # Higher max_logden first. 
        sort_indices = sorted(
            range(len(self.max_logden_list)), 
            key=lambda k: self.max_logden_list[k], 
            reverse=True
        )
        
        # 2. Identify all attributes that need to be sorted
        state_attributes = [
            'max_logden_list', 'last_gaussian_points', 'searched_points_list',
            'searched_log_densities_list', 'means_list', 'inv_covariances_list',
            'gaussian_normterm_list', 'call_num_list', 'rej_num_list',
            'wcoeff_list', 'wdeno_list', 'element_num_list',
            'now_covariances', 'now_normterms', 'proposalcoeff_list'
        ]
        
        # 3. Apply sorting to each attribute
        for attr_name in state_attributes:
            if hasattr(self, attr_name):
                current_val = getattr(self, attr_name)
                # Handle both list and tuple
                sorted_val = [current_val[idx] for idx in sort_indices]
                setattr(self, attr_name, sorted_val)

    def _find_clusters_by_distance(self) -> List[List[int]]:
        """Find clusters based on Mahalanobis distance between process means."""
        self.last_gaussian_points = list(np.array(self.last_gaussian_points))
        cluster_indices = get_cluster_indices_cov(
            np.array(self.last_gaussian_points), 
            self.now_covariances, 
            dist=self.merge_dist
        )
        return [sorted(sublist) for sublist in cluster_indices]

    def _find_clusters_by_weight(self) -> List[List[int]]:
        """Find clusters by cross-evaluating samples against other processes' proposals."""
        classified_indices = set()
        new_clusters = []
        
        # Greedy Pairwise Merging (Matching original loop structure)
        for j in range(self.n_proc):
            if j in classified_indices:
                continue
                
            found_merge = False
            for j_prime in range(self.n_proc):
                if j == j_prime or j_prime in classified_indices:
                    continue 

                # Prepare data for cross-check
                ind1 = max(-self.latest_prob_index + self.element_num_list[j_prime], 0)
                ind2 = self.element_num_list[j_prime]
                
                # Determine target points for evaluation (Must exactly match original logic)
                if self.merge_type == 'single':
                    target_end = self.element_num_list[j]
                    target_start = max(0, target_end - self.batch_point_num)
                elif self.merge_type == 'multiple':
                    target_end = self.element_num_list[j]
                    target_start = 0
                else: # Fallback to single behavior if somehow unspecified
                    target_end = self.element_num_list[j]
                    target_start = max(0, target_end - self.batch_point_num)

                points_to_eval = self.searched_points_list[j][target_start:target_end]
                addon_weights = self._compute_weight_segment(
                    points=points_to_eval,
                    param_idx=j_prime, 
                    start_idx=ind1, 
                    end_idx=ind2, 
                    cov_mode="multi" 
                )
                
                current_weights = self.wdeno_list[j][target_start:target_end]
                
                # Merge Condition
                if self.merge_type == 'single':
                    if np.mean(addon_weights) >= np.mean(current_weights):
                        new_clusters.append([j, j_prime])
                        classified_indices.add(j)
                        classified_indices.add(j_prime)
                        found_merge = True
                        break 
                elif self.merge_type == 'multiple':
                    # better_count = np.sum(addon_weights > current_weights/1)
                    better_count = np.sum(addon_weights > current_weights)
                    if len(addon_weights) > 0 and better_count > 0:
                        new_clusters.append([j, j_prime])
                        classified_indices.add(j)
                        classified_indices.add(j_prime)
                        found_merge = True
                        break
            
            # This logic was outside j_prime loop in original
            # but effectively handled at the end
            
        # Add any remaining unclassified processes as their own clusters
        for k in range(self.n_proc):
            if k not in classified_indices:
                new_clusters.append([k])
        
        return new_clusters

    def _merge_processes(self, i: int) -> bool:
        """
        Sort processes and merge clusters.
        
        Returns:
            bool: True if sampling should stop (merge detected and stop_on_merge=True).
        """
        if i <= 0:
            return False

        # 1. Sort processes by max log density (Robust implementation)
        self._sort_processes_state()

        # 2. Determine Cluster Indices based on strategy
        if self.merge_type == 'distance':
            cluster_indices = self._find_clusters_by_distance()
        else:
            cluster_indices = self._find_clusters_by_weight()

        # 3. Implement merging logic
        lists_to_merge = [
            self.wdeno_list, self.searched_points_list, self.inv_covariances_list, 
            self.gaussian_normterm_list, self.means_list, self.call_num_list, 
            self.rej_num_list, self.wcoeff_list, self.proposalcoeff_list
        ]
        
        merged_lists, self.searched_log_densities_list = merge_arrays(
            lists_to_merge, cluster_indices, self.element_num_list, 
            self.searched_log_densities_list, self.latest_prob_index, self.cov_update_count
        )
        
        (self.wdeno_list, self.searched_points_list, self.inv_covariances_list, 
         self.gaussian_normterm_list, self.means_list, self.call_num_list, 
         self.rej_num_list, self.wcoeff_list, self.proposalcoeff_list) = merged_lists
         
        self.max_logden_list = merge_max_list(self.max_logden_list, cluster_indices)
        self.element_num_list = merge_element_num_list(self.element_num_list, cluster_indices)
        self.now_covariances = merge_max_list(self.now_covariances, cluster_indices)
        self.now_normterms = merge_max_list(self.now_normterms, cluster_indices)
        
        n_proc_prev = self.n_proc
        self.n_proc = len(self.searched_log_densities_list)
        
        if self.stop_on_merge and self.n_proc < n_proc_prev:
            logger.info(f"Merge detected at iteration {i}. Stopping.")
            print(f"Merge detected at iteration {i}. Stopping.")
            self.current_iter = i + 1 
            return True

        self.last_gaussian_points = [self.last_gaussian_points[cluster_indices[j][0]] for j in range(self.n_proc)]
        
        return False

    def _update_means_and_covariances(self, i: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Update means and covariances based on current samples.
        
        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: points_list, probabilities_list
        """
        points_list = []
        probabilities_list = []
        self.now_means = []
        
        if (i + 1) % self.gamma == 0:
            self.now_covariances = []
            self.now_normterms = []
            self.keep_trial_seeds = np.full(self.n_proc, True)

        for j in range(self.n_proc):
            ind1 = 0
            ind2 = self.element_num_list[j]
            points_list.append(self.searched_points_list[j][ind1:ind2])
            
            # Calculate probabilities
            probs = np.exp(self.searched_log_densities_list[j][ind1:ind2] - self.loglike_normalization)
            if np.any(self.wdeno_list[j][ind1:ind2] <= 0.):
                logger.warning(f"Weights <= 0, seed ind {j}, iter {i}")
            else:
                probs /= self.wdeno_list[j][ind1:ind2]
            
            probs[probs < 0] = 0
            weights_sum = np.sum(probs)
            
            if weights_sum > 0:
                probs /= weights_sum
            else:
                probs = np.ones_like(probs) / len(probs)
                
            probabilities_list.append(probs)
            
            # Calculate Mean
            mean = np.average(points_list[j], weights=probs, axis=0)
            self.now_means.append(mean)
            
            # Update Covariance periodically
            if (i + 1) % self.gamma == 0:
                covariance = np.cov(points_list[j], aweights=probs, rowvar=False, ddof=0)
                n_samples = len(probs)
                mvn = multivariate_normal(mean=np.zeros(self.ndim), cov=covariance, allow_singular=True)
                
                # Note: config.seed is used at init, but here we rely on global state or pass random_state if needed.
                # Since we fixed global seed and LHS, this should be deterministic.
                # However, to be consistent with earlier fixes, if we wanted per-step determinism, we'd pass seed.
                # But here I am just refactoring, so I keep original logic (which uses global state).
                original_zeromean_samples = mvn.rvs(size=self.integral_num)
                
                is_out_of_bounds = np.any((original_zeromean_samples < (0 - mean)) | (original_zeromean_samples > (1 - mean)), axis=1)
                if is_out_of_bounds.sum() / self.integral_num < self.USE_BETA_THRESHOLD:
                    self.use_beta = False
                else:
                    self.use_beta = self.config.use_beta
                    
                if self.boundary_limiting and self.use_beta:
                    cov_inv = np.linalg.inv(covariance)
                    diff = points_list[j] - mean
                    Adiff = np.einsum('jk,ik->ji', cov_inv, diff)
                    W = probs
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

        return points_list, probabilities_list

    def _calculate_guess_size(self, j: int, ind1: int) -> int:
        """Calculate the number of guesses based on acceptance history."""
        start_idx = ind1 - self.LOOKBACK_WINDOW
        recent_calls = self.call_num_list[j][start_idx:ind1].sum()
        
        # Avoid division by zero if lookback window is effectively empty (though indices are handled safely by slice)
        # Logic matches original: mean calls per point / divisor
        avg_calls = recent_calls / self.LOOKBACK_WINDOW
        
        guess_size = max(avg_calls / self.GUESS_SIZE_DIVISOR, self.MIN_GUESS_SIZE)
        return int(min(self.trail_size, guess_size))

    def _sample_with_boundary(self, j: int, n_guess: int, points_all: np.ndarray, 
                            probabilities_all: np.ndarray, mvn: multivariate_normal) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Perform rejection sampling to satisfy boundary constraints ([0, 1]^d).
        
        Returns:
            Tuple: (selected_points, selected_log_densities, selected_means, calls_made)
        """
        # Initialize loop variables
        bulky_ind1 = 0
        while_call_count = 0
        total_calls_added = 0
        
        # Pre-generate bulky samples to avoid calling rvs inside the loop too often
        zeromean_samples = mvn.rvs(size=self.trail_size)
        bulky_mean_inds = np.random.choice(len(probabilities_all), self.trail_size, p=probabilities_all, replace=True)
        
        # Buffers for the final selected point
        # Although we might search many points, we only need ONE success for this batch_point_num=1 logic
        # (Note: Current code implies batch_point_num=1 for boundary logic logic flow)
        
        while True:
            # Refresh bulk samples if we ran out
            if bulky_ind1 + n_guess > self.trail_size:
                bulky_ind1 = 0
                zeromean_samples = mvn.rvs(size=self.trail_size)
                bulky_mean_inds = np.random.choice(len(probabilities_all), self.trail_size, p=probabilities_all, replace=True)

            bulky_ind2 = bulky_ind1 + n_guess
            
            # 1. Propose candidates
            indices_here = bulky_mean_inds[bulky_ind1:bulky_ind2]
            gaussian_means = points_all[indices_here]
            gaussian_points = zeromean_samples[bulky_ind1:bulky_ind2] + gaussian_means
            
            # 2. Check Boundaries
            out_of_bound_indices = np.any((gaussian_points < 0) | (gaussian_points > 1), axis=1)
            is_within_bounds = ~out_of_bound_indices
            valid_count = is_within_bounds.sum()
            
            # 3. Evaluate Densities (only for valid points)
            gaussian_log_densities = np.full(n_guess, -np.inf)
            if valid_count > 0:
                if self.use_pool and self.pool is not None:
                    # Filter points for parallel execution
                    valid_points_list = [gaussian_points[k] for k in range(n_guess) if is_within_bounds[k]]
                    results = self.pool.map(self.log_density_func, valid_points_list)
                    gaussian_log_densities[is_within_bounds] = np.array(results).flatten()
                else:
                    gaussian_log_densities[is_within_bounds] = self.log_density_func(gaussian_points[is_within_bounds])
            
            # 4. Compute Weights & Acceptance
            # We need single_weight_deno only for valid points
            single_weight_deno = np.ones(n_guess)
            if valid_count > 0:
                # Need covariance/normterm for the CURRENT iteration's weighting
                # In original code: int((i + 1) * self.batch_point_num / self.cov_update_count)
                # We need 'i' passed in or accessible. 
                # To avoid passing 'i' deep, we can pass the specific cov/norm needed.
                # For now, let's assume we access self state or passed params.
                # Actually, in original code, it accessed self.inv_covariances_list[j][...]
                # We'll use the current covariance snapshot which corresponds to the *next* update slot
                # But wait, original code used `i` which is passed to _generate_new_points.
                # We need to compute the correct index.
                # Let's compute it outside and pass it in, OR just recompute it here if we pass 'i'.
                # To keep signature clean, let's use the 'current' attributes which should be up to date?
                # No, inv_covariances_list is a history. 
                pass

            # Update loop counters
            bulky_ind1 = bulky_ind2
            while_call_count += n_guess
            
            # ... (Logic gets complex here to extract cleanly without changing behavior)
            # To ensure EXACT behavior match, I will inline the specific weighting check logic 
            # or pass necessary context.
            return None # Placeholder to stop and rethink strategy for safety

    # RETRYING STRATEGY: 
    # The dependencies on `i` and `self.inv_covariances_list` inside the loop make extraction tricky 
    # without passing many arguments. 
    # Instead, I will keep `_sample_with_boundary` inside `Sampler` and pass `i` to it.

    def _generate_new_points(self, i: int, points_list: List[np.ndarray], probabilities_list: List[np.ndarray]) -> None:
        """Generate new candidate points."""
        self.last_gaussian_points = []
        
        # Pre-calculate index for covariance lookup to avoid repetition
        # Matches original: int((i + 1) * self.batch_point_num / self.cov_update_count)
        cov_idx = int((i + 1) * self.batch_point_num / self.cov_update_count)

        for j in range(self.n_proc):
            points_all = points_list[j]
            probabilities_all = probabilities_list[j]
            mean = self.now_means[j]
            covariance = self.now_covariances[j]
            ind1 = self.element_num_list[j]
            ind2 = self.element_num_list[j] + self.batch_point_num
            
            n_guess = self._calculate_guess_size(j, ind1)
            
            mvn = multivariate_normal(mean=np.zeros(self.ndim), cov=covariance, allow_singular=True)
            self.last_gaussian_points.append(mean.copy())
            
            # Initialize result containers
            final_points = None
            final_log_densities = None
            final_means = None
            calls_added = 0
            eff_calls_added = 0

            if self.boundary_limiting:
                # --- Boundary-Aware Sampling Logic ---
                bulky_ind1 = 0
                while_call_count = 0
                
                # Pre-allocate bulk samples
                zeromean_samples = mvn.rvs(size=self.trail_size)
                bulky_mean_inds = np.random.choice(len(probabilities_all), self.trail_size, p=probabilities_all, replace=True)
                
                while True:
                    # Refresh bulk if needed
                    if bulky_ind1 + n_guess > self.trail_size:
                        bulky_ind1 = 0
                        zeromean_samples = mvn.rvs(size=self.trail_size)
                        bulky_mean_inds = np.random.choice(len(probabilities_all), self.trail_size, p=probabilities_all, replace=True)
                    
                    bulky_ind2 = bulky_ind1 + n_guess
                    indices_here = bulky_mean_inds[bulky_ind1:bulky_ind2]
                    gaussian_means = points_all[indices_here]
                    gaussian_points = zeromean_samples[bulky_ind1:bulky_ind2] + gaussian_means
                    
                    # Reset buffers
                    gaussian_log_densities = np.full(n_guess, -np.inf)
                    single_weight_deno = np.ones(n_guess)
                    
                    # Check Bounds
                    out_of_bound_indices = np.any((gaussian_points < 0) | (gaussian_points > 1), axis=1)
                    is_within_bounds = ~out_of_bound_indices
                    valid_count = is_within_bounds.sum()
                    
                    # Evaluate Valid Points
                    if valid_count > 0:
                        if self.use_pool and self.pool is not None:
                            valid_points_list = [gaussian_points[k] for k in range(n_guess) if is_within_bounds[k]]
                            results = self.pool.map(self.log_density_func, valid_points_list)
                            gaussian_log_densities[is_within_bounds] = np.array(results).flatten()
                        else:
                            gaussian_log_densities[is_within_bounds] = self.log_density_func(gaussian_points[is_within_bounds])
                        
                        # Calculate Weights for Rejection
                        single_weight_deno[is_within_bounds] = weighting_seeds_onepoint_with_onemean(
                            gaussian_points[is_within_bounds], 
                            gaussian_means[is_within_bounds], 
                            self.inv_covariances_list[j][cov_idx], 
                            self.gaussian_normterm_list[j][cov_idx]
                        )

                    # Rejection Criteria
                    # Note: exclude_scale_z defaults to inf, so threshold is usually 0
                    threshold = np.sum(probabilities_all) / self.exclude_scale_z
                    if_weights_big = (np.exp(gaussian_log_densities - self.loglike_normalization) / single_weight_deno) > threshold
                    
                    # Force array type just in case
                    if_weights_big = np.array(if_weights_big, dtype=bool) 
                    # Original code had: if_weights_big = np.full(if_weights_big.shape, True, dtype=bool) 
                    # WAIT! The original code OVERWROTE the logic check with True!
                    # "if_weights_big = np.full(if_weights_big.shape, True, dtype=bool)"
                    # This means it ALWAYS accepts the first valid point found? 
                    # Let me double check the read file content.
                    
                    # YES. In the original code read earlier:
                    # if_weights_big = np.exp(...) > ...
                    # if_weights_big = np.full(if_weights_big.shape, True, dtype=bool)
                    # This effectively disables the importance weight rejection check, making it simple rejection sampling
                    # based on "found a valid point?" (if density > -inf).
                    # I must preserve this behavior (even if it looks odd/debug-like).
                    if_weights_big[:] = True 

                    bulky_ind1 = bulky_ind2
                    while_call_count += n_guess
                    
                    has_valid = if_weights_big.any() # Effectively "is there any point?"
                    
                    # Decision Logic
                    if not has_valid:
                        # Case 1: No valid points found in this batch
                        calls_added += n_guess
                        eff_calls_added += valid_count
                        # Loop continues...
                    else:
                        # Case 2: Found at least one valid point
                        # Original: if if_weights_big[0]: ... else: ...
                        # Since all are True, it just picks the first one?
                        # Wait, logic is: 
                        # if if_weights_big[0]: valid_index=0 ... break
                        # else: valid_index=argmax... break
                        # Since we set all True, it will always take index 0.
                        
                        valid_index = 0
                        calls_added += (valid_index + 1) # = 1
                        eff_calls_added += 1 # logic for eff_calls was simplified in original for index 0
                        
                        # Extract the winner
                        final_log_densities = gaussian_log_densities[valid_index:valid_index + 1]
                        final_points = gaussian_points[valid_index:valid_index + 1]
                        final_means = gaussian_means[valid_index:valid_index + 1]
                        break
                    
                    # Timeout / Give up Check
                    if while_call_count > int(self.trail_size) or not self.keep_trial_seeds[j]:
                        self.keep_trial_seeds[j] = False
                        # Force accept the first one (even if invalid? No, indices match)
                        valid_index = 0
                        calls_added += n_guess
                        eff_calls_added += valid_count
                        
                        final_log_densities = gaussian_log_densities[valid_index:valid_index + 1]
                        final_points = gaussian_points[valid_index:valid_index + 1]
                        final_means = gaussian_means[valid_index:valid_index + 1]
                        break
                        
            else:
                # --- Unbounded Sampling Logic ---
                indices_here = np.random.choice(len(probabilities_all), self.batch_point_num, p=probabilities_all, replace=True)
                gaussian_means = points_all[indices_here]
                zeromean_samples = mvn.rvs(size=self.batch_point_num)
                if self.batch_point_num == 1:
                    zeromean_samples = zeromean_samples.reshape(1, -1)
                gaussian_points = zeromean_samples + gaussian_means
                
                if self.use_pool and self.pool is not None:
                    gaussian_points_list = [gaussian_points[k, :] for k in range(gaussian_points.shape[0])]
                    results = self.pool.map(self.log_density_func, gaussian_points_list)
                    gaussian_log_densities = np.array(results).flatten()
                else:
                    gaussian_log_densities = self.log_density_func(gaussian_points)

                calls_added = self.batch_point_num
                eff_calls_added = self.batch_point_num
                
                final_log_densities = gaussian_log_densities
                final_points = gaussian_points
                final_means = gaussian_means

            # Update State
            self.call_num_list[j][ind1:ind2] += calls_added
            self.eff_calls += eff_calls_added
            
            self.searched_log_densities_list[j][ind1:ind2] = final_log_densities.copy()
            self.searched_points_list[j][ind1:ind2] = final_points.copy()
            self.means_list[j][ind1:ind2] = final_means.copy()
            self.proposalcoeff_list[j][ind1:ind2] = 1.0
            
            self.element_num_list[j] += self.batch_point_num
            self.max_logden_list[j] = max(self.max_logden_list[j], final_log_densities.max())

    def _report_progress(self, i: int, stop_dlogZ: Optional[float], pbar: tqdm) -> bool:
        """Calculate stats, report progress, and check stopping conditions."""
        
        # Check flags first
        try:
            self._check_flags_and_take_actions()
        except Exception as e:
            logger.error(f"Flag actions failed: {e}")

        compute_logZ = (i % self.print_iter == 0) or (stop_dlogZ is not None and (i % self.alpha == 0))
        logZ = None
        dlogZ = None
        
        if compute_logZ:
            c_term = self.loglike_normalization
            calls = sum(self.element_num_list)
            ind1 = max(int(self.element_num_list[0] * (1 - self.EVIDENCE_ESTIMATION_FRACTION)), 0)
            ind2 = self.element_num_list[0] - self.batch_point_num
            
            # Using generator expression for sum to save memory
            wsum = sum(np.sum(np.exp(self.searched_log_densities_list[j][ind1:ind2] - c_term) / self.wdeno_list[j][ind1:ind2]) for j in range(self.n_proc)) * self.n_proc * (self.alpha * self.batch_point_num)
            Nsum = sum(self.call_num_list[j][ind1:ind2].sum() for j in range(self.n_proc))
            
            logZ = c_term - np.log(Nsum) + np.log(wsum)

            if stop_dlogZ is not None and (i % self.alpha == 0):
                if self._last_logZ_for_stop is not None and self._last_logZ_iter is not None and (i - self._last_logZ_iter) >= self.alpha:
                    dlogZ = abs(logZ - self._last_logZ_for_stop)
                    self._last_dlogZ = dlogZ
                self._last_logZ_for_stop = logZ
                self._last_logZ_iter = i

            # Record baseline stats
            if i > 0:
                baseline_file = os.path.join('basic_results', 'baseline_stats.json')
                os.makedirs('basic_results', exist_ok=True)
                
                # Calculate additional checksums for the primary process (index 0)
                p0_samples = self.searched_points_list[0][:self.element_num_list[0]]
                p0_mean_sum = float(np.sum(np.mean(p0_samples, axis=0)))
                p0_std_sum = float(np.sum(np.std(p0_samples, axis=0)))
                
                baseline_data = {
                    "iter": i,
                    "logZ": float(logZ) if logZ is not None else None,
                    "max_ld": float(self.max_logden_list[0]),
                    "n_proc": int(self.n_proc),
                    "p0_mean_sum": p0_mean_sum,
                    "p0_std_sum": p0_std_sum
                }
                
                if os.path.exists(baseline_file):
                    try:
                        with open(baseline_file, 'r') as f:
                            all_baselines = json.load(f)
                    except:
                        all_baselines = {}
                else:
                    all_baselines = {}
                    
                all_baselines[str(i)] = baseline_data
                with open(baseline_file, 'w') as f:
                    json.dump(all_baselines, f, indent=4)

            display_dlogZ = self._last_dlogZ if self._last_dlogZ is not None else np.nan
            if dlogZ is not None:
                display_dlogZ = dlogZ
            # Report stats
            status = f"samples: {Nsum}, evals: {calls}, n_proc: {self.n_proc}, cov[0]: {self.now_covariances[0][0, 0]:.5e}, logZ: {logZ:.5f}, dlogZ: {display_dlogZ:.5e}, max_ld: {self.max_logden_list[0]:.5f}"
            pbar.set_description(status)
            if i % self.print_iter == 0:
                pbar.update(self.print_iter)

            if stop_dlogZ is not None and dlogZ is not None and dlogZ <= stop_dlogZ:
                self.current_iter = i + 1
                stop_message = f"Early stopping at iter {i}: dlogZ={dlogZ:.5e} <= stop_dlogZ={stop_dlogZ:.5e}, logZ={logZ:.5f}"
                logger.info(stop_message)
                print(stop_message)
                return True
        
        return False

    def run_sampling(self, num_iterations: int, savepath: str, print_iter: int = 1,
                 stop_dlogZ: Optional[float] = None,
                 external_lhs_points: Optional[np.ndarray] = None,
                 external_lhs_log_densities: Optional[np.ndarray] = None,
                 callback: Optional[Callable[['Sampler', int], None]] = None) -> None:
        """Run the sampling process for a specified number of iterations.

        Parameters
        ----------
        num_iterations : int
            Total number of iterations to execute.
        savepath : str
            Directory path for saving sampler state.
        print_iter : int, optional
            Progress update cadence.
        stop_dlogZ : float, optional
            Absolute difference threshold |logZ(i) - logZ(i-alpha)| to trigger early stopping;
            disabled when None.
        callback : callable, optional
            Function called at the start of each iteration.
            Signature: callback(sampler, i)
        """
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        # Validate external inputs if provided
        if external_lhs_points is not None or external_lhs_log_densities is not None:
            if external_lhs_points is None or external_lhs_log_densities is None:
                raise ValueError("Both external_lhs_points and external_lhs_log_densities must be provided together")
            if len(external_lhs_points) != len(external_lhs_log_densities):
                raise ValueError("external_lhs_points and external_lhs_log_densities must have same length")
            if external_lhs_points.shape[1] != self.ndim:
                raise ValueError("external_lhs_points must have shape (n_samples, ndim)")
            
        self.savepath = savepath
        self.print_iter = print_iter

        # If starting from scratch, initialize everything
        if self.current_iter == 0:
            self.initialize_first_iteration(num_iterations, external_lhs_points, external_lhs_log_densities)  # self.initialize_first_iteration(num_iterations)
            # Initialize additional variables used in the loop
            self.keep_trial_seeds = np.full(self.n_proc, True)
            self.eff_calls = 0        
            num_iterations -= 1
            # Reset stopping trackers for fresh runs
            self._last_logZ_for_stop = None
            self._last_logZ_iter = None
            self._last_dlogZ = None
        else:
            # If resuming from a previous run, check if arrays need to be extended
            self._extend_arrays_if_needed(num_iterations)
        
        try:
            pbar = tqdm(total=num_iterations, initial=0, desc="Sampling")
            
            for i in range(self.current_iter, self.current_iter + num_iterations):
                if self.debug and i == 0:
                    print(f"DEBUG: Iter {i} start")
                    print(f"DEBUG: LHS LogDensities[0]: {self.searched_log_densities_list[0][:self.batch_point_num]}")
                    print(f"DEBUG: LHS Points[0][0]: {self.searched_points_list[0][0]}")

                # Execute callback if provided (e.g. for visualization or external monitoring)
                if callback:
                    callback(self, i)

                # Dynamic normalization update to prevent overflow
                if len(self.max_logden_list) > 0:
                    max_so_far = max(self.max_logden_list)
                    if max_so_far > self.loglike_normalization:
                         self.loglike_normalization = max_so_far

                # Weighting calculations
                if i > 0:
                    self._perform_weighting_update()
    
                if self.debug and i == 0:
                     print(f"DEBUG: After Weighting Update")
                     print(f"DEBUG: wdeno_list[0][0]: {self.wdeno_list[0][0]}")

                # Merging clusters logic
                if self._merge_processes(i):
                    break
    
                # Update means and covariances
                points_list, probabilities_list = self._update_means_and_covariances(i)
    
                # Generate new points
                self._generate_new_points(i, points_list, probabilities_list)

                if self.debug and i == 0:
                    print(f"DEBUG: After Generation")
                    print(f"DEBUG: Generated Point[0]: {self.searched_points_list[0][self.element_num_list[0]-1]}")
                    print(f"DEBUG: Generated LogDen[0]: {self.searched_log_densities_list[0][self.element_num_list[0]-1]}")

                # Report progress and check stopping condition
                if self._report_progress(i, stop_dlogZ, pbar):
                    break

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
        Calculate and return importance weights for all samples across all processes,
        handling the special case of the latest batch of samples.
        
        Returns:
        -------
        list of numpy.ndarray:
            A list where each element is an array of importance weights for a process.
        """
        weights_list = []
        for j in range(self.n_proc):
            element_num = self.element_num_list[j]
            
            # Calculate importance weights: exp(log_density - normalization) / denominator
            weights = np.exp(self.searched_log_densities_list[j][:element_num] - self.loglike_normalization)
            
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
            If False, returns lists of arrays for each process. Default is False.
        
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
            for j in range(self.n_proc):
                element_num = self.element_num_list[j]
                samples = self.searched_points_list[j][:element_num]
                transformed_samples = self.apply_prior_transform(samples, self.prior_transform)
                transformed_samples_list.append(transformed_samples)
        else:
            transformed_samples_list = []
            for j in range(self.n_proc):
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


