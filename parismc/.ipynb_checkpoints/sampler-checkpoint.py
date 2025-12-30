import numpy as np
from smt.sampling_methods import LHS
from multiprocessing import Pool
import pickle
import os
import json
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
    merge_type: str = 'distance' # 'distance', 'single', or 'multiple'

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
        sampling = LHS(xlimits=xlimits)
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

    def run_sampling(self, num_iterations: int, savepath: str, print_iter: int = 1,
                 stop_dlogZ: Optional[float] = None,
                 external_lhs_points: Optional[np.ndarray] = None,
                 external_lhs_log_densities: Optional[np.ndarray] = None) -> None:
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
                # Dynamic normalization update to prevent overflow
                if len(self.max_logden_list) > 0:
                    max_so_far = max(self.max_logden_list)
                    if max_so_far > self.loglike_normalization:
                         self.loglike_normalization = max_so_far

                points_list = []
                probabilities_list = []
    
                # Weighting calculations
                if i > 0:
                    for j in range(self.n_proc):
                        ind1 = max(-self.latest_prob_index + self.element_num_list[j], 0)
                        ind2 = self.element_num_list[j]
                        points_list.append(self.searched_points_list[j][ind1:ind2])
                        probabilities_list.append(np.exp(self.searched_log_densities_list[j][ind1:ind2] - self.loglike_normalization))
    
                        # 1. New proposals (Single Covariance)
                        ind1_newproposals = max(ind2 - self.batch_point_num, 0)
                        addon_weights = self._compute_weight_segment(
                            points=points_list[j], 
                            param_idx=j,  # <--- param source is j
                            start_idx=ind1_newproposals, end_idx=ind2,
                            cov_mode="single", cov_ref_idx=int((ind2 - 1) / self.cov_update_count)
                        )
                        self.wdeno_list[j][ind1:ind2] += addon_weights
                
                        # 2. New points with old proposals (Multi Covariance)
                        ind2_oldproposals = self.element_num_list[j] - self.batch_point_num
                        addon_weights = self._compute_weight_segment(
                            points=self.searched_points_list[j][ind2_oldproposals:ind2],
                            param_idx=j,  # <--- param source is j
                            start_idx=ind1, end_idx=ind2_oldproposals,
                            cov_mode="multi"
                        )
                        self.wdeno_list[j][ind2_oldproposals:ind2] += addon_weights
                
                        # 3. Subtracting for old proposals (Single Covariance)
                        if ind1 > 0:
                            ind1_oldproposals = max(ind1 - self.batch_point_num, 0)
                            addon_weights = self._compute_weight_segment(
                                points=self.searched_points_list[j][ind1:ind2_oldproposals],
                                param_idx=j,  # <--- param source is j
                                start_idx=ind1_oldproposals, end_idx=ind1,
                                cov_mode="single", cov_ref_idx=int((ind1 - 1) / self.cov_update_count)
                            )
                            self.wdeno_list[j][ind1:ind2_oldproposals] -= addon_weights

                # Merging clusters logic
                if i > 0:
                    # 1. Sort processes by max log density
                    combined = sorted(zip(self.max_logden_list, self.last_gaussian_points, self.searched_points_list, self.searched_log_densities_list, self.means_list, self.inv_covariances_list, self.gaussian_normterm_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.wdeno_list, self.element_num_list, self.now_covariances, self.now_normterms, self.proposalcoeff_list), reverse=True, key=lambda x: x[0])
                    
                    self.max_logden_list, self.last_gaussian_points, self.searched_points_list, self.searched_log_densities_list, self.means_list, self.inv_covariances_list, self.gaussian_normterm_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.wdeno_list, self.element_num_list, self.now_covariances, self.now_normterms, self.proposalcoeff_list = zip(*combined)

                    # 2. Determine Cluster Indices (Strategy Selection)
                    cluster_indices = []

                    # 2. Determine Cluster Indices (Strategy Selection)
                    cluster_indices = []

                    if self.merge_type == 'distance':
                        # --- Strategy A: Distance-based Merging ---
                        self.last_gaussian_points = list(np.array(self.last_gaussian_points))
                        cluster_indices = get_cluster_indices_cov(np.array(self.last_gaussian_points), self.now_covariances, dist=self.merge_dist)
                        cluster_indices = [sorted(sublist) for sublist in cluster_indices]
                    else:
                        # --- Strategy B/C: Weight-based Merging (Cross-proc check) ---
                        classified_indices = set()
                        new_clusters = []
                        
                        # Greedy Pairwise Merging
                        for j in range(self.n_proc):
                            if j in classified_indices:
                                continue # Already handled
                                
                            found_merge = False
                            for j_prime in range(self.n_proc):
                                if j == j_prime or j_prime in classified_indices:
                                    continue 

                                # Prepare data for cross-check
                                ind1 = max(-self.latest_prob_index + self.element_num_list[j_prime], 0)
                                ind2 = self.element_num_list[j_prime]
                                
                                # Evaluate j's latest points using j_prime's parameters
                                if self.merge_type == 'single':
                                    target_end = self.element_num_list[j]
                                    target_start = max(0, target_end - self.batch_point_num)
                                elif self.merge_type == 'multiple':
                                    target_end = self.element_num_list[j]
                                    target_start = 0#max(0, target_end - self.alpha)
                                else:
                                    # Should not happen given outer check, but safe fallback
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
                                
                                # Compare j_prime's contribution vs j's own contribution
                                current_weights = self.wdeno_list[j][target_start:target_end]
                                
                                # Merge Condition
                                if self.merge_type == 'single':
                                    # Strategy B: Average Weight Comparison
                                    if np.mean(addon_weights) >= np.mean(current_weights):
                                        new_clusters.append([j, j_prime])
                                        classified_indices.add(j)
                                        classified_indices.add(j_prime)
                                        found_merge = True
                                        break 
                                elif self.merge_type == 'multiple':
                                    # Strategy C: Pointwise Density Comparison (>= 10% points)
                                    # Check if proposal_j'(x) > proposal_j(x) for at least 10% of points
                                    # addon_weights corresponds to proposal_j'(x) (approx/unnormalized density contribution)
                                    # current_weights corresponds to proposal_j(x)
                                    better_count = np.sum(addon_weights > current_weights/1)
                                    if len(addon_weights) > 0 and better_count>0:#better_count / len(addon_weights) >= 0.1:
                                        new_clusters.append([j, j_prime])
                                        classified_indices.add(j)
                                        classified_indices.add(j_prime)
                                        found_merge = True
                                        break
                            
                            if not found_merge:
                                # If no merge partner found, j stands alone (for now, will be added below if not in classified)
                                pass

                        # Add any remaining unclassified processes as their own clusters
                        for k in range(self.n_proc):
                            if k not in classified_indices:
                                new_clusters.append([k])
                        
                        cluster_indices = new_clusters
                        

    

                    #implement merging according to clusters
                    lists_to_merge = [self.wdeno_list, self.searched_points_list, self.inv_covariances_list, self.gaussian_normterm_list, self.means_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.proposalcoeff_list]
                    merged_lists, self.searched_log_densities_list = merge_arrays(lists_to_merge, cluster_indices, self.element_num_list, self.searched_log_densities_list, self.latest_prob_index, self.cov_update_count)
                    (self.wdeno_list, self.searched_points_list, self.inv_covariances_list, self.gaussian_normterm_list, self.means_list, self.call_num_list, self.rej_num_list, self.wcoeff_list, self.proposalcoeff_list) = merged_lists
                    self.max_logden_list = merge_max_list(self.max_logden_list, cluster_indices)
                    self.element_num_list = merge_element_num_list(self.element_num_list, cluster_indices)
                    self.now_covariances = merge_max_list(self.now_covariances, cluster_indices)
                    self.now_normterms = merge_max_list(self.now_normterms, cluster_indices)
                    n_proc_prev = self.n_proc
                    self.n_proc = len(self.searched_log_densities_list)
                    
                    if self.stop_on_merge and self.n_proc < n_proc_prev:
                        print(f"Merge detected at iteration {i}. Stopping.")
                        logger.info(f"Merge detected at iteration {i}. Stopping.")
                        self.current_iter = i + 1 # update iter count
                        break

                    last_gaussian_points_cache = [self.last_gaussian_points[cluster_indices[j][0]] for j in range(self.n_proc)]
                    self.last_gaussian_points = last_gaussian_points_cache
    
                # Update means and covariances
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
                    probabilities_list.append(np.exp(self.searched_log_densities_list[j][ind1:ind2] - self.loglike_normalization))
                    if np.any(self.wdeno_list[j][ind1:ind2] <= 0.):
                        logger.warning(f"Weights <= 0, seed ind {j}, iter {i}")
                    else:
                        probabilities_list[j] /= self.wdeno_list[j][ind1:ind2]
                    probabilities_list[j][probabilities_list[j] < 0] = 0
                    weights_sum = np.sum(probabilities_list[j])
                    if weights_sum > 0:
                        probabilities_list[j] /= weights_sum
                    else:
                        probabilities_list[j] = np.ones_like(probabilities_list[j]) / len(probabilities_list[j])
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
                for j in range(self.n_proc):
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
                    gaussian_log_densities = -np.inf * np.ones((n_guess,))
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
                            gaussian_log_densities[:] = -np.inf
                            single_weight_deno[:] = 1
                            out_of_bound_indices = np.any((gaussian_points < 0) | (gaussian_points > 1), axis=1)
                            is_within_bounds = ~out_of_bound_indices
                            
                            if self.use_pool and self.pool is not None and is_within_bounds.sum() > 0:
                                gaussian_points_list = [gaussian_points[k, :] for k in range(gaussian_points.shape[0]) if is_within_bounds[k]]
                                results = self.pool.map(self.log_density_func, gaussian_points_list)
                                gaussian_log_densities[is_within_bounds] = np.array(results).flatten()
                            elif is_within_bounds.sum() > 0:
                                gaussian_log_densities[is_within_bounds] = self.log_density_func(gaussian_points[is_within_bounds])
                            #print('gaussian_log_densities[is_within_bounds]',repr(gaussian_points[is_within_bounds]),gaussian_log_densities[is_within_bounds])
                                
                            single_weight_deno[is_within_bounds] = weighting_seeds_onepoint_with_onemean(gaussian_points[is_within_bounds], gaussian_means[is_within_bounds], self.inv_covariances_list[j][int((i + 1) * self.batch_point_num / self.cov_update_count)], self.gaussian_normterm_list[j][int((i + 1) * self.batch_point_num / self.cov_update_count)])
                            if_weights_big = np.exp(gaussian_log_densities - self.loglike_normalization) / single_weight_deno > weights_sum / self.exclude_scale_z
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
                                gaussian_log_densities = gaussian_log_densities[valid_index:valid_index + 1]
                                gaussian_points = gaussian_points[valid_index:valid_index + 1]
                                gaussian_means = gaussian_means[valid_index:valid_index + 1]
                                break
                            else:
                                valid_index = np.argmax(if_weights_big)
                                self.call_num_list[j][ind1:ind2] += valid_index + 1
                                true_indices = np.flatnonzero(is_within_bounds)
                                pos = np.where(true_indices == valid_index)[0][0] + 1
                                self.eff_calls += pos
                                gaussian_log_densities = gaussian_log_densities[valid_index:valid_index + 1]
                                gaussian_points = gaussian_points[valid_index:valid_index + 1]
                                gaussian_means = gaussian_means[valid_index:valid_index + 1]
                                break
                            if while_call_count > int(self.trail_size) or not self.keep_trial_seeds[j]:
                                self.keep_trial_seeds[j] = False
                                valid_index = 0
                                self.call_num_list[j][ind1:ind2] += n_guess
                                self.eff_calls += is_within_bounds.sum()
                                gaussian_log_densities = gaussian_log_densities[valid_index:valid_index + 1]
                                gaussian_points = gaussian_points[valid_index:valid_index + 1]
                                gaussian_means = gaussian_means[valid_index:valid_index + 1]
                                break

                    else:
                        # No boundary limiting: draw directly without truncation
                        # Select mean indices according to current weights
                        indices_here = np.random.choice(len(probabilities_all), self.batch_point_num, p=probabilities_all, replace=True)
                        gaussian_means = points_all[indices_here]
                        # Draw zero-mean samples and shift by selected means
                        zeromean_samples = mvn.rvs(size=self.batch_point_num)
                        # Ensure correct shape when batch_point_num == 1
                        if self.batch_point_num == 1:
                            zeromean_samples = zeromean_samples.reshape(1, -1)
                        gaussian_points = zeromean_samples + gaussian_means
                        # Evaluate log density
                        if self.use_pool and self.pool is not None:
                            gaussian_points_list = [gaussian_points[k, :] for k in range(gaussian_points.shape[0])]
                            results = self.pool.map(self.log_density_func, gaussian_points_list)
                            gaussian_log_densities = np.array(results).flatten()
                        else:
                            gaussian_log_densities = self.log_density_func(gaussian_points)
                        # Single-point denominator with current covariance snapshot
                        single_weight_deno = weighting_seeds_onepoint_with_onemean(
                            gaussian_points,
                            gaussian_means,
                            self.inv_covariances_list[j][int((i + 1) * self.batch_point_num / self.cov_update_count)],
                            self.gaussian_normterm_list[j][int((i + 1) * self.batch_point_num / self.cov_update_count)]
                        )
                        # Accounting for calls (no rejection cycle here)
                        self.call_num_list[j][ind1:ind2] += self.batch_point_num
                        self.eff_calls += self.batch_point_num

                    proposalcoeffs = np.ones((len(gaussian_means)))
                    self.searched_log_densities_list[j][ind1:ind2] = gaussian_log_densities.copy()
                    self.searched_points_list[j][ind1:ind2] = gaussian_points.copy()
                    self.means_list[j][ind1:ind2] = gaussian_means.copy()
                    self.proposalcoeff_list[j][ind1:ind2] = proposalcoeffs.copy()
                    self.element_num_list[j] += self.batch_point_num
                    self.max_logden_list[j] = max(self.max_logden_list[j], gaussian_log_densities.max())

                # After updating element counts, check flags and optionally perform actions
                try:
                    self._check_flags_and_take_actions()
                except Exception as e:
                    logger.error(f"Flag actions failed: {e}")

                # Update diagnostics
                compute_logZ = (i % self.print_iter == 0) or (stop_dlogZ is not None and (i % self.alpha == 0))
                logZ = None
                dlogZ = None
                if compute_logZ:
                    c_term = self.loglike_normalization
                    calls = sum(self.element_num_list)
                    ind1 = max(int(self.element_num_list[0] * (1 - self.EVIDENCE_ESTIMATION_FRACTION)), 0)
                    ind2 = self.element_num_list[0] - self.batch_point_num
                    wsum = sum(np.sum(np.exp(self.searched_log_densities_list[j][ind1:ind2] - c_term) / self.wdeno_list[j][ind1:ind2]) for j in range(self.n_proc)) * self.n_proc * (self.alpha * self.batch_point_num)
                    Nsum = sum(self.call_num_list[j][ind1:ind2].sum() for j in range(self.n_proc))
                    logZ = c_term - np.log(Nsum) + np.log(wsum)

                    if stop_dlogZ is not None and (i % self.alpha == 0):
                        if self._last_logZ_for_stop is not None and self._last_logZ_iter is not None and (i - self._last_logZ_iter) >= self.alpha:
                            dlogZ = abs(logZ - self._last_logZ_for_stop)
                            self._last_dlogZ = dlogZ
                        self._last_logZ_for_stop = logZ
                        self._last_logZ_iter = i

                    display_dlogZ = self._last_dlogZ if self._last_dlogZ is not None else np.nan
                    if dlogZ is not None:
                        display_dlogZ = dlogZ
                    # Report stats for the top process (processes are periodically sorted by max log-likelihood)
                    status = f"samples: {Nsum}, evals: {calls}, n_proc: {self.n_proc}, cov[0]: {self.now_covariances[0][0, 0]:.5e}, logZ: {logZ:.5f}, dlogZ: {display_dlogZ:.5e}, max_ld: {self.max_logden_list[0]:.5f}"
                    pbar.set_description(status)
                    if i % self.print_iter == 0:
                        pbar.update(self.print_iter)

                    if stop_dlogZ is not None and dlogZ is not None and dlogZ <= stop_dlogZ:
                        self.current_iter = i + 1
                        stop_message = f"Early stopping at iter {i}: dlogZ={dlogZ:.5e} <= stop_dlogZ={stop_dlogZ:.5e}, logZ={logZ:.5f}"
                        logger.info(stop_message)
                        print(stop_message)
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


