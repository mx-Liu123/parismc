from scipy.special import gammainc
from scipy.optimize import bisect
import numpy as np

def weighting_seeds_manypoint(points_array, means_array, inv_covariance, norm_term, proposalcoeff_array):
    weights = np.zeros((points_array.shape[0]))  # Initialize the weights for each point    
    for k in range(means_array.shape[0]):  # Iterate over each mean
        diff = points_array - means_array[k]  # Calculate difference for each point and the current mean
        exponent = -0.5 * np.einsum('ij,jk,ik->i', diff, inv_covariance, diff)  # Exponent part for each mean
        #print('exponent == 0',exponent == 0)
        exponent[exponent == 0] = -points_array.shape[1]/2#-np.inf 
        weights += norm_term * np.exp(exponent) * proposalcoeff_array[k]    
    return weights

def weighting_seeds_onepoint_with_onemean(points_array, means_array, inv_covariance, norm_term):
    #assuming points and means have same shape
    weights = np.zeros((points_array.shape[0]))  # Initialize the weights for each point    
    for k in range(means_array.shape[0]):  # Iterate over each mean
        diff = points_array[k] - means_array[k]  # Calculate difference for each point and the current mean
        exponent = -0.5 * np.einsum('j,jk,k->', diff, inv_covariance, diff)  # Exponent part for each mean
        weights[k] = norm_term * np.exp(exponent)
    return weights    

    
def weighting_seeds_manycov(points_array, means_array, inv_covariances_array, norm_terms_array, proposalcoeff_array):
    weights = np.zeros((points_array.shape[0]))
    for k in range(points_array.shape[0]):
        diff = points_array[k] - means_array  # Calculate the difference (x - mu) for each point
        exponent = -0.5 * np.einsum('ij,ijk,ik->i', diff, inv_covariances_array, diff)  # Calculate the exponent part
        #exponent[exponent == 0] = -np.inf 
        weights[k] = (norm_terms_array * np.exp(exponent) * proposalcoeff_array).sum()
    return weights


def find_sigma_level(ndim, prob):
    """Helper method to compute sigma level for a given probability."""
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be between 0 and 1.")        
    def analytical_probability(k):
        return gammainc(ndim / 2, k**2 / 2) - prob
    return bisect(analytical_probability, 0, 10, xtol=1e-6)


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