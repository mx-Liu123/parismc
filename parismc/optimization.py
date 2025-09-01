import numpy as np
from scipy.optimize import minimize

def negative_BN_log_like(beta, diff, Adiff, W, W_sum, original_log_det_cov, original_zeromean_samples, mean, ndim, integral_num):
    #if beta > 0:
    #integral_points = original_zeromean_samples / beta[None, :] + mean 
    #is_within_bounds = np.all((integral_points >= 0) & (integral_points <= 1), axis=1)
    is_out_of_bounds = np.any(
        (original_zeromean_samples < (0 - mean) * beta[None, :]) | 
        (original_zeromean_samples > (1 - mean) * beta[None, :]),
        axis=1
    )
    
    portion = (integral_num - is_out_of_bounds.sum())/integral_num

    A = np.einsum('ij,ji->i', diff*beta[None, :]*beta[None, :], Adiff)
    A_inner_W = A@W

    scaling_factor = np.prod(beta**2)
    gaussian_normterm = np.exp(-0.5 * original_log_det_cov ) * np.sqrt(scaling_factor) / np.sqrt((2 * np.pi) ** ndim)
    return 0.5 * A_inner_W + W_sum * np.log( 1 / gaussian_normterm * portion )
    #else:
    #    return np.inf

def find_max_beta(diff, Adiff, W, W_sum, original_log_det_cov, original_zeromean_samples, mean, ndim, integral_num):
    # Objective function: expects beta as a vector, so beta[i] represents each element in the beta vector
    obj_func = lambda beta: negative_BN_log_like(beta, diff, Adiff, W, W_sum, original_log_det_cov, original_zeromean_samples, mean, ndim, integral_num)
    # Set initial guesses and bounds for each element in the beta vector
    vector_size = ndim  # Example: set size based on the length of A_inner_W
    initial_guess = [1] * vector_size  # Start with a vector close to 1 for each component
    bounds = [(0.5, 1+1e-3)] * vector_size  # Ensure all elements in beta are within (0.1, 1)
    # Run optimization with bounds using L-BFGS-B
    result = minimize(obj_func, initial_guess, method='Powell',tol=1e-2, bounds=bounds)#CG#Nelder-Mead
    # Check if optimization was successful
    if result.success:
        max_beta = result.x  # This will be a vector now
        return np.array(max_beta)
    else:
        #print("Optimization failed.")
        return np.array([1] * vector_size)

def oracle_approximating_shrinkage(cov_matrix, n_samples):
    """
    Estimate the Oracle Approximating Shrinkage covariance matrix.

    Parameters
    ----------
    cov_matrix : array-like of shape (n_features, n_features)
        The sample covariance matrix.

    n_samples : int
        The number of samples used to compute the sample covariance.

    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        The Oracle Approximating Shrinkage covariance matrix.

    shrinkage : float
        Shrinkage coefficient used in the computation.
    """


    p = cov_matrix.shape[0]
    mu = np.trace(cov_matrix) / p
    sum_of_squares = (cov_matrix**2).sum() #which is Tr(\Sigma^2)
    squared_trace = (mu * p)**2 #which is Tr(\Sigma)^2
    
    numerator = (1 - 2/p)*sum_of_squares + squared_trace
    denominator = (n_samples + 1 - 2/p)*sum_of_squares + (1 - n_samples/p)*squared_trace
    shrinkage = 1.0 if denominator == 0 else min(numerator / denominator, 1.0)

    # Shrink covariance matrix towards the identity matrix scaled by mu    
    shrunk_cov = (1.0 - shrinkage) * cov_matrix
    shrunk_cov.flat[::p + 1] += shrinkage * mu

    return shrunk_cov, shrinkage