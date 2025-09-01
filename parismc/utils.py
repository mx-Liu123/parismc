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