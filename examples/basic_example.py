"""
Basic example of using the Paris Monte Carlo Sampler.

This example demonstrates:
1. Setting up a simple multivariate Gaussian target distribution
2. Configuring the sampler with minimal parameters
3. Running a short sampling process
4. Analyzing basic results
"""

import numpy as np
from parismc import Sampler, SamplerConfig

# Define target distribution parameters at module level
TRUE_MEAN = np.array([0.3, 0.7])
TRUE_COV = np.array([[0.01, 0.005], [0.005, 0.02]])
INV_COV = np.linalg.inv(TRUE_COV)

def log_density(x):
    """
    Log-density for multivariate Gaussian in [0,1]^2 space.
    
    Parameters:
    ----------
    x : array-like, shape (n_samples, 2)
        Sample points in [0,1]^2
        
    Returns:
    -------
    array-like, shape (n_samples,)
        Log-density values
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Compute Mahalanobis distance
    diff = x - TRUE_MEAN
    mahal_dist = np.einsum('ij,jk,ik->i', diff, INV_COV, diff)
    
    # Return log-density (without normalization constant)
    return -0.5 * mahal_dist

def main():
    
    # Configure sampler with minimal settings
    config = SamplerConfig()  # Use all default values
    
    # Initialize sampler
    ndim = 2
    n_walkers = 3
    init_cov_list = [np.eye(ndim) * 0.05] * n_walkers
    
    print("Initializing sampler...")
    sampler = Sampler(
        ndim=ndim,
        n_seed=n_walkers,
        log_reward_func=log_density,
        init_cov_list=init_cov_list
        # No prior_transform needed for this simple example
    )
    
    # Prepare initial LHS samples
    print("Preparing LHS samples...")
    sampler.prepare_lhs_samples(lhs_num=1000, batch_size=100)
    
    # Run sampling
    print("Running sampling...")
    sampler.run_sampling(num_iterations=100, savepath='./basic_results', print_iter=20)
    
    # Get results
    print("Extracting results...")
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    
    # Basic analysis
    print(f"\nResults Summary:")
    print(f"Total samples: {len(samples)}")
    print(f"Effective sample size: {1/np.sum(weights**2):.1f}")
    
    # Weighted statistics
    weighted_mean = np.average(samples, weights=weights, axis=0)
    weighted_cov = np.cov(samples.T, aweights=weights)
    
    print(f"\nTrue mean: {TRUE_MEAN}")
    print(f"Estimated mean: {weighted_mean}")
    print(f"Mean error: {np.linalg.norm(weighted_mean - TRUE_MEAN):.6f}")
    
    print(f"\nTrue covariance diagonal: {np.diag(TRUE_COV)}")
    print(f"Estimated covariance diagonal: {np.diag(weighted_cov)}")
    
    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        
        # Plot 1: Sample scatter plot
        plt.subplot(1, 2, 1)
        # Color samples by their weights
        scatter = plt.scatter(samples[:, 0], samples[:, 1], 
                            c=weights, s=30, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='Sample Weight')
        plt.scatter(TRUE_MEAN[0], TRUE_MEAN[1], color='red', s=100, 
                   marker='x', label='True mean', linewidth=3)
        plt.scatter(weighted_mean[0], weighted_mean[1], color='orange', s=100, 
                   marker='+', label='Estimated mean', linewidth=3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Weighted Samples')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Weight distribution
        plt.subplot(1, 2, 2)
        plt.hist(weights, bins=30, alpha=0.7, density=True, edgecolor='black')
        plt.xlabel('Weight')
        plt.ylabel('Density')
        plt.title('Weight Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('basic_results/sampling_results.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: basic_results/sampling_results.png")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except:
            pass
        
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"\nPlotting failed: {e}")
    
    print("\nBasic example completed successfully!")
    print("For more complex examples, see multimodal_example.py")

if __name__ == "__main__":
    main()
