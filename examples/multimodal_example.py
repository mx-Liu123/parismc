"""
Multimodal optimization example using the Paris Monte Carlo Sampler.

This example demonstrates:
1. Setting up a complex 10D multimodal target distribution
2. Using prior transforms
3. Configuring the sampler for challenging problems
4. Running large-scale sampling
5. Analyzing multimodal results
"""

import numpy as np
import os
from scipy.special import logsumexp
from parismc import Sampler, SamplerConfig

# Define modes at module level to avoid pickle issues
MODES = np.array([
    [0.85, 0.05, 0.85, 0.65, 0.05, 0.45, 0.55, 0.85, 0.45, 0.65],
    [0.45, 0.75, 0.25, 0.35, 0.45, 0.95, 0.45, 0.65, 0.95, 0.85],
    [0.05, 0.95, 0.35, 0.15, 0.95, 0.75, 0.65, 0.25, 0.25, 0.35],
    [0.15, 0.65, 0.55, 0.05, 0.55, 0.25, 0.25, 0.95, 0.15, 0.75],
    [0.95, 0.45, 0.15, 0.25, 0.15, 0.35, 0.85, 0.45, 0.65, 0.55],
    [0.65, 0.55, 0.75, 0.95, 0.35, 0.55, 0.05, 0.75, 0.75, 0.25],
    [0.35, 0.35, 0.95, 0.55, 0.65, 0.15, 0.35, 0.55, 0.05, 0.95],
    [0.75, 0.15, 0.05, 0.85, 0.85, 0.85, 0.75, 0.15, 0.55, 0.05],
    [0.55, 0.25, 0.45, 0.45, 0.25, 0.05, 0.15, 0.05, 0.35, 0.45],
    [0.25, 0.85, 0.65, 0.75, 0.75, 0.65, 0.95, 0.35, 0.85, 0.15]
])

WEIGHTS = np.array([1.0] * len(MODES))
LOG_WEIGHTS = np.log(WEIGHTS)

def log_density(final_states_raw):
    """
    Complex 10D multimodal log-density function with 10 modes.
    
    This function creates a challenging multimodal landscape with:
    - 10 distinct modes in 10D space
    - Sharp peaks (high curvature)
    - Coordinate transformation from [0,1] to [−0.2,1.2]
    
    Parameters:
    ----------
    final_states_raw : array-like, shape (n_samples, 10) or (10,)
        Sample points in [0,1]^10 space
        
    Returns:
    -------
    array-like, shape (n_samples,) or scalar
        Log-density values
    """
    # Transform coordinates: [0,1] -> [-0.2, 1.2]
    final_states_in = (final_states_raw * 1.4) - 0.2
    
    # Equal weights for all modes (already defined at module level)
    
    # Ensure input is 2D
    if final_states_in.ndim == 1:
        final_states_in = final_states_in[None, :]  # shape: (1, ndim)
    
    # Compute log-density for each mode
    log_likelihoods = []
    for mode in MODES:
        # Euclidean distance to mode center
        distance = np.linalg.norm(final_states_in - mode, axis=1)
        # Handle numerical issues
        distance = np.nan_to_num(distance, nan=1e6, posinf=1e6, neginf=1e6)
        # Sharp Gaussian-like peaks (note: very high curvature with factor 400)
        log_prob = -400 * distance**2
        log_likelihoods.append(log_prob)
    
    # Combine all modes using log-sum-exp trick
    log_likelihoods = np.array(log_likelihoods)  # shape: (num_modes, batch_size)
    log_likelihoods = LOG_WEIGHTS[:, None] + log_likelihoods  # add log weights
    total_log_likelihood = logsumexp(log_likelihoods, axis=0)  # shape: (batch_size,)
    
    # Return scalar if single sample, array otherwise
    if total_log_likelihood.shape[0] == 1:
        return total_log_likelihood[0]
    return total_log_likelihood

def prior_transform(u):
    """
    Transform from unit cube [0,1]^ndim to parameter space.
    
    In this case, we keep the same domain [0,1]^ndim since
    the coordinate transformation is handled inside log_density.
    
    Parameters:
    ----------
    u : array-like, shape (n_samples, ndim) or (ndim,)
        Points in unit cube [0,1]^ndim
        
    Returns:
    -------
    array-like, same shape as input
        Transformed points (in this case, unchanged)
    """
    return u

def analyze_results(sampler, savepath):
    """
    Analyze and summarize the sampling results.
    
    Parameters:
    ----------
    sampler : Sampler
        The sampler instance after running
    savepath : str
        Path where results are saved
    """
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Get samples and weights
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    
    # Basic statistics
    print(f"Total samples generated: {len(samples):,}")
    print(f"Effective sample size: {1/np.sum(weights**2):.1f}")
    print(f"Weight coefficient of variation: {np.std(weights)/np.mean(weights):.3f}")
    
    # Transform samples to the coordinate system used in log_density
    transformed_samples = (samples * 1.4) - 0.2
    
    # Compute weighted statistics
    weighted_mean = np.average(transformed_samples, weights=weights, axis=0)
    weighted_std = np.sqrt(np.average((transformed_samples - weighted_mean)**2, 
                                    weights=weights, axis=0))
    
    print(f"\nWeighted mean (transformed coordinates): {weighted_mean}")
    print(f"Weighted std (transformed coordinates): {weighted_std}")
    
    # Find best samples
    log_densities = sampler.log_density_func(samples)
    best_idx = np.argmax(log_densities)
    best_sample = samples[best_idx]
    best_log_density = log_densities[best_idx]
    
    print(f"\nBest sample found:")
    print(f"  Coordinates (unit cube): {best_sample}")
    print(f"  Coordinates (transformed): {(best_sample * 1.4) - 0.2}")
    print(f"  Log-density: {best_log_density:.2f}")
    
    # Mode detection (simple clustering based on high-density samples)
    high_likelihood_threshold = np.percentile(log_densities, 95)
    high_likelihood_mask = log_densities >= high_likelihood_threshold
    high_likelihood_samples = transformed_samples[high_likelihood_mask]
    
    print(f"\nHigh-density regions (top 5%):")
    print(f"  Number of samples: {np.sum(high_likelihood_mask)}")
    
    if np.sum(high_likelihood_mask) > 0:
        # Simple mode detection: find cluster centers
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(10, np.sum(high_likelihood_mask))
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(high_likelihood_samples)
                cluster_centers = kmeans.cluster_centers_
                
                print(f"  Detected {n_clusters} high-density clusters:")
                for i, center in enumerate(cluster_centers):
                    cluster_size = np.sum(cluster_labels == i)
                    print(f"    Cluster {i+1}: center = {center}, samples = {cluster_size}")
        except ImportError:
            print("  (sklearn not available for clustering analysis)")
    
    # Save additional analysis results
    analysis_file = os.path.join(savepath, 'analysis_summary.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"Multimodal Sampling Analysis Results\n")
        f.write(f"=====================================\n\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Effective sample size: {1/np.sum(weights**2):.1f}\n")
        f.write(f"Best log-density: {best_log_density:.6f}\n")
        f.write(f"Best sample (unit cube): {best_sample}\n")
        f.write(f"Best sample (transformed): {(best_sample * 1.4) - 0.2}\n")
    
    print(f"\nDetailed analysis saved to: {analysis_file}")

def visualize_marginal_distributions(sampler, savepath):
    """
    Create marginal distribution plots for each dimension.
    
    Parameters:
    ----------
    sampler : Sampler
        The sampler instance after running
    savepath : str
        Path where results are saved
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import norm
    except ImportError:
        print("Matplotlib/seaborn not available. Skipping visualization.")
        return
    
    print("\nCreating marginal distribution plots...")
    
    # Get samples and weights
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    ndim = samples.shape[1]
    
    # Visualization parameters
    bin_num = 50
    decay = 3  # For exponential smoothing
    
    def exponential_smoothing(hist, decay=1.0):
        """Apply exponential smoothing to histogram."""
        smoothed = np.zeros_like(hist)
        for i in range(len(hist)):
            weights_exp = np.exp(-decay * np.abs(np.arange(len(hist)) - i))
            weights_exp /= np.sum(weights_exp)  # normalize
            smoothed[i] = np.sum(hist * weights_exp)
        return smoothed
    
    def gaussian_mixture_likelihood(x, mode, variance=1/(2*400)):
        """Calculate the density for a single mode in the GMM for a 1D projection."""
        x_transformed = 1.4*x - 0.2
        return norm.pdf(x_transformed, loc=mode, scale=np.sqrt(variance))
    
    # Mode positions for each dimension (from the global MODES array)
    mode_positions = MODES  # shape: (10, 10)
    
    # Set up the plot
    sns.set(style="white", context="talk")
    
    # Create subplot grid
    n_rows = (ndim + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.5*n_rows))
    
    # Flatten axes for easier indexing
    if ndim == 1:
        axes = [axes]
    elif ndim <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    x_range = np.linspace(0, 1, 1000)
    
    for i in range(ndim):
        ax = axes[i]
        
        # PARIS samples (our method)
        param_samples = samples[:, i]
        hist, bin_edges = np.histogram(param_samples, bins=bin_num, weights=weights, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist = exponential_smoothing(hist, decay=decay)
        ax.plot(bin_centers, hist, color='green', linewidth=2, label='PARIS')
        
        # Analytical marginal density
        # Sum over all modes for this dimension
        gmm_curve = np.sum([gaussian_mixture_likelihood(x_range, mode=mode_positions[j, i]) 
                           for j in range(len(mode_positions))], axis=0) / len(mode_positions) * 1.4
        ax.plot(x_range, gmm_curve, color='grey', linestyle='-', linewidth=1.5, 
                label='True marginal')
        
        # Mark the true mode positions for this dimension
        true_modes_this_dim = mode_positions[:, i]
        # Transform back to unit cube: mode = (x + 0.2) / 1.4
        true_modes_unit = (true_modes_this_dim + 0.2) / 1.4
        # Filter modes that are within [0,1]
        valid_modes = true_modes_unit[(true_modes_unit >= 0) & (true_modes_unit <= 1)]
        
        for mode_pos in valid_modes:
            ax.axvline(x=mode_pos, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
        # Formatting
        ax.set_title(f'Dimension {i+1} Marginal Distribution', fontsize=14)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add legend to first subplot
        if i == 0:
            ax.legend(fontsize=10, frameon=True)
    
    # Remove empty subplots if ndim is odd
    if ndim % 2 == 1 and ndim > 1:
        fig.delaxes(axes[-1])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(savepath, 'marginal_distributions.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Marginal distribution plot saved to: {plot_filename}")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass
    
    # Create a 2D corner plot for the first few dimensions
    if ndim >= 2:
        create_corner_plot(samples, weights, savepath, max_dims=min(4, ndim))

def create_corner_plot(samples, weights, savepath, max_dims=4):
    """
    Create a corner plot showing 2D marginals.
    
    Parameters:
    ----------
    samples : array-like
        Sample points
    weights : array-like
        Sample weights
    savepath : str
        Save path
    max_dims : int
        Maximum number of dimensions to include
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return
    
    print(f"Creating corner plot for first {max_dims} dimensions...")
    
    # Use only first max_dims dimensions
    samples_subset = samples[:, :max_dims]
    
    fig, axes = plt.subplots(max_dims, max_dims, figsize=(12, 12))
    
    for i in range(max_dims):
        for j in range(max_dims):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: 1D marginal
                hist, bin_edges = np.histogram(samples_subset[:, i], bins=30, weights=weights, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ax.plot(bin_centers, hist, color='green', linewidth=2)
                ax.set_xlim(0, 1)
                ax.set_title(f'Dim {i+1}')
                
            elif i > j:
                # Lower triangle: 2D scatter
                scatter = ax.scatter(samples_subset[:, j], samples_subset[:, i], 
                                   c=weights, s=1, alpha=0.6, cmap='viridis')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
            else:
                # Upper triangle: empty
                ax.axis('off')
            
            if i == max_dims - 1:
                ax.set_xlabel(f'Dimension {j+1}')
            if j == 0 and i > 0:
                ax.set_ylabel(f'Dimension {i+1}')
    
    plt.tight_layout()
    corner_filename = os.path.join(savepath, 'corner_plot.png')
    plt.savefig(corner_filename, dpi=300, bbox_inches='tight')
    print(f"Corner plot saved to: {corner_filename}")
    
    try:
        plt.show()
    except:
        pass

def main():
    """
    Main function to run the multimodal sampling example.
    """
    print("Multimodal 10D Sampling Example")
    print("===============================")
    
    # Problem setup
    ndim = 10
    n_seed = 100  # Number of initial walkers/chains
    sigma = 0.01  # Initial covariance scale
    savepath = './multimodal_results/'
    
    # Create save directory
    os.makedirs(savepath, exist_ok=True)
    
    # Initialize covariance matrices for each walker
    init_cov_list = []
    for i in range(n_seed):
        init_cov_list.append(sigma**2 * np.eye(ndim))
    
    # Configure sampler for challenging multimodal problem
    config = SamplerConfig(
        proc_merge_prob=0.9,           # High probability for merging similar chains
        alpha=1000,                    # Use recent samples for weighting
        trail_size=int(1e3),          # Maximum trials per iteration
        boundary_limiting=True,        # Enable boundary constraints
        use_beta=True,                # Use beta correction for boundaries
        integral_num=int(1e5),        # MC samples for beta estimation
        gamma=100,                    # Covariance update frequency
        exclude_scale_z=np.inf,       # No exclusion based on weights
        use_pool=False,               # Set to True for multiprocessing
        n_pool=4                      # Number of processes (if use_pool=True)
    )
    
    print(f"Problem dimension: {ndim}")
    print(f"Number of walkers: {n_seed}")
    print(f"Initial covariance scale: {sigma}")
    print(f"Save path: {savepath}")
    print(f"Multiprocessing: {config.use_pool}")
    
    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = Sampler(
        ndim=ndim, 
        n_seed=n_seed,
        log_density_func=log_density,
        init_cov_list=init_cov_list,
        prior_transform=prior_transform,
        config=config
    )
    
    # Prepare initial samples using Latin Hypercube Sampling
    print("Preparing LHS samples...")
    sampler.prepare_lhs_samples(lhs_num=int(1e5), batch_size=100)
    
    # Run the sampling process
    print("Starting sampling process...")
    print("(This may take several minutes for 10,000 iterations)")
    
    try:
        sampler.run_sampling(
            num_iterations=10000, 
            savepath=savepath,
            print_iter=100  # Print progress every 100 iterations
        )
        
        print("\nSampling completed successfully!")
        
        # Analyze results
        analyze_results(sampler, savepath)
        
        # Create visualizations
        visualize_marginal_distributions(sampler, savepath)
        
    except KeyboardInterrupt:
        print("\nSampling interrupted by user.")
        print("Partial results have been saved.")
        if hasattr(sampler, 'searched_points_list'):
            analyze_results(sampler, savepath)
            try:
                visualize_marginal_distributions(sampler, savepath)
            except:
                print("Could not create visualizations with partial results.")
    
    except Exception as e:
        print(f"\nError during sampling: {e}")
        raise
    
    print(f"\nResults saved to: {savepath}")
    print("Example completed!")

if __name__ == "__main__":

    main()


