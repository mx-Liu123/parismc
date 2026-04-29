import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parismc.sampler import Sampler, SamplerConfig

def log_density_func(x):
    """Simple 2D Gaussian log-density."""
    # Mode 1 at [0.5, 0.5]
    log_p1 = -0.5 * np.sum((x - 0.5)**2, axis=1) / 0.01
    # Mode 2 at [0.1, 0.1] - This one we might want to terminate if it gets stuck
    log_p2 = -0.5 * np.sum((x - 0.1)**2, axis=1) / 0.005
    return np.logaddexp(log_p1, log_p2)

def custom_terminate_condition(sampler, proc_idx):
    """
    Custom protocol:
    Terminate if max_logden stable for > 20 iters AND top 10 thetas' 
    first dimension is < 0.2.
    """
    # 1. Check stability count (built-in tracker)
    stable_count = sampler._proc_max_ld_stable_count[proc_idx]
    
    if stable_count > 20:
        # 2. Get Top 10 samples for this process
        element_num = sampler.element_num_list[proc_idx]
        log_dens = sampler.searched_log_densities_list[proc_idx][:element_num]
        points = sampler.searched_points_list[proc_idx][:element_num]
        
        # Sort by log-density descending
        top_indices = np.argsort(-log_dens)[:10]
        top_thetas = points[top_indices]
        
        # 3. Condition: first dimension of all top 10 thetas < 0.2
        if np.all(top_thetas[:, 0] < 0.2):
            print(f"\n[Terminator] Process {proc_idx} matches condition! "
                  f"Stable for {stable_count} iters. Top theta[0] max: {np.max(top_thetas[:, 0]):.4f}")
            return True
            
    return False

def run_example():
    ndim = 2
    n_seed = 10
    
    # Initialize with identity covariances (much smaller to slow down merging)
    init_cov_list = [np.eye(ndim) * 0.0001 for _ in range(n_seed)]
    
    # Setup config with custom termination protocol
    config = SamplerConfig(
        n_pool=4,
        use_pool=False, # Set to True for parallel
        keep_dead_processes=True, # Archive samples of terminated processes
        terminate_proc_condition=custom_terminate_condition,
        merge_type='distance',
        merge_confidence=0.1, # Small confidence means smaller merge radius
        gamma=50,
        debug=True
    )
    
    sampler = Sampler(
        ndim=ndim,
        n_seed=n_seed,
        log_density_func=log_density_func,
        init_cov_list=init_cov_list,
        config=config
    )
    
    # Prepare initial samples
    sampler.prepare_lhs_samples(lhs_num=1000, batch_size=200)
    
    print("\nStarting sampling with custom termination condition...")
    print("Processes targeting the [0.1, 0.1] mode should be terminated once they stabilize.")
    
    # Run sampling
    sampler.run_sampling(
        num_iterations=500,
        savepath='./custom_terminate_results',
        print_iter=10
    )
    
    print("\nSampling complete.")
    print(f"Final number of active processes: {sampler.n_proc}")
    print(f"Number of archived (terminated) processes: {len(sampler.archived_points)}")

if __name__ == "__main__":
    run_example()
