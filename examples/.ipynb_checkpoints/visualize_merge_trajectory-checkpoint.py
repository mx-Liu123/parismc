import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.special import logsumexp
from parismc import Sampler, SamplerConfig
import copy

# ==============================================================================
# 1. Problem Definition (Same as merge_exp.py)
# ==============================================================================

MODES = np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
])
WEIGHTS = np.array([1.0] * len(MODES))
LOG_WEIGHTS = np.log(WEIGHTS)

def log_density(final_states_raw):
    """
    10D unimodal log-density (masked as multimodal structure).
    Peak at 0.5, 0.5, ...
    Transformation: x_in = 1.4 * x_raw - 0.2
    """
    final_states_in = (final_states_raw * 1.4) - 0.2
    
    log_likelihoods = []
    for mode in MODES:
        distance = np.linalg.norm(final_states_in - mode, axis=1)
        distance = np.nan_to_num(distance, nan=1e6, posinf=1e6, neginf=1e6)
        log_prob = -400 * distance**2
        log_likelihoods.append(log_prob)
    
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = LOG_WEIGHTS[:, None] + log_likelihoods
    total_log_likelihood = logsumexp(log_likelihoods, axis=0)
    
    if total_log_likelihood.shape[0] == 1:
        return total_log_likelihood[0]
    return total_log_likelihood

def prior_transform(u):
    return u

# ==============================================================================
# 2. Experiment Logic
# ==============================================================================

def run_vis_experiment(merge_type, run_id, total_iters=10000):
    """
    Runs the sampler, collecting samples and detecting merge events.
    Strategy:
    - Run chunks of 100 iters until iter 500 (burn-in).
    - Then run 1 iter at a time to strictly monitor merges.
    - Check n_proc changes to detect merges.
    """
    ndim = 10
    n_seed = 2
    sigma = 0.01
    savepath = f'./vis_results/run_{run_id}_{merge_type}/'
    os.makedirs(savepath, exist_ok=True)
    
    # Init Covariance
    init_cov_list = []
    for i in range(n_seed):
        init_cov_list.append(sigma**2 * np.eye(ndim))
    
    # Config: stop_on_merge=False to continue after merge
    config = SamplerConfig(
        merge_confidence=0.9,
        alpha=1000,
        trail_size=int(1e3),
        boundary_limiting=True,
        use_beta=False, # Set False for stability
        integral_num=int(1e5),
        gamma=100,
        exclude_scale_z=np.inf,
        use_pool=False,
        n_pool=4,
        merge_type=merge_type,
        stop_on_merge=False  # Continue sampling after merge
    )
    
    sampler = Sampler(
        ndim=ndim, 
        n_seed=n_seed,
        log_density_func=log_density,
        init_cov_list=init_cov_list,
        prior_transform=prior_transform,
        config=config
    )
    
    # Initial points: symmetric 0.4 and 0.6
    external_lhs_points = np.zeros((n_seed, ndim)) + 0.4
    external_lhs_points[1, :] = 0.6
    external_lhs_log_densities = log_density(external_lhs_points)
    
    all_samples = [] # List of (iter, x, y, proc_id)
    merge_events = [] # List of (iter, x, y)
    
    # Initialize run internally by calling with 0 iters first? No, run_sampling handles it.
    # We will just start the loop.
    
    current_iter = 0
    
    # Setup loop phases
    # Phase 1: Burn-in (0 to 500)
    burn_in_limit = 500
    burn_in_step = 100
    
    # Track previous state for merge detection
    # We need to know previous process count and their locations to ID who died
    prev_n_proc = n_seed
    # We can track the means or last points to know where they were
    prev_means = [external_lhs_points[i] for i in range(n_seed)]

    # Helper to collect samples
    def collect_samples(sampler_obj, start_iter, end_iter, last_counts_dict):
        """
        Collect new samples from sampler.
        last_counts_dict: maps proc_index (0 to n_proc-1) to previous count
        Returns: new samples list, updated counts dict
        """
        new_data = []
        current_counts = {}
        
        # Iterate over currently active processes
        # Note: sampler.n_proc changes! Indices 0..n_proc-1 are valid.
        for p_idx in range(sampler_obj.n_proc):
            total_count = sampler_obj.element_num_list[p_idx]
            prev_cnt = last_counts_dict.get(p_idx, 0)
            
            # If we had a merge, indices might shift or content might change.
            # But usually parismc keeps the "surviving" processes at 0..n_new-1
            # We just grab whatever is new at this index.
            
            if total_count > prev_cnt:
                # Grab new points
                # searched_points_list[p_idx] is (N, ndim)
                pts = sampler_obj.searched_points_list[p_idx][prev_cnt:total_count]
                
                # Assign iteration roughly
                # Distribute them evenly over the step? Or just mark as end_iter?
                # Mark as end_iter for simplicity
                for pt in pts:
                    new_data.append((end_iter, pt[0], pt[1], p_idx))
            
            current_counts[p_idx] = total_count
            
        return new_data, current_counts

    last_counts = {i: 0 for i in range(n_seed)}
    
    # Master Loop
    while current_iter < total_iters:
        # Determine step size
        if current_iter < burn_in_limit:
            step = burn_in_step
        else:
            step = 1
            
        # Cap at total_iters
        if current_iter + step > total_iters:
            step = total_iters - current_iter
        
        # Snapshot before run
        before_n_proc = sampler.n_proc if sampler.n_proc is not None else n_seed
        # Snapshot locations (means of active processes) to detect who dies
        # sampler.now_means is available after first run.
        # If not available, use last known or external_lhs initially.
        if hasattr(sampler, 'now_means') and len(sampler.now_means) > 0:
             before_means = [m.copy() for m in sampler.now_means]
        else:
             before_means = copy.deepcopy(prev_means)

        # Run Sampler
        try:
            # For first call, pass external points
            kwargs = {
                'num_iterations': step,
                'print_iter': total_iters + 100, # Silence it
                'savepath': savepath
            }
            if current_iter == 0:
                kwargs['external_lhs_points'] = external_lhs_points
                kwargs['external_lhs_log_densities'] = external_lhs_log_densities
                
            sampler.run_sampling(**kwargs)
            
        except Exception as e:
            print(f"Sampling stopped at {current_iter} due to error: {e}")
            break
            
        current_iter += step
        
        # Collect Samples
        new_samples, last_counts = collect_samples(sampler, current_iter - step, current_iter, last_counts)
        all_samples.extend(new_samples)
        
        # Check for Merge
        # Logic: If n_proc decreased, a merge happened.
        # The sampler sorts processes by importance (max_logden) and keeps top ones.
        # The ones at the END of the list usually die or get merged into others.
        # But we need to know WHERE the dead one was to plot the star.
        # We look at `before_means`.
        # If we went from N to N-1, the process that disappeared is likely the one
        # that was at index N-1 (if sorted) or we need to find which mean is "missing" in new means.
        # Actually, parismc re-sorts.
        # Robust way: Find which `before_mean` is furthest from any `current_mean`.
        # That one likely died.
        
        after_n_proc = sampler.n_proc
        if after_n_proc < before_n_proc:
            # Merge Detected!
            num_merged = before_n_proc - after_n_proc
            
            # Identify which means disappeared
            # Current means
            current_means = sampler.now_means
            
            # For each old mean, find min distance to any new mean
            # If distance is small, it survived (or moved slightly).
            # If distance is large, it likely disappeared/merged.
            # But if it merged, it might have merged INTO a survivor.
            # Let's simplify:
            # The "Dead" process location is what we want.
            # In `parismc`, processes are sorted.
            # Usually the 'merger' (survivor) and 'mergee' (dead) are close.
            # We can just plot the star at the location of the process that was removed.
            # If we assume the list was sorted by log-density, the lower ones die.
            # But sorting happens inside.
            
            # Heuristic: The `before_means` has `before_n_proc` entries.
            # `current_means` has `after_n_proc`.
            # We want to find the `num_merged` entries in `before_means` that are NOT represented in `current_means`.
            
            # Simple matching
            matched_indices = []
            for cm in current_means:
                dists = [np.linalg.norm(cm - bm) for bm in before_means]
                # Match to closest previous mean
                best_idx = np.argmin(dists)
                matched_indices.append(best_idx)
            
            # Unmatched indices in `before_means` are the dead ones
            for idx in range(before_n_proc):
                if idx not in matched_indices:
                    dead_loc = before_means[idx]
                    merge_events.append((current_iter, dead_loc[0], dead_loc[1]))
            
            # Also handle the case where multiple current map to same previous (split? unlikely here)
            # or strictly 1-to-1.
            
            # Fallback: if we can't match perfectly, just take the last ones?
            # No, let's stick to the unmatched heuristic.
            
            print(f"Merge detected at iter {current_iter}. Active: {before_n_proc} -> {after_n_proc}")

        # Update tracking
        prev_n_proc = after_n_proc
        if hasattr(sampler, 'now_means'):
            prev_means = [m.copy() for m in sampler.now_means]

        if after_n_proc == 0:
            break

    return np.array(all_samples), merge_events

# ==============================================================================
# 3. Main Visualization
# ==============================================================================

def main():
    merge_types = ['distance', 'single', 'multiple']
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 18))
    
    # Target Mode Parameters for Plotting
    target_mode = [0.5, 0.5]
    target_sigma = 0.02525
    
    for i, m_type in enumerate(merge_types):
        print(f"Running visualization for merge_type: {m_type}")
        samples, merges = run_vis_experiment(m_type, i, total_iters=10000)
        
        ax = axes[i]
        
        # 1. Scatter Plot with Colorbar
        if len(samples) > 0:
            # samples columns: [iter, x, y, proc_id]
            # Plot x, y, color by iter
            sc = ax.scatter(samples[:, 1], samples[:, 2], c=samples[:, 0], 
                            cmap='viridis', s=10, alpha=0.6, edgecolors='none')
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Iteration')
        
        # 2. Target 1-sigma Circle
        circle = Circle(target_mode, target_sigma, color='red', fill=False, 
                        linewidth=2, linestyle='--', label='Target 1-$\sigma$')
        ax.add_patch(circle)
        
        # 3. Merge Events (Stars)
        for me in merges:
            ax.scatter(me[1], me[2], marker='*', s=300, color='crimson', 
                       edgecolors='black', zorder=10, label='Merge Event')
            ax.text(me[1]+0.02, me[2]+0.02, f"Iter {int(me[0])}", fontsize=10, color='crimson', weight='bold')
        
        # Formatting
        ax.set_title(f"Merge Strategy: {m_type}")
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.2, 0.8)
        ax.set_xlabel("Dim 0")
        ax.set_ylabel("Dim 1")
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        print(f"Finished {m_type}. Total samples: {len(samples)}. Merges: {len(merges)}")

    plt.tight_layout()
    plt.savefig('visualize_merge_trajectory.png', dpi=300)
    print("Plot saved to visualize_merge_trajectory.png")

if __name__ == "__main__":
    main()