User Guide
==========

Configuration & Tuning
----------------------

Primary Tuning Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

These are the main knobs you might need to adjust for your specific problem:

- **n_lhs ($N_{\text{LHS}}$)**: Total number of LHS points covering the prior. This is the "global search" phase. If your modes are very small relative to the prior volume, increase this.
- **n_seed ($N_{\text{seed}}$)**: Number of parallel seeds to start. A good rule of thumb is $10 \times$ the number of modes you expect to find.
- **alpha ($\alpha$)**: The sliding window size. It determines how many of the most recent samples are used to build the local proposal mixture. $\alpha=1000$ is usually sufficient.

Stability & Advanced Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following parameters usually work well with their defaults:

- **cov_jitter ($\epsilon$)**: A tiny diagonal offset (default $10^{-10}$) added to the covariance matrix to ensure it remains positive-definite and invertible in high dimensions.
- **gamma ($\gamma$)**: How often the local covariance is updated (in iterations). Default is 100.
- **merge_confidence (p)**: *(Distance-merging only)* Used to calculate the Mahalanobis threshold. Default $p=0.9$.
- **trail_size**: Maximum number of attempts to find a valid point within prior boundaries during rejection sampling.
- **keep_dead_processes**: If set to ``True``, the sampler will gracefully archive the trimmed sample histories of merged (dead) processes instead of discarding them. This is extremely memory-efficient and allows advanced users to analyze the entire exploration trajectory by calling ``sampler.get_samples_with_weights(include_dead=True)``. Defaults to ``False``.
- **Automatic Anomaly Detection**: The sampler internally tracks ``bad_logden_count``. If your ``log_density`` function returns ``NaN`` or ``Inf``, the sampler will log a warning at the first occurrence and every 1000 occurrences thereafter, helping you diagnose numerical issues without immediate collapse.

Tuning Tips
~~~~~~~~~~~

- **n_lhs ($N_{\text{LHS}}$)**: Number of LHS points covering the prior for a global search of good start points. Estimate from the relative size of a typical mode to the prior region. If a mode occupies fraction $f$ of the prior volume, pick $N_{\text{LHS}} \gtrsim 50/f$ to get several hits per mode; $10^3$--$10^5$ is common depending on dimension.
- **n_seed ($N_{\text{seed}}$)**: Depends on a conservative estimate of total mode count. Recommended $N_{\text{seed}} = 10 \times$ expected modes to avoid missing weaker modes.
- **init_cov_list ($\Sigma_{\text{init}}$)**: Initial covariance for each process. Use a conservative small estimate of mode size, or the inverse Fisher matrix when available. On a unit cube, $\text{diag}((0.05\text{--}0.1)^2)$ per dimension is a reasonable start.
- **Less sensitive**: $\alpha$ and $\gamma$ are typically robust. Defaults often suffice; try $\alpha=10000$ for a safe, general setting.

Using Prior Transforms
----------------------

PARIS is designed to sample from the unit hypercube $[0, 1]^d$. For physical problems with specific bounds or priors (e.g., Uniform $[-10, 10]$, Gaussian priors), you must provide a ``prior_transform`` function.

Workflow
~~~~~~~~

1.  **Sampler Proposal**: Generates a point $u \in [0, 1]^d$.
2.  **Transform**: Calls $x = \text{prior\_transform}(u)$.
3.  **Likelihood**: Calls $\ln L = \text{log\_density}(x)$.

**Important**: Your ``log_density`` function must expect **physical parameters** $x$, not the unit cube parameters $u$.

.. code-block:: python

   def prior_transform(u):
       # Map [0, 1] to [-5, 5]
       return u * 10 - 5

   def log_density(x):
       # Calculate density using physical x in [-5, 5]
       return -0.5 * np.sum(x**2)

Advanced Usage
--------------

Progress Bar Output
~~~~~~~~~~~~~~~~~~~

During ``run_sampling``, the terminal progress bar provides real-time updates:

- **samples**: The number of valid samples currently held in the active sliding windows.
- **evals**: The total cumulative number of likelihood function evaluations performed, including those from rejected trials and previously merged processes.
- **n_proc**: The current number of active parallel processes. This naturally decreases as redundant modes are merged.
- **logZ**: The current estimate of the log-evidence ($\ln \mathcal{Z}$). Returns ``NULL`` if not yet calculated.
- **dlogZ**: The absolute difference in the log-evidence estimate compared to its value 1000 iterations ago. This is used to trigger early stopping if ``stop_dlogZ`` is provided. Returns ``NULL`` during the first 1000 iterations.
- **max_ld**: The highest log-density (log-likelihood) value discovered so far across all active processes.

Runtime Flags
~~~~~~~~~~~~~

The sampler watches a JSON flag file (``sampler_flags.json``) in the working directory during ``run_sampling``. Set a flag to ``true`` while it runs; the sampler performs the action once and resets the flag to ``false``.

- ``output_latest_samples``: write ``latest_samples.npy`` and ``latest_weights.npy``.
- ``plot_latest_samples``: write ``latest_corner.png`` (requires ``corner``).
- ``print_latest_infos``: write ``latest_infos.txt`` with per-process diagnostics.

.. code-block:: python

   import json
   # Example: Toggle flag during run
   with open("sampler_flags.json", "r") as f:
       flags = json.load(f)
   flags["output_latest_samples"] = True
   with open("sampler_flags.json", "w") as f:
       json.dump(flags, f)

Custom Initialization
---------------------

By default, PARIS uses Latin Hypercube Sampling (LHS) internally via ``prepare_lhs_samples()``. However, you can provide your own starting points (e.g., from a Sobol sequence, or pre-computed physical locations) using the **External LHS** interface.

To do this, skip the ``prepare_lhs_samples()`` call and pass your points and their corresponding log-densities directly to ``run_sampling()``:

.. code-block:: python

   # 1. Generate external points in [0, 1] unit cube
   ext_points = my_custom_qmc_generator(n_samples)
   
   # 2. Calculate their log-densities (use physical parameters if transform is set)
   ext_log_densities = np.array([log_density(prior_transform(p)) for p in ext_points])
   
   # 3. Pass to run_sampling
   sampler.run_sampling(
       num_iterations=1000,
       savepath='./results',
       external_lhs_points=ext_points,
       external_lhs_log_densities=ext_log_densities
   )

This interface is useful when you want to ensure the sampler starts from specific regions of interest or when using specialized space-filling designs.

