"""
Resume a saved PARIS sampler and continue sampling for more iterations.

This complements examples/load_sampler_example.py which focuses on
loading and analysis only. Here we load, then continue running.

Usage:
  python examples/resume_sampler_example.py \
    --path ./multimodal_results/sampler_state.pkl \
    --more-iters 2000 \
    --out ./multimodal_results \
    --print-iter 100

Notes on pickle/imports:
- If your saved state was produced by running a script directly
  (e.g., `python examples/multimodal_example.py`), the stored
  log_density/prior_transform may be referenced from `__main__`.
  To make unpickling robust, this script defines wrapper functions
  named `log_density` and `prior_transform` at module scope.
  They delegate to `examples.multimodal_example` by default so
  the attribute lookup during unpickle succeeds.
- If your state came from a different script, adjust the wrappers
  to point to your original functions or import that module here.
"""

import os
import argparse
from parismc import Sampler


# Wrappers to help unpickling when the original functions were in __main__
# and created by examples/multimodal_example.py. Adjust if needed.
def log_density(x):
    from examples.multimodal_example import log_density as _ld
    return _ld(x)


def prior_transform(u):
    from examples.multimodal_example import prior_transform as _pt
    return _pt(u)


def summarize(sampler: Sampler) -> None:
    print("Sampler State")
    print("-------------")
    print(f"ndim: {sampler.ndim}")
    print(f"n_proc: {sampler.n_seed}")
    print(f"current_iter: {getattr(sampler, 'current_iter', None)}")
    print(f"savepath: {getattr(sampler, 'savepath', '(unset)')}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a saved PARIS sampler and continue sampling.")
    parser.add_argument(
        "--path",
        type=str,
        default="./multimodal_results/sampler_state.pkl",
        help="Path to sampler_state.pkl (defaults to multimodal example output)",
    )
    parser.add_argument(
        "--more-iters",
        type=int,
        default=1000,
        help="Number of additional iterations to run after loading",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory to continue saving results (default: same folder as --path)",
    )
    parser.add_argument(
        "--print-iter",
        type=int,
        default=100,
        help="Progress bar update frequency (iterations)",
    )
    args = parser.parse_args()

    state_path = args.path
    if not os.path.isfile(state_path):
        print(f"Sampler state not found at: {state_path}")
        print("Please run the multimodal example first:")
        print("  python examples/multimodal_example.py")
        return

    out_dir = args.out or os.path.dirname(os.path.abspath(state_path))
    os.makedirs(out_dir, exist_ok=True)

    # Load saved sampler
    # The presence of top-level `log_density`/`prior_transform` wrappers above
    # helps unpickling when the original module was `__main__`.
    sampler = Sampler.load_state(state_path)

    # Optionally, rebind the functions to the wrappers for clarity
    # (unpickling may already have set them to these wrappers).
    try:
        sampler.log_density_func_original = log_density
        if hasattr(sampler, "prior_transform") and sampler.prior_transform is not None:
            sampler.prior_transform = prior_transform
        # If prior_transform was set, ensure transformed log-density hook is in place
        if getattr(sampler, "prior_transform", None) is not None:
            sampler.log_density_func = sampler.transformed_log_density_func
        else:
            sampler.log_density_func = sampler.log_density_func_original
    except Exception:
        # Keep going even if attributes differ in older states
        pass

    print("Loaded state. Summary before resume:")
    summarize(sampler)

    # Continue sampling
    print("\nResuming sampling...")
    sampler.run_sampling(
        num_iterations=args.more_iters,
        savepath=out_dir,
        print_iter=args.print_iter,
        stop_dlogZ=0.1,
    )

    print("\nResume completed. Summary after resume:")
    summarize(sampler)

    # Optional: quick analysis similar to load_sampler_example
    try:
        import numpy as np
        samples, weights = sampler.get_samples_with_weights(flatten=True)
        ess = 1.0 / (weights ** 2).sum()
        wmean = (samples * weights[:, None]).sum(axis=0) / weights.sum()
        print("\nQuick analysis")
        print("--------------")
        print(f"Total samples: {len(samples)}")
        print(f"Effective sample size (ESS): {ess:.1f}")
        print(f"Weighted mean (first 5 dims): {wmean[:5]}")
    except Exception as e:
        print(f"Analysis skipped: {e}")


if __name__ == "__main__":
    main()
