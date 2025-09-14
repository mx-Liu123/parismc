Welcome to PARIS's documentation!
=================================

PARIS: Parallel Adaptive Reweighting Importance Sampling
--------------------------------------------------------
PARIS implements adaptive importance sampling with parallel seeds and
clustering for efficient multi-modal Bayesian inference.

Key Hyperparameters
-------------------

- merge_confidence (p): Controls the merge radius between seeds via a Mahalanobis threshold. p=0.9 is generally fine for most problems.
- alpha: Number of most recent samples used for importance weighting. Using alpha=10000 is a safe, conservative choice (older docs referred to this as latest_prob_index).

Tuning Tips
-----------

- n_lhs (``lhs_num`` in API): Number of LHS points covering the prior for a global search of good start points. Estimate from the relative size of a typical mode to the prior region. If a mode occupies fraction f of the prior volume, pick ``lhs_num ≳ c/f`` (``c≈50–200``) to get several hits per mode; 10³–10⁵ is common depending on dimension.
- n_seed: Depends on a conservative estimate of total mode count. Recommended ``n_seed = 10 ×`` expected modes to avoid missing weaker modes.
- init_cov_list: Initial covariance for each process. Use a conservative small estimate of mode size, or the inverse Fisher matrix when available. On a unit cube, ``diag((0.05–0.1)^2)`` per dimension is a reasonable start.
- Less sensitive: ``alpha`` and ``merge_confidence`` are typically robust. Defaults often suffice; try ``alpha=10000`` and ``p=0.9`` for safe, general settings.

Getting Started
---------------

Install:

.. code-block:: bash

   pip install parismc

Import:

.. code-block:: python

   from parismc import Sampler  # adjust to your API


.. toctree::
   :caption: Documentation:
   :maxdepth: 2
   :glob:

   user/*


.. toctree::
   :caption: Tutorial:
   :maxdepth: 2
   :glob:

   examples/*


.. toctree::
   :caption: General Information:
   :maxdepth: 1
   :glob:

   general/*


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
