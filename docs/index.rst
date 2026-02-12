PARIS: Parallel Adaptive Reweighting Importance Sampling
========================================================

**PARIS** is an efficient adaptive importance sampler designed for high-dimensional, multi-modal Bayesian inference. It combines global exploration with local adaptation to tackle complex posteriors in astrophysics and beyond.

Installation
------------

.. code-block:: bash

   pip install parismc

Getting Started
---------------

.. code-block:: python

   import numpy as np
   from parismc import Sampler, SamplerConfig

   # 1. Define log-density
   def log_density(x):
       return -0.5 * np.sum(x**2, axis=1)

   # 2. Configure & Initialize
   config = SamplerConfig(alpha=1000)
   sampler = Sampler(ndim=2, n_seed=5, log_density_func=log_density, 
                     init_cov_list=[np.eye(2)*0.1]*5, config=config)

   # 3. Run
   sampler.prepare_lhs_samples(1000, 100)
   sampler.run_sampling(500, './results')

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   user/guide
   api

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: General

   general/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`