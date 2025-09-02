from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read long description from README
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "An advanced Monte Carlo sampler with adaptive covariance and clustering capabilities."

setup(
    name="parismc",
    version="0.1.0",
    author="Miaoxin Liu, Alvin J. K. Chua",
    author_email="miaoxin.liu@u.nus.edu",
    description="PARIS: Parallel Adaptive Reweighting Importance Sampling for high-dimensional multi-modal Bayesian inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mx-Liu123/parismc",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="monte carlo, bayesian inference, importance sampling, multimodal, adaptive sampling, MCMC",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
            'isort>=5.0',
        ],
        'plotting': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'ipython>=7.0.0',
        ],
        'full': [
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
            'jupyter>=1.0.0',
            'ipython>=7.0.0',
        ],
    },    
    project_urls={
        "Bug Reports": "https://github.com/mx-Liu123/parismc/issues",
        "Source": "https://github.com/mx-Liu123/parismc",
        "Documentation": "https://github.com/mx-Liu123/parismc/blob/main/README.md",
    },

)
