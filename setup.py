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
    author="Your Name",
    author_email="your.email@example.com",
    description="An advanced Monte Carlo sampler with adaptive covariance and clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/parismc",  # Update with your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/parismc/issues",
        "Source": "https://github.com/yourusername/parismc",
    },
)