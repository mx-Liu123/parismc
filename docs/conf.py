import os
import sys

# -- Path setup --------------------------------------------------------------

# Add the project root so autodoc can find the package when needed
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "PARIS"
author = "PARIS contributors"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "PARIS documentation"

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# Use MathJax v3 like the Eryn docs
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# nbsphinx options (safe defaults)
nbsphinx_allow_errors = False
