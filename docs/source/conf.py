"""Configuration file for the Sphinx documentation builder."""

import os
import sys

project = "Nereus"
copyright = "2025, Joshua J. Wakefield, Finley Boulton"
author = "Joshua J. Wakefield, Finley Boulton"
release = "2.2.1"

sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/__init__.py"]

html_theme = "sphinx_rtd_theme"
suppress_warnings = ["myst.header", "toc.not_readable"]
