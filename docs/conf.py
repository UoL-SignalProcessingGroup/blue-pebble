"""Sphinx configuration for the Blue Pebble documentation."""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

try:
    from nbformat.warnings import MissingIDFieldWarning
except ImportError:  # pragma: no cover - older nbformat versions
    MissingIDFieldWarning = None

try:
    from sphinx.deprecation import RemovedInSphinx10Warning  # type: ignore[attr-defined]  # noqa: E402
except ImportError:  # pragma: no cover - older Sphinx versions
    RemovedInSphinx10Warning = Warning

from bluepebble import __version__  # noqa: E402

project = "Blue Pebble"
author = "Joshua J Wakefield and Finley Boulton"
copyright = "2026, Joshua J Wakefield and Finley Boulton"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nb_execution_mode = "off"
suppress_warnings = [
    "mystnb.unknown_mime_type",
]

warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)
if MissingIDFieldWarning is not None:
    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
