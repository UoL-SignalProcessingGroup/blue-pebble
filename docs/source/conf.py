"""Sphinx configuration for the Blue Pebble documentation."""
# ruff: noqa: I001

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(ROOT))
sys.path.insert(0, os.fspath(Path(__file__).parent))

try:
    from sphinx.deprecation import RemovedInSphinx10Warning  # type: ignore[attr-defined]  # noqa: E402
except ImportError:  # pragma: no cover - older Sphinx versions
    RemovedInSphinx10Warning = Warning

from _plotly_scraper import PlotlyScraper  # noqa: E402
from bluepebble import __version__  # noqa: E402

project = "Blue Pebble"
author = "Joshua J Wakefield and Finley Boulton"
copyright = "2026, Joshua J Wakefield and Finley Boulton"
release = __version__
root_doc = "source/index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]

intersphinx_mapping = {
    "stonesoup": ("https://stonesoup.readthedocs.io/en/stable/", None),
}


templates_path = ["_templates"]
html_extra_path = ["_extra"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
    # Exclude the raw gallery source dirs — Sphinx should only see the
    # sphinx-gallery-generated RST output under source/auto_examples/ etc.
    "examples",
    "tutorials",
]

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


_plotly_scraper = PlotlyScraper()

sphinx_gallery_conf = {
    "examples_dirs": ["examples", "tutorials"],
    "gallery_dirs": ["source/auto_examples", "source/auto_tutorials"],
    "filename_pattern": r"\.py",
    "ignore_pattern": r"using_measured_data\.py",
    "abort_on_example_error": False,
    "image_scrapers": ("matplotlib", _plotly_scraper),
    "reset_modules": (_plotly_scraper.reset,),
    "plot_gallery": True,
    # Disable repr capture — all Plotly figures are handled exclusively by
    # PlotlyScraper.  Without this, a go.Figure that is the last expression in
    # a cell is captured twice (once by PlotlyScraper, once by capture_repr).
    "capture_repr": (),
    # Resolve bluepebble class links in code blocks against the local build.
    # Without this entry, Sphinx-Gallery falls back to intersphinx and links
    # bluepebble classes to Stone Soup's Base class instead.
    "reference_url": {
        "bluepebble": None,
    },
}

warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)
