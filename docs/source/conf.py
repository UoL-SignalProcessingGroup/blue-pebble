"""Sphinx configuration for the Blue Pebble documentation."""
# ruff: noqa: I001

from __future__ import annotations

import os
import re
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
root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
    "_sgscript",
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
    # The raw gallery sources now live outside this source dir (docs/examples,
    # docs/tutorials), so Sphinx never walks them and no exclude is needed. Only the
    # leftover header in source/examples/ has to be hidden: that directory holds the
    # hand-written pages for the data-dependent scripts, which _patch_gallery_index
    # links into the gallery, so excluding the directory wholesale made those links
    # dangle.
    "examples/GALLERY_HEADER.rst",
    # Exclude RST stubs for WAV-dependent scripts that are in ignore_pattern.
    # These files may persist from earlier builds and would otherwise cause
    # toc.not_included warnings.
    "auto_examples/comparing_simulators.rst",
    "auto_examples/modelling_acoustic_sources.rst",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
# Headings napoleon doesn't know are not section boundaries to it: anything after a standard
# section (e.g. Attributes) runs on into it and renders as garbled fields. Registering them makes
# each one end the previous section and render as its own rubric.
napoleon_custom_sections = [
    "Threshold modes",
    "Noise calibration",
    "Sidelobe false alarms",
    "Peak consolidation",
    "Assumptions",
    "Choosing an equation",
    "Relation to sonar noise normalisation",
]

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
    # Relative to this conf.py (the Sphinx source dir). The gallery sources live in
    # docs/examples and docs/tutorials, one level up -- "examples" without the ../ resolved
    # to docs/source/examples, which holds only hand-written stubs and no scripts, so the
    # whole examples gallery silently built empty.
    "examples_dirs": ["../examples", "../tutorials"],
    "gallery_dirs": ["auto_examples", "auto_tutorials"],
    # Execute only the tutorials. Running every example as well takes about sixteen minutes,
    # beyond Read the Docs' fifteen-minute build limit, so the examples render with their code
    # and text but no output. Set this back to r"\.py" to execute them too.
    "filename_pattern": r"[\\/]tutorials[\\/]",
    # Exclude scripts that require external data not bundled with the repository.
    # These examples have hand-written RST pages under docs/source/examples/.
    "ignore_pattern": (
        r"using_measured_data\.py"
        r"|modelling_acoustic_sources\.py"
        r"|comparing_simulators\.py"
    ),
    "abort_on_example_error": False,
    "image_scrapers": ("matplotlib", _plotly_scraper),
    "reset_modules": (_plotly_scraper.reset,),
    "plot_gallery": True,
    # Disable repr capture — all Plotly figures are handled exclusively by
    # PlotlyScraper.  Without this, a go.Figure that is the last expression in
    # a cell is captured twice (once by PlotlyScraper, once by capture_repr).
    "capture_repr": (),
    # Strip "# sphinx_gallery_thumbnail_path = ..." and similar lines from the rendered code.
    "remove_config_comments": True,
    # Resolve bluepebble class links in code blocks against the local build.
    # Without this entry, Sphinx-Gallery falls back to intersphinx and links
    # bluepebble classes to Stone Soup's Base class instead.
    "reference_url": {
        "bluepebble": None,
    },
}

warnings.filterwarnings("ignore", category=RemovedInSphinx10Warning)

# ---------------------------------------------------------------------------
# Patch the SG-generated gallery index to include the measured-data example.
#
# Sphinx-Gallery regenerates source/auto_examples/index.rst on every build,
# so any manual edits are lost.  The setup() hook below runs in builder-inited
# AFTER SG's own handler (same priority 500, FIFO order) and appends a
# "Measured Data Examples" section to the end of the generated index.
# ---------------------------------------------------------------------------

_MEASURED_DATA_SECTION = """\
External Data Examples
----------------------

Examples that require external data files not bundled with the repository.
Pre-generated figures are embedded so the pages render without re-running
the scripts.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Previews whale call, snapping shrimp, commercial vessel tonals, measured vessel noise, and ambient white noise models, then combines them into a composite soundscape.">

.. only:: html

  .. image:: /_static/acoustic_source_figs/modelling_acoustic_sources_whale.png
    :alt:

  :doc:`/examples/modelling_acoustic_sources`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modelling Acoustic Sources</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Runs one scenario in the environment of a real area: GEBCO bathymetry and Copernicus temperature/salinity converted to sound speed via the NPL equation.  Demonstrates how to wire real geophysical datasets into the Blue Pebble pipeline.">

.. only:: html

  .. image:: /_static/measured_data_figs/using_measured_data_world.png
    :alt:

  :doc:`/examples/using_measured_data`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using GEBCO and Copernicus Environmental Data</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /examples/modelling_acoustic_sources
   /examples/using_measured_data


"""


def _patch_gallery_index(app: object) -> None:
    """Append the measured-data section to the SG-generated gallery index."""
    gallery_index = Path(app.srcdir) / "auto_examples" / "index.rst"  # type: ignore[attr-defined]
    if not gallery_index.exists():
        return
    content = gallery_index.read_text(encoding="utf-8")
    if _MEASURED_DATA_SECTION in content:
        return  # already patched (shouldn't happen, but be safe)
    content = content + "\n" + _MEASURED_DATA_SECTION
    gallery_index.write_text(content, encoding="utf-8")


_TIMING_ROW_RE = re.compile(
    r"   \* - :ref:`[^`]+`[^\n]*\n     - [^\n]*\n     - [^\n]*\n"
)
_IGNORED_IN_TIMING = re.compile(
    r"comparing_simulators|modelling_acoustic_sources|using_measured_data"
)


def _patch_timing_files(app: object) -> None:
    """Remove rows for ignored scripts from SG-generated execution-time tables.

    Sphinx-Gallery includes ignored scripts (with 0s timing) in the timing
    tables but generates no gallery page for them, leaving broken :ref: targets.
    This hook strips those rows after SG has written the files.
    """
    candidates = [
        Path(app.srcdir) / "auto_examples" / "sg_execution_times.rst",  # type: ignore[attr-defined]
        Path(app.srcdir) / "auto_tutorials" / "sg_execution_times.rst",  # type: ignore[attr-defined]
        Path(app.srcdir) / "sg_execution_times.rst",  # type: ignore[attr-defined]
    ]
    for timing_file in candidates:
        if not timing_file.exists():
            continue
        content = timing_file.read_text(encoding="utf-8")
        patched = _TIMING_ROW_RE.sub(
            lambda m: "" if _IGNORED_IN_TIMING.search(m.group(0)) else m.group(0),
            content,
        )
        if patched != content:
            timing_file.write_text(patched, encoding="utf-8")


def setup(app: object) -> None:
    """Register Blue Pebble's Sphinx extensions with the application."""
    app.connect("builder-inited", _patch_gallery_index)  # type: ignore[attr-defined]
    app.connect("builder-inited", _patch_timing_files)  # type: ignore[attr-defined]
