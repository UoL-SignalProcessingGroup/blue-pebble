"""Regenerate pre-generated figures for the using_measured_data example.

Run from the repository root:

    python docs/scripts/generate_using_measured_data_figs.py

The GEBCO and Copernicus NetCDF files must be present in
``docs/examples/measured_data/`` before running this script.  Data can be
obtained from:

- GEBCO Compilation Group, The GEBCO Grid (GEBCO_2024 Grid):
  https://www.gebco.net
- E.U. Copernicus Marine Service Information:
  https://doi.org/10.48670/moi-00016

Figures are written to ``docs/source/_static/measured_data_figs/``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_here = Path(__file__).resolve().parent
_example = _here.parent / "examples" / "using_measured_data.py"
_figs_dir = _here.parent / "source" / "_static" / "measured_data_figs"
_figs_dir.mkdir(parents=True, exist_ok=True)

ns = runpy.run_path(str(_example))

_figures = [
    ("world", ns["fig_world"]),
    ("results", ns["fig_results"]),
]

for name, fig in _figures:
    stem = f"using_measured_data_{name}"
    fig.write_image(_figs_dir / f"{stem}.png", scale=2)
    html_fragment = fig.to_html(include_plotlyjs="cdn", full_html=False)
    (_figs_dir / f"{stem}.html").write_text(
        f'<div style="overflow-x: auto;">{html_fragment}</div>', encoding="utf-8"
    )
    print(f"Saved {stem}")
