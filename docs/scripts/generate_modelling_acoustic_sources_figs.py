"""Regenerate pre-generated figures for the modelling_acoustic_sources example.

Run from the repository root:

    python docs/scripts/generate_modelling_acoustic_sources_figs.py

The WAV file (``SanctSound_CI05_03_largeship_20190925T135956Z.wav``) must be
present in ``docs/examples/measured_data/`` before running this script.

Figures are written to ``docs/source/_static/acoustic_source_figs/``.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_here = Path(__file__).resolve().parent
_example = _here.parent / "examples" / "modelling_acoustic_sources.py"
_figs_dir = _here.parent / "source" / "_static" / "acoustic_source_figs"
_figs_dir.mkdir(parents=True, exist_ok=True)

ns = runpy.run_path(str(_example))

_figures = [
    ("whale", ns["fig_whale"]),
    ("shrimp", ns["fig_shrimp"]),
    ("vessel_tonal", ns["fig_vessel_tonal"]),
    ("vessel_measured", ns["fig_vessel_measured"]),
    ("ambient", ns["fig_ambient"]),
    ("composite", ns["fig_composite"]),
]

for name, fig in _figures:
    stem = f"modelling_acoustic_sources_{name}"
    fig.write_image(_figs_dir / f"{stem}.png", scale=2)
    html_fragment = fig.to_html(include_plotlyjs="cdn", full_html=False)
    (_figs_dir / f"{stem}.html").write_text(
        f'<div style="overflow-x: auto;">{html_fragment}</div>', encoding="utf-8"
    )
    print(f"Saved {stem}")
