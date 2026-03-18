"""Custom sphinx-gallery scraper for Plotly figures."""

from __future__ import annotations


def _write_placeholder_png(path: str) -> None:
    """Write a matplotlib placeholder PNG when kaleido is unavailable."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(
        0.5,
        0.5,
        "Interactive Plotly figure\n(kaleido required for static export)",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=12,
        color="grey",
    )
    ax.axis("off")
    fig.savefig(path, dpi=72)
    plt.close(fig)


class PlotlyScraper:
    """Capture Plotly figures from gallery script globals.

    For each new Plotly figure found in ``example_globals``, the scraper:

    - Writes a static PNG thumbnail via kaleido (falling back to a matplotlib
      placeholder if kaleido is unavailable).
    - Writes an HTML fragment via ``fig.to_html()`` alongside the PNG.
    - Emits RST that embeds the HTML for full interactivity in HTML builds,
      with a static PNG fallback for other output formats (e.g. PDF).

    Previously-seen figures are skipped to avoid counting the same object
    twice across blocks. The tracking set is reset between scripts via
    ``reset_modules``.
    """

    def __init__(self) -> None:
        """Initialise the set of already-captured figure object IDs."""
        self._seen_ids: set[int] = set()

    def reset(self, _gallery_conf: dict, _fname: str) -> None:
        """Clear captured-figure tracking between gallery scripts."""
        self._seen_ids = set()

    def __repr__(self) -> str:
        """Return string representation."""
        return "PlotlyScraper"

    def __call__(self, _block: object, block_vars: dict, gallery_conf: dict) -> str:
        """Scrape Plotly figures from the current block's globals."""
        import os

        import plotly.io as pio
        from sphinx_gallery.scrapers import figure_rst  # type: ignore[import-untyped]

        seen_this_block: set[int] = set()
        new_figs = []
        for val in block_vars.get("example_globals", {}).values():
            if (
                hasattr(val, "to_html")
                and id(val) not in self._seen_ids
                and id(val) not in seen_this_block
            ):
                new_figs.append(val)
                seen_this_block.add(id(val))
        if not new_figs:
            return ""

        image_path_iterator = block_vars["image_path_iterator"]
        rst_parts: list[str] = []

        for fig in new_figs:
            png_path: str = next(image_path_iterator)
            html_path = os.path.splitext(png_path)[0] + ".html"

            # Write PNG thumbnail.
            try:
                pio.write_image(fig, png_path)
            except Exception:
                _write_placeholder_png(png_path)

            # Write HTML fragment for interactive embedding.
            # Wrap in a scrollable container so figures wider than the page
            # content column scroll horizontally rather than overflowing.
            html_fragment = fig.to_html(  # type: ignore[union-attr]
                include_plotlyjs="cdn", full_html=False
            )
            html_content = f'<div style="overflow-x: auto;">{html_fragment}</div>'
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            self._seen_ids.add(id(fig))

            html_basename = os.path.basename(html_path)
            interactive_rst = f"\n.. raw:: html\n   :file: images/{html_basename}\n\n"

            static_rst = figure_rst([png_path], gallery_conf["src_dir"])
            indented = "\n".join(
                ("   " + line) if line.strip() else "" for line in static_rst.strip().splitlines()
            )
            only_block = f"\n.. only:: not html\n\n{indented}\n\n"

            rst_parts.append(interactive_rst + only_block)

        return "".join(rst_parts)
