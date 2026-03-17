"""Custom sphinx-gallery scraper for Plotly figures."""

from __future__ import annotations


class PlotlyScraper:
    """Capture Plotly figures produced by fig.show() calls.

    Writes a static PNG thumbnail (via kaleido) for each new figure so that
    sphinx-gallery's image-path accounting is satisfied.  Previously-seen
    figures are skipped to avoid counting the same object twice across blocks.
    The scraper instance is reset at the start of each script via
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

    def __call__(self, _block, block_vars, gallery_conf):
        """Scrape Plotly figures from the current block's globals."""
        import matplotlib.pyplot as plt
        import plotly.io as pio
        from sphinx_gallery.scrapers import figure_rst  # type: ignore[import-untyped]

        new_figs = [
            fig
            for fig in block_vars.get("example_globals", {}).values()
            if hasattr(fig, "to_html") and id(fig) not in self._seen_ids
        ]
        if not new_figs:
            return ""

        image_path_iterator = block_vars["image_path_iterator"]
        image_paths = []
        for fig in new_figs:
            png_path = next(image_path_iterator)
            try:
                pio.write_image(fig, png_path)  # type: ignore[arg-type]
            except Exception:
                # Kaleido v1.x requires Chrome; fall back to a matplotlib placeholder
                # so that sphinx-gallery's file-existence check still passes.
                mpl_fig, ax = plt.subplots(figsize=(6, 4))
                ax.text(
                    0.5,
                    0.5,
                    "Interactive Plotly figure\n(install Chrome for static export)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="grey",
                )
                ax.axis("off")
                mpl_fig.savefig(png_path, dpi=72)
                plt.close(mpl_fig)
            self._seen_ids.add(id(fig))
            image_paths.append(png_path)

        return figure_rst(image_paths, gallery_conf["src_dir"])
