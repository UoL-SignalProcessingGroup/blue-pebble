"""Sphinx extension: ``gallery-script`` directive.

Renders a Sphinx-Gallery-format Python script as interleaved narrative and code
blocks, without executing the script.  Used for examples that require external
data not bundled with the repository.

Each ``# %%`` cell boundary is honoured:

- Leading ``# `` comment lines in a cell become narrative RST (headings,
  paragraphs, directives, etc.).
- Remaining lines become a ``.. code-block:: python`` block.

The module docstring (before the first ``# %%``) is skipped; any introductory
text should live in the containing RST page.
"""

from __future__ import annotations

import re
from pathlib import Path

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

logger = logging.getLogger(__name__)


def _parse_cells(source: str) -> list[tuple[str, str]]:
    """Return a list of ``(narrative, code)`` pairs, one per ``# %%`` cell."""
    cell_re = re.compile(r"^# %%[^\n]*\n", re.MULTILINE)
    matches = list(cell_re.finditer(source))

    cells: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        cell_text = source[start:end]

        narrative_lines: list[str] = []
        code_lines: list[str] = []
        in_narrative = True

        for line in cell_text.split("\n"):
            if in_narrative:
                if line.startswith("# "):
                    narrative_lines.append(line[2:])
                elif line == "#":
                    narrative_lines.append("")
                elif not line.strip() and not code_lines:
                    # Blank line before any code — part of narrative spacing.
                    narrative_lines.append("")
                else:
                    in_narrative = False
                    if line.strip():
                        code_lines.append(line)
            else:
                code_lines.append(line)

        # Strip trailing blank lines.
        while narrative_lines and not narrative_lines[-1].strip():
            narrative_lines.pop()
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()

        narrative = "\n".join(narrative_lines)
        code = "\n".join(code_lines)
        if narrative or code:
            cells.append((narrative, code))

    return cells


class GalleryScriptDirective(SphinxDirective):
    """Render a Sphinx-Gallery Python script as narrative + code cells."""

    required_arguments = 1
    optional_arguments = 0
    has_content = False

    def run(self) -> list[nodes.Node]:
        rst_path = Path(self.env.doc2path(self.env.docname))
        script_path = (rst_path.parent / self.arguments[0]).resolve()

        if not script_path.exists():
            logger.warning(
                "gallery-script: file not found: %s",
                script_path,
                location=(self.env.docname, self.lineno),
            )
            return []

        source = script_path.read_text(encoding="utf-8")
        cells = _parse_cells(source)

        # Use the RST file's path as the StringList source so that relative
        # file paths inside directives (e.g. ``.. raw:: html :file:``) are
        # resolved relative to the RST document, not the Python script.
        rst_source = str(rst_path)

        result: list[nodes.Node] = []
        for narrative, code in cells:
            if narrative:
                container = nodes.container()
                container.document = self.state.document
                vl = StringList(narrative.splitlines(), source=rst_source)
                nested_parse_with_titles(self.state, vl, container)
                result.extend(container.children)

            if code:
                block = nodes.literal_block(code, code)
                block["language"] = "python"
                result.append(block)

        return result


def setup(app: object) -> dict[str, object]:
    app.add_directive("gallery-script", GalleryScriptDirective)  # type: ignore[attr-defined]
    return {"version": "0.1", "parallel_read_safe": True}
