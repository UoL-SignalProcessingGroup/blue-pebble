"""Locating and running the Acoustics Toolbox KRAKEN, KRAKENC and FIELD programs.

The Acoustics Toolbox is an external program, not a Python dependency. Build it from
http://oalib.hlsresearch.com/AcousticsToolbox/ and put ``kraken.exe``, ``krakenc.exe`` and
``field.exe`` on ``PATH``, or set ``BLUEPEBBLE_AT_BIN`` to their directory or to the toolbox
root.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: Environment variable listing directories (``os.pathsep`` separated) that hold the
#: Acoustics Toolbox programs, or the root of a built toolbox.
BINARY_PATH_VARIABLE = "BLUEPEBBLE_AT_BIN"

PROGRAMS = ("kraken", "krakenc", "field")


class AcousticsToolboxError(RuntimeError):
    """A toolbox program failed or produced no output."""


def _candidate_directories(extra: Iterable[str | Path] = ()) -> list[Path]:
    """Return directories to search, most specific first."""
    roots = [Path(p) for p in extra]
    roots += [Path(p) for p in os.environ.get(BINARY_PATH_VARIABLE, "").split(os.pathsep) if p]
    directories: list[Path] = []
    for root in roots:
        # A toolbox root keeps the programs in subdirectories named after them.
        directories += [root, root / "Kraken", root / "KrakenField", root / "bin"]
    return directories


def find_program(name: str, search: Iterable[str | Path] = ()) -> Path | None:
    """Find a toolbox program by name.

    Parameters
    ----------
    name : str
        ``"kraken"``, ``"krakenc"`` or ``"field"``.
    search : iterable of path-like, optional
        Directories to try first. Then come those in :data:`BINARY_PATH_VARIABLE`, and
        finally ``PATH``.

    Returns
    -------
    pathlib.Path or None
        The executable, or ``None`` if it was not found.

    """
    filenames = (f"{name}.exe", name)
    for directory in _candidate_directories(search):
        for filename in filenames:
            path = directory / filename
            if path.is_file() and os.access(path, os.X_OK):
                return path.resolve()
    for filename in filenames:
        found = shutil.which(filename)
        if found:
            return Path(found).resolve()
    return None


@dataclass(frozen=True)
class AcousticsToolboxBinaries:
    """Paths to the toolbox programs that are available.

    Use :meth:`find` to locate them. A program that was not found is ``None``.
    """

    kraken: Path | None = None
    krakenc: Path | None = None
    field: Path | None = None

    @classmethod
    def find(cls, search: Iterable[str | Path] = ()) -> AcousticsToolboxBinaries:
        """Locate whichever programs can be found.

        Parameters
        ----------
        search : iterable of path-like, optional
            Directories to try first, before :data:`BINARY_PATH_VARIABLE` and ``PATH``.

        Returns
        -------
        AcousticsToolboxBinaries
            The programs found.

        """
        search = list(search)
        return cls(**{name: find_program(name, search) for name in PROGRAMS})

    def require(self, name: str) -> Path:
        """Return a program's path, or raise if it was not found.

        Parameters
        ----------
        name : str
            ``"kraken"``, ``"krakenc"`` or ``"field"``.

        Returns
        -------
        pathlib.Path
            The executable.

        Raises
        ------
        FileNotFoundError
            If the program is not available.

        """
        if name not in PROGRAMS:
            raise ValueError(f"Unknown program {name!r}; expected one of {PROGRAMS}")
        path = getattr(self, name)
        if path is None:
            raise FileNotFoundError(
                f"Acoustics Toolbox program {name!r} not found. Build the Acoustics Toolbox "
                "from http://oalib.hlsresearch.com/AcousticsToolbox/ and put it on PATH, or "
                f"set {BINARY_PATH_VARIABLE} to its directory or to the toolbox root."
            )
        return Path(path)


def _tail(path: Path, lines: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "<no print file>"


def run_program(program: Path, work_dir: Path, root: str, output_suffix: str) -> Path:
    """Run a toolbox program on ``work_dir/root`` and return its output file.

    Parameters
    ----------
    program : pathlib.Path
        The executable.
    work_dir : pathlib.Path
        Directory holding the input files; the program runs there.
    root : str
        File root without extension, e.g. ``"modes"`` for ``modes.env``.
    output_suffix : str
        Extension of the expected output, e.g. ``".mod"``.

    Returns
    -------
    pathlib.Path
        The output file.

    Raises
    ------
    AcousticsToolboxError
        If the program fails, writes no output, or reports an error in its print file.

    """
    work_dir = Path(work_dir)
    output = work_dir / f"{root}{output_suffix}"
    result = subprocess.run(
        [str(Path(program).resolve()), root],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    # FIELD always names its print file field.prt; KRAKEN uses the file root.
    print_file = work_dir / ("field.prt" if output_suffix == ".shd" else f"{root}.prt")
    printed = _tail(print_file)
    failed = result.returncode != 0 or not output.is_file() or "FATAL ERROR" in printed
    if failed:
        raise AcousticsToolboxError(
            f"{Path(program).name} failed in {work_dir} (exit code {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}\n"
            f"Last lines of {print_file.name}:\n{printed}"
        )
    return output
