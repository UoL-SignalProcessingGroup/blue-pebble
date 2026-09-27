"""Smoke tests keeping the benchmark scripts runnable as the package changes."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from .support import REPO_ROOT


def test_das_beamformer_benchmark_runs_and_saves_its_outputs(tmp_path) -> None:
    """The DAS benchmark should run at minimal sizes and save one output per configuration."""
    saved = tmp_path / "outputs.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "das_beamformer.py"),
            "--sensors",
            "8",
            "--beams",
            "5",
            "--repeats",
            "1",
            "--save",
            str(saved),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with np.load(saved) as outputs:
        assert list(outputs) == ["broadband_power_8x5"]
        assert outputs["broadband_power_8x5"].shape[0] == 5
