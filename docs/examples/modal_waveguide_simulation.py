r"""
==========================================================
Normal-Mode Simulation of a Range-Independent Waveguide
==========================================================

In a range-independent waveguide the normal modes ``k_m(f)``, ``phi_m(z)`` depend only on
the environment and frequency. KRAKENC computes them **once**, for every frequency of the
simulator's filter grid. Every later evaluation of the channel is then a mode sum at the
current ranges and depths. That costs milliseconds, so the channel can be evaluated
densely enough for each mode to keep its own Doppler shift and for modal fading to be
resolved as the range changes.

This example:

1. builds a shallow-water waveguide over a layered seabed;
2. checks the Python mode sum against FIELD, the Acoustics Toolbox's own mode summation;
3. simulates a source opening in range past a horizontal line array, and shows the
   interference striations that the modes produce in the received spectrum.

It needs the Acoustics Toolbox programs ``krakenc`` and ``field``, found on ``PATH`` or
through ``BLUEPEBBLE_AT_BIN`` (see the installation page).

Assumptions
-----------
- The waveguide is range-independent: flat seabed and one sound speed profile throughout.
- The array is fixed and its elements' positions are known exactly.
"""  # noqa: D205, D212, D400, D415
# sphinx_gallery_thumbnail_path = "_static/thumbnails/modal_waveguide_simulation.png"

# %%
# Imports
# -------
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
from scipy import signal as scipy_signal
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from bluepebble.models.environment import (
    FlatBathymetry,
    LayeredSeabed,
    SeabedLayer,
    TabulatedSoundSpeedProfile,
)
from bluepebble.models.propagation import KrakenModalPropagationModel
from bluepebble.models.propagation.kraken import (
    AcousticsToolboxBinaries,
    compute_mode_set,
    field_parameters_text,
    read_shade_file,
    run_program,
)
from bluepebble.models.propagation.kraken.env import write_text
from bluepebble.signal import SpectrumLevelSignal
from bluepebble.simulator import ContinuousModalPassiveSonarArraySimulator

# %%
# The waveguide
# -------------
# 120 m of water with a downward-refracting summer profile, over a graded sediment layer
# and a faster halfspace. Attenuation is in dB/(m kHz), as geoacoustic models publish it.
water_depth = 120.0
ssp = TabulatedSoundSpeedProfile(
    depths_m=[0.0, 20.0, 40.0, 80.0, 120.0],
    speeds_m_s=[1522.0, 1520.0, 1500.0, 1490.0, 1489.0],
)
seabed = LayeredSeabed(
    layers=(SeabedLayer(20.0, 1580.0, 1620.0, 1.7, 0.2, name="sediment"),),
    halfspace=SeabedLayer(None, 1750.0, 1750.0, 2.0, 0.1, name="basement"),
)
model = KrakenModalPropagationModel(
    ssp=ssp,
    bathymetry=FlatBathymetry(depth=-water_depth),
    seabed=seabed,
    program="krakenc",
)

# %%
# Mode sum against FIELD
# ----------------------
# Run KRAKENC once for three frequencies, keeping its files, then compare the Python mode
# sum with FIELD's pressure computed from the very same mode file. They agree to the
# single precision of the file.
binaries = AcousticsToolboxBinaries.find()
frequencies = np.array([100.0, 200.0, 300.0])
source_depth, receiver_depth = 30.0, 60.0
ranges = np.linspace(200.0, 10000.0, 500)

figure = go.Figure()
with tempfile.TemporaryDirectory() as work:
    work = Path(work)
    modes = compute_mode_set(
        model.kraken_environment(),
        frequencies,
        np.linspace(0.0, water_depth, 241),
        binaries=binaries,
        frequencies_per_run=frequencies.size,
        work_dir=work,
    )
    run_dir = work / "chunk_0000"
    write_text(
        run_dir / "modes.flp",
        field_parameters_text([source_depth], [receiver_depth], ranges / 1000.0),
    )
    field = read_shade_file(run_program(binaries.require("field"), run_dir, "modes", ".shd"))
ours = modes.pressure_grid(source_depth, [receiver_depth], ranges)[:, 0]
for index, frequency in enumerate(frequencies):
    reference = field.pressure[index, 0, 0]
    error = np.max(np.abs(ours[index] - reference)) / np.max(np.abs(reference))
    figure.add_trace(
        go.Scatter(
            x=ranges / 1000,
            y=-20 * np.log10(np.abs(reference)),
            name=f"FIELD, {frequency:g} Hz",
            line={"width": 6},
            opacity=0.35,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=ranges / 1000,
            y=-20 * np.log10(np.abs(ours[index])),
            name=f"mode sum, {frequency:g} Hz (error {error:.0e})",
        )
    )
    print(f"{frequency:g} Hz: largest difference from FIELD {error:.1e} of the peak pressure")
figure.update_layout(
    xaxis_title="Range (km)",
    yaxis_title="Transmission loss (dB re 1 m)",
    yaxis_autorange="reversed",
)
figure.show()

# %%
# A source opening in range
# -------------------------
# A broadband source at 30 m depth opens from 1.5 km at 6 m/s past an eight-element
# horizontal line array at 60 m. The simulator plans its own knots from the modes: the
# printout gives the group-speed spread, the speed that centres the arrivals in the
# filter, and the knot interval that keeps every mode's residual phase step under
# ``pi/16``.
fs = 2000
start = datetime(2024, 1, 1, tzinfo=UTC)
duration_s, preroll_s = 120.0, 6.0
metadata = {"position_mapping": [0, 2, 4], "velocity_mapping": [1, 3, 5]}
states = [
    GroundTruthState(
        StateVector([1500.0 + 6.0 * t, 6.0, 200.0, 0.0, 30.0, 0.0]),
        timestamp=start + timedelta(seconds=float(t)),
        metadata=metadata,
    )
    for t in np.arange(-preroll_s, duration_s + 1.0, 1.0)
]
# The array is fixed, so it is described directly by its element positions: anything with
# ``num_sensors`` and ``get_platform_state_at`` serves.
elements = np.vstack([np.zeros(8), 4.0 * np.arange(8), np.full(8, 60.0)])
fixed_array = SimpleNamespace(
    num_sensors=8,
    get_platform_state_at=lambda timestamp: SimpleNamespace(
        array=SimpleNamespace(state_vector=elements, ref_state_vector=elements[:, :1])
    ),
)
simulator = ContinuousModalPassiveSonarArraySimulator(
    array=fixed_array,
    propagation_model=model,
    signal_models=[
        SpectrumLevelSignal(
            duration_s=duration_s + preroll_s,
            sampling_rate_hz=fs,
            level_db=140.0,
            band_hz=(100.0, 450.0),
            seed=2,
        )
    ],
    ground_truth_paths=[GroundTruthPath(states)],
    start_time=start,
    duration_s=duration_s,
    source_preroll_s=preroll_s,
    frequency_band_hz=(100.0, 450.0),
    band_taper_hz=20.0,
    filter_length=4096,
)
knots = simulator.plan_knots(0)
dispersion = knots.dispersion
print(
    f"group speeds {dispersion.min_group_speed_m_s:.1f}-{dispersion.max_group_speed_m_s:.1f} "
    f"m/s, c_ref {dispersion.reference_speed_m_s:.1f} m/s, "
    f"{knots.knot_times_s.size} knots every {np.diff(knots.knot_times_s)[0]:.3f} s"
)
output = simulator.render(include_noise=False)

# %%
# Interference striations
# -----------------------
# The received spectrogram shows the waveguide-invariant striations: the modes interfere,
# and the frequency of each fade changes smoothly with range. The output is one-sided, so its
# spectrogram is taken two-sided and only the positive frequencies are shown. The first 2 s are
# left out, while the channel filter fills.
frequencies_hz, times_s, power = scipy_signal.spectrogram(
    output[0], fs=fs, nperseg=1024, noverlap=768, return_onesided=False
)
band = (frequencies_hz >= 100) & (frequencies_hz <= 450)
settled = times_s >= 2.0
figure = go.Figure(
    go.Heatmap(
        x=frequencies_hz[band],
        y=times_s[settled],
        z=10 * np.log10(power[np.ix_(band, settled)].T + 1e-30),
        colorscale="Viridis",
        colorbar={"title": {"text": "dB", "side": "right"}},
    )
)
figure.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Time (s)")
figure.show()
