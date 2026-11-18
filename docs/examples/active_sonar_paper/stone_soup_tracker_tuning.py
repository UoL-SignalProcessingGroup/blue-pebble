r"""
=============================
Tuning the Stone Soup Tracker
=============================

This script tunes the paper's tracker with Optuna and compares it with the untuned tracker, on
the measurements ``sonar_signal_processing_chain_comparison.py`` saved. It re-runs only the
tracker, so each trial takes seconds.

The untuned tracker's settings come from the sensors' specification
(``paper_scenario.TRACKER_DEFAULTS``): bearing standard deviations from each array's
beamwidth, a 15 m range standard deviation, a small process noise, a Mahalanobis gate of 4,
confirmation after two updates. The tuned tracker's settings come from the data.

Six processing chains are tuned, chosen before any tuning by the compute cost of their
propagation and signal processing configuration alone: on each seabed, the cheapest, the
median and the most expensive of the chains the comparison script ran there. Each
gets its own Optuna study, which minimises the chain's mean GOSPA over its training runs. Every
result is then reported on holdout runs, generated with other seeds, which the tuning never
saw, so that tuning can't fit one realisation of the noise.

- **Untuned against tuned**: each chain's GOSPA, false tracks and missed targets on the test
  runs, with both trackers.
- **Transfer**: each chain's tuned settings applied to the other five, to show whether one
  tuning serves every chain, or every chain on a seabed, or each needs its own.
- **What mattered**: each study's history and its settings' importances.
"""  # noqa: D205, D212, D400, D415

# %%
# Imports
# -------

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import paper_scenario as scenario
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# Configuration
# -------------

# Set to "plotly" to save interactive HTML figures, or "pdf" to save static PDF figures. Either
# way, figures are written to a folder named after this script.
PLOTTING_MODE = "pdf"
FIGURE_DIR = Path(__file__).with_name(f"{Path(__file__).stem}_figures")
MEASUREMENTS_PATH = Path(__file__).with_name(
    "sonar_signal_processing_chain_comparison_measurements.json"
)
TUNED_PATH = Path(__file__).with_name(f"{Path(__file__).stem}_results.json")

N_TRIALS = 200  # per chain
SAMPLER_SEED = 0  # the studies are reproducible
FAILED_TRIAL_GOSPA_M = 10 * scenario.GOSPA_C_M  # a trial whose tracker fails scores this


def save_figure(figure: go.Figure, name: str) -> None:
    """Write ``figure`` to ``FIGURE_DIR`` in the format set by ``PLOTTING_MODE``."""
    FIGURE_DIR.mkdir(exist_ok=True)
    if PLOTTING_MODE == "plotly":
        figure.write_html(FIGURE_DIR / f"{name}.html", include_plotlyjs="directory")
    elif PLOTTING_MODE == "pdf":
        figure.write_image(FIGURE_DIR / f"{name}.pdf")
    else:
        raise ValueError(f'PLOTTING_MODE must be "plotly" or "pdf", not {PLOTTING_MODE!r}')


# %%
# Search space
# ------------
#
# Each setting of ``paper_scenario.TRACKER_DEFAULTS`` is searched over a range around its
# untuned value, on a log scale where it spans orders of magnitude.

SEARCH_SPACE = {
    "process_noise": ("float", 1e-4, 1.0, True),
    "bearing_sd_scale": ("float", 0.25, 4.0, True),
    "range_sd_m": ("float", 1.0, 50.0, True),
    "gate": ("float", 2.0, 10.0, False),
    "prior_speed_sd_mps": ("float", 1.0, 10.0, False),
    "min_points": ("int", 2, 4, False),
    "merge_distance_m": ("float", 50.0, 300.0, False),
}


def suggest(trial):
    """Return one trial's tracker settings, drawn from ``SEARCH_SPACE``."""
    params = {}
    for name, (kind, low, high, log) in SEARCH_SPACE.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, low, high)
        else:
            params[name] = trial.suggest_float(name, low, high, log=log)
    return params


# %%
# Measurements
# ------------
#
# Each of the comparison script's runs is the scenario with its own noise, from a seed, and is
# marked for tuning or holdout there (``TUNING_SEEDS``, ``HOLDOUT_SEEDS``). The studies see
# only the tuning runs; every result below is on the holdout runs.

data = json.loads(MEASUREMENTS_PATH.read_text())
ping_times = [datetime.fromisoformat(t) for t in data["ping_times"]]
RUNS = data["runs"]
TRAIN_RUNS = [i for i, run in enumerate(RUNS) if run["role"] == "tuning"]
TEST_RUNS = [i for i, run in enumerate(RUNS) if run["role"] == "holdout"]
if not TRAIN_RUNS or not TEST_RUNS:
    raise ValueError(
        f"{MEASUREMENTS_PATH.name} needs runs of both roles, tuning and holdout; set "
        "TUNING_SEEDS and HOLDOUT_SEEDS in the comparison script and rerun it."
    )
_, arrays = scenario.ship(ping_times)
target_paths = scenario.targets(ping_times)
print(
    f"{len(data['chains'])} chains, {len(ping_times)} pings; tuning on seeds "
    f"{[RUNS[i]['seed'] for i in TRAIN_RUNS]}, evaluating on holdout seeds "
    f"{[RUNS[i]['seed'] for i in TEST_RUNS]}"
)

# %%
# Chains to tune
# --------------
#
# On each seabed, the cheapest, the median and the most expensive chain by the compute cost of
# its propagation and signal processing configuration, as if it were run on its own: its
# simulation (propagation and echo synthesis) plus its beamforming, matched filtering,
# detection and grouping. Tracking is left out, since it depends on the settings being tuned.

SEABEDS = list(dict.fromkeys(chain["config"]["seabed"] for chain in data["chains"]))
selected, roles = [], {}
for seabed in SEABEDS:
    on_seabed = sorted(
        (c for c in data["chains"] if c["config"]["seabed"] == seabed),
        key=lambda chain: chain["cost_s"],
    )
    for role, chain in (
        ("Cheapest", on_seabed[0]),
        ("Median", on_seabed[len(on_seabed) // 2]),
        ("Most expensive", on_seabed[-1]),
    ):
        if chain["label"] not in roles:
            selected.append(chain)
            roles[chain["label"]] = role
print(f"\n{'Seabed':>8} {'Role':>15} {'Chain':>45} {'Cost (s)':>10}")
for chain in selected:
    print(
        f"{chain['config']['seabed']:>8} {roles[chain['label']]:>15} {chain['label']:>45}"
        f" {chain['cost_s']:>10.0f}"
    )


def scores(chain, runs, params=None):
    """Return the chain's tracking scores on ``runs`` with tracker settings ``params``.

    Each score is the mean over the runs and the settled pings; ``gospa_by_ping_m`` is the mean
    over the runs at each ping.
    """
    per_run = []
    for run in runs:
        _, states_at = scenario.run_tracker(
            arrays,
            ping_times,
            chain["measurements"][run],
            chain["config"]["arrays"],
            params,
        )
        per_run.append(scenario.score_tracks(target_paths, ping_times, states_at))
    settle = scenario.SETTLE_PINGS
    summary = {
        measure: float(np.mean([r[measure][settle:] for r in per_run]))
        for measure in ("gospa_m", "localisation_m", "false_tracks", "missed_targets")
    }
    summary["gospa_by_ping_m"] = np.mean([r["gospa_m"] for r in per_run], axis=0).tolist()
    return summary


# %%
# Tuning
# ------
#
# One study per chain, with Optuna's tree-structured Parzen estimator, seeded so it can be
# repeated. Each trial runs the tracker on the chain's training runs and returns its mean GOSPA
# after the settling pings. A trial whose tracker fails numerically scores 10 times the GOSPA
# cut-off.

optuna.logging.set_verbosity(optuna.logging.WARNING)
studies, tuned_params, tuning_s = {}, {}, {}
for chain in selected:
    label = chain["label"]

    def objective(trial, chain=chain):
        """Return the chain's mean training GOSPA with one trial's tracker settings."""
        try:
            return scores(chain, TRAIN_RUNS, suggest(trial))["gospa_m"]
        except np.linalg.LinAlgError:  # an ill-conditioned covariance from extreme settings
            return FAILED_TRIAL_GOSPA_M

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    )
    # The untuned settings are the first trial, so the tuned ones can only match or beat them
    # on the training runs.
    study.enqueue_trial(dict(scenario.TRACKER_DEFAULTS))
    start = time.perf_counter()
    study.optimize(objective, n_trials=N_TRIALS)
    tuning_s[label] = time.perf_counter() - start
    studies[label] = study
    tuned_params[label] = study.best_params
    print(
        f"\n{label}: best training GOSPA {study.best_value:.1f} m after {N_TRIALS} trials "
        f"({tuning_s[label]:.0f} s)"
    )
    for name, value in study.best_params.items():
        print(f"  {name:>20}: {value:<10.4g} (untuned {scenario.TRACKER_DEFAULTS[name]:g})")

# %%
# Untuned against tuned
# ---------------------
#
# Every number here is on the test runs.

results = {}
print(
    f"\n{'Chain':>45} {'Tracker':>8} {'GOSPA (m)':>10} {'Localisation (m)':>17}"
    f" {'False tracks':>13} {'Missed':>7}"
)
for chain in selected:
    label = chain["label"]
    results[label] = {
        "untuned": scores(chain, TEST_RUNS),
        "tuned": scores(chain, TEST_RUNS, tuned_params[label]),
    }
    for tracker, s in results[label].items():
        print(
            f"{label:>45} {tracker:>8} {s['gospa_m']:>10.1f} {s['localisation_m']:>17.1f}"
            f" {s['false_tracks']:>13.2f} {s['missed_targets']:>7.2f}"
        )

# %%
# Transfer
# --------
#
# Each chain tracked with each chain's tuned settings, on the test runs: the diagonal is each
# chain's own tuning. If the off-diagonal entries are close to the diagonal, one tuning serves
# every chain.

transfer = {
    tracked["label"]: {
        source["label"]: scores(tracked, TEST_RUNS, tuned_params[source["label"]])["gospa_m"]
        for source in selected
    }
    for tracked in selected
}
print("\nTest GOSPA (m): rows tracked with the settings tuned on each column's chain")
labels = [chain["label"] for chain in selected]
print(f"{'':>45}", *(f"{i + 1:>10}" for i in range(len(labels))))
for i, row in enumerate(labels):
    print(f"{f'{i + 1}. {row}':>45}", *(f"{transfer[row][col]:>10.1f}" for col in labels))

# %%
# Importance
# ----------
#
# How much each setting mattered to each study's GOSPA, by Optuna's fANOVA importance.

importances = {}
for label, study in studies.items():
    try:
        importances[label] = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError) as error:  # too few distinct completed trials
        print(f"{label}: no importances ({error})")
for label, values in importances.items():
    print(f"\n{label}")
    for name, value in values.items():
        print(f"  {name:>20}: {value:.2f}")

TUNED_PATH.write_text(
    json.dumps(
        {
            "measurements_file": MEASUREMENTS_PATH.name,
            "tuning_seeds": [RUNS[i]["seed"] for i in TRAIN_RUNS],
            "holdout_seeds": [RUNS[i]["seed"] for i in TEST_RUNS],
            "n_trials": N_TRIALS,
            "sampler_seed": SAMPLER_SEED,
            "untuned": scenario.TRACKER_DEFAULTS,
            "search_space": {k: list(v) for k, v in SEARCH_SPACE.items()},
            "chains": [
                {
                    "label": chain["label"],
                    "role": roles[chain["label"]],
                    "config": chain["config"],
                    "cost_s": chain["cost_s"],
                    "tuned": tuned_params[chain["label"]],
                    "best_training_gospa_m": studies[chain["label"]].best_value,
                    "tuning_s": tuning_s[chain["label"]],
                    "test": results[chain["label"]],
                    "importance": importances.get(chain["label"]),
                }
                for chain in selected
            ],
            "transfer_gospa_m": transfer,
        },
        indent=2,
    )
)
print(f"\nResults saved to {TUNED_PATH.name}")

# %%
# Figures
# -------
#
# The first figure shows each study's best training GOSPA so far against the trial number; the
# untuned settings are trial 0. The second shows each chain's test GOSPA at every ping, untuned
# (dashed) and tuned (solid).

figure = go.Figure()
for label, study in studies.items():
    values = [t.value for t in study.trials if t.value is not None]
    figure.add_trace(go.Scatter(y=np.minimum.accumulate(values), mode="lines", name=label))
figure.update_layout(
    template="plotly_white",
    title="Tuning: Best Training GOSPA so Far",
    xaxis_title="Trial",
    yaxis_title="GOSPA (m)",
    height=500,
    width=900,
)
save_figure(figure, "tuning_history")

cols = 3
rows = -(-len(selected) // cols)
figure = make_subplots(
    rows=rows, cols=cols, subplot_titles=[f"{roles[label]}: {label}" for label in labels]
)
ping_numbers = np.arange(1, len(ping_times) + 1)
for index, label in enumerate(labels):
    row, col = index // cols + 1, index % cols + 1
    for tracker, dash in (("untuned", "dash"), ("tuned", "solid")):
        figure.add_trace(
            go.Scatter(
                x=ping_numbers,
                y=results[label][tracker]["gospa_by_ping_m"],
                mode="lines",
                line={"dash": dash, "color": "#2a78d6" if tracker == "tuned" else "#7f7f7f"},
                name=tracker.capitalize(),
                showlegend=index == 0,
            ),
            row=row,
            col=col,
        )
    figure.update_xaxes(title_text="Ping", row=row, col=col)
    figure.update_yaxes(title_text="GOSPA (m)" if col == 1 else None, row=row, col=col)
figure.update_layout(
    template="plotly_white",
    title="Test GOSPA over Time: Untuned against Tuned Tracker",
    height=420 * rows,
    width=450 * cols,
)
save_figure(figure, "untuned_against_tuned")
