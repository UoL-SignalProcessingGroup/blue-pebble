"""Tests for converge_settings, with a stand-in model whose error shrinks with its settings."""

import numpy as np
import pytest

from bluepebble.models.propagation import SpectrumPropagationModel, converge_settings

FREQUENCIES_HZ = np.linspace(1000.0, 3000.0, 50)
LEGS = [(np.array([0.0, 0.0, -50.0]), np.array([[1000.0, 1000.0], [0.0, 10.0], [-50.0, -60.0]]))]


class StandInModel(SpectrumPropagationModel):
    """A delay of 1 s plus an error of 1 ms / n**2, and an amplitude of 1 + 0.1 / m."""

    def __init__(self, n=1, m=1e9):
        """Set the delay and amplitude from the two settings."""
        self.delay_s = 1.0 + 1e-3 / n**2
        self.amplitude = 1.0 + 0.1 / m

    def propagate_spectrum(self, platform, source, frequencies_hz):
        """Return the same transfer function at every sensor, whatever the source."""
        num_sensors = platform.array.state_vector.shape[1]
        row = self.amplitude * np.exp(-2j * np.pi * frequencies_hz * self.delay_s)
        return np.tile(row, (num_sensors, 1)), np.full(num_sensors, self.delay_s)


def test_keeps_the_coarsest_value_that_agrees_with_the_next() -> None:
    """The answer stops changing (below -30 dB) between n = 32 and 64, so 32 is kept.

    Between n and 2n the delay moves by 0.75 ms / n**2, a relative change of about
    2 pi f times that: -25 dB from 16 to 32 at 3 kHz, -37 dB from 32 to 64.
    """
    result = converge_settings(
        lambda s: StandInModel(**s), {"n": [1, 2, 4, 8, 16, 32, 64, 128]}, LEGS, FREQUENCIES_HZ
    )

    assert result.settings == {"n": 32}
    assert [step.converged for step in result.steps] == [False] * 5 + [True]
    assert result.steps[-1].difference_db <= -30.0 < result.steps[-2].difference_db
    assert result.unconverged == []


def test_a_tuning_field_that_runs_out_keeps_its_finest_value_and_says_so() -> None:
    """If the answer is still changing at the end, the finest value is used and reported."""
    result = converge_settings(lambda s: StandInModel(**s), {"n": [1, 2, 4]}, LEGS, FREQUENCIES_HZ)

    assert result.settings == {"n": 4}
    assert result.unconverged == ["n"]
    assert "Still changing" in result.report()


def test_settings_are_tuned_in_order_with_earlier_ones_held_at_their_choice() -> None:
    """The second setting is tuned with the first at its converged value."""
    seen = []

    def make_model(settings):
        seen.append(dict(settings))
        return StandInModel(**settings)

    result = converge_settings(
        make_model,
        {"n": [8, 16, 32, 64], "m": [10.0, 100.0, 1000.0, 10000.0]},
        LEGS,
        FREQUENCIES_HZ,
    )

    assert result.settings["n"] == 32
    tuning_m = [s for s in seen if s["m"] != 10.0]
    assert tuning_m and all(s["n"] == 32 for s in tuning_m)


def test_each_combination_of_settings_is_run_once() -> None:
    """Comparisons share answers, so no combination of settings is computed twice."""
    seen = []

    def make_model(settings):
        seen.append(tuple(sorted(settings.items())))
        return StandInModel(**settings)

    converge_settings(make_model, {"n": [1, 2, 4, 8, 16, 32, 64]}, LEGS, FREQUENCIES_HZ)

    assert len(seen) == len(set(seen)) == 7


def test_identical_answers_converge_at_once() -> None:
    """A setting that changes nothing is kept at its first value."""
    result = converge_settings(
        lambda s: StandInModel(), {"unused": [1, 2, 3]}, LEGS, FREQUENCIES_HZ
    )

    assert result.settings == {"unused": 1}
    assert result.steps[0].difference_db == -np.inf


@pytest.mark.parametrize(
    ("tuning_fields", "legs", "match"),
    [({}, LEGS, "tuning_fields"), ({"n": [1, 2]}, [], "legs"), ({"n": []}, LEGS, "no values")],
)
def test_empty_inputs_are_rejected(tuning_fields, legs, match) -> None:
    """No settings, no legs, or an empty tuning field raise a ValueError."""
    with pytest.raises(ValueError, match=match):
        converge_settings(lambda s: StandInModel(**s), tuning_fields, legs, FREQUENCIES_HZ)
