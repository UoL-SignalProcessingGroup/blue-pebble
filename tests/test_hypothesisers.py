"""Unit tests for the BasicHypothesiser class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.stats import chi2

from nereus.association.hypothesisers import (
    BasicHypothesiser,
    Hypothesiser,
    JPDAHypothesiser,
    PDAHypothesiser,
    ProbabilisticHypothesiser,
)
from nereus.models.measurement import MeasurementModel
from nereus.types.detections import Detection, MissedDetection
from nereus.types.hypotheses import Hypothesis, ProbabilityHypothesis
from nereus.types.states import GaussianState, ParticleState, State


class TestBasicHypothesiser:
    """Test suite for BasicHypothesiser class."""

    def test_init_valid_measurement_model(self):
        """Test initialization with a valid measurement model."""
        mock_model = Mock(spec=MeasurementModel)
        hypothesiser = BasicHypothesiser(mock_model)

        assert hypothesiser.measurement_model is mock_model

    def test_init_invalid_measurement_model(self):
        """Test initialization with invalid measurement model."""
        # The BasicHypothesiser doesn't validate the measurement_model in __init__
        # So we expect it to fail when hypothesise is called
        hypothesiser = BasicHypothesiser(None)

        mock_state = Mock(spec=State)
        with pytest.raises(AttributeError):
            hypothesiser.hypothesise(mock_state, [])

    def test_hypothesise_empty_detections(self):
        """Test hypothesise with empty detections list."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        hypothesiser = BasicHypothesiser(mock_model)

        hypotheses = hypothesiser.hypothesise(mock_state, [])

        assert len(hypotheses) == 1
        assert isinstance(hypotheses[0], Hypothesis)
        assert hypotheses[0].prediction == mock_state
        assert isinstance(hypotheses[0].measurement, MissedDetection)
        mock_model.function.assert_called_once_with(mock_state, noise=False)

    def test_hypothesise_single_detection(self):
        """Test hypothesise with a single detection."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        mock_detection = Mock(spec=Detection)

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        assert len(hypotheses) == 2

        # First hypothesis should be missed detection
        assert isinstance(hypotheses[0], Hypothesis)
        assert hypotheses[0].prediction == mock_state
        assert isinstance(hypotheses[0].measurement, MissedDetection)

        # Second hypothesis should be detection association
        assert isinstance(hypotheses[1], Hypothesis)
        assert hypotheses[1].prediction == mock_state
        assert hypotheses[1].measurement == mock_detection
        np.testing.assert_array_equal(
            hypotheses[1].measurement_prediction, np.array([1.0, 2.0])
        )

    def test_hypothesise_multiple_detections(self):
        """Test hypothesise with multiple detections."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        mock_detections = [Mock(spec=Detection) for _ in range(3)]

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, mock_detections)

        assert len(hypotheses) == 4  # 1 missed + 3 detections

        # First hypothesis should be missed detection
        assert isinstance(hypotheses[0].measurement, MissedDetection)

        # Remaining hypotheses should correspond to detections
        for i, detection in enumerate(mock_detections):
            assert hypotheses[i + 1].measurement == detection
            assert hypotheses[i + 1].prediction == mock_state
            np.testing.assert_array_equal(
                hypotheses[i + 1].measurement_prediction, np.array([1.0, 2.0])
            )

    def test_hypothesise_with_gaussian_state(self):
        """Test hypothesise with GaussianState."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([3.0, 4.0])

        mock_state = Mock(spec=GaussianState)
        mock_detection = Mock(spec=Detection)

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        assert len(hypotheses) == 2
        mock_model.function.assert_called_once_with(mock_state, noise=False)

    def test_hypothesise_with_particle_state(self):
        """Test hypothesise with ParticleState."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([5.0, 6.0])

        mock_state = Mock(spec=ParticleState)
        mock_detection = Mock(spec=Detection)

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        assert len(hypotheses) == 2
        mock_model.function.assert_called_once_with(mock_state, noise=False)

    def test_hypothesise_with_iterator_detections(self):
        """Test hypothesise with detections as iterator."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        mock_detections = [Mock(spec=Detection) for _ in range(2)]

        hypothesiser = BasicHypothesiser(mock_model)
        # Pass as generator
        hypotheses = hypothesiser.hypothesise(mock_state, iter(mock_detections))

        assert len(hypotheses) == 3  # 1 missed + 2 detections

    def test_hypothesise_none_state(self):
        """Test hypothesise with None state raises appropriate error."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.side_effect = AttributeError("'NoneType' has no attribute")

        hypothesiser = BasicHypothesiser(mock_model)

        with pytest.raises(AttributeError):
            hypothesiser.hypothesise(None, [])

    def test_hypothesise_none_detections(self):
        """Test hypothesise with None detections raises appropriate error."""
        mock_model = Mock(spec=MeasurementModel)
        mock_state = Mock(spec=State)

        hypothesiser = BasicHypothesiser(mock_model)

        with pytest.raises(TypeError):
            hypothesiser.hypothesise(mock_state, None)

    def test_measurement_prediction_conversion_to_float64(self):
        """Test that measurement predictions are converted to float64."""
        mock_model = Mock(spec=MeasurementModel)
        # Return non-float64 array
        mock_model.function.return_value = np.array([1, 2], dtype=np.int32)

        mock_state = Mock(spec=State)
        mock_detection = Mock(spec=Detection)

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        # Check that the prediction was converted to float64
        prediction = hypotheses[1].measurement_prediction
        assert prediction.dtype == np.float64
        np.testing.assert_array_equal(prediction, np.array([1.0, 2.0]))

    def test_measurement_model_called_correctly(self):
        """Test that measurement model is called with correct parameters."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        mock_detection = Mock(spec=Detection)

        hypothesiser = BasicHypothesiser(mock_model)
        hypothesiser.hypothesise(mock_state, [mock_detection])

        mock_model.function.assert_called_once_with(mock_state, noise=False)

    def test_hypothesis_objects_structure(self):
        """Test that returned Hypothesis objects have correct structure."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        mock_detection = Mock(spec=Detection)

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        # Check missed detection hypothesis
        missed_hyp = hypotheses[0]
        assert hasattr(missed_hyp, "prediction")
        assert hasattr(missed_hyp, "measurement")
        assert missed_hyp.prediction == mock_state

        # Check detection hypothesis
        det_hyp = hypotheses[1]
        assert hasattr(det_hyp, "prediction")
        assert hasattr(det_hyp, "measurement")
        assert hasattr(det_hyp, "measurement_prediction")
        assert det_hyp.prediction == mock_state
        assert det_hyp.measurement == mock_detection

    def test_hypothesise_large_detection_set(self):
        """Test hypothesise with a large number of detections."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        num_detections = 100
        mock_detections = [Mock(spec=Detection) for _ in range(num_detections)]

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, mock_detections)

        assert len(hypotheses) == num_detections + 1
        assert isinstance(hypotheses[0].measurement, MissedDetection)

        for i in range(num_detections):
            assert hypotheses[i + 1].measurement == mock_detections[i]

    def test_measurement_model_exception_propagation(self):
        """Test that exceptions from measurement model are properly propagated."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.side_effect = ValueError("Model error")

        mock_state = Mock(spec=State)

        hypothesiser = BasicHypothesiser(mock_model)

        with pytest.raises(ValueError, match="Model error"):
            hypothesiser.hypothesise(mock_state, [])

    def test_inheritance_from_hypothesiser(self):
        """Test that BasicHypothesiser properly inherits from Hypothesiser."""
        mock_model = Mock(spec=MeasurementModel)
        hypothesiser = BasicHypothesiser(mock_model)

        assert isinstance(hypothesiser, Hypothesiser)
        assert hasattr(hypothesiser, "hypothesise")

    def test_hypothesise_return_type(self):
        """Test that hypothesise returns the correct type."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        mock_detection = Mock(spec=Detection)

        hypothesiser = BasicHypothesiser(mock_model)
        result = hypothesiser.hypothesise(mock_state, [mock_detection])

        assert isinstance(result, list)
        assert all(isinstance(hyp, Hypothesis) for hyp in result)

    @pytest.mark.parametrize("num_detections", [0, 1, 5, 10])
    def test_hypothesise_parametrized_detection_counts(self, num_detections):
        """Test hypothesise with various numbers of detections."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.function.return_value = np.array([1.0, 2.0])

        mock_state = Mock(spec=State)
        mock_detections = [Mock(spec=Detection) for _ in range(num_detections)]

        hypothesiser = BasicHypothesiser(mock_model)
        hypotheses = hypothesiser.hypothesise(mock_state, mock_detections)

        # Always 1 missed detection + number of actual detections
        assert len(hypotheses) == num_detections + 1
        assert isinstance(hypotheses[0].measurement, MissedDetection)


class TestPDAHypothesiser:
    """Test suite for PDAHypothesiser class."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]  # 2D measurement

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
            clutter_spatial_density=1e-3,
        )

        assert hypothesiser.measurement_model is mock_model
        assert hypothesiser.detection_probability == 0.9
        assert hypothesiser.gate_probability == 0.99
        assert hypothesiser.clutter_spatial_density == 1e-3

        # Check that gate threshold is calculated correctly
        expected_threshold = chi2.ppf(0.99, df=2)
        assert hypothesiser.gate_threshold == expected_threshold

    def test_init_without_clutter_density(self):
        """Test initialization without clutter spatial density."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.8,
            gate_probability=0.95,
        )

        assert hypothesiser.clutter_spatial_density is None

    def test_inheritance_from_probabilistic_hypothesiser(self):
        """Test that PDAHypothesiser inherits from ProbabilisticHypothesiser."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0]

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        assert isinstance(hypothesiser, ProbabilisticHypothesiser)
        assert hasattr(hypothesiser, "_predict_measurement_and_covar")
        assert hasattr(hypothesiser, "_validation_region_volume")

    @patch("nereus.association.hypothesisers.mahalanobis")
    @patch("nereus.association.hypothesisers.multivariate_normal.pdf")
    def test_hypothesise_empty_detections(self, mock_pdf, mock_mahalanobis):
        """Test hypothesise with empty detections list."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        # Mock the measurement prediction
        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            hypotheses = hypothesiser.hypothesise(mock_state, [])

        # Should only have missed detection hypothesis
        assert len(hypotheses) == 1
        assert isinstance(hypotheses[0], ProbabilityHypothesis)
        assert isinstance(hypotheses[0].measurement, MissedDetection)
        assert hypotheses[0].probability == 1.0  # Only possibility
        assert hypotheses[0].prediction == mock_state

    @patch("nereus.association.hypothesisers.mahalanobis")
    @patch("nereus.association.hypothesisers.multivariate_normal.pdf")
    def test_hypothesise_single_detection_inside_gate(self, mock_pdf, mock_mahalanobis):
        """Test hypothesise with single detection inside validation gate."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        mock_detection = Mock(spec=Detection)
        mock_detection.state_vector = np.array([[1.1], [2.1]])

        # Mock measurement prediction
        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        # Mock that detection is inside gate
        mock_mahalanobis.return_value = 5.0  # Less than gate threshold
        mock_pdf.return_value = 0.5

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
            clutter_spatial_density=1e-3,
        )

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        # Should have missed detection + 1 detection hypothesis
        assert len(hypotheses) == 2
        assert isinstance(hypotheses[0], ProbabilityHypothesis)
        assert isinstance(hypotheses[0].measurement, MissedDetection)
        assert isinstance(hypotheses[1], ProbabilityHypothesis)
        assert hypotheses[1].measurement == mock_detection

        # Probabilities should sum to 1
        total_prob = sum(hyp.probability for hyp in hypotheses)
        assert abs(total_prob - 1.0) < 1e-10

    @patch("nereus.association.hypothesisers.mahalanobis")
    @patch("nereus.association.hypothesisers.multivariate_normal.pdf")
    def test_hypothesise_single_detection_outside_gate(
        self, mock_pdf, mock_mahalanobis
    ):
        """Test hypothesise with single detection outside validation gate."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        mock_detection = Mock(spec=Detection)
        mock_detection.state_vector = np.array([[5.0], [6.0]])

        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        # Mock that detection is outside gate
        mock_mahalanobis.return_value = 50.0  # Greater than gate threshold

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        # Should have missed detection + 1 detection hypothesis
        assert len(hypotheses) == 2
        assert isinstance(hypotheses[0].measurement, MissedDetection)
        assert hypotheses[0].probability == 1.0  # Only missed detection possible
        assert hypotheses[1].probability == 0.0  # Detection outside gate

    @patch("nereus.association.hypothesisers.mahalanobis")
    @patch("nereus.association.hypothesisers.multivariate_normal.pdf")
    def test_hypothesise_multiple_detections_mixed_gating(
        self, mock_pdf, mock_mahalanobis
    ):
        """Test hypothesise with multiple detections, some inside and outside gate."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        # Create 3 detections
        detections = []
        for i in range(3):
            det = Mock(spec=Detection)
            det.state_vector = np.array([[i], [i]])
            detections.append(det)

        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        # Mock gating: first and third inside, second outside
        mock_mahalanobis.side_effect = [5.0, 50.0, 6.0]
        mock_pdf.side_effect = [0.3, 0.4]  # Only for gated detections

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
            clutter_spatial_density=1e-3,
        )

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            hypotheses = hypothesiser.hypothesise(mock_state, detections)

        # Should have 1 missed + 3 detection hypotheses
        assert len(hypotheses) == 4
        assert isinstance(hypotheses[0].measurement, MissedDetection)

        # Check that probabilities sum to 1
        total_prob = sum(hyp.probability for hyp in hypotheses)
        assert abs(total_prob - 1.0) < 1e-10

        # Detection outside gate should have zero probability
        assert hypotheses[2].probability == 0.0  # Second detection

    def test_hypothesise_with_gaussian_state(self):
        """Test hypothesise with GaussianState input."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]
        mock_model.function.return_value = np.array([[1.0], [2.0]])
        mock_model.jacobian.return_value = np.eye(2)
        mock_model.R = np.eye(2) * 0.01

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2) * 0.1

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        # Test that _predict_measurement_and_covar works with GaussianState
        pred_mean, pred_covar = hypothesiser._predict_measurement_and_covar(mock_state)

        assert pred_mean.shape == (2, 1)
        assert pred_covar.shape == (2, 2)
        mock_model.function.assert_called_once_with(mock_state, noise=False)
        mock_model.jacobian.assert_called_once_with(mock_state)

    def test_hypothesise_with_particle_state(self):
        """Test hypothesise with ParticleState input."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        # Mock particle predictions
        particle_predictions = np.array([[0.9, 1.1, 1.0], [1.9, 2.1, 2.0]])
        mock_model.function.side_effect = [
            particle_predictions,
            np.array([[1.0], [2.0]]),
        ]

        mock_state = Mock(spec=ParticleState)
        mock_state.mean = Mock()

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        # Test that _predict_measurement_and_covar works with ParticleState
        pred_mean, pred_covar = hypothesiser._predict_measurement_and_covar(mock_state)

        assert pred_mean.shape == (2, 1)
        assert pred_covar.shape == (2, 2)

    def test_hypothesise_with_unsupported_state_type(self):
        """Test hypothesise with unsupported state type raises TypeError."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock()  # Not GaussianState or ParticleState

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        with pytest.raises(
            TypeError, match="State must be either ParticleState or GaussianState"
        ):
            hypothesiser._predict_measurement_and_covar(mock_state)

    def test_validation_region_volume_calculation(self):
        """Test validation region volume calculation."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        measurement_prediction = np.array([[1.0], [2.0]])
        covar = np.eye(2) * 0.1

        volume = hypothesiser._validation_region_volume(measurement_prediction, covar)

        # Volume should be positive
        assert volume > 0
        # For 2D, should involve pi and gate threshold
        assert isinstance(volume, float)

    @patch("nereus.association.hypothesisers.mahalanobis")
    @patch("nereus.association.hypothesisers.multivariate_normal.pdf")
    def test_hypothesise_zero_total_likelihood(self, mock_pdf, mock_mahalanobis):
        """Test hypothesise when total likelihood is zero."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        mock_detection = Mock(spec=Detection)
        mock_detection.state_vector = np.array([[1.0], [2.0]])

        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        # Mock zero likelihood scenario
        mock_mahalanobis.return_value = 50.0  # Outside gate

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.0,  # Zero detection probability
            gate_probability=0.99,
        )

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        # Missed detection should have probability 1, detection should have 0
        assert hypotheses[0].probability == 1.0
        assert hypotheses[1].probability == 0.0

    @patch("nereus.association.hypothesisers.mahalanobis")
    @patch("nereus.association.hypothesisers.multivariate_normal.pdf")
    def test_hypothesise_with_uniform_clutter_assumption(
        self, mock_pdf, mock_mahalanobis
    ):
        """Test hypothesise without explicit clutter density (uniform assumption)."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        mock_detection = Mock(spec=Detection)
        mock_detection.state_vector = np.array([[1.0], [2.0]])

        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        # Mock that detection is inside gate
        mock_mahalanobis.return_value = 5.0
        mock_pdf.return_value = 0.5

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
            # No clutter_spatial_density provided
        )

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            with patch.object(
                hypothesiser, "_validation_region_volume", return_value=1.0
            ):
                hypotheses = hypothesiser.hypothesise(mock_state, [mock_detection])

        # Should handle uniform clutter assumption
        assert len(hypotheses) == 2
        total_prob = sum(hyp.probability for hyp in hypotheses)
        assert abs(total_prob - 1.0) < 1e-10

    def test_hypothesise_return_type(self):
        """Test that hypothesise returns correct type."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            result = hypothesiser.hypothesise(mock_state, [])

        assert isinstance(result, list)
        assert all(isinstance(hyp, ProbabilityHypothesis) for hyp in result)

    @pytest.mark.parametrize("detection_prob", [0.1, 0.5, 0.9, 1.0])
    def test_hypothesise_parametrized_detection_probabilities(self, detection_prob):
        """Test hypothesise with various detection probabilities."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=detection_prob,
            gate_probability=0.99,
        )

        pred_mean = np.array([[1.0], [2.0]])
        pred_covar = np.eye(2) * 0.1

        with patch.object(
            hypothesiser,
            "_predict_measurement_and_covar",
            return_value=(pred_mean, pred_covar),
        ):
            hypotheses = hypothesiser.hypothesise(mock_state, [])

        # Should always have exactly one hypothesis for empty detections
        assert len(hypotheses) == 1
        assert isinstance(hypotheses[0].measurement, MissedDetection)
        assert hypotheses[0].probability == 1.0

    @pytest.mark.parametrize("gate_prob", [0.9, 0.95, 0.99, 0.999])
    def test_hypothesise_parametrized_gate_probabilities(self, gate_prob):
        """Test hypothesise with various gate probabilities."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=gate_prob,
        )

        # Gate threshold should increase with gate probability
        expected_threshold = chi2.ppf(gate_prob, df=2)
        assert abs(hypothesiser.gate_threshold - expected_threshold) < 1e-10

    def test_invalid_parameters_handling(self):
        """Test handling of invalid parameter values."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]

        # Test initialization doesn't validate parameters in constructor
        # But invalid values should cause issues during use
        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=-0.1,  # Invalid but allowed in constructor
            gate_probability=0.99,
        )

        # Just verify the hypothesiser was created
        assert hypothesiser.detection_probability == -0.1

    def test_edge_case_measurement_model_error_propagation(self):
        """Test that measurement model errors are properly propagated."""
        mock_model = Mock(spec=MeasurementModel)
        mock_model.mapping = [0, 1]
        mock_model.function.side_effect = ValueError("Model error")

        mock_state = Mock(spec=GaussianState)
        mock_state.covar = np.eye(2)

        hypothesiser = PDAHypothesiser(
            measurement_model=mock_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        with pytest.raises(ValueError, match="Model error"):
            hypothesiser._predict_measurement_and_covar(mock_state)


@pytest.fixture
def mock_measurement_model():
    """Fixture for a mock measurement model."""
    model = Mock(spec=MeasurementModel)
    model.mapping = [0, 1]  # 2D measurement
    model.jacobian.return_value = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # 2x4 jacobian
    model.R = np.eye(2) * 0.1

    def function_side_effect(state, noise=False):
        # Extract position from state vector
        if hasattr(state, "state_vector"):
            return state.state_vector[:2]  # Take first 2 elements (position)
        else:
            return np.array([[0], [0]])  # Default

    model.function.side_effect = function_side_effect
    return model


@pytest.fixture
def mock_states():
    """Fixture for a list of mock states."""
    states = []
    for i in range(2):
        state = Mock(spec=GaussianState, name=f"State_{i}")
        state.state_vector = np.array([[i * 10], [i * 10], [0], [0]])
        state.covar = np.eye(4) * 0.5
        states.append(state)
    return states


@pytest.fixture
def mock_detections():
    """Fixture for a list of mock detections."""
    detections = []
    for i in range(3):
        det = Mock(spec=Detection, name=f"Detection_{i}")
        det.name = f"Detection_{i}"  # Add name attribute
        det.state_vector = np.array([[i * 2], [i * 2]])
        detections.append(det)
    return detections


class TestJPDAHypothesiser:
    """Test suite for JPDAHypothesiser class."""

    @pytest.mark.parametrize("use_ehm", [True, False])
    def test_init_valid_parameters(self, mock_measurement_model, use_ehm):
        """Test initialization with valid parameters."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
            clutter_spatial_density=1e-3,
            use_ehm=use_ehm,
        )
        assert isinstance(hypothesiser, ProbabilisticHypothesiser)
        assert hypothesiser.use_ehm is use_ehm

    @patch(
        "nereus.association.hypothesisers.JPDAHypothesiser._predict_measurement_and_covar"
    )
    def test_hypothesise_empty_states(
        self, mock_predict, mock_measurement_model, mock_detections
    ):
        """Test hypothesise with an empty list of states."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )
        hypotheses, unassociated = hypothesiser.hypothesise([], mock_detections)

        assert hypotheses == {}
        assert unassociated == set(mock_detections)
        mock_predict.assert_not_called()

    @patch(
        "nereus.association.hypothesisers.JPDAHypothesiser._predict_measurement_and_covar"
    )
    def test_hypothesise_empty_detections(
        self, mock_predict, mock_measurement_model, mock_states
    ):
        """Test hypothesise with an empty list of detections."""
        mock_predict.return_value = (np.zeros((2, 1)), np.eye(2))
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )
        hypotheses, unassociated = hypothesiser.hypothesise(mock_states, [])

        assert unassociated == set()
        assert len(hypotheses) == len(mock_states)
        for state in mock_states:
            assert len(hypotheses[state]) == 1  # Only missed detection
            missed_hyp = hypotheses[state][0]
            assert isinstance(missed_hyp.measurement, MissedDetection)
            assert missed_hyp.probability == 1.0

    @pytest.mark.parametrize("use_ehm", [True, False])
    @patch("nereus.association.hypothesisers.EHM2.run")
    def test_hypothesise_dispatch(
        self,
        mock_ehm_run,
        use_ehm,
        mock_measurement_model,
        mock_states,
        mock_detections,
    ):
        """Test that the correct association method is called based on use_ehm."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
            use_ehm=use_ehm,
        )

        # Mock internal methods to focus on the dispatch logic
        with (
            patch.object(
                hypothesiser,
                "_compute_validation_matrix",
                return_value=(np.ones((2, 4), dtype=bool), set()),
            ),
            patch.object(
                hypothesiser,
                "_compute_likelihood_matrix",
                return_value=np.random.rand(2, 4),
            ),
            patch.object(
                hypothesiser,
                "_compute_association_combinatorial",
                return_value=np.random.rand(2, 4),
            ) as mock_comb,
        ):
            hypothesiser.hypothesise(mock_states, mock_detections)

            if use_ehm:
                mock_ehm_run.assert_called_once()
                mock_comb.assert_not_called()
            else:
                mock_ehm_run.assert_not_called()
                mock_comb.assert_called_once()

    def test_compute_validation_matrix(
        self, mock_measurement_model, mock_states, mock_detections
    ):
        """Test the validation matrix computation."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        pred_means = [np.array([[0], [0]]), np.array([[10], [10]])]
        pred_covars = [np.eye(2), np.eye(2)]

        # Detections at (0,0), (2,2), (4,4).
        # State 1 at (0,0) should validate det 0 and 1.
        # State 2 at (10,10) should validate none.
        hypothesiser.gate_threshold = 9.21  # chi2.ppf(0.99, df=2)

        matrix, unassociated = hypothesiser._compute_validation_matrix(
            mock_detections, pred_means, pred_covars
        )

        expected_matrix = np.array(
            [
                [True, True, True, False],  # State 1
                [True, False, False, False],  # State 2
            ],
            dtype=bool,
        )

        np.testing.assert_array_equal(matrix, expected_matrix)
        assert len(unassociated) == 1  # Detection at (4,4)
        assert list(unassociated)[0].name == "Detection_2"

    def test_compute_likelihood_matrix(
        self, mock_measurement_model, mock_states, mock_detections
    ):
        """Test the likelihood matrix computation."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
            clutter_spatial_density=1e-3,
        )

        num_states = len(mock_states)
        pred_means = [np.array([[0], [0]]), np.array([[10], [10]])]
        pred_covars = [np.eye(2), np.eye(2)]
        validation_matrix = np.array(
            [[True, True, True, False], [True, False, False, False]], dtype=bool
        )

        likelihood_matrix = hypothesiser._compute_likelihood_matrix(
            num_states, mock_detections, validation_matrix, pred_means, pred_covars
        )

        # Check shape
        assert likelihood_matrix.shape == (num_states, len(mock_detections) + 1)
        # Check missed detection likelihood
        assert likelihood_matrix[0, 0] == pytest.approx(1 - 0.9 * 0.99)
        # Check likelihood for validated detection is positive
        assert likelihood_matrix[0, 1] > 0
        # Check likelihood for non-validated detection is zero
        assert likelihood_matrix[0, 3] == 0

    def test_compute_association_combinatorial(self, mock_measurement_model):
        """Test the combinatorial association logic."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
            use_ehm=False,
        )

        # 2 tracks, 1 detection
        validation_matrix = np.array([[True, True], [True, True]], dtype=bool)
        likelihood_matrix = np.array([[0.1, 0.5], [0.2, 0.8]])

        assoc_matrix = hypothesiser._compute_association_combinatorial(
            validation_matrix, likelihood_matrix
        )

        # Probabilities for each track must sum to 1
        assert np.allclose(assoc_matrix.sum(axis=1), 1.0)
        # Check shape
        assert assoc_matrix.shape == (2, 2)

    def test_generate_hypotheses(
        self, mock_measurement_model, mock_states, mock_detections
    ):
        """Test the final hypothesis generation step."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
        )

        association_matrix = np.random.rand(len(mock_states), len(mock_detections) + 1)
        association_matrix /= association_matrix.sum(axis=1, keepdims=True)  # Normalize

        pred_means = [np.zeros((2, 1)) for _ in mock_states]

        hypotheses_dict = hypothesiser._generate_hypotheses(
            mock_states, mock_detections, association_matrix, pred_means
        )

        assert len(hypotheses_dict) == len(mock_states)
        for state in mock_states:
            assert len(hypotheses_dict[state]) == len(mock_detections) + 1
            assert all(
                isinstance(h, ProbabilityHypothesis) for h in hypotheses_dict[state]
            )
            assert np.isclose(sum(h.probability for h in hypotheses_dict[state]), 1.0)

    def test_full_hypothesise_run_no_ehm(
        self, mock_measurement_model, mock_states, mock_detections
    ):
        """Test a full run of the hypothesise method without EHM."""
        hypothesiser = JPDAHypothesiser(
            measurement_model=mock_measurement_model,
            detection_probability=0.9,
            gate_probability=0.99,
            clutter_spatial_density=1e-4,
            use_ehm=False,
        )

        # Mock prediction to control inputs
        predict_returns = [
            (np.array([[0], [0]]), np.eye(2)),
            (np.array([[2], [2]]), np.eye(2)),
        ]

        with patch.object(
            hypothesiser, "_predict_measurement_and_covar", side_effect=predict_returns
        ):
            hypotheses, unassociated = hypothesiser.hypothesise(
                mock_states, mock_detections
            )

        assert isinstance(hypotheses, dict)
        assert isinstance(unassociated, set)
        assert len(hypotheses) == len(mock_states)

        # Detections are at (0,0), (2,2), (4,4)
        # With predictions at (0,0) and (2,2), all detections are within gates
        # Detection 2 at (4,4) has Mahalanobis distance 8.0 from prediction at (2,2)
        # which is less than gate threshold 9.21, so it's associated
        assert len(unassociated) == 0

        # Check probabilities sum to 1 for each state
        for state in mock_states:
            assert np.isclose(sum(h.probability for h in hypotheses[state]), 1.0)
