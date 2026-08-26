"""Bearing-only target motion analysis using a particle filter."""

import numpy as np

from stonesoup.base import Base, Property
from datetime import datetime

from stonesoup.models.measurement.nonlinear import Cartesian2DToBearing
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.array import StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState

from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import ESSResampler
from stonesoup.updater.particle import ParticleUpdater

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track


class BearingOnlyTargetMotionAnalysis(Base):
    start_time: datetime = Property(doc="Start time of the filter")
    platform: object = Property(doc="TowedArraySensor object")
    theta_0: float = Property(doc="Initial bearing measurement in radians")
    n_particles: int = Property(default=1000, doc="Number of particles")
    q_x: float = Property(default=0.001, doc="Process noise for x")
    q_y: float = Property(default=0.001, doc="Process noise for y")
    r_min: float = Property(default=500.0, doc="Minimum initial range in metres")
    r_max: float = Property(default=8000.0, doc="Maximum initial range in metres")
    bearing_noise_std: float = Property(default=5.0, doc="Bearing noise std in degrees")

    def tma_pf_init(self):
        # Build transition model
        self.transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(self.q_x), ConstantVelocity(self.q_y)]
        )

        # Get initial platform position
        platform_state = self.platform.get_platform_state_at(self.start_time)
        ref_pos = np.mean(platform_state.array.state_vector, axis=1)
        platform_x = ref_pos[0]
        platform_y = ref_pos[1]

        # Initialise particles along the bearing line
        r = np.random.uniform(self.r_min, self.r_max, self.n_particles)
        x = platform_x + r * np.cos(self.theta_0)
        y = platform_y + r * np.sin(self.theta_0)

        # Sample velocities uniformly over speed and course
        speed = np.random.uniform(0, 15, self.n_particles)
        course = np.random.uniform(0, 2 * np.pi, self.n_particles)
        vx = speed * np.cos(course)
        vy = speed * np.sin(course)

        # Stack into (4, N) state vectors [x, xdot, y, ydot]
        samples = np.vstack([x, vx, y, vy])

        self.prior = ParticleState(
            state_vector=StateVectors(samples),
            weight=np.array([Probability(1 / self.n_particles)] * self.n_particles),
            timestamp=self.start_time
        )

        self.predictor = ParticlePredictor(self.transition_model)
        self.resampler = ESSResampler()
        self.track = Track()

    def tma_pf_step(self, detections, timestamp):

        platform_state = self.platform.get_platform_state_at(timestamp)
        ref_pos = np.mean(platform_state.array.state_vector, axis=1)
        platform_x = ref_pos[0]
        platform_y = ref_pos[1]

        # Build measurement model
        self.measurement_model = Cartesian2DToBearing(
            ndim_state=4,
            mapping=(0, 2),
            noise_covar=np.array([[np.radians(self.bearing_noise_std) ** 2]]),
            translation_offset=np.array([[platform_x], [platform_y]])
        )

        self.updater = ParticleUpdater(self.measurement_model, self.resampler)
        
        prediction = self.predictor.predict(self.prior, timestamp=timestamp)

        if detections:
            # Associate with the detection closest to the predicted bearing,
            # since bearing-only TMA has no gating/hypothesiser of its own.
            pred_bearing = float(self.measurement_model.function(
                prediction, noise=False).mean())
            detection = min(detections,
                        key=lambda d: abs(float(d.state_vector[0]) - pred_bearing))
            hypothesis = SingleHypothesis(prediction, detection)
            post = self.updater.update(hypothesis)
            self.track.append(post)
            self.prior = post
        else:
            self.prior = prediction
            self.track.append(prediction)
