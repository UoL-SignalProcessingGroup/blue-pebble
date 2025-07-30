"""Defines utility functions.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from pathlib import Path

import numpy as np
from scipy.stats import ncx2


def cartesian_to_bearing_state(state: np.ndarray) -> np.ndarray:
    """Convert a Cartesian position and velocity state to bearing and bearing rate.

    Args:
        state (np.ndarray): A 1D array of shape (4,) representing [x, vx, y, vy].

    Returns:
        np.ndarray: A 1D array of shape (2,) representing [bearing, bearing_rate] in
        radians.

    """
    x, vx, y, vy = state

    # Compute bearing (theta) in radians
    theta = np.arctan2(y, x)

    # Compute bearing rate (dtheta/dt)
    denominator = x**2 + y**2
    if denominator == 0:
        return np.array([theta, 0.0])  # Bearing rate is 0 at the origin
    theta_dot = (x * vy - y * vx) / denominator

    return np.array([theta, theta_dot])


def p_fa_from_snr_thresh(snr_thresh_db: float) -> float:
    """Calculate the probability of false alarm given a detection threshold in SNR (dB).

    Args:
        snr_thresh_db (float): Detection threshold in decibels.

    Returns:
        float: Probability of false alarm.

    Notes:
        The probability of false alarm is calculated using the formula:
            p_fa = exp(-SNR_linear / 2)
        This formula is derived assuming:
            SNR = A_T^2 / sigma^2
        and that the amplitude detection threshold A_T lies in a Rayleigh-distributed
        noise background with zero mean Gaussian components.

    References:
        M. A. Ainslie, Principles of Sonar Performance Modelling, Springer, 2010.
        See Equation (7.7).

    """
    snr_linear = 10 ** (snr_thresh_db / 10)
    return np.exp(-snr_linear / 2)


def p_d_rician(snr_db: float, p_fa: float) -> float:
    """Calculate the probability of detection (P_d) under a Rician signal model.

    Args:
        snr_db (float): Signal-to-noise ratio (SNR) in decibels.
        p_fa (float): Probability of false alarm.

    Returns:
        float: Probability of detection (P_d).

    Raises:
        ValueError: If p_fa is not between 0 and 1.

    Notes:
        The probability of detection for a Rician-distributed signal in Gaussian noise
        is approximated using the noncentral chi-squared survival function.
        Specifically, the Marcum Q-function is approximated as:

            P_d ≈ Q₁(√(2 * SNR), √(-2 * log(p_fa)))
                ≈ P(X > b²),   where X ~ χ²(df=2, nc=2 * SNR)

        This approximation interprets the detection problem in terms of the cumulative
        distribution function of the noncentral chi-squared distribution with 2 degrees
        of freedom and noncentrality parameter 2 * SNR.

    References:
        M. A. Ainslie, Principles of Sonar Performance Modelling, Springer, 2010.
        See Equation (7.12).

    """
    if not (0 < p_fa < 1):
        raise ValueError("Probability of false alarm (p_fa) must be between 0 and 1.")

    snr_linear = 10 ** (snr_db / 10)
    a = np.sqrt(2 * snr_linear)
    b = np.sqrt(-2 * np.log(p_fa))

    # Marcum Q1(a, b) ≈ P(X > b^2), X ~ chi2(df=2, nc=a^2)
    return ncx2.sf(b**2, df=2, nc=a**2)


def p_d_rayleigh(snr_db: float, p_fa: float) -> float:
    """Calculate the probability of detection (P_d) under a Rayleigh model.

    Args:
        snr_db (float): Signal-to-noise ratio in decibels.
        p_fa (float): Probability of false alarm.

    Returns:
        float: Probability of detection.

    Raises:
        ValueError: If p_fa is not between 0 and 1.

    Notes:
        The probability of detection for a Rayleigh-distributed signal in Gaussian noise
        is given by:

            P_d = (P_fa)^(1 / (1 + SNR_linear))

        where SNR_linear is the signal-to-noise ratio in linear scale.

        This model assumes that the received signal is a Rayleigh-distributed random
        variable and the detection threshold is based on amplitude detection in white
        Gaussian noise.

    References:
        M. A. Ainslie, Principles of Sonar Performance Modelling, Springer, 2010.
        See Equation (7.34).

    """
    if not (0 < p_fa < 1):
        raise ValueError("Probability of false alarm (p_fa) must be between 0 and 1.")

    snr_linear = 10 ** (snr_db / 10)
    return p_fa ** (1 / (1 + snr_linear))


def gaussian_regularisation(
    state_vectors: np.ndarray,
    alpha: float = 0.5,
    angle_indices: list[int] | None = None,
) -> np.ndarray:
    """Regularise the state vectors by adding Gaussian noise.

    This function applies noise based on the data's empirical covariance
    and uses Silverman's rule of thumb to determine the noise bandwidth.
    It can also handle angular data by wrapping specified components to the
    [-pi, pi] range.

    Args:
        state_vectors (np.ndarray): A 2D array of shape (d, n) where 'd' is
            the dimension of the state and 'n' is the number of samples.
        alpha (float, optional): A scaling factor for Silverman's rule of
            thumb. Defaults to 0.5.
        angle_indices (list[int] | None, optional): A list of indices for the
            rows that represent angles in radians. These will be wrapped to
            the range [-pi, pi]. Defaults to None.

    Returns:
        np.ndarray: The regularised state vectors.

    Usage:
        Using functools.partial
        This creates a new function with the 'angle_indices' argument "frozen" in.
        regularisation_callable = partial(
            gaussian_regularisation,
            angle_indices=angle_indices_for_state
        )

    """
    if state_vectors.ndim != 2:
        raise ValueError("state_vectors must be a 2D array.")

    d, n = state_vectors.shape

    # Calculate the bandwidth using Silverman's rule of thumb
    h = alpha * (4 / ((d + 2) * n)) ** (1 / (d + 4))

    # Calculate the empirical covariance matrix
    # Note: np.cov expects rows to be variables, which matches our input shape
    cov = np.cov(state_vectors)
    cov += np.eye(d) * 1e-6  # Add a small value for numerical stability

    # Generate scaled, correlated multivariate noise
    # The Cholesky decomposition helps generate noise with the same covariance
    # as the input data.
    chol = np.linalg.cholesky(cov)
    noise = h * chol @ np.random.randn(d, n)

    # Perturb the original state vectors
    perturbed_vectors = state_vectors + noise

    # If there are angular components, wrap them to the correct range
    if angle_indices:
        for i in angle_indices:
            perturbed_vectors[i, :] = (perturbed_vectors[i, :] + np.pi) % (
                2 * np.pi
            ) - np.pi

    return perturbed_vectors


def read_shade_file(
    filename: str | Path, xs: float | None = None, ys: float | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a Bellhop shade file (.shd).

    The file contains acoustic transmission loss or pressure fields.

    Args:
        filename (str | Path): The path to the shade file (.shd).
        xs (float | None): Specific source x-position (in km) to extract. If None,
            the entire field is read.
        ys (float | None): Specific source y-position (in km) to extract. If None,
            the entire field is read.

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray]]:
            - pressure: A complex NumPy array with the pressure field.
            - geometry: A dictionary with dimension information (frequencies, depths,
                ranges, etc.).

    """
    try:
        with open(filename, "rb") as f:
            # --- Read Header Information ---
            # The first 4-byte integer is the record length in words.
            record_len_bytes = np.fromfile(f, dtype=np.int32, count=1)[0] * 4

            # Record 2: Plot Type
            f.seek(1 * record_len_bytes)
            plot_type = f.read(10).strip().decode("utf-8")

            # Record 3: Dimensions
            f.seek(2 * record_len_bytes)
            dims = np.fromfile(f, dtype=np.int32, count=7)
            n_freq, n_theta, n_sx, n_sy, n_sd, n_rd, n_rr = dims

            # Record 4: Frequencies
            f.seek(3 * record_len_bytes)
            frequencies = np.fromfile(f, dtype=np.float32, count=n_freq)

            # Record 5: Thetas (bearing angles)
            f.seek(4 * record_len_bytes)
            thetas = np.fromfile(f, dtype=np.float32, count=n_theta)

            # Record 6 & 7: Source X and Y positions
            f.seek(5 * record_len_bytes)
            if "TL" in plot_type:  # Compressed format
                pos_sx = np.fromfile(f, dtype=np.float32, count=2)
                source_x = np.linspace(pos_sx[0], pos_sx[1], n_sx)
                f.seek(6 * record_len_bytes)
                pos_sy = np.fromfile(f, dtype=np.float32, count=2)
                source_y = np.linspace(pos_sy[0], pos_sy[1], n_sy)
            else:  # Standard format
                source_x = np.fromfile(f, dtype=np.float32, count=n_sx)
                f.seek(6 * record_len_bytes)
                source_y = np.fromfile(f, dtype=np.float32, count=n_sy)

            # Record 8: Source Depths
            f.seek(7 * record_len_bytes)
            source_depths = np.fromfile(f, dtype=np.float32, count=n_sd)

            # Record 9: Receiver Depths
            f.seek(8 * record_len_bytes)
            receiver_depths = np.fromfile(f, dtype=np.float32, count=n_rd)

            # Record 10: Receiver Ranges
            f.seek(9 * record_len_bytes)
            receiver_ranges = np.fromfile(f, dtype=np.float32, count=n_rr)

            # --- Prepare for reading pressure data ---
            n_rcvrs_per_range = 1 if plot_type == "irregular" else n_rd
            pressure_shape = (n_theta, n_sd, n_rcvrs_per_range, n_rr)
            pressure = np.zeros(pressure_shape, dtype=np.complex64)

            # --- Read Pressure Field ---
            # This section reads the main data payload, which can be very large.
            # The file format requires seeking to each data slice individually.
            if xs is None or ys is None:
                # Read the entire 4D or 5D pressure field
                for i_theta in range(n_theta):
                    for i_sd in range(n_sd):
                        for i_rd in range(n_rcvrs_per_range):
                            record_num = (
                                10
                                + (i_theta * n_sd * n_rcvrs_per_range)
                                + (i_sd * n_rcvrs_per_range)
                                + i_rd
                            )
                            f.seek(record_num * record_len_bytes)
                            pressure[i_theta, i_sd, i_rd, :] = np.fromfile(
                                f, dtype=np.complex64, count=n_rr
                            )
            else:
                # Read a specific slice for a given source position (xs, ys)
                idx_x = np.abs(source_x - xs * 1000).argmin()
                idx_y = np.abs(source_y - ys * 1000).argmin()

                for i_theta in range(n_theta):
                    for i_sd in range(n_sd):
                        for i_rd in range(n_rcvrs_per_range):
                            # The record number calculation for 5D fields
                            # (x, y, theta, z, r)
                            base_offset = 10
                            slice_size_per_src_pos = n_theta * n_sd * n_rcvrs_per_range
                            sx_offset = idx_x * n_sy * slice_size_per_src_pos
                            sy_offset = idx_y * slice_size_per_src_pos
                            theta_offset = i_theta * n_sd * n_rcvrs_per_range
                            sd_offset = i_sd * n_rcvrs_per_range
                            record_num = (
                                base_offset
                                + sx_offset
                                + sy_offset
                                + theta_offset
                                + sd_offset
                                + i_rd
                            )
                            f.seek(record_num * record_len_bytes)
                            pressure[i_theta, i_sd, i_rd, :] = np.fromfile(
                                f, dtype=np.complex64, count=n_rr
                            )

    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None

    geometry = {
        "plot_type": plot_type,
        "frequencies": frequencies,
        "thetas": thetas,
        "source_x": source_x,
        "source_y": source_y,
        "source_depths": source_depths,
        "receiver_depths": receiver_depths,
        "receiver_ranges": receiver_ranges,
    }

    return pressure, geometry
