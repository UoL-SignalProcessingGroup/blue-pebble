"""Defines utility functions.

© Copyright 2025 Joshua J. Wakefield.
Licensed under the MIT License.
"""

from pathlib import Path

import numpy as np


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
