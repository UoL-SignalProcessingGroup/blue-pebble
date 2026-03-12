"""Defines Bellhop-specific utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

Float32Array: TypeAlias = NDArray[np.float32]
Complex64Array: TypeAlias = NDArray[np.complex64]


class ShadeGeometry(TypedDict):
    """Metadata parsed from a Bellhop shade file."""

    plot_type: str
    frequencies: Float32Array
    thetas: Float32Array
    source_x: Float32Array
    source_y: Float32Array
    source_depths: Float32Array
    receiver_depths: Float32Array
    receiver_ranges: Float32Array


ShadeReadResult: TypeAlias = tuple[Complex64Array, ShadeGeometry] | tuple[None, None]


def read_shade_file(
    filename: str | Path, xs: float | None = None, ys: float | None = None
) -> ShadeReadResult:
    """Read a Bellhop shade file (.shd).

    This function parses the binary format of a Bellhop shade file, which contains acoustic
    transmission loss or pressure fields. It can read the entire multi-dimensional field or extract
    a specific slice corresponding to a source position.

    Parameters
    ----------
    filename : str | Path
        The path to the shade file (.shd).
    xs : float | None
        Specific source x-position in kilometers to extract. If None, the entire field is read.
    ys : float | None
        Specific source y-position in kilometers to extract. If None, the entire field is read.

    Returns
    -------
    ShadeReadResult
        A tuple ``(pressure, geometry)`` where ``pressure`` is a complex NumPy
        array containing the pressure field and ``geometry`` stores the
        associated frequencies, depths, ranges, and related dimensions.
        Returns ``(None, None)`` if the file cannot be read.

    """
    try:
        with open(filename, "rb") as f:
            # --- Read Header Information ---
            # The first 4-byte integer is the record length in words.
            record_len_bytes = int(np.fromfile(f, dtype=np.int32, count=1)[0]) * 4

            # Record 2: Plot Type
            f.seek(1 * record_len_bytes)
            plot_type = f.read(10).strip().decode("utf-8")

            # Record 3: Dimensions
            f.seek(2 * record_len_bytes)
            dims = np.fromfile(f, dtype=np.int32, count=7)
            n_freq, n_theta, n_sx, n_sy, n_sd, n_rd, n_rr = (int(v) for v in dims)

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
                source_x = np.linspace(pos_sx[0], pos_sx[1], n_sx, dtype=np.float32)
                f.seek(6 * record_len_bytes)
                pos_sy = np.fromfile(f, dtype=np.float32, count=2)
                source_y = np.linspace(pos_sy[0], pos_sy[1], n_sy, dtype=np.float32)
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
                            f.seek(int(record_num * record_len_bytes))
                            pressure[i_theta, i_sd, i_rd, :] = np.fromfile(
                                f, dtype=np.complex64, count=n_rr
                            )
            else:
                # Read a specific slice for a given source position (xs, ys)
                source_x_array = np.asarray(source_x, dtype=np.float64)
                source_y_array = np.asarray(source_y, dtype=np.float64)
                idx_x = np.abs(np.subtract(source_x_array, float(xs) * 1000.0)).argmin()
                idx_y = np.abs(np.subtract(source_y_array, float(ys) * 1000.0)).argmin()

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
                            file_offset = int(np.asarray(record_num * record_len_bytes).item())
                            f.seek(file_offset)
                            pressure[i_theta, i_sd, i_rd, :] = np.fromfile(
                                f, dtype=np.complex64, count=n_rr
                            )

    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        return None, None
    except Exception as exc:
        print(f"An error occurred while reading the file: {exc}")
        return None, None

    geometry: ShadeGeometry = {
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
