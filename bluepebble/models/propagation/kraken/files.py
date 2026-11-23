"""Readers for the binary KRAKEN mode file and FIELD shade file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HalfspaceRecord:
    """Top or bottom boundary of one mode set, as KRAKEN wrote it.

    Attributes
    ----------
    boundary : str
        Boundary-condition letter, e.g. ``'V'`` (vacuum) or ``'A'`` (acoustic halfspace).
    cp, cs : complex
        Compressional and shear speed, with attenuation in the imaginary part.
    density_g_cm3 : float
        Density.
    depth_m : float
        Depth of the boundary.

    """

    boundary: str
    cp: complex
    cs: complex
    density_g_cm3: float
    depth_m: float


@dataclass(frozen=True)
class ModeFileSet:
    """The modes at one frequency.

    Attributes
    ----------
    frequency_hz : float
        Frequency.
    k : numpy.ndarray
        Horizontal wavenumbers, shape ``(M,)``, complex with ``Im k <= 0``.
    phi : numpy.ndarray
        Mode shapes at the tabulated depths, shape ``(M, Nz)`` complex64.
    top, bottom : HalfspaceRecord
        Boundaries.

    """

    frequency_hz: float
    k: np.ndarray
    phi: np.ndarray
    top: HalfspaceRecord
    bottom: HalfspaceRecord


@dataclass(frozen=True)
class ModeFile:
    """Every mode set in a single-profile KRAKEN ``.mod`` file.

    Attributes
    ----------
    title : str
        File title; starts with ``KRAKENC`` or ``KRAKEN``.
    depths_m : numpy.ndarray
        Tabulation depths, shape ``(Nz,)``.
    sets : list of ModeFileSet
        One per frequency, in file order.

    """

    title: str
    depths_m: np.ndarray
    sets: list[ModeFileSet]

    @property
    def is_krakenc(self) -> bool:
        """Whether KRAKENC wrote the file, which changes the halfspace tail rule."""
        return self.title.upper().startswith("KRAKENC")


def _halfspace(raw: bytes, offset: int) -> tuple[HalfspaceRecord, int]:
    boundary = raw[offset : offset + 1].decode("ascii", errors="replace")
    values = np.frombuffer(raw, dtype=np.float32, count=6, offset=offset + 1)
    record = HalfspaceRecord(
        boundary=boundary,
        cp=complex(values[0], values[1]),
        cs=complex(values[2], values[3]),
        density_g_cm3=float(values[4]),
        depth_m=float(values[5]),
    )
    return record, offset + 1 + 24


def read_mode_file(path: str | Path) -> ModeFile:
    """Read every frequency's modes from a single-profile KRAKEN or KRAKENC ``.mod`` file.

    The record layout follows ``KrakenField/ReadModes.f90``. Records are ``4 * LRecL``
    bytes, with ``LRecL`` the file's first word. Each frequency's set starts with its mode
    count, then the halfspace record, one record per mode shape, and the wavenumbers
    packed ``LRecL / 2`` complex values to a record.

    Parameters
    ----------
    path : path-like
        The ``.mod`` file.

    Returns
    -------
    ModeFile
        Every mode set in the file.

    Raises
    ------
    ValueError
        If the file is truncated or holds an elastic problem whose modes need extracting.

    """
    raw = Path(path).read_bytes()
    words = int(np.frombuffer(raw, dtype=np.int32, count=1)[0])
    record = 4 * words

    def at(index: int) -> int:
        return index * record

    title = raw[4:84].decode("ascii", errors="replace").strip()
    num_freq, num_media, num_tab, num_mat = (
        int(v) for v in np.frombuffer(raw, dtype=np.int32, count=4, offset=84)
    )
    materials = []
    for medium in range(num_media):
        offset = at(1) + medium * 12
        materials.append(raw[offset + 4 : offset + 12].decode("ascii", errors="replace").strip())
    if any(m.upper().startswith("ELASTIC") for m in materials):
        raise ValueError("Elastic media in the mode file are not supported by this reader")
    frequencies = np.frombuffer(raw, dtype=np.float64, count=num_freq, offset=at(3))
    depths = np.frombuffer(raw, dtype=np.float32, count=num_tab, offset=at(4)).astype(float)

    sets: list[ModeFileSet] = []
    index = 5
    for frequency in frequencies:
        if at(index) + 4 > len(raw):
            raise ValueError(f"{path} is truncated at frequency {frequency:g} Hz")
        num_modes = int(np.frombuffer(raw, dtype=np.int32, count=1, offset=at(index))[0])
        top, offset = _halfspace(raw, at(index + 1))
        bottom, _ = _halfspace(raw, offset)
        if num_modes > 0:
            first_mode = at(index + 2)
            # Mode records are evenly spaced, so read them as a strided view.
            block = np.frombuffer(raw, dtype=np.uint8, count=num_modes * record, offset=first_mode)
            phi = (
                block.reshape(num_modes, record)[:, : 8 * num_mat]
                .copy()
                .view(np.complex64)[:, :num_tab]
            )
            # Wavenumber records are filled contiguously, LRecL / 2 values each.
            k = np.frombuffer(
                raw, dtype=np.complex64, count=num_modes, offset=at(index + 2 + num_modes)
            ).astype(np.complex128)
            index += 3 + num_modes + (2 * num_modes - 1) // words
        else:
            phi = np.zeros((0, num_tab), dtype=np.complex64)
            k = np.zeros(0, dtype=np.complex128)
            # Fortran's integer division of -1 by LRecL is 0, unlike Python's.
            index += 3
        sets.append(
            ModeFileSet(frequency_hz=float(frequency), k=k, phi=phi, top=top, bottom=bottom)
        )
    return ModeFile(title=title, depths_m=depths, sets=sets)


@dataclass(frozen=True)
class ShadeFile:
    """Complex pressure from a FIELD ``.shd`` file.

    Attributes
    ----------
    frequencies_hz : numpy.ndarray
        Frequencies, shape ``(F,)``.
    source_depths_m, receiver_depths_m, ranges_m : numpy.ndarray
        Grid axes.
    pressure : numpy.ndarray
        Shape ``(F, n_source_depths, n_receiver_depths, n_ranges)`` complex64.

    """

    frequencies_hz: np.ndarray
    source_depths_m: np.ndarray
    receiver_depths_m: np.ndarray
    ranges_m: np.ndarray
    pressure: np.ndarray


def read_shade_file(path: str | Path) -> ShadeFile:
    """Read a FIELD shade file, written by ``misc/RWSHDFile.f90``.

    Parameters
    ----------
    path : path-like
        The ``.shd`` file.

    Returns
    -------
    ShadeFile
        Pressure for every frequency in the file.

    """
    raw = Path(path).read_bytes()
    words = int(np.frombuffer(raw, dtype=np.int32, count=1)[0])
    record = 4 * words

    def at(index: int) -> int:
        return index * record

    num_freq, num_theta, num_sx, num_sy, num_sz, num_rz, num_rr = (
        int(v) for v in np.frombuffer(raw, dtype=np.int32, count=7, offset=at(2))
    )
    frequencies = np.frombuffer(raw, dtype=np.float64, count=num_freq, offset=at(3)).copy()
    source_depths = np.frombuffer(raw, dtype=np.float32, count=num_sz, offset=at(7)).astype(float)
    receiver_depths = np.frombuffer(raw, dtype=np.float32, count=num_rz, offset=at(8)).astype(
        float
    )
    ranges = np.frombuffer(raw, dtype=np.float64, count=num_rr, offset=at(9)).copy()
    if num_theta != 1 or num_sx != 1 or num_sy != 1:
        raise ValueError("Only single-bearing, single-source-position shade files are supported")

    num_records = num_freq * num_sz * num_rz
    block = np.frombuffer(raw, dtype=np.uint8, count=num_records * record, offset=at(10))
    pressure = (
        block.reshape(num_records, record)[:, : 8 * num_rr]
        .copy()
        .view(np.complex64)
        .reshape(num_freq, num_sz, num_rz, num_rr)
    )
    return ShadeFile(
        frequencies_hz=frequencies,
        source_depths_m=source_depths,
        receiver_depths_m=receiver_depths,
        ranges_m=ranges,
        pressure=pressure,
    )
