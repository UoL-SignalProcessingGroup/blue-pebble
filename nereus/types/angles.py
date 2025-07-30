"""Defines the Angle and Bearing classes.

This module is a derivative work of Stone Soup, available at
https://github.com/dstl/StoneSoup.

The original work is licensed under the MIT License.
© Crown Copyright 2017-2025 Defence Science and Technology Laboratory UK
© Crown Copyright 2018-2025 Defence Research and Development Canada / Recherche et développement pour la défense Canada
© Copyright 2018-2025 University of Liverpool UK
© Copyright 2020-2025 Fraunhofer FKIE
© Copyright 2020-2025 John Hiles
© Copyright 2020-2025 Riskaware Ltd
© Copyright 2021-2025 Roke Manor Research Ltd UK
© Copyright 2023-2025 Loughborough University UK
© Copyright 2025 Joshua J. Wakefield.
"""  # noqa: E501

from __future__ import annotations

from math import ceil, floor, trunc
from numbers import Real
from typing import Self

import numpy as np


class Angle(Real):
    """A class to represent an angle.

    This class stores an angle value in radians and provides methods for
    arithmetic operations and trigonometric functions. It is designed to be
    subclassed for specific angle types, such as bearings, that require
    modulo arithmetic.

    """

    _value: float

    def __init__(self: Self, value: float) -> None:
        """Initialise the Angle.

        Args:
            value (float): The angle value in radians.

        """
        self._value = self.mod_angle(value)

    @staticmethod
    def mod_angle(value: float) -> float:
        """Apply modulo arithmetic to the angle.

        Note:
            This base implementation does not apply any modulo. It should be
            overridden by subclasses that require angle wrapping (e.g., into
            the range [-pi, pi]).

        Args:
            value (float): The raw angle value in radians.

        Returns:
            float: The angle value after applying modulo logic.

        """
        return float(value)

    @property
    def degrees(self: Self) -> float:
        """Return the angle in degrees."""
        return np.rad2deg(self._value)

    def __hash__(self: Self) -> int:
        """Return a hash of the angle value."""
        return hash(self._value)

    def __add__(self: Self, other: Angle | float) -> Self:
        """Add another angle or a float to this angle."""
        if isinstance(other, Angle):
            other = other._value
        return self.__class__(self._value + other)

    def __radd__(self: Self, other: float) -> Self:
        """Return the result of adding a float to this angle."""
        return self.__add__(other)

    def __sub__(self: Self, other: Angle | float) -> Self:
        """Subtract another angle or a float from this angle."""
        if isinstance(other, Angle):
            other = other._value
        return self.__class__(self._value - other)

    def __rsub__(self: Self, other: float) -> Self:
        """Return the result of subtracting this angle from a float."""
        # other - self is equivalent to -(self - other)
        return self.__class__(other - self._value)

    def __float__(self: Self) -> float:
        """Return the angle value as a float."""
        return self._value

    def __mul__(self: Self, other: float) -> float:
        """Multiply the angle by a float or another angle."""
        if isinstance(other, Angle):
            other = other._value
        return self._value * other

    def __rmul__(self: Self, other: float) -> float:
        """Return the result of multiplying this angle by a float."""
        return self._value * other

    def __str__(self: Self) -> str:
        """Return a string representation of the angle."""
        return str(self._value)

    def __repr__(self: Self) -> str:
        """Return a detailed string representation of the angle."""
        return f"{self.__class__.__name__}({float(self)!r})"

    def __neg__(self: Self) -> Self:
        """Return the negation of the angle."""
        return self.__class__(-self._value)

    def __truediv__(self: Self, other: float) -> float:
        """Divide the angle by a float or another angle."""
        if isinstance(other, Angle):
            other = other._value
        return self._value / other

    def __rtruediv__(self: Self, other: float) -> float:
        """Return the result of dividing a float by this angle."""
        return other / self._value

    def __eq__(self: Self, other: object) -> bool:
        """Check if this angle is equal to another angle or a float."""
        if not isinstance(other, Real):
            return NotImplemented
        return self._value == float(other)

    def __ne__(self: Self, other: object) -> bool:
        """Check if this angle is not equal to another angle or a float."""
        if not isinstance(other, Real):
            return NotImplemented
        return self._value != float(other)

    def __abs__(self: Self) -> Self:
        """Return the absolute value of the angle."""
        return self.__class__(abs(self._value))

    def __le__(self: Self, other: Real) -> bool:
        """Check if this angle is less than or equal to another angle or a float."""
        return self._value <= float(other)

    def __lt__(self: Self, other: Real) -> bool:
        """Check if this angle is less than another angle or a float."""
        return self._value < float(other)

    def __ge__(self: Self, other: Real) -> bool:
        """Check if this angle is greater than or equal to another angle or a float."""
        return self._value >= float(other)

    def __gt__(self: Self, other: Real) -> bool:
        """Check if this angle is greater than another angle or a float."""
        return self._value > float(other)

    def __floor__(self: Self) -> int:
        """Return the floor of the angle value."""
        return floor(self._value)

    def __ceil__(self: Self) -> int:
        """Return the ceiling of the angle value."""
        return ceil(self._value)

    def __floordiv__(self: Self, other: float) -> float:
        """Divide the angle by a float or another angle using floor division."""
        if isinstance(other, Angle):
            other = other._value
        return self._value // other

    def __mod__(self: Self, other: float) -> float:
        """Return the remainder of dividing the angle by a float or another angle."""
        return self._value % other

    def __pos__(self: Self) -> Self:
        """Return the angle unchanged (positive)."""
        return self.__class__(+self._value)

    def __pow__(self: Self, value: float) -> float:
        """Raise the angle to the power of a float or another angle."""
        return pow(self._value, value)

    def __rfloordiv__(self: Self, other: float) -> float:
        """Return the result of floor dividing a float by this angle."""
        return other // self._value

    def __rmod__(self: Self, other: float) -> float:
        """Return the remainder of dividing a float by this angle."""
        return other % self._value

    def __round__(self: Self, ndigits: int | None = None) -> float:
        """Round the angle value to a specified number of decimal places."""
        return round(self._value, ndigits)

    def __rpow__(self: Self, base: float) -> float:
        """Raise a float to the power of this angle."""
        return NotImplemented

    def __trunc__(self: Self) -> int:
        """Return the truncated integer value of the angle."""
        return trunc(self._value)

    def cos(self: Self) -> float:
        """Return the cosine of the angle."""
        return np.cos(self._value)

    def sin(self: Self) -> float:
        """Return the sine of the angle."""
        return np.sin(self._value)

    def tan(self: Self) -> float:
        """Return the tangent of the angle."""
        return np.tan(self._value)

    def rad2deg(self) -> float:
        """Convert the angle from radians to degrees."""
        return np.rad2deg(self._value)

    @classmethod
    def average(cls, angles: list[Angle], weights: list[float] | None = None) -> Self:
        """Calculate the circular mean for a list of angles.

        Args:
            angles (list[Angle]): A list of angles to be averaged.
            weights (list[float] | None, optional): A list of weights for
                each angle. Defaults to None, which implies equal weighting.

        Returns:
            Self: The circular mean of the angles as a new angle object.

        """
        if weights is None:
            # Use mean for unweighted average, which is more direct
            s_bar = np.mean(np.sin(angles), axis=0)
            c_bar = np.mean(np.cos(angles), axis=0)
        else:
            # Use weighted average
            s_bar = np.average(np.sin(angles), axis=0, weights=weights)
            c_bar = np.average(np.cos(angles), axis=0, weights=weights)

        return cls(np.arctan2(s_bar, c_bar))


class Bearing(Angle):
    """A bearing angle class that wraps angles to the range [-pi, pi].

    This class inherits from :class:`~.Angle` and overrides the modulo
    arithmetic to ensure that bearing values are always kept within the
    standard [-pi, pi] radian range.
    """

    @staticmethod
    def mod_angle(value: float) -> float:
        """Wrap an angle to the range [-pi, pi].

        Args:
            value (float): The raw angle value in radians.

        Returns:
            float: The angle value wrapped to the range [-pi, pi].

        """
        return (value + np.pi) % (2.0 * np.pi) - np.pi
