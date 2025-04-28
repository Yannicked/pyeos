"""
Implementation of the ideal gamma-law equation of state.

This module provides an implementation of the ideal gamma-law equation of state,
which is commonly used in gas dynamics and astrophysical simulations.
"""

import numpy as np
from scipy.constants import gas_constant

from ..eos import Eos
from ..types import EOSReal


class IdealGamma(Eos):
    """
    Ideal gamma-law equation of state.

    This class implements an ideal gas equation of state with a constant
    adiabatic index (gamma). It is commonly used for simple gas dynamics
    and as a baseline for more complex equation of state models.
    """

    def __init__(self, gamma, A, Z) -> None:
        """
        Initialize the ideal gamma-law equation of state.

        Parameters
        ----------
        gamma : float
            Adiabatic index (ratio of specific heats)
        A : float
            Atomic mass number
        Z : float
            Atomic number
        """
        self.gamma = gamma
        self.const = gas_constant / A * 1e7
        self._A = A
        self._Z = Z

    def InternalEnergyFromDensityTemperature(self, rho, temperature) -> EOSReal:
        """
        Calculate internal energy from density and temperature.

        For an ideal gas, internal energy depends only on temperature.

        Parameters
        ----------
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Internal energy value(s)
        """
        return np.maximum(0.0, self.const / (self.gamma - 1) * temperature)

    def PressureFromDensityTemperature(self, rho, temperature) -> EOSReal:
        """
        Calculate pressure from density and temperature.

        For an ideal gas, P = ρ·R·T/A.

        Parameters
        ----------
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Pressure value(s)
        """
        return np.maximum(0.0, self.const * rho * temperature)

    def HelmholtzFreeEnergyFromDensityTemperature(self, rho, temperature) -> EOSReal:
        """
        Calculate Helmholtz free energy from density and temperature.

        Parameters
        ----------
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Helmholtz free energy value(s)
        """
        return self.InternalEnergyFromDensityTemperature(rho, temperature) * np.log(
            rho ** (self.gamma - 1) / temperature
        )

    @property
    def A(self) -> float:
        """
        Get the atomic mass number.

        Returns
        -------
        float
            Atomic mass number
        """
        return self._A

    @property
    def Z(self) -> float:
        """
        Get the atomic number.

        Returns
        -------
        float
            Atomic number
        """
        return self._Z
