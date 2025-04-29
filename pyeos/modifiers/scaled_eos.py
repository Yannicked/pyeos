"""
Scaled equation of state modifier.

This module provides a modifier that scales the output of an existing
equation of state model by a scaling function.
"""

from typing import Callable

from ..eos import Eos
from ..types import EOSReal


class ScaledEos(Eos):
    """
    Scaled equation of state modifier.

    This class wraps an existing equation of state and applies a scaling
    function to its outputs. This can be used to modify the behavior of
    an equation of state without changing its implementation.
    """

    def __init__(self, my_eos: Eos, scale_fn: Callable[[EOSReal, EOSReal], EOSReal]):
        """
        Initialize the scaled equation of state.

        Parameters
        ----------
        my_eos : Eos
            The base equation of state to be scaled
        scale_fn : Callable[[EOSReal, EOSReal], EOSReal]
            A function that takes density and temperature and returns a scaling factor
        """
        self.eos = my_eos
        self.scale_fn = scale_fn

    def InternalEnergyFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Calculate scaled internal energy from density and temperature.

        Parameters
        ----------
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Scaled internal energy value(s)
        """
        return self.eos.InternalEnergyFromDensityTemperature(
            rho, temperature
        ) * self.scale_fn(rho, temperature)

    def PressureFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Calculate scaled pressure from density and temperature.

        Parameters
        ----------
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Scaled pressure value(s)
        """
        return self.eos.PressureFromDensityTemperature(
            rho, temperature
        ) * self.scale_fn(rho, temperature)

    def HelmholtzFreeEnergyFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Calculate scaled Helmholtz free energy from density and temperature.

        Parameters
        ----------
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Scaled Helmholtz free energy value(s)
        """
        return self.eos.HelmholtzFreeEnergyFromDensityTemperature(
            rho, temperature
        ) * self.scale_fn(rho, temperature)

    @property
    def A(self) -> float:
        """
        Get the atomic mass number from the base equation of state.

        Returns
        -------
        float
            Atomic mass number
        """
        return self.eos.A

    @property
    def Z(self) -> float:
        """
        Get the atomic number from the base equation of state.

        Returns
        -------
        float
            Atomic number
        """
        return self.eos.Z
