"""
Bilinear ramp equation of state modifier.
"""

import numpy as np

from ..eos import Eos
from ..types import EOSReal


class BilinearRampEos(Eos):
    """
    Bilinear ramp equation of state modifier.
    This class wraps an existing equation of state and applies a
    bilinear pressure ramp.
    """

    def __init__(self, my_eos: Eos, r0: float, a: float, b: float, c: float):
        """
        Initialize the bilinear ramp equation of state.
        Parameters
        ----------
        my_eos : Eos
            The base equation of state to be modified
        r0 : float
            Reference density
        a : float
            Ramp coefficient
        b : float
            Ramp coefficient
        c : float
            Ramp coefficient
        """
        self.eos = my_eos
        self.r0 = r0
        self.a = a
        self.b = b
        self.c = c

        if a == b:
            self.rmid = np.inf
        else:
            self.rmid = self.r0 * (self.a - self.b * self.c) / (self.a - self.b)
        self.pmid = self.a * (self.rmid / self.r0 - 1.0)

    def get_ramp_pressure(self, rho: EOSReal) -> EOSReal:
        """Calculate the ramp pressure."""

        p_ramp = np.piecewise(
            rho,
            [rho < self.r0, (rho >= self.r0) & (rho < self.rmid)],
            [
                0.0,
                lambda rho_in: self.a * (rho_in / self.r0 - 1.0),
                lambda rho_in: self.b * (rho_in / self.r0 - self.c),
            ],
        )

        return p_ramp

    def get_ramp_dpdrho(self, rho: EOSReal) -> EOSReal:
        """Calculate the derivative of ramp pressure with respect to density."""
        dpdr = np.piecewise(
            rho, [rho < self.rmid], [self.a / self.r0, self.b / self.r0]
        )
        return dpdr

    def BulkModulusFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Calculate bulk modulus from density and temperature with ramp.
        """
        p_eos = self.eos.PressureFromDensityTemperature(rho, temperature)
        p_ramp = self.get_ramp_pressure(rho)

        bmod_eos = self.eos.BulkModulusFromDensityTemperature(rho, temperature)

        return np.where(p_eos < p_ramp, rho * self.get_ramp_dpdrho(rho), bmod_eos)

    def PressureFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Calculate ramped pressure from density and temperature.

        Parameters
        ----------
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Ramped pressure value(s)
        """
        p_eos = self.eos.PressureFromDensityTemperature(rho, temperature)
        p_ramp = self.get_ramp_pressure(rho)
        return np.maximum(p_eos, p_ramp)

    def InternalEnergyFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
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
        return self.eos.InternalEnergyFromDensityTemperature(rho, temperature)

    def HelmholtzFreeEnergyFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
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
        return self.eos.HelmholtzFreeEnergyFromDensityTemperature(rho, temperature)

    @property
    def A(self) -> float:
        """
        Get the atomic mass number from the base equation of state.
        """
        return self.eos.A

    @property
    def Z(self) -> float:
        """
        Get the atomic number from the base equation of state.
        """
        return self.eos.Z
