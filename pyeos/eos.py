"""
Core module defining the abstract base class for all equation of state models.
"""

from abc import ABC, abstractmethod

from .types import EOSReal


class Eos(ABC):
    """
    Abstract base class for all equation of state (EOS) models.

    This class defines the interface that all EOS implementations must follow.
    It provides methods to calculate thermodynamic properties such as internal energy,
    pressure, and Helmholtz free energy from density and temperature.

    All EOS implementations should inherit from this class and implement
    the abstract methods.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the equation of state model."""
        pass

    @abstractmethod
    def InternalEnergyFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Calculate internal energy from density and temperature.

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
        pass

    @abstractmethod
    def PressureFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Calculate pressure from density and temperature.

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
        pass

    @abstractmethod
    def HelmholtzFreeEnergyFromDensityTemperature(
        self, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
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
        pass

    @property
    @abstractmethod
    def A(self) -> float:
        """
        Get the atomic mass number.

        Returns
        -------
        float
            Atomic mass number
        """
        pass

    @property
    @abstractmethod
    def Z(self) -> float:
        """
        Get the atomic number.

        Returns
        -------
        float
            Atomic number
        """
        pass
