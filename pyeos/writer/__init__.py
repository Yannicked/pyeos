"""
Writer module for equation of state data.

This module provides abstract and concrete classes for writing equation of state
data to various file formats.
"""

from abc import ABC, abstractmethod

from ..types import EOSArray


class Writer(ABC):
    """
    Abstract base class for equation of state data writers.

    This class defines the interface that all writer implementations must follow.
    Writers are responsible for outputting equation of state data to specific
    file formats.
    """

    @abstractmethod
    def write_ion_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """
        Write ion component equation of state data.

        Parameters
        ----------
        density : EOSArray
            Array of density values
        temperature : EOSArray
            Array of temperature values
        energy : EOSArray
            Array of internal energy values
        pressure : EOSArray
            Array of pressure values
        helmholtz : EOSArray
            Array of Helmholtz free energy values
        """
        pass

    @abstractmethod
    def write_electron_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """
        Write electron component equation of state data.

        Parameters
        ----------
        density : EOSArray
            Array of density values
        temperature : EOSArray
            Array of temperature values
        energy : EOSArray
            Array of internal energy values
        pressure : EOSArray
            Array of pressure values
        helmholtz : EOSArray
            Array of Helmholtz free energy values
        """
        pass

    @abstractmethod
    def write_total_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """
        Write total (combined) equation of state data.

        Parameters
        ----------
        density : EOSArray
            Array of density values
        temperature : EOSArray
            Array of temperature values
        energy : EOSArray
            Array of internal energy values
        pressure : EOSArray
            Array of pressure values
        helmholtz : EOSArray
            Array of Helmholtz free energy values
        """
        pass
