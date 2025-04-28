"""
Reader module for equation of state data.

This module provides abstract and concrete classes for reading equation of state
data from various file formats.
"""

from abc import ABC, abstractmethod


class Reader(ABC):
    """
    Abstract base class for equation of state data readers.

    This class defines the interface that all reader implementations must follow.
    Readers are responsible for parsing equation of state data from specific
    file formats.
    """

    @abstractmethod
    def read_ion_data(self):
        """
        Read ion component equation of state data.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the ion component.
        """
        pass

    @abstractmethod
    def read_electron_data(self):
        """
        Read electron component equation of state data.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the electron component.
        """
        pass

    @abstractmethod
    def read_total_data(self):
        """
        Read total (combined) equation of state data.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the total equation of state.
        """
        pass

    @abstractmethod
    def read_material_data(self):
        """
        Read material data.

        Returns
        -------
        dict
            A dictionary containing material properties such as atomic number,
            atomic mass, etc.
        """
        pass
