"""
Reader module for equation of state data.

This module provides abstract and concrete classes for reading equation of state
data from various file formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..types import EOSArray

__all__ = ["MaterialData", "MaterialProperties"]


@dataclass
class MaterialData:
    density: EOSArray
    temperature: EOSArray
    energy: EOSArray
    pressure: EOSArray
    helmholtz: EOSArray


@dataclass
class MaterialProperties:
    atomic_number: float
    atomic_mass: float
    normal_density: float
    solid_bulk_modulus: float
    exchange_coefficient: float


class Reader(ABC):
    """
    Abstract base class for equation of state data readers.

    This class defines the interface that all reader implementations must follow.
    Readers are responsible for parsing equation of state data from specific
    file formats.
    """

    @abstractmethod
    def read_ion_data(
        self,
    ) -> MaterialData:
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
    def read_electron_data(
        self,
    ) -> MaterialData:
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
    def read_total_data(
        self,
    ) -> MaterialData:
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
    def read_material_properties(self) -> MaterialProperties:
        """
        Read material data.

        Returns
        -------
        dict
            A dictionary containing material properties such as atomic number,
            atomic mass, etc.
        """
        pass
