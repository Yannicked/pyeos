"""
IONMIX table interpolation equation of state.

This module provides an implementation of an equation of state that
interpolates over IONMIX format tables using xarray.
"""

import xarray as xr

from ..reader.ionmix import IonmixReader
from .xarray_eos import XArrayEos


class IonmixEos(XArrayEos):
    """
    IONMIX table interpolation equation of state.

    This class implements an equation of state that interpolates over
    IONMIX format tables using xarray. It provides a way to use tabulated
    equation of state data with efficient interpolation.
    """

    def __init__(
        self,
        file_name: str,
        atomic_number: float,
        atomic_mass: float,
        component: str = "total",
        twot: bool = False,
    ) -> None:
        """
        Initialize the IONMIX table interpolation equation of state.

        Parameters
        ----------
        file_name : str
            Path to the IONMIX format file
        atomic_number : float
            Atomic number (Z) of the material.
        atomic_mass : float
            Atomic mass (A) of the material in AMU.
        component : str, optional
            Which component to use ('total', 'ion', or 'electron'),
            by default "total"
        twot : bool, optional
            Flag for two-temperature data, by default False.
        """
        self.file_name = file_name
        self.component = component
        self.reader = IonmixReader(file_name, atomic_number, atomic_mass, twot=twot)

        # Read material data
        material_properties = self.reader.read_material_properties()
        self._Z = material_properties.atomic_number
        self._A = material_properties.atomic_mass

        # Read EOS data based on component
        if component == "total":
            material_data = self.reader.read_total_data()
        elif component == "ion":
            material_data = self.reader.read_ion_data()
        elif component == "electron":
            material_data = self.reader.read_electron_data()
        else:
            raise ValueError(
                f"Invalid component: {component}. "
                "Must be 'total', 'ion', or 'electron'."
            )

        # Create xarray DataArrays for each property
        self.energy_da = xr.DataArray(
            material_data.energy,
            dims=["temperature", "density"],
            coords={
                "temperature": material_data.temperature,
                "density": material_data.density,
            },
            name="internal_energy",
        )

        self.pressure_da = xr.DataArray(
            material_data.pressure,
            dims=["temperature", "density"],
            coords={
                "temperature": material_data.temperature,
                "density": material_data.density,
            },
            name="pressure",
        )

        self.helmholtz_da = xr.DataArray(
            material_data.helmholtz,
            dims=["temperature", "density"],
            coords={
                "temperature": material_data.temperature,
                "density": material_data.density,
            },
            name="helmholtz_free_energy",
        )

        # Create a dataset for convenient access
        self.dataset = xr.Dataset(
            {
                "internal_energy": self.energy_da,
                "pressure": self.pressure_da,
                "helmholtz_free_energy": self.helmholtz_da,
            }
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
