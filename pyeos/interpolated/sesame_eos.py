"""
SESAME table interpolation equation of state.

This module provides an implementation of an equation of state that
interpolates over SESAME format tables using xarray.
"""

import xarray as xr

from ..reader.sesame import SesameReader
from ..types import EOSReal
from .xarray_eos import XArrayEos


class SesameEos(XArrayEos):
    """
    SESAME table interpolation equation of state.

    This class implements an equation of state that interpolates over
    SESAME format tables using xarray. It provides a way to use tabulated
    equation of state data with efficient interpolation.
    """

    def __init__(self, file_name, component="total") -> None:
        """
        Initialize the SESAME table interpolation equation of state.

        Parameters
        ----------
        file_name : str
            Path to the SESAME format file
        component : str, optional
            Which component to use ('total', 'ion', or 'electron'),
            by default "total"
        """
        self.file_name = file_name
        self.component = component
        self.reader = SesameReader(file_name)

        # Read material data
        material_data = self.reader.read_material_data()
        self._Z = material_data["atomic_number"]
        self._A = material_data["atomic_mass"]

        # Read EOS data based on component
        if component == "total":
            density, temperature, energy, pressure, helmholtz = (
                self.reader.read_total_data()
            )
        elif component == "ion":
            density, temperature, energy, pressure, helmholtz = (
                self.reader.read_ion_data()
            )
        elif component == "electron":
            density, temperature, energy, pressure, helmholtz = (
                self.reader.read_electron_data()
            )
        else:
            raise ValueError(
                f"Invalid component: {component}. "
                "Must be 'total', 'ion', or 'electron'."
            )

        # Create xarray DataArrays for each property
        # Note: SESAME tables store data with temperature as the first dimension
        # and density as the second dimension
        self.energy_da = xr.DataArray(
            energy,
            dims=["temperature", "density"],
            coords={"temperature": temperature, "density": density},
            name="internal_energy",
        )

        self.pressure_da = xr.DataArray(
            pressure,
            dims=["temperature", "density"],
            coords={"temperature": temperature, "density": density},
            name="pressure",
        )

        self.helmholtz_da = xr.DataArray(
            helmholtz,
            dims=["temperature", "density"],
            coords={"temperature": temperature, "density": density},
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
