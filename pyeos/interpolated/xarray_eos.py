"""
Base class for xarray based EOS classes

"""

import numpy as np
import xarray as xr

from ..eos import Eos
from ..types import EOSArray, EOSReal


class XArrayEos(Eos):
    dataset: xr.Dataset

    @property
    def xarray_handle(self) -> xr.Dataset:
        return self.dataset

    def _interpolate(
        self, data_array: xr.DataArray, rho: EOSReal, temperature: EOSReal
    ) -> EOSReal:
        """
        Interpolate a data array at the given density and temperature.

        Parameters
        ----------
        data_array : xr.DataArray
            The data array to interpolate
        rho : EOSReal
            Density value(s)
        temperature : EOSReal
            Temperature value(s)

        Returns
        -------
        EOSReal
            Interpolated value(s)
        """
        # Convert inputs to numpy arrays if they're not already
        rho_array = np.asarray(rho)
        temp_array = np.asarray(temperature)

        # Check if we're dealing with scalar or array inputs
        scalar_input = rho_array.ndim == 0 and temp_array.ndim == 0

        # Reshape for interpolation if needed
        if scalar_input:
            rho_array = rho_array.reshape(1)
            temp_array = temp_array.reshape(1)

        # Create a new coordinates dictionary for interpolation
        coords = {
            "density": ("points", rho_array.flatten()),
            "temperature": ("points", temp_array.flatten()),
        }

        # Perform the interpolation
        result = data_array.interp(
            coords,
            method="linear",
            assume_sorted=True,
            kwargs={"fill_value": None},
        )
        values: EOSArray = result.values  # type: ignore

        # Return scalar or array based on input type
        if scalar_input:
            # For scalar input, return a float
            return float(values[0])
        else:
            # For array input, reshape and return as numpy array
            return values.reshape(rho_array.shape)

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
        return self._interpolate(self.dataset.internal_energy, rho, temperature)

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
        return self._interpolate(self.dataset.pressure, rho, temperature)

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
        return self._interpolate(self.dataset.helmholtz_free_energy, rho, temperature)
