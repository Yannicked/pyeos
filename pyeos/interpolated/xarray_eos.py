"""
Base class for xarray based EOS classes

"""

import numpy as np

from ..eos import Eos
from ..types import EOSReal


class XArrayEos(Eos):
    @property
    def xarray_handle(self):
        return self.dataset

    def _interpolate(self, data_array, rho, temperature) -> EOSReal:
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

        # Return scalar or array based on input type
        if scalar_input:
            return float(result.values[0])
        else:
            return result.values.reshape(rho_array.shape)

    def InternalEnergyFromDensityTemperature(self, rho, temperature) -> EOSReal:
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

    def PressureFromDensityTemperature(self, rho, temperature) -> EOSReal:
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
        return self._interpolate(self.dataset.helmholtz_free_energy, rho, temperature)
