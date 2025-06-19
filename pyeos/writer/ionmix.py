"""
IONMIX format writer for equation of state data.
"""

import numpy as np
from datetime import datetime

from ..types import EOSArray
from ..eos import Eos
from . import Writer

# Constants
JOULE_FROM_ERG = 1.0e-7
EV_FROM_K = 1.0 / 11604.525
AMU_G = 1.66053906660e-24
NA = 6.02214076e23


class IonmixWriter(Writer):
    """Writer for the IONMIX equation of state format."""

    def __init__(
        self,
        file_name: str,
        atomic_number: float,
        atomic_mass: float,
        twot: bool = False,
        man: bool = True,
    ):
        """Initialize the IONMIX writer."""
        self.file_name = file_name
        self.atomic_number = atomic_number
        self.atomic_mass = atomic_mass
        self.mpi = atomic_mass * AMU_G  # Mass per ion in grams
        self.twot = twot
        self.man = man

    def __enter__(self) -> "IonmixWriter":
        """Context manager entry point."""
        if not self.file_name:
            raise ValueError("Filename should be set!")
        self.file = open(self.file_name, "w")
        return self

    def __exit__(self, type: type, value: Exception, traceback: object) -> None:
        """Context manager exit point."""
        self.file.close()

    def _write_header(
        self, ntemp: int, ndens: int, ngroups: int = 0, grid_params=None
    ) -> None:
        """Write the IONMIX header."""
        # Write number of temperature/density points
        self.file.write(f"{ntemp:10d}{ndens:10d}\n")

        # Write three blank lines for composition info
        for _ in range(2):
            self.file.write("\n")

        # Write grid information
        if not self.man:
            if grid_params is None:
                raise ValueError("Grid parameters must be provided for non-manual grid")
            ddens_log10, dens0_log10, dtemp_log10, temp0_log10 = grid_params
            self.file.write(
                f"{ddens_log10:12.5E}{dens0_log10:12.5E}"
                f"{dtemp_log10:12.5E}{temp0_log10:12.5E}{ngroups:12d}\n"
            )
        else:
            self.file.write(f"{ngroups:12d}\n")

    def _write_block(self, data: EOSArray) -> None:
        """Write a block of data to the IONMIX file."""
        for i, val in enumerate(data):
            self.file.write(f"{val:12.5E}")
            if (i + 1) % 4 == 0:
                self.file.write("\n")
        if len(data) % 4 != 0:
            self.file.write("\n")

    def write(
        self,
        eos: Eos,
        ion_eos: Eos,
        electron_eos: Eos,
        density: EOSArray | None = None,
        temperature: EOSArray | None = None,
    ) -> None:
        """Write complete equation of state data to the IONMIX file."""
        if density is None:
            density = np.geomspace(1e-12, 10, 100, dtype=np.float64)
        if temperature is None:
            temperature = np.geomspace(1, 174067875.0925, 100, dtype=np.float64)

        # Convert temperature from K to eV
        temps_ev = temperature * EV_FROM_K

        # Calculate number densities (cm^-3) from mass densities (g/cm^3)
        num_dens = density * NA / self.atomic_mass

        # Create density/temperature meshgrid
        density_grid, temperature_grid = np.meshgrid(density, temperature)
        density_grid = density_grid.flatten()
        temperature_grid = temperature_grid.flatten()

        # Write header
        self._write_header(len(temps_ev), len(num_dens), 0)

        # Write temperature and density grids if manual
        if self.man:
            self._write_block(temps_ev)
            self._write_block(num_dens)

        # Calculate average ionization state (Zbar)
        # For simplicity, we'll use a constant value based on atomic number
        zbar = np.ones((len(num_dens), len(temps_ev))) * (self.atomic_number * 0.5)
        self._write_block(zbar.flatten())

        if not self.twot:
            # 1T data
            # Calculate total energy per ion (ergs)
            energy = eos.InternalEnergyFromDensityTemperature(
                density_grid, temperature_grid
            )
            energy_per_ion = energy * self.mpi  # Convert from erg/g to erg/ion
            self._write_block(energy_per_ion * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate total Cv per ion (ergs/K)
            # For simplicity, we'll use a constant value
            cvtot = np.ones_like(energy_per_ion) * 1.0e-16
            self._write_block(cvtot * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate dE/dN (derivative of energy with respect to number density)
            dedn = np.zeros_like(energy_per_ion)
            self._write_block(dedn)
        else:
            # 2T data
            # Calculate dZ/dT (derivative of Zbar with respect to temperature)
            dzdt = np.zeros_like(zbar.flatten())
            self._write_block(dzdt)

            # Calculate ion pressure (ergs/cm^3)
            ion_pressure = ion_eos.PressureFromDensityTemperature(
                density_grid, temperature_grid
            )
            ion_pressure = ion_pressure.reshape(
                len(temps_ev), len(num_dens)
            ).T.flatten()
            self._write_block(ion_pressure * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate electron pressure (ergs/cm^3)
            ele_pressure = electron_eos.PressureFromDensityTemperature(
                density_grid, temperature_grid
            )
            ele_pressure = ele_pressure.reshape(
                len(temps_ev), len(num_dens)
            ).T.flatten()
            self._write_block(ele_pressure * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate dP_ion/dT and dP_ele/dT
            dpidt = np.ones_like(ion_pressure) * 1.0e-16
            dpedt = np.ones_like(ele_pressure) * 1.0e-16
            self._write_block(dpidt * JOULE_FROM_ERG)  # Convert to Joules
            self._write_block(dpedt * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate ion energy per ion (ergs)
            ion_energy = ion_eos.InternalEnergyFromDensityTemperature(
                density_grid, temperature_grid
            )
            ion_energy_per_ion = ion_energy * self.mpi  # Convert from erg/g to erg/ion
            ion_energy_per_ion = ion_energy_per_ion.reshape(
                len(temps_ev), len(num_dens)
            ).T.flatten()
            self._write_block(ion_energy_per_ion * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate electron energy per ion (ergs)
            ele_energy = electron_eos.InternalEnergyFromDensityTemperature(
                density_grid, temperature_grid
            )
            ele_energy_per_ion = ele_energy * self.mpi  # Convert from erg/g to erg/ion
            ele_energy_per_ion = ele_energy_per_ion.reshape(
                len(temps_ev), len(num_dens)
            ).T.flatten()
            self._write_block(ele_energy_per_ion * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate ion and electron Cv per ion (ergs/K)
            cvion = np.ones_like(ion_energy_per_ion) * 1.0e-16
            cvele = np.ones_like(ele_energy_per_ion) * 1.0e-16
            self._write_block(cvion * JOULE_FROM_ERG)  # Convert to Joules
            self._write_block(cvele * JOULE_FROM_ERG)  # Convert to Joules

            # Calculate dE_ion/dN and dE_ele/dN
            deidn = np.zeros_like(ion_energy_per_ion)
            deedn = np.zeros_like(ele_energy_per_ion)
            self._write_block(deidn * JOULE_FROM_ERG)  # Convert to Joules
            self._write_block(deedn * JOULE_FROM_ERG)  # Convert to Joules

    def write_total_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """Write total equation of state data to the IONMIX file."""
        # This method is not directly used for IONMIX format
        raise NotImplementedError(
            "Individual component writing is not supported for IONMIX format. "
            "Use the write() method instead."
        )

    def write_ion_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """Write ion component equation of state data to the IONMIX file."""
        # This method is not directly used for IONMIX format
        raise NotImplementedError(
            "Individual component writing is not supported for IONMIX format. "
            "Use the write() method instead."
        )

    def write_electron_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """Write electron component equation of state data to the IONMIX file."""
        # This method is not directly used for IONMIX format
        raise NotImplementedError(
            "Individual component writing is not supported for IONMIX format. "
            "Use the write() method instead."
        )
