# -*- coding: utf-8 -*-
"""
IONMIX format reader for equation of state data.

This module provides a reader implementation for the IONMIX format,
often used in radiation-hydrodynamics codes.
"""

import re
from io import StringIO

import numpy as np

from ..types import EOSArray
from . import MaterialData, MaterialProperties, Reader

# Constants
JOULE_TO_ERG = 1.0e7
EV_TO_K = 11604.525  # Approximate value for k_B = 1 eV/K
AMU_G = 1.66053906660e-24  # Atomic mass unit in grams
NA = 6.02214076e23  # Avogadro constant


class IonmixReader(Reader):
    """
    Reader for the IONMIX equation of state format.

    Reads EOS data (pressure, energy) and optionally opacity data
    from IONMIX files. Requires atomic number and mass to be provided.

    Handles both single-temperature (1T) and two-temperature (2T) formats.
    """

    def __init__(
        self,
        file_name: str,
        atomic_number: float,
        atomic_mass: float,  # in AMU
        twot: bool = False,
        man: bool = True,
    ):
        """
        Initialize the IONMIX reader.

        Parameters
        ----------
        file_name : str
            Path to the input IONMIX file.
        atomic_number : float
            Atomic number (Z) of the material.
        atomic_mass : float
            Atomic mass (A) of the material in AMU.
        twot : bool, optional
            Flag for two-temperature data, by default False.
        man : bool, optional
            Flag for manually specified temperature/density points, by default False.
        """
        self.file_name = file_name
        self.atomic_number = atomic_number
        self.atomic_mass = atomic_mass
        self.mpi = atomic_mass * AMU_G  # Mass per ion in grams
        self.twot = twot
        self.man = man

        # Data storage attributes (initialized to None or empty)
        self.ntemp: int = 0
        self.ndens: int = 0
        self.temps_ev: EOSArray = np.array([])  # Temperatures in eV
        self.num_dens: EOSArray = np.array([])  # Number densities in cm^-3
        self.dens: EOSArray = np.array([])  # Mass densities in g/cm^3
        self.ngroups: int = 0
        self.zbar: EOSArray = np.array([])
        # 1T data
        self.etot: EOSArray = np.array([])  # Total energy per ion (ergs)
        self.cvtot: EOSArray = np.array([])  # Total Cv per ion (ergs/K or ergs/eV?)
        self.dedn: EOSArray = np.array([])
        # 2T data
        self.dzdt: EOSArray = np.array([])
        self.pion: EOSArray = np.array([])  # Ion pressure (ergs/cm^3)
        self.pele: EOSArray = np.array([])  # Electron pressure (ergs/cm^3)
        self.dpidt: EOSArray = np.array([])
        self.dpedt: EOSArray = np.array([])
        self.eion: EOSArray = np.array([])  # Ion energy per ion (ergs)
        self.eele: EOSArray = np.array([])  # Electron energy per ion (ergs)
        self.cvion: EOSArray = np.array([])  # Ion Cv per ion (ergs/K or ergs/eV?)
        self.cvele: EOSArray = np.array([])  # Electron Cv per ion (ergs/K or ergs/eV?)
        self.deidn: EOSArray = np.array([])
        self.deedn: EOSArray = np.array([])
        # Electron entropy per ion (ergs/K or ergs/eV?)
        self.sele: EOSArray = np.array([])
        # Opacity data (optional, not used by base EOS)
        self.opac_bounds: EOSArray = np.array([])
        self.rosseland: EOSArray = np.array([])
        self.planck_absorb: EOSArray = np.array([])
        self.planck_emiss: EOSArray = np.array([])

        self._data_stream: StringIO | None = None

        self._read_file()
        self._parse_eos()
        # Opacity parsing can be added here if needed later
        # self._parse_opac()

    def _read_file(self) -> None:
        """
        Read the IONMIX file content and parse the header and grid.
        """
        with open(self.file_name, "r") as f:
            # Read the number of temperatures/densities:
            try:
                self.ntemp = int(f.read(10))
                self.ndens = int(f.read(10))
            except ValueError as e:
                raise ValueError(f"Could not read ntemp/ndens from header: {e}") from e

            # Skip the next three lines (composition info):
            for _ in range(3):
                f.readline()

            # Setup temperature/density grid:
            if not self.man:
                # Read information about the temperature/density grid:
                try:
                    ddens_log10 = float(f.read(12))
                    dens0_log10 = float(f.read(12))
                    dtemp_log10 = float(f.read(12))
                    temp0_log10 = float(f.read(12))
                except ValueError as e:
                    raise ValueError(f"Could not read grid parameters: {e}") from e

                # Compute number densities (cm^-3) and temperatures (eV):
                # Assuming log10(Number Density [cm^-3]) and log10(Temperature [eV])
                self.num_dens = np.logspace(
                    dens0_log10,
                    dens0_log10 + ddens_log10 * (self.ndens - 1),
                    self.ndens,
                    dtype=np.float64,
                )
                self.temps_ev = np.logspace(
                    temp0_log10,
                    temp0_log10 + dtemp_log10 * (self.ntemp - 1),
                    self.ntemp,
                    dtype=np.float64,
                )

                # Read number of groups:
                try:
                    self.ngroups = int(f.read(12))
                except ValueError as e:
                    raise ValueError(f"Could not read ngroups: {e}") from e
            else:
                # Manual grid specification
                try:
                    self.ngroups = int(f.read(12))
                except ValueError as e:
                    raise ValueError(
                        f"Could not read ngroups (manual grid): {e}"
                    ) from e
                f.readline()  # Consume rest of line

            # Read the rest of the file into a stream for block reading
            txt = re.sub(r"\s", "", f.read())
            self._data_stream = StringIO(txt)

            if self.man:
                # Read manual temperature/density values
                self.temps_ev = self._get_block(self.ntemp)
                self.num_dens = self._get_block(self.ndens)

            # Calculate mass density (g/cm^3)
            self.dens = self.num_dens * self.mpi / NA

    def _get_block(self, n: int) -> EOSArray:
        """Reads a block of n numbers (12 chars each) from the data stream."""
        if self._data_stream is None:
            raise RuntimeError("Data stream not initialized.")
        arr = np.zeros(n)
        for i in range(n):
            try:
                val_str = self._data_stream.read(12)
                if not val_str:
                    raise EOFError(
                        "Unexpected end of file while "
                        f"reading block element {i + 1}/{n}"
                    )
                arr[i] = float(val_str)
            except ValueError as e:
                raise ValueError(
                    f"Could not parse float from '{val_str}' at "
                    f"element {i + 1}/{n}: {e}"
                ) from e
            except EOFError as e:
                raise EOFError(f"Could not read element {i + 1}/{n}: {e}") from e
        return arr

    def _parse_eos(self) -> None:
        """
        Parse the EOS data blocks from the data stream.
        """
        nt = self.ntemp
        nd = self.ndens

        try:
            self.zbar = self._get_block(nd * nt).reshape(nd, nt)

            if not self.twot:
                # Read 1T data (convert Joules -> ergs)
                self.etot = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.cvtot = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.dedn = self._get_block(nd * nt).reshape(nd, nt)
            else:
                # Read 2T data (convert Joules -> ergs)
                self.dzdt = self._get_block(nd * nt).reshape(nd, nt)
                self.pion = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.pele = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.dpidt = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.dpedt = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.eion = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.eele = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.cvion = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.cvele = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.deidn = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG
                self.deedn = self._get_block(nd * nt).reshape(nd, nt) * JOULE_TO_ERG

        except (ValueError, EOFError) as e:
            raise IOError(f"Error parsing EOS data blocks from IONMIX file: {e}") from e

    def read_material_properties(self) -> MaterialProperties:
        """
        Return material properties provided during initialization.

        Note: IONMIX format does not typically store these properties directly.
        """
        return MaterialProperties(
            atomic_number=self.atomic_number,
            atomic_mass=self.atomic_mass,
            normal_density=np.nan,  # Not available in IONMIX
            solid_bulk_modulus=np.nan,  # Not available in IONMIX
            exchange_coefficient=np.nan,  # Not available in IONMIX
        )

    def _prepare_material_data(
        self, energy_per_g: EOSArray, pressure: EOSArray, helmholtz: EOSArray
    ) -> MaterialData:
        """Helper to create MaterialData with correct units and array shapes."""
        # Ensure arrays are (num_temperature, num_density)
        # IONMIX data is read as (nd, nt), so transpose is needed.
        temps_k = self.temps_ev * EV_TO_K
        return MaterialData(
            density=self.dens,  # 1D array
            temperature=temps_k,  # 1D array
            energy=energy_per_g.T,  # Transpose to (nt, nd)
            pressure=pressure.T,  # Transpose to (nt, nd)
            helmholtz=helmholtz.T,  # Transpose to (nt, nd)
        )

    def read_total_data(self) -> MaterialData:
        """
        Read total equation of state data.

        Combines ion and electron contributions if reading 2T data.
        Returns specific energy (ergs/g) and pressure (ergs/cm^3).
        Helmholtz energy is currently returned as zeros.
        """
        zeros = np.zeros((self.ndens, self.ntemp))

        if self.twot:
            total_energy_per_ion = self.eion + self.eele
            total_pressure = self.pion + self.pele
        else:
            total_energy_per_ion = self.etot
            # Total pressure not directly available in 1T IONMIX
            total_pressure = zeros  # Return zeros for pressure

        total_energy_per_g = total_energy_per_ion / self.mpi
        # Helmholtz not directly available
        total_helmholtz = zeros

        return self._prepare_material_data(
            total_energy_per_g, total_pressure, total_helmholtz
        )

    def read_ion_data(self) -> MaterialData:
        """
        Read ion component equation of state data.

        Returns data only if the file is in 2T format, otherwise returns zeros.
        Returns specific energy (ergs/g) and pressure (ergs/cm^3).
        Helmholtz energy is currently returned as zeros.
        """
        zeros = np.zeros((self.ndens, self.ntemp))
        if self.twot:
            ion_energy_per_g = self.eion / self.mpi
            ion_pressure = self.pion
            ion_helmholtz = zeros  # Ion Helmholtz not available
            return self._prepare_material_data(
                ion_energy_per_g, ion_pressure, ion_helmholtz
            )
        else:
            raise ValueError("No ion data available: table is not 2T")

    def read_electron_data(self) -> MaterialData:
        """
        Read electron component equation of state data.

        Returns data only if the file is in 2T format, otherwise returns zeros.
        Returns specific energy (ergs/g) and pressure (ergs/cm^3).
        Calculates Helmholtz energy if electron entropy (`sele`) is available.
        """
        zeros = np.zeros((self.ndens, self.ntemp))
        if self.twot:
            ele_energy_per_g = self.eele / self.mpi
            ele_pressure = self.pele

            ele_helmholtz = zeros  # Electron Helmholtz not available/calculable

            return self._prepare_material_data(
                ele_energy_per_g, ele_pressure, ele_helmholtz
            )
        else:
            raise ValueError("No electron data available: table is not 2T")
