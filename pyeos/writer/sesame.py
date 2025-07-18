"""
SESAME format writer for equation of state data.

This module provides a writer implementation for the SESAME format,
which is a standard format for equation of state data used in
high-energy-density physics simulations.
"""

from datetime import datetime

import numpy as np

from ..eos import Eos
from ..types import EOSArray
from . import Writer


class SesameWriter(Writer):
    """
    Writer for the SESAME equation of state format.

    This class implements the Writer interface for the SESAME format,
    which is a standard format for equation of state data used in
    high-energy-density physics simulations.
    """

    description_record = (
        "{file_number:>2}{material_id:>6}{table_id:>6}{num_words:>6}"
        "   r{creation_date:>9}{update_date:>9}{version:>4}"
        "                                {file_number:>2}\n"
    )

    CGS_DENSITY_TO_SESAME = 1.0
    CGS_ENERGY_TO_SESAME = 1e-10
    CGS_PRESSURE_TO_SESAME = 1e-10

    def __init__(self, file_name: str, material_id: int):
        """
        Initialize the SESAME writer.

        Parameters
        ----------
        file_name : str
            Path to the output file
        material_id : int
            Material ID for the SESAME file
        """
        self.file_name = file_name
        self.material_id = material_id
        self.comment_table = 101
        self.extra_comments: list[str] = []

    def __enter__(self) -> "SesameWriter":
        """
        Context manager entry point.

        Opens the output file for writing.

        Returns
        -------
        SesameWriter
            Self reference for context manager
        """
        if not self.file_name or not self.material_id:
            raise ValueError("Filename and material id should be set!")
        self.file = open(self.file_name, "w")
        return self

    def __exit__(self, type: type, value: Exception, traceback: object) -> None:
        """
        Context manager exit point.

        Closes the output file.
        """
        self.file.close()

    def add_comment(self, comment: str) -> None:
        """
        Add a comment to be written to the SESAME file.

        Parameters
        ----------
        comment : str
            Comment text
        """
        self.extra_comments.append(comment)

    def write(
        self,
        eos: Eos,
        ion_eos: Eos,
        electron_eos: Eos,
        density: EOSArray | None = None,
        temperature: EOSArray | None = None,
    ) -> None:
        """
        Write complete equation of state data to the SESAME file.

        This method writes all components (total, ion, and electron)
        of the equation of state to the file.

        Parameters
        ----------
        eos : Eos
            Total equation of state
        ion_eos : Eos
            Ion component equation of state
        electron_eos : Eos
            Electron component equation of state
        density : EOSArray, optional
            Density grid to use, by default None
        temperature : EOSArray, optional
            Temperature grid to use, by default None
        """
        self.write_comment("This table was generated with pyeos", 101)
        for comment in self.extra_comments:
            self.write_comment(comment)
        self.write_material_data(eos.Z, eos.A, 1.0, 1.0, 1.0)

        if density is None:
            density = np.geomspace(1e-12, 10, 1000, dtype=np.float64)
        if temperature is None:
            temperature = np.geomspace(1, 174067875.0925, 100, dtype=np.float64)
        density_grid, temperature_grid = np.meshgrid(density, temperature)
        density_grid = density_grid.flatten()
        temperature_grid = temperature_grid.flatten()
        # Calculate EOS values and reshape to match the grid
        energy = eos.InternalEnergyFromDensityTemperature(
            density_grid, temperature_grid
        )
        pressure = eos.PressureFromDensityTemperature(density_grid, temperature_grid)
        helmholtz = eos.HelmholtzFreeEnergyFromDensityTemperature(
            density_grid, temperature_grid
        )

        # Convert to numpy arrays if they are scalars
        energy_array = np.atleast_1d(energy).reshape(temperature.size, density.size)
        pressure_array = np.atleast_1d(pressure).reshape(temperature.size, density.size)
        helmholtz_array = np.atleast_1d(helmholtz).reshape(
            temperature.size, density.size
        )

        self.write_total_data(
            density,
            temperature,
            energy_array,
            pressure_array,
            helmholtz_array,
        )

        # Calculate ion EOS values
        ion_energy = ion_eos.InternalEnergyFromDensityTemperature(
            density_grid, temperature_grid
        )
        ion_pressure = ion_eos.PressureFromDensityTemperature(
            density_grid, temperature_grid
        )
        ion_helmholtz = ion_eos.HelmholtzFreeEnergyFromDensityTemperature(
            density_grid, temperature_grid
        )

        # Convert to numpy arrays if they are scalars
        ion_energy_array = np.atleast_1d(ion_energy).reshape(
            temperature.size, density.size
        )
        ion_pressure_array = np.atleast_1d(ion_pressure).reshape(
            temperature.size, density.size
        )
        ion_helmholtz_array = np.atleast_1d(ion_helmholtz).reshape(
            temperature.size, density.size
        )

        self.write_ion_data(
            density,
            temperature,
            ion_energy_array,
            ion_pressure_array,
            ion_helmholtz_array,
        )

        # Calculate electron EOS values
        electron_energy = electron_eos.InternalEnergyFromDensityTemperature(
            density_grid, temperature_grid
        )
        electron_pressure = electron_eos.PressureFromDensityTemperature(
            density_grid, temperature_grid
        )
        electron_helmholtz = electron_eos.HelmholtzFreeEnergyFromDensityTemperature(
            density_grid, temperature_grid
        )

        # Convert to numpy arrays if they are scalars
        electron_energy_array = np.atleast_1d(electron_energy).reshape(
            temperature.size, density.size
        )
        electron_pressure_array = np.atleast_1d(electron_pressure).reshape(
            temperature.size, density.size
        )
        electron_helmholtz_array = np.atleast_1d(electron_helmholtz).reshape(
            temperature.size, density.size
        )

        self.write_electron_data(
            density,
            temperature,
            electron_energy_array,
            electron_pressure_array,
            electron_helmholtz_array,
        )

    def write_header(
        self,
        file_number: int,
        material_id: int,
        table_id: int,
        num_words: int,
        creation_date: str,
        update_date: str,
        version: int,
    ) -> None:
        """
        Write a SESAME header record.

        Parameters
        ----------
        file_number : int
            File number (0 or 1)
        material_id : int
            Material ID
        table_id : int
            Table ID
        num_words : int
            Number of words in the table
        creation_date : str
            Creation date string (format: MMDDYY)
        update_date : str
            Update date string (format: MMDDYY)
        version : int
            Version number
        """
        self.file.write(
            SesameWriter.description_record.format(
                file_number=file_number,
                material_id=material_id,
                table_id=table_id,
                num_words=num_words,
                creation_date=creation_date,
                update_date=update_date,
                version=version,
            )
        )

    def write_words(self, data: EOSArray) -> None:
        """
        Write an array of data words to the SESAME file.

        Parameters
        ----------
        data : EOSArray
            Array of data values to write
        """
        data_lines = [data[i : i + 5] for i in range(0, len(data), 5)]
        # data_lines = np.array_split(data, np.ceil(len(data) / 5))
        for line in data_lines:
            for word in line:
                self.file.write(f"{word:22.15E}")
            count = len(line)
            self.file.write(
                f"{' ' * 22 * (5 - count)}{'1' * count}{'0' * (5 - count)}\n"
            )

    def write_comment(self, comment: str, comment_table_idx: int | None = None) -> None:
        """
        Write a comment to the SESAME file.

        Parameters
        ----------
        comment : str
            Comment text
        comment_table_idx : int | None, optional
            Comment table index, by default None
        """
        if comment_table_idx is not None:
            self.comment_table = comment_table_idx
        if self.comment_table > 199 or self.comment_table < 101:
            raise ValueError("Comment talbe ID should be between 101 and 199")
        comment_lines = [comment[i : i + 80] for i in range(0, len(comment), 80)]
        if len(comment_lines) > 0:
            comment_lines[-1] = f"{comment_lines[-1]:<80}"
        else:
            comment_lines.append(" " * 80)
        data = "\n".join(comment_lines)
        current_date = datetime.now()
        date_str = current_date.strftime("%m%d%y")
        self.write_header(
            0 if self.comment_table == 101 else 1,
            self.material_id,
            self.comment_table,
            len(data) - len(comment_lines) + 1,
            date_str,
            date_str,
            1,
        )
        self.file.write(data)
        self.file.write("\n")  # Don't forget!
        self.comment_table += 1

    def write_material_data(
        self,
        atomic_number: float,
        atomic_mass: float,
        normal_density: float,
        solid_bulk_modulus: float,
        exchange_coefficient: float,
    ) -> None:
        """
        Write material data to the SESAME file.

        Parameters
        ----------
        atomic_number : float
            Atomic number (Z)
        atomic_mass : float
            Atomic mass number (A)
        normal_density : float
            Normal density
        solid_bulk_modulus : float
            Solid bulk modulus
        exchange_coefficient : float
            Exchange coefficient
        """
        data = np.array(
            [
                atomic_number,
                atomic_mass,
                normal_density,
                solid_bulk_modulus,
                exchange_coefficient,
            ]
        )
        current_date = datetime.now()
        date_str = current_date.strftime("%m%d%y")
        self.write_header(1, self.material_id, 201, len(data), date_str, date_str, 1)
        self.write_words(data)

    def write_data(
        self,
        table: int,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """
        Write equation of state data to the SESAME file.

        Parameters
        ----------
        table : int
            Table ID
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
        # Flatten 2D arrays before concatenation
        pressure_flat = pressure.flatten() if pressure.ndim > 1 else pressure
        energy_flat = energy.flatten() if energy.ndim > 1 else energy
        helmholtz_flat = helmholtz.flatten() if helmholtz.ndim > 1 else helmholtz

        data = np.concatenate(
            [
                np.array([len(density), len(temperature)]),
                density * self.CGS_DENSITY_TO_SESAME,
                temperature,
                pressure_flat * self.CGS_PRESSURE_TO_SESAME,
                energy_flat * self.CGS_ENERGY_TO_SESAME,
                helmholtz_flat * self.CGS_ENERGY_TO_SESAME,
            ]
        )
        current_date = datetime.now()
        date_str = current_date.strftime("%m%d%y")
        self.write_header(1, self.material_id, table, len(data), date_str, date_str, 1)
        self.write_words(data)

    def write_total_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """
        Write total equation of state data to the SESAME file.

        Implements the Writer.write_total_data method.

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
        self.write_data(301, density, temperature, energy, pressure, helmholtz)

    def write_ion_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """
        Write ion component equation of state data to the SESAME file.

        Implements the Writer.write_ion_data method.

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
        self.write_data(303, density, temperature, energy, pressure, helmholtz)

    def write_electron_data(
        self,
        density: EOSArray,
        temperature: EOSArray,
        energy: EOSArray,
        pressure: EOSArray,
        helmholtz: EOSArray,
    ) -> None:
        """
        Write electron component equation of state data to the SESAME file.

        Implements the Writer.write_electron_data method.

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
        self.write_data(304, density, temperature, energy, pressure, helmholtz)

    def write_table_end(self) -> None:
        """
        Write the end marker for the SESAME file.
        """
        self.file.write(" 2" + " " * 67 + "2\n")
