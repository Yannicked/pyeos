"""
SESAME format reader for equation of state data.

This module provides a reader implementation for the SESAME format,
which is a standard format for equation of state data used in
high-energy-density physics simulations.
"""

import numpy as np

from . import Reader


class SesameReader(Reader):
    """
    Reader for the SESAME equation of state format.

    This class implements the Reader interface for the SESAME format,
    which is a standard format for equation of state data used in
    high-energy-density physics simulations.
    """

    # Constants for unit conversion
    CGS_DENSITY_FROM_SESAME = 1.0
    CGS_ENERGY_FROM_SESAME = 1e10
    CGS_PRESSURE_FROM_SESAME = 1e10

    def __init__(self, file_name):
        """
        Initialize the SESAME reader.

        Parameters
        ----------
        file_name : str
            Path to the input file
        """
        self.file_name = file_name
        self.file_content = None
        self.tables = {}
        self.material_id = None
        self._read_file()
        self._parse_tables()

    def _read_file(self):
        """
        Read the SESAME file content.
        """
        with open(self.file_name, "r") as f:
            self.file_content = f.readlines()

    def _parse_tables(self):
        """
        Parse the SESAME file content into tables.
        """
        i = 0
        while i < len(self.file_content):
            line = self.file_content[i]
            if len(line) < 20:  # Skip short lines
                i += 1
                continue

            # Parse header line
            try:
                # file_number = int(line[0:2].strip())
                material_id = int(line[2:8].strip())
                table_id = int(line[8:14].strip())
                num_words = int(line[14:20].strip())
            except ValueError:
                i += 1
                continue

            if self.material_id is None:
                self.material_id = material_id

            # Process different table types
            if 101 <= table_id <= 199:  # Comment tables
                # Skip comment tables for now
                i += (num_words // 80) + (1 if num_words % 80 > 0 else 0) + 1
            elif table_id == 201:  # Material data
                data = self._read_data_block(i + 1, num_words)
                self.tables[table_id] = data
                i += self._count_data_lines(num_words) + 1
            elif table_id in [301, 303, 304]:  # EOS data tables
                data = self._read_data_block(i + 1, num_words)
                self.tables[table_id] = data
                i += self._count_data_lines(num_words) + 1
            else:
                i += 1

    def _read_data_block(self, start_line, num_words):
        """
        Read a block of data from the file.

        Parameters
        ----------
        start_line : int
            Line number to start reading from
        num_words : int
            Number of words to read

        Returns
        -------
        NDArray[np.float64]
            Array of data values
        """
        data = []
        lines_to_read = self._count_data_lines(num_words)

        for i in range(lines_to_read):
            line = self.file_content[start_line + i]
            # Each line has up to 5 values, each 22 characters wide
            for j in range(5):
                if len(line) >= (j + 1) * 22:
                    word_str = line[j * 22 : (j + 1) * 22].strip()
                    if word_str:
                        try:
                            data.append(float(word_str))
                        except ValueError:
                            # Skip invalid values
                            pass

                    if len(data) >= num_words:
                        break

            if len(data) >= num_words:
                break

        return np.array(data)

    def _count_data_lines(self, num_words):
        """
        Calculate the number of lines needed for a given number of words.

        Parameters
        ----------
        num_words : int
            Number of words

        Returns
        -------
        int
            Number of lines
        """
        return (num_words + 4) // 5  # 5 words per line, rounded up

    def _extract_table_data(self, table_id):
        """
        Extract and process data from a specific table.

        Parameters
        ----------
        table_id : int
            Table ID to extract

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the specified table.
        """
        if table_id not in self.tables:
            raise ValueError(f"Table {table_id} not found in SESAME file")

        data = self.tables[table_id]

        # First two values are the number of density and temperature points
        num_density = int(data[0])
        num_temperature = int(data[1])

        # Extract the density and temperature arrays
        density = data[2 : 2 + num_density] * self.CGS_DENSITY_FROM_SESAME
        temperature = data[2 + num_density : 2 + num_density + num_temperature]

        # Calculate the total number of points in the grid
        total_points = num_density * num_temperature

        # Extract pressure, energy, and helmholtz arrays
        start_idx = 2 + num_density + num_temperature
        pressure = (
            data[start_idx : start_idx + total_points].reshape(
                num_temperature, num_density
            )
            * self.CGS_PRESSURE_FROM_SESAME
        )

        start_idx += total_points
        energy = (
            data[start_idx : start_idx + total_points].reshape(
                num_temperature, num_density
            )
            * self.CGS_ENERGY_FROM_SESAME
        )

        start_idx += total_points
        helmholtz = (
            data[start_idx : start_idx + total_points].reshape(
                num_temperature, num_density
            )
            * self.CGS_ENERGY_FROM_SESAME
        )

        return density, temperature, energy, pressure, helmholtz

    def read_material_data(self):
        """
        Read material data from the SESAME file.

        Returns
        -------
        dict
            A dictionary containing material properties
        """
        if 201 not in self.tables:
            raise ValueError("Material data table (201) not found in SESAME file")

        data = self.tables[201]

        return {
            "atomic_number": data[0],
            "atomic_mass": data[1],
            "normal_density": data[2],
            "solid_bulk_modulus": data[3],
            "exchange_coefficient": data[4] if len(data) > 4 else None,
        }

    def read_total_data(self):
        """
        Read total equation of state data from the SESAME file.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the total equation of state.
        """
        return self._extract_table_data(301)

    def read_ion_data(self):
        """
        Read ion component equation of state data from the SESAME file.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the ion component.
        """
        return self._extract_table_data(303)

    def read_electron_data(self):
        """
        Read electron component equation of state data from the SESAME file.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the electron component.
        """
        return self._extract_table_data(304)
