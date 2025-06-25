"""
SESAME format reader for equation of state data.

This module provides a reader implementation for the SESAME format,
which is a standard format for equation of state data used in
high-energy-density physics simulations.
"""

from typing import Dict, List

import numpy as np

from ..types import EOSArray
from . import MaterialData, MaterialProperties, Reader


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

    def __init__(self, file_name: str, material_id: int | None = None):
        """
        Initialize the SESAME reader.

        Parameters
        ----------
        file_name : str
            Path to the input file
        """
        self.file_name = file_name
        self.tables: Dict[int, EOSArray] = {}
        self.material_id: int | None = material_id
        self._parse_tables()

    def _parse_tables(self) -> None:
        """
        Parse the SESAME file content into tables.
        """
        with open(self.file_name, "r") as f:
            for material_id, table_id, data in self._record_iterator(f):
                if self.material_id is None:
                    self.material_id = material_id
                elif self.material_id != material_id:
                    continue
                self.tables[table_id] = data
            if not self.tables:
                raise ValueError(
                    f"Material {self.material_id} not found in SESAME file"
                )

    def _record_iterator(self, file_handle) -> iter:
        """
        Create an iterator for the records in the SESAME file.
        """
        file_iterator = iter(file_handle)
        for line in file_iterator:
            # Parse header line
            try:
                file_number = int(line[0:2].strip())
                if file_number == 2:
                    continue
                material_id = int(line[2:8].strip())
                table_id = int(line[8:14].strip())
                num_words = int(line[14:20].strip())
            except (ValueError, IndexError):
                continue

            # Process different table types
            if 101 <= table_id <= 199:  # Comment tables
                # Skip comment tables for now
                num_lines = (num_words // 80) + (1 if num_words % 80 > 0 else 0)
                for _ in range(num_lines):
                    next(file_iterator)
            elif table_id in [201, 301, 303, 304]:  # EOS data tables
                data = self._read_data_block(file_iterator, num_words)
                yield material_id, table_id, data

    def _read_data_block(self, file_iterator: iter, num_words: int) -> EOSArray:
        """
        Read a block of data from the file.

        Parameters
        ----------
        file_iterator : iter
            Iterator for the file lines
        num_words : int
            Number of words to read

        Returns
        -------
        EOSArray
            Array of data values
        """
        data: List[float] = []
        lines_to_read = (num_words + 4) // 5  # 5 words per line, rounded up

        for i, line in enumerate(file_iterator):
            if i >= lines_to_read:
                break
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

    def _extract_table_data(self, table_id: int) -> MaterialData:
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

        data: EOSArray = self.tables[table_id]

        # First two values are the number of density and temperature points
        num_density = int(data[0])
        num_temperature = int(data[1])

        # Extract the density and temperature arrays
        density: EOSArray = data[2 : 2 + num_density] * self.CGS_DENSITY_FROM_SESAME
        temperature: EOSArray = data[
            2 + num_density : 2 + num_density + num_temperature
        ]

        # Calculate the total number of points in the grid
        total_points = num_density * num_temperature

        # Extract pressure, energy, and helmholtz arrays
        start_idx = 2 + num_density + num_temperature
        pressure: EOSArray = (
            data[start_idx : start_idx + total_points].reshape(
                num_temperature, num_density
            )
            * self.CGS_PRESSURE_FROM_SESAME
        )

        start_idx += total_points
        energy: EOSArray = (
            data[start_idx : start_idx + total_points].reshape(
                num_temperature, num_density
            )
            * self.CGS_ENERGY_FROM_SESAME
        )

        start_idx += total_points
        helmholtz: EOSArray = (
            data[start_idx : start_idx + total_points].reshape(
                num_temperature, num_density
            )
            * self.CGS_ENERGY_FROM_SESAME
        )

        return MaterialData(density, temperature, energy, pressure, helmholtz)

    def read_material_properties(self) -> MaterialProperties:
        """
        Read material data from the SESAME file.

        Returns
        -------
        dict
            A dictionary containing material properties
        """
        if 201 not in self.tables:
            raise ValueError("Material data table (201) not found in SESAME file")

        data: EOSArray = self.tables[201]

        return MaterialProperties(*data)

    def read_total_data(
        self,
    ) -> MaterialData:
        """
        Read total equation of state data from the SESAME file.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the total equation of state.
        """
        return self._extract_table_data(301)

    def read_ion_data(
        self,
    ) -> MaterialData:
        """
        Read ion component equation of state data from the SESAME file.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the ion component.
        """
        return self._extract_table_data(303)

    def read_electron_data(
        self,
    ) -> MaterialData:
        """
        Read electron component equation of state data from the SESAME file.

        Returns
        -------
        tuple
            A tuple containing (density, temperature, energy, pressure, helmholtz)
            arrays for the electron component.
        """
        return self._extract_table_data(304)
