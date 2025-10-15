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

    def __init__(self, file_name: str):
        """
        Initialize the SESAME reader.

        Parameters
        ----------
        file_name : str
            Path to the input file
        """
        self.file_name = file_name
        self.tables: Dict[int, EOSArray] = {}
        self.material_id: int | None = None
        self.comments: List[str] = []
        self._read_file()
        self._parse_tables()

    def _read_file(self) -> None:
        """
        Read the SESAME file content.
        """
        self.file = open(self.file_name, "r")

    def _consume_header(self):
        header_line = next(self.file)
        header_pieces = header_line.split()
        file_number = int(header_pieces[0])
        material_id = int(header_pieces[1])
        table_id = int(header_pieces[2])
        num_words = int(header_pieces[3])
        create_date = str(header_pieces[4])
        update_date = str(header_pieces[5])
        version_number = str(header_pieces[6])
        return (
            file_number,
            material_id,
            table_id,
            num_words,
            create_date,
            update_date,
            version_number,
        )

    def _parse_tables(self) -> None:
        """
        Parse the SESAME file content into tables.
        """
        version = next(self.file)
        while True:
            # Parse header line
            try:
                (
                    _,
                    material_id,
                    table_id,
                    num_words,
                    _,
                    _,
                    _,
                ) = self._consume_header()
            except ValueError:
                continue
            except StopIteration:
                break

            if self.material_id is None:
                self.material_id = material_id

            # Process different table types
            if 101 <= table_id <= 199:  # Comment tables
                comments = self._read_comment_block(num_words)
                if "unclassified" not in comments.lower() and table_id == 101:
                    print(material_id)
                if self.material_id == material_id:
                    self.comments.append(comments)
            elif table_id == 201:  # Material data
                data = self._read_data_block(num_words)
                if self.material_id == material_id:
                    self.tables[table_id] = data
            elif table_id in [301, 303, 304]:  # EOS data tables
                data = self._read_data_block(num_words)
                if self.material_id == material_id:
                    self.tables[table_id] = data
        self.file.close()

    def _read_comment_block(self, num_words: int):
        n_read = 0
        comment = ""
        while True:
            line = next(self.file)
            line = line.replace("\n", "")
            comment += line
            n_read += len(line)
            if n_read >= num_words:
                break
        return comment

    def _read_data_block(self, num_words: int) -> EOSArray:
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
        EOSArray
            Array of data values
        """
        data: List[float] = []

        while True:
            line = next(self.file)
            # Each line has up to 5 values, each 22 characters wide
            for word_str in line.split():
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
