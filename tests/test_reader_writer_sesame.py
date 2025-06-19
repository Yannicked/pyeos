"""
Tests for the SesameReader and SesameWriter classes.
"""

import os
import tempfile

import numpy as np
import pytest

from pyeos.analytical import IdealGamma
from pyeos.interpolated.sesame_eos import SesameEos
from pyeos.modifiers import ZSplit
from pyeos.reader.sesame import SesameReader
from pyeos.writer.sesame import SesameWriter


def test_sesame_writer_initialization():
    """Test that SesameWriter initializes correctly."""
    file_name = "test.sesame"
    material_id = 9999

    writer = SesameWriter(file_name, material_id)

    assert writer.file_name == file_name
    assert writer.material_id == material_id
    assert writer.comment_table == 101
    assert writer.extra_comments == []


def test_sesame_writer_context_manager():
    """Test that SesameWriter works as a context manager."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Use the context manager
        with SesameWriter(file_name, 9999) as writer:
            assert hasattr(writer, "file")
            assert not writer.file.closed

        # After exiting the context, the file should be closed
        assert writer.file.closed

        # File should exist
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) == 0  # No data written
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_add_comment():
    """Test adding comments to the writer."""
    writer = SesameWriter("test.sesame", 9999)

    # Add some comments
    writer.add_comment("Test comment 1")
    writer.add_comment("Test comment 2")

    assert len(writer.extra_comments) == 2
    assert writer.extra_comments[0] == "Test comment 1"
    assert writer.extra_comments[1] == "Test comment 2"


def test_write_full_file():
    """Test writing a complete SESAME file."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create EOS objects
        eos = IdealGamma(5 / 3, 1.008, 1)
        ion_eos, electron_eos = ZSplit(eos)

        # Write the file
        with SesameWriter(file_name, 9999) as writer:
            writer.add_comment("Test SESAME file")
            writer.write(eos, ion_eos, electron_eos)

        # Check that the file exists and has content
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0

        # Basic check of file content
        with open(file_name, "r") as f:
            content = f.read()
            # Check for expected header elements
            assert " 0  9999   101" in content  # Comment table header
            assert " 1  9999   201" in content  # Material data header
            assert " 1  9999   301" in content  # Total EOS data header
            assert " 1  9999   303" in content  # Ion EOS data header
            assert " 1  9999   304" in content  # Electron EOS data header
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_write_data_methods():
    """Test the individual data writing methods."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create test data
        density = np.array([0.1, 1.0, 10.0])
        temperature = np.array([100.0, 1000.0, 10000.0])
        energy = np.ones_like(density) * 1e6
        pressure = np.ones_like(density) * 1e5
        helmholtz = np.ones_like(density) * 1e4

        # Write the file with individual methods
        with SesameWriter(file_name, 9999) as writer:
            writer.write_total_data(density, temperature, energy, pressure, helmholtz)
            writer.write_ion_data(density, temperature, energy, pressure, helmholtz)
            writer.write_electron_data(
                density, temperature, energy, pressure, helmholtz
            )
            writer.write_table_end()

        # Check that the file exists and has content
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0

        # Basic check of file content
        with open(file_name, "r") as f:
            content = f.read()
            # Check for expected table headers
            assert " 1  9999   301" in content  # Total EOS data header
            assert " 1  9999   303" in content  # Ion EOS data header
            assert " 1  9999   304" in content  # Electron EOS data header
            assert " 2" in content  # Table end marker
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_sesame_reader_initialization():
    """Test that SesameReader initializes correctly."""
    # Create a temporary SESAME file
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create EOS objects
        eos = IdealGamma(5 / 3, 1.008, 1)  # Hydrogen
        ion_eos, electron_eos = ZSplit(eos)

        # Write the file
        with SesameWriter(file_name, 9999) as writer:
            writer.write(eos, ion_eos, electron_eos)

        # Initialize the reader
        reader = SesameReader(file_name)

        # Check that the reader initialized correctly
        assert reader.file_name == file_name
        assert reader.material_id == 9999
        assert 301 in reader.tables  # Total EOS data
        assert 303 in reader.tables  # Ion EOS data
        assert 304 in reader.tables  # Electron EOS data
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_sesame_reader_error_handling():
    """Test error handling in SesameReader."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        SesameReader("nonexistent_file.sesame")


def test_sesame_write_and_read():
    """Test writing and then reading a SESAME file."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create smaller density and temperature arrays to avoid formatting issues
        density = np.array([0.1, 1.0, 10.0])
        temperature = np.array([100.0, 1000.0, 10000.0])

        # Create an EOS object to write
        eos = IdealGamma(5 / 3, 1.008, 1)  # Hydrogen
        ion_eos, electron_eos = ZSplit(eos)

        # Write the file
        with SesameWriter(file_name, 9999) as writer:
            writer.write(eos, ion_eos, electron_eos, density, temperature)

        # Check that the file exists and has content
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0

        # Read the file back
        reader = SesameReader(file_name)

        # Read material properties
        material_props = reader.read_material_properties()
        assert material_props.atomic_number == 1.0
        assert material_props.atomic_mass == 1.008

        # Read total data
        total_data = reader.read_total_data()

        # Check that the data has the expected structure
        assert hasattr(total_data, "density")
        assert hasattr(total_data, "temperature")
        assert hasattr(total_data, "energy")
        assert hasattr(total_data, "pressure")
        assert hasattr(total_data, "helmholtz")

        # Check that the arrays have the expected shapes
        assert total_data.density.shape == (len(density),)
        assert total_data.temperature.shape == (len(temperature),)
        assert total_data.energy.shape == (len(temperature), len(density))
        assert total_data.pressure.shape == (len(temperature), len(density))
        assert total_data.helmholtz.shape == (len(temperature), len(density))

        # Check that the density and temperature values match what we wrote
        np.testing.assert_allclose(total_data.density, density, rtol=1e-5)
        np.testing.assert_allclose(total_data.temperature, temperature, rtol=1e-5)

        # Read ion data
        ion_data = reader.read_ion_data()
        assert ion_data.density.shape == (len(density),)
        assert ion_data.temperature.shape == (len(temperature),)

        # Read electron data
        electron_data = reader.read_electron_data()
        assert electron_data.density.shape == (len(density),)
        assert electron_data.temperature.shape == (len(temperature),)
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_sesame_eos_from_written_file():
    """Test creating a SesameEos from a written file."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create smaller density and temperature arrays to avoid formatting issues
        density = np.array([0.1, 1.0, 10.0])
        temperature = np.array([100.0, 1000.0, 10000.0])

        # Create an EOS object to write
        eos = IdealGamma(5 / 3, 1.008, 1)  # Hydrogen
        ion_eos, electron_eos = ZSplit(eos)

        # Write the file
        with SesameWriter(file_name, 9999) as writer:
            writer.write(eos, ion_eos, electron_eos, density, temperature)

        # Create SesameEos objects for each component
        total_eos = SesameEos(file_name, component="total")
        ion_eos_read = SesameEos(file_name, component="ion")
        electron_eos_read = SesameEos(file_name, component="electron")

        # Check that the EOS has the expected properties
        assert total_eos.Z == 1.0
        assert total_eos.A == 1.008
        assert total_eos.material_id == 9999

        # Check that the dataset has the expected structure
        assert "internal_energy" in total_eos.dataset
        assert "pressure" in total_eos.dataset
        assert "helmholtz_free_energy" in total_eos.dataset

        # Test interpolation at a point
        test_density = 1.0
        test_temperature = 1000.0
        energy = total_eos.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )
        pressure = total_eos.PressureFromDensityTemperature(
            test_density, test_temperature
        )

        # Values should be finite
        assert np.isfinite(energy)
        assert np.isfinite(pressure)

        # For IdealGamma, we can calculate the expected values
        expected_energy = eos.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )
        expected_pressure = eos.PressureFromDensityTemperature(
            test_density, test_temperature
        )

        # Check that the interpolated values are close to the expected values
        np.testing.assert_allclose(energy, expected_energy, rtol=1e-1)
        np.testing.assert_allclose(pressure, expected_pressure, rtol=1e-1)

        # Test ion and electron components
        ion_energy = ion_eos_read.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )
        electron_energy = electron_eos_read.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )

        # Total energy should be approximately the sum of ion and electron energies
        np.testing.assert_allclose(energy, ion_energy + electron_energy, rtol=1e-1)
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)
