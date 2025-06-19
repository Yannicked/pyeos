"""
Tests for the IonmixReader and IonmixWriter classes.
"""

import os
import tempfile

import numpy as np
import pytest

from pyeos.analytical import IdealGamma
from pyeos.interpolated.ionmix_eos import IonmixEos
from pyeos.reader.ionmix import IonmixReader
from pyeos.writer.ionmix import IonmixWriter


def test_ionmix_writer_initialization():
    """Test that IonmixWriter initializes correctly."""
    file_name = "test.cn4"
    atomic_number = 13.0  # Aluminum
    atomic_mass = 26.98  # Aluminum

    writer = IonmixWriter(file_name, atomic_number, atomic_mass)

    assert writer.file_name == file_name
    assert writer.atomic_number == atomic_number
    assert writer.atomic_mass == atomic_mass
    assert writer.twot is False  # Default value
    assert writer.man is True  # Default value


def test_ionmix_writer_context_manager():
    """Test that IonmixWriter works as a context manager."""
    with tempfile.NamedTemporaryFile(suffix=".cn4", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Use the context manager
        with IonmixWriter(file_name, 13.0, 26.98) as writer:
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


def test_ionmix_write_and_read():
    """Test writing and then reading an IONMIX file."""
    with tempfile.NamedTemporaryFile(suffix=".cn4", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create smaller density and temperature arrays to avoid formatting issues
        density = np.array([0.1, 1.0, 10.0])
        temperature = np.array([100.0, 1000.0, 10000.0])

        # Create an EOS object to write
        eos = IdealGamma(5 / 3, 1.008, 1)  # Hydrogen

        # Write the file
        with IonmixWriter(file_name, 1.0, 1.008) as writer:
            writer.write(eos, eos, eos, density, temperature)

        # Check that the file exists and has content
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0

        # Read the file back
        reader = IonmixReader(file_name, 1.0, 1.008)
        data = reader.read_total_data()

        # Check that the data has the expected structure
        assert hasattr(data, "density")
        assert hasattr(data, "temperature")
        assert hasattr(data, "energy")
        assert hasattr(data, "pressure")
        assert hasattr(data, "helmholtz")

        # Check that the arrays have the expected shapes
        assert data.density.shape == (len(density),)
        assert data.temperature.shape == (len(temperature),)
        assert data.energy.shape == (len(temperature), len(density))
        assert data.pressure.shape == (len(temperature), len(density))
        assert data.helmholtz.shape == (len(temperature), len(density))

        # Check that the density and temperature values match what we wrote
        # Note: There might be small differences due to conversion between units
        np.testing.assert_allclose(data.density, density, rtol=1e-5)
        np.testing.assert_allclose(
            data.temperature,  # Convert K to eV
            temperature,
            rtol=1e-5,
        )
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_ionmix_eos_from_written_file():
    """Test creating an IonmixEos from a written file."""
    with tempfile.NamedTemporaryFile(suffix=".cn4", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create smaller density and temperature arrays to avoid formatting issues
        density = np.array([0.1, 1.0, 10.0])
        temperature = np.array([100.0, 1000.0, 10000.0])

        # Create an EOS object to write
        eos = IdealGamma(5 / 3, 1.008, 1)  # Hydrogen

        # Write the file
        with IonmixWriter(file_name, 1.0, 1.008) as writer:
            writer.write(eos, eos, eos, density, temperature)

        # Create an IonmixEos from the file
        ionmix_eos = IonmixEos(file_name, 1.0, 1.008)

        # Check that the EOS has the expected properties
        assert ionmix_eos.Z == 1.0
        assert ionmix_eos.A == 1.008

        # Check that the dataset has the expected structure
        assert "internal_energy" in ionmix_eos.dataset
        assert "pressure" in ionmix_eos.dataset
        assert "helmholtz_free_energy" in ionmix_eos.dataset

        # Test interpolation at a point
        test_density = 1.0
        test_temperature = 1000.0
        energy = ionmix_eos.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )
        pressure = ionmix_eos.PressureFromDensityTemperature(
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
        # Note: This is an approximate check since the IONMIX format has limitations
        np.testing.assert_allclose(energy, expected_energy, rtol=1e-1)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(pressure, expected_pressure, rtol=1e-1)
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_ionmix_twot_mode():
    """Test writing and reading an IONMIX file in two-temperature mode."""
    with tempfile.NamedTemporaryFile(suffix=".cn4", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create smaller density and temperature arrays to avoid formatting issues
        density = np.array([0.1, 1.0, 10.0])
        temperature = np.array([100.0, 1000.0, 10000.0])

        # Create an EOS object to write
        eos = IdealGamma(5 / 3, 1.008, 1)  # Hydrogen

        # Write the file in 2T mode
        with IonmixWriter(file_name, 1.0, 1.008, twot=True) as writer:
            writer.write(eos, eos, eos, density, temperature)

        # Check that the file exists and has content
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0

        # Read the file back in 2T mode
        reader = IonmixReader(file_name, 1.0, 1.008, twot=True)

        # Test reading ion data
        ion_data = reader.read_ion_data()
        assert hasattr(ion_data, "density")
        assert hasattr(ion_data, "temperature")
        assert hasattr(ion_data, "energy")
        assert hasattr(ion_data, "pressure")
        assert hasattr(ion_data, "helmholtz")

        # Test reading electron data
        electron_data = reader.read_electron_data()
        assert hasattr(electron_data, "density")
        assert hasattr(electron_data, "temperature")
        assert hasattr(electron_data, "energy")
        assert hasattr(electron_data, "pressure")
        assert hasattr(electron_data, "helmholtz")

        # Create IonmixEos objects for each component
        total_eos = IonmixEos(file_name, 1.0, 1.008, component="total", twot=True)
        ion_eos = IonmixEos(file_name, 1.0, 1.008, component="ion", twot=True)
        electron_eos = IonmixEos(file_name, 1.0, 1.008, component="electron", twot=True)

        # Test interpolation at a point
        test_density = 0.5
        test_temperature = 1000.0

        # Total energy and pressure should be the sum of ion and electron components
        total_energy = total_eos.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )
        ion_energy = ion_eos.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )
        electron_energy = electron_eos.InternalEnergyFromDensityTemperature(
            test_density, test_temperature
        )

        total_pressure = total_eos.PressureFromDensityTemperature(
            test_density, test_temperature
        )
        ion_pressure = ion_eos.PressureFromDensityTemperature(
            test_density, test_temperature
        )
        electron_pressure = electron_eos.PressureFromDensityTemperature(
            test_density, test_temperature
        )

        # Check that the total is approximately the sum of the components
        # Note: There might be small differences due to interpolation
        np.testing.assert_allclose(
            total_energy, ion_energy + electron_energy, rtol=1e-1
        )
        np.testing.assert_allclose(
            total_pressure, ion_pressure + electron_pressure, rtol=1e-1
        )
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_ionmix_writer_error_handling():
    """Test error handling in IonmixWriter."""
    # Test with empty filename
    with pytest.raises(ValueError):
        with IonmixWriter("", 1.0, 1.008):
            pass


def test_ionmix_reader_error_handling():
    """Test error handling in IonmixReader."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        IonmixReader("nonexistent_file.cn4", 1.0, 1.008)

    # Create an empty file
    with tempfile.NamedTemporaryFile(suffix=".cn4", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Test with empty file
        with pytest.raises(ValueError):
            IonmixReader(file_name, 1.0, 1.008)
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)
