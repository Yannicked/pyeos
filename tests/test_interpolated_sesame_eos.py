"""
Tests for the SesameEos class.
"""

import os
import tempfile

import numpy as np
import pytest

from pyeos.analytical import IdealGamma
from pyeos.interpolated import SesameEos
from pyeos.modifiers import ZSplit
from pyeos.writer.sesame import SesameWriter


@pytest.fixture
def temp_sesame_file():
    """Create a temporary SESAME file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create EOS objects
        eos = IdealGamma(5 / 3, 1.008, 1)
        ion_eos, electron_eos = ZSplit(eos)

        # Write the file
        with SesameWriter(file_name, 9999) as writer:
            writer.add_comment("Test SESAME file for interpolation tests")
            writer.write(eos, ion_eos, electron_eos)

        yield file_name
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_sesame_eos_initialization(temp_sesame_file):
    """Test that SesameEos initializes correctly."""
    # Test with default component (total)
    eos = SesameEos(temp_sesame_file)
    assert eos.component == "total"
    assert eos.file_name == temp_sesame_file
    assert eos.Z == 1.0
    assert eos.A == 1.008

    # Test with ion component
    ion_eos = SesameEos(temp_sesame_file, component="ion")
    assert ion_eos.component == "ion"

    # Test with electron component
    electron_eos = SesameEos(temp_sesame_file, component="electron")
    assert electron_eos.component == "electron"

    # Test with invalid component
    with pytest.raises(ValueError):
        SesameEos(temp_sesame_file, component="invalid")


def test_sesame_eos_xarray_dataset(temp_sesame_file):
    """Test that SesameEos creates proper xarray dataset."""
    eos = SesameEos(temp_sesame_file)

    # Check that dataset has the expected variables
    assert "internal_energy" in eos.dataset
    assert "pressure" in eos.dataset
    assert "helmholtz_free_energy" in eos.dataset

    # Check that dataset has the expected dimensions
    assert "temperature" in eos.dataset.dims
    assert "density" in eos.dataset.dims

    # Check that dataset has the expected coordinates
    assert "temperature" in eos.dataset.coords
    assert "density" in eos.dataset.coords

    # Check that data arrays have the expected attributes
    assert eos.energy_da.name == "internal_energy"
    assert eos.pressure_da.name == "pressure"
    assert eos.helmholtz_da.name == "helmholtz_free_energy"


def test_sesame_eos_interpolation(temp_sesame_file):
    """Test that SesameEos interpolates correctly."""
    eos = SesameEos(temp_sesame_file)

    # Test scalar interpolation
    rho = 1.0
    temperature = 1000.0

    energy = eos.InternalEnergyFromDensityTemperature(rho, temperature)
    pressure = eos.PressureFromDensityTemperature(rho, temperature)
    helmholtz = eos.HelmholtzFreeEnergyFromDensityTemperature(rho, temperature)

    assert isinstance(energy, float)
    assert isinstance(pressure, float)
    assert isinstance(helmholtz, float)

    # Test array interpolation
    rho_array = np.array([0.1, 1.0, 10.0])
    temp_array = np.array([100.0, 1000.0, 10000.0])

    rho_grid, temp_grid = np.meshgrid(rho_array, temp_array)

    energy_array = eos.InternalEnergyFromDensityTemperature(rho_grid, temp_grid)
    pressure_array = eos.PressureFromDensityTemperature(rho_grid, temp_grid)
    helmholtz_array = eos.HelmholtzFreeEnergyFromDensityTemperature(rho_grid, temp_grid)

    assert energy_array.shape == rho_grid.shape
    assert pressure_array.shape == rho_grid.shape
    assert helmholtz_array.shape == rho_grid.shape


def test_sesame_eos_properties(temp_sesame_file):
    """Test that SesameEos properties match expected values."""
    # Create reference EOS
    ref_eos = IdealGamma(5 / 3, 1.008, 1)

    # Create SesameEos
    eos = SesameEos(temp_sesame_file)

    # Test at a few points
    rho_values = [0.1, 1.0, 10.0]
    temp_values = [100.0, 1000.0, 10000.0]

    for rho in rho_values:
        for temp in temp_values:
            # Calculate reference values
            ref_energy = ref_eos.InternalEnergyFromDensityTemperature(rho, temp)
            ref_pressure = ref_eos.PressureFromDensityTemperature(rho, temp)

            # Calculate interpolated values
            interp_energy = eos.InternalEnergyFromDensityTemperature(rho, temp)
            interp_pressure = eos.PressureFromDensityTemperature(rho, temp)

            # Check that values are close (within 5%)
            assert np.isclose(interp_energy, ref_energy, rtol=0.05)
            assert np.isclose(interp_pressure, ref_pressure, rtol=0.05)
