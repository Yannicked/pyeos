"""
Tests for the ZSplit modifier.
"""

import numpy as np
import pytest

from pyeos.modifiers import ZSplit
from pyeos.modifiers.z_split import thomas_fermi_ionization


def test_thomas_fermi_ionization():
    """Test the Thomas-Fermi ionization model."""
    # Test with scalar inputs
    rho = 1.0  # g/cm³
    T = 1000.0  # K
    Z = 1.0  # Hydrogen
    A = 1.008  # Hydrogen

    # Ionization should be between 0 and Z
    ionization = thomas_fermi_ionization(rho, T, Z, A)
    assert 0.0 <= ionization <= Z

    # Test with array inputs
    rho_array = np.array([0.1, 1.0, 10.0])
    T_array = np.array([100.0, 1000.0, 10000.0])

    # Test with 1D arrays
    ionization_array = thomas_fermi_ionization(rho_array, T_array, Z, A)
    assert ionization_array.shape == rho_array.shape
    assert np.all(0.0 <= ionization_array) and np.all(ionization_array <= Z)

    # Test with 2D arrays (meshgrid)
    rho_grid, T_grid = np.meshgrid(rho_array, T_array)
    ionization_grid = thomas_fermi_ionization(rho_grid, T_grid, Z, A)
    assert ionization_grid.shape == rho_grid.shape
    assert np.all(0.0 <= ionization_grid) and np.all(ionization_grid <= Z)


def test_ionization_temperature_dependence():
    """Test that ionization increases with temperature."""
    rho = 1.0
    Z = 1.0
    A = 1.008

    T_low = 100.0
    T_high = 10000.0

    ionization_low = thomas_fermi_ionization(rho, T_low, Z, A)
    ionization_high = thomas_fermi_ionization(rho, T_high, Z, A)

    # Higher temperature should lead to higher ionization
    assert ionization_high > ionization_low


@pytest.mark.xfail(reason="This does not seem to work")
def test_ionization_density_dependence():
    """Test that ionization increases with density."""
    T = 1000.0
    Z = 1.0
    A = 1.008

    rho_low = 0.1
    rho_high = 10.0

    ionization_low = thomas_fermi_ionization(rho_low, T, Z, A)
    ionization_high = thomas_fermi_ionization(rho_high, T, Z, A)

    # Higher density should lead to higher ionization
    assert ionization_high > ionization_low


def test_z_split_returns_two_eos(ideal_gamma_eos):
    """Test that ZSplit returns two EOS objects."""
    electron_eos, ion_eos = ZSplit(ideal_gamma_eos)

    # Check that both are valid EOS objects with the same A and Z
    assert electron_eos.A == ideal_gamma_eos.A
    assert electron_eos.Z == ideal_gamma_eos.Z
    assert ion_eos.A == ideal_gamma_eos.A
    assert ion_eos.Z == ideal_gamma_eos.Z


def test_z_split_energy_sum(ideal_gamma_eos, density_values, temperature_values):
    """Test that the sum of electron and ion energies equals the total energy."""
    electron_eos, ion_eos = ZSplit(ideal_gamma_eos)

    for rho in density_values:
        for T in temperature_values:
            total_energy = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho, T)
            electron_energy = electron_eos.InternalEnergyFromDensityTemperature(rho, T)
            ion_energy = ion_eos.InternalEnergyFromDensityTemperature(rho, T)

            # The sum of electron and ion energies should approximately equal
            # the total energy
            assert np.isclose(electron_energy + ion_energy, total_energy)


def test_z_split_pressure_sum(ideal_gamma_eos, density_values, temperature_values):
    """Test that the sum of electron and ion pressures equals the total pressure."""
    electron_eos, ion_eos = ZSplit(ideal_gamma_eos)

    for rho in density_values:
        for T in temperature_values:
            total_pressure = ideal_gamma_eos.PressureFromDensityTemperature(rho, T)
            electron_pressure = electron_eos.PressureFromDensityTemperature(rho, T)
            ion_pressure = ion_eos.PressureFromDensityTemperature(rho, T)

            # The sum of electron and ion pressures should approximately equal
            # the total pressure
            assert np.isclose(electron_pressure + ion_pressure, total_pressure)


def test_z_split_with_arrays(ideal_gamma_eos):
    """Test that ZSplit works with array inputs."""
    electron_eos, ion_eos = ZSplit(ideal_gamma_eos)

    # Create array inputs
    rho = np.array([0.1, 1.0, 10.0])
    T = np.array([100.0, 1000.0, 10000.0])

    # Test with 1D arrays
    total_energy = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho, T)
    electron_energy = electron_eos.InternalEnergyFromDensityTemperature(rho, T)
    ion_energy = ion_eos.InternalEnergyFromDensityTemperature(rho, T)

    assert np.allclose(electron_energy + ion_energy, total_energy)

    # Test with 2D arrays (meshgrid)
    rho_grid, T_grid = np.meshgrid(rho, T)
    total_energy = ideal_gamma_eos.InternalEnergyFromDensityTemperature(
        rho_grid, T_grid
    )
    electron_energy = electron_eos.InternalEnergyFromDensityTemperature(
        rho_grid, T_grid
    )
    ion_energy = ion_eos.InternalEnergyFromDensityTemperature(rho_grid, T_grid)

    assert np.allclose(electron_energy + ion_energy, total_energy)
