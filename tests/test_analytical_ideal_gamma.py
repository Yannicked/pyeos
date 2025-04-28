"""
Tests for the IdealGamma equation of state.
"""

import numpy as np
from scipy.constants import gas_constant

from pyeos.analytical import IdealGamma


def test_ideal_gamma_initialization():
    """Test that IdealGamma initializes with correct parameters."""
    gamma = 5 / 3
    A = 1.008
    Z = 1
    eos = IdealGamma(gamma, A, Z)

    assert eos.gamma == gamma
    assert eos.A == A
    assert eos.Z == Z
    assert eos.const == gas_constant / A * 1e7


def test_internal_energy(ideal_gamma_eos, density_values, temperature_values):
    """Test internal energy calculation for ideal gamma law."""
    # For ideal gas, internal energy depends only on temperature
    for rho in density_values:
        for T in temperature_values:
            energy = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho, T)
            expected = ideal_gamma_eos.const / (ideal_gamma_eos.gamma - 1) * T
            assert np.isclose(energy, expected)


def test_pressure(ideal_gamma_eos, density_values, temperature_values):
    """Test pressure calculation for ideal gamma law."""
    # For ideal gas, P = rho * R * T / A
    for rho in density_values:
        for T in temperature_values:
            pressure = ideal_gamma_eos.PressureFromDensityTemperature(rho, T)
            expected = ideal_gamma_eos.const * rho * T
            assert np.isclose(pressure, expected)


def test_helmholtz_free_energy(ideal_gamma_eos, density_values, temperature_values):
    """Test Helmholtz free energy calculation for ideal gamma law."""
    for rho in density_values:
        for T in temperature_values:
            energy = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho, T)
            helmholtz = ideal_gamma_eos.HelmholtzFreeEnergyFromDensityTemperature(
                rho, T
            )
            expected = energy * np.log(rho ** (ideal_gamma_eos.gamma - 1) / T)
            assert np.isclose(helmholtz, expected)


def test_array_inputs(ideal_gamma_eos):
    """Test that the EOS works with array inputs."""
    rho = np.array([0.1, 1.0, 10.0])
    T = np.array([100.0, 1000.0, 10000.0])

    # Test with 1D arrays
    energy_1d = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho, T)
    pressure_1d = ideal_gamma_eos.PressureFromDensityTemperature(rho, T)
    helmholtz_1d = ideal_gamma_eos.HelmholtzFreeEnergyFromDensityTemperature(rho, T)

    assert energy_1d.shape == rho.shape
    assert pressure_1d.shape == rho.shape
    assert helmholtz_1d.shape == rho.shape

    # Test with 2D arrays (meshgrid)
    rho_grid, T_grid = np.meshgrid(rho, T)
    energy_2d = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho_grid, T_grid)
    pressure_2d = ideal_gamma_eos.PressureFromDensityTemperature(rho_grid, T_grid)
    helmholtz_2d = ideal_gamma_eos.HelmholtzFreeEnergyFromDensityTemperature(
        rho_grid, T_grid
    )

    assert energy_2d.shape == rho_grid.shape
    assert pressure_2d.shape == rho_grid.shape
    assert helmholtz_2d.shape == rho_grid.shape


def test_zero_temperature_behavior(ideal_gamma_eos):
    """Test behavior at zero temperature."""
    rho = np.array([0.1, 1.0, 10.0])
    T = 0.0

    energy = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho, T)
    pressure = ideal_gamma_eos.PressureFromDensityTemperature(rho, T)

    # At zero temperature, energy and pressure should be zero
    assert np.all(energy == 0.0)
    assert np.all(pressure == 0.0)
