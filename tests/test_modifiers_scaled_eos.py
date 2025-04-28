"""
Tests for the ScaledEos modifier.
"""

import numpy as np

from pyeos.analytical import IdealGamma
from pyeos.modifiers import ScaledEos


def test_scaled_eos_initialization(ideal_gamma_eos):
    """Test that ScaledEos initializes correctly."""

    # Define a simple scaling function
    def scale_fn(rho, T):
        return 0.5

    scaled_eos = ScaledEos(ideal_gamma_eos, scale_fn)

    # Check that the base EOS is stored correctly
    assert scaled_eos.eos is ideal_gamma_eos
    assert scaled_eos.scale_fn is scale_fn

    # Check that properties are passed through
    assert scaled_eos.A == ideal_gamma_eos.A
    assert scaled_eos.Z == ideal_gamma_eos.Z


def test_scaled_internal_energy(ideal_gamma_eos, density_values, temperature_values):
    """Test that internal energy is correctly scaled."""

    # Define a scaling function that depends on density
    def scale_fn(rho, T):
        return 0.5 * rho

    scaled_eos = ScaledEos(ideal_gamma_eos, scale_fn)

    for rho in density_values:
        for T in temperature_values:
            base_energy = ideal_gamma_eos.InternalEnergyFromDensityTemperature(rho, T)
            scaled_energy = scaled_eos.InternalEnergyFromDensityTemperature(rho, T)
            expected = base_energy * scale_fn(rho, T)

            assert np.isclose(scaled_energy, expected)


def test_scaled_pressure(ideal_gamma_eos, density_values, temperature_values):
    """Test that pressure is correctly scaled."""

    # Define a scaling function that depends on temperature
    def scale_fn(rho, T):
        return 2.0 * T / 1000.0

    scaled_eos = ScaledEos(ideal_gamma_eos, scale_fn)

    for rho in density_values:
        for T in temperature_values:
            base_pressure = ideal_gamma_eos.PressureFromDensityTemperature(rho, T)
            scaled_pressure = scaled_eos.PressureFromDensityTemperature(rho, T)
            expected = base_pressure * scale_fn(rho, T)

            assert np.isclose(scaled_pressure, expected)


def test_scaled_helmholtz(ideal_gamma_eos, density_values, temperature_values):
    """Test that Helmholtz free energy is correctly scaled."""

    # Define a scaling function that depends on both density and temperature
    def scale_fn(rho, T):
        return 1.0 + 0.1 * rho * T / 1000.0

    scaled_eos = ScaledEos(ideal_gamma_eos, scale_fn)

    for rho in density_values:
        for T in temperature_values:
            base_helmholtz = ideal_gamma_eos.HelmholtzFreeEnergyFromDensityTemperature(
                rho, T
            )
            scaled_helmholtz = scaled_eos.HelmholtzFreeEnergyFromDensityTemperature(
                rho, T
            )
            expected = base_helmholtz * scale_fn(rho, T)

            assert np.isclose(scaled_helmholtz, expected)


def test_array_scaling():
    """Test that scaling works with array inputs."""
    # Create a base EOS
    base_eos = IdealGamma(5 / 3, 1.0, 1.0)

    # Define a scaling function
    def scale_fn(rho, T):
        return rho / T

    scaled_eos = ScaledEos(base_eos, scale_fn)

    # Create array inputs
    rho = np.array([0.1, 1.0, 10.0])
    T = np.array([100.0, 1000.0, 10000.0])

    # Test with 1D arrays
    base_energy = base_eos.InternalEnergyFromDensityTemperature(rho, T)
    scaled_energy = scaled_eos.InternalEnergyFromDensityTemperature(rho, T)
    expected = base_energy * scale_fn(rho, T)

    assert np.allclose(scaled_energy, expected)

    # Test with 2D arrays (meshgrid)
    rho_grid, T_grid = np.meshgrid(rho, T)
    base_energy = base_eos.InternalEnergyFromDensityTemperature(rho_grid, T_grid)
    scaled_energy = scaled_eos.InternalEnergyFromDensityTemperature(rho_grid, T_grid)
    expected = base_energy * scale_fn(rho_grid, T_grid)

    assert np.allclose(scaled_energy, expected)
