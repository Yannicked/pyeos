#!/usr/bin/env python3
"""
Example script demonstrating the use of SesameEos for interpolating SESAME tables.

This script shows how to read a SESAME table file and create an equation of state
that interpolates over the tabulated data using xarray.
"""

import click
import matplotlib.pyplot as plt
import numpy as np

from pyeos.interpolated import SesameEos


def plot_eos_properties(eos, density_range, temperature_range, output_prefix=None):
    """
    Plot EOS properties over a range of densities and temperatures.

    Parameters
    ----------
    eos : SesameEos
        The equation of state object
    density_range : tuple
        (min_density, max_density, num_points)
    temperature_range : tuple
        (min_temperature, max_temperature, num_points)
    output_prefix : str, optional
        Prefix for output files, by default None
    """
    # Create density and temperature grids
    rho_min, rho_max, rho_points = density_range
    temp_min, temp_max, temp_points = temperature_range

    rho = np.geomspace(rho_min, rho_max, rho_points)
    temperature = np.geomspace(temp_min, temp_max, temp_points)

    rho_grid, temp_grid = np.meshgrid(rho, temperature)

    # Calculate EOS properties
    energy = eos.InternalEnergyFromDensityTemperature(rho_grid, temp_grid)
    pressure = eos.PressureFromDensityTemperature(rho_grid, temp_grid)
    helmholtz = eos.HelmholtzFreeEnergyFromDensityTemperature(rho_grid, temp_grid)

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot internal energy
    im0 = axes[0].pcolormesh(
        rho_grid, temp_grid, np.log10(np.abs(energy)), shading="auto"
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Density (g/cm³)")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Internal Energy (log10, erg/g)")
    plt.colorbar(im0, ax=axes[0])

    # Plot pressure
    im1 = axes[1].pcolormesh(
        rho_grid, temp_grid, np.log10(np.abs(pressure)), shading="auto"
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Density (g/cm³)")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_title("Pressure (log10, dyn/cm²)")
    plt.colorbar(im1, ax=axes[1])

    # Plot Helmholtz free energy
    im2 = axes[2].pcolormesh(
        rho_grid, temp_grid, np.log10(np.abs(helmholtz)), shading="auto"
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Density (g/cm³)")
    axes[2].set_ylabel("Temperature (K)")
    axes[2].set_title("Helmholtz Free Energy (log10, erg/g)")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    # Save or show the plot
    if output_prefix:
        plt.savefig(f"{output_prefix}_properties.png", dpi=300)
        click.echo(f"Plot saved to {output_prefix}_properties.png")
    else:
        plt.show()


@click.command()
@click.argument("sesame_file", type=click.Path(exists=True))
@click.option(
    "--component",
    type=click.Choice(["total", "ion", "electron"]),
    default="total",
    help="Which component to use",
)
@click.option(
    "--rho-min",
    type=float,
    default=1e-4,
    help="Minimum density (g/cm³)",
)
@click.option(
    "--rho-max",
    type=float,
    default=100.0,
    help="Maximum density (g/cm³)",
)
@click.option(
    "--temp-min",
    type=float,
    default=1e2,
    help="Minimum temperature (K)",
)
@click.option(
    "--temp-max",
    type=float,
    default=1e8,
    help="Maximum temperature (K)",
)
@click.option(
    "--points",
    type=int,
    default=100,
    help="Number of points in each dimension",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output file prefix",
)
def main(
    sesame_file,
    component,
    rho_min,
    rho_max,
    temp_min,
    temp_max,
    points,
    output,
):
    """
    Interpolate and visualize SESAME table data.

    This script loads a SESAME format file, creates an equation of state
    that interpolates over the tabulated data, and generates plots of
    the EOS properties.
    """
    # Create the EOS object
    click.echo(f"Loading SESAME file: {sesame_file}")
    eos = SesameEos(sesame_file, component=component)
    click.echo(f"Loaded EOS for material with Z={eos.Z}, A={eos.A}")

    # Plot EOS properties
    density_range = (rho_min, rho_max, points)
    temperature_range = (temp_min, temp_max, points)

    click.echo("Generating plots...")
    plot_eos_properties(eos, density_range, temperature_range, output)
    click.echo("Done!")


if __name__ == "__main__":
    main()
