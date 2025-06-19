#!/usr/bin/env python3
"""
Script to convert between IONMIX and SESAME equation of state tables.

This script reads an input file in either IONMIX or SESAME format and
converts it to the other format, attempting to preserve the grid structure.
"""

import os
import click
import numpy as np

from pyeos.reader.ionmix import IonmixReader
from pyeos.reader.sesame import SesameReader
from pyeos.writer.ionmix import IonmixWriter
from pyeos.writer.sesame import SesameWriter
from pyeos.interpolated.ionmix_eos import IonmixEos
from pyeos.interpolated.sesame_eos import SesameEos


def detect_format(file_path):
    """Detect the format of the input file based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".cn4", ".imx", ".ionmix"]:
        return "ionmix"
    elif ext in [".ses", ".sesame"]:
        return "sesame"
    else:
        # Default to sesame if can't determine
        return "sesame"


def ionmix_to_sesame(
    input_file, output_file, atomic_number, atomic_mass, material_id, twot=False
):
    """Convert IONMIX format to SESAME format."""
    click.echo(f"Converting IONMIX file {input_file} to SESAME format...")

    # Create EOS objects for total, ion, and electron components
    try:
        total_eos = IonmixEos(
            input_file, atomic_number, atomic_mass, component="total", twot=twot
        )
    except Exception as e:
        click.echo(f"Error: Could not read total EOS data: {e}", err=True)
        return

    try:
        ion_eos = IonmixEos(
            input_file, atomic_number, atomic_mass, component="ion", twot=twot
        )
    except Exception as e:
        click.echo(f"Warning: Could not read ion EOS data: {e}", err=True)
        ion_eos = None

    try:
        electron_eos = IonmixEos(
            input_file, atomic_number, atomic_mass, component="electron", twot=twot
        )
    except Exception as e:
        click.echo(f"Warning: Could not read electron EOS data: {e}", err=True)
        electron_eos = None

    # Extract the density and temperature grids from the xarray dataset
    density = total_eos.dataset.density.values
    temperature = total_eos.dataset.temperature.values

    # Write to SESAME format
    with SesameWriter(output_file, material_id) as writer:
        writer.add_comment(f"Converted from IONMIX file: {input_file}")
        writer.add_comment(
            f"Atomic number: {atomic_number}, Atomic mass: {atomic_mass}"
        )

        if twot:
            writer.add_comment("Two-temperature data")
        else:
            writer.add_comment("Single-temperature data")

        # If we have all components, use the write method
        if twot and ion_eos and electron_eos:
            writer.write(total_eos, ion_eos, electron_eos, density, temperature)
        else:
            # For 1T data, we can only use the total EOS
            writer.write(total_eos, total_eos, total_eos, density, temperature)

    click.echo(f"Conversion complete. Output written to {output_file}")


def sesame_to_ionmix(
    input_file, output_file, atomic_number=None, atomic_mass=None, twot=False
):
    """Convert SESAME format to IONMIX format."""
    click.echo(f"Converting SESAME file {input_file} to IONMIX format...")

    # Create EOS objects for total, ion, and electron components
    try:
        total_eos = SesameEos(input_file, component="total")

        # Use provided atomic number/mass or get from the EOS object
        atomic_number = atomic_number or total_eos.Z
        atomic_mass = atomic_mass or total_eos.A
    except Exception as e:
        click.echo(f"Error: Could not read total EOS data: {e}", err=True)
        return

    try:
        ion_eos = SesameEos(input_file, component="ion")
    except Exception as e:
        click.echo(f"Warning: Could not read ion EOS data: {e}", err=True)
        ion_eos = None

    try:
        electron_eos = SesameEos(input_file, component="electron")
    except Exception as e:
        click.echo(f"Warning: Could not read electron EOS data: {e}", err=True)
        electron_eos = None

    # Extract the density and temperature grids from the xarray dataset
    density = total_eos.dataset.density.values
    temperature = total_eos.dataset.temperature.values

    # Write to IONMIX format
    with IonmixWriter(output_file, atomic_number, atomic_mass, twot=twot) as writer:
        # If we have all components and twot is True, use the write method
        if twot and ion_eos and electron_eos:
            writer.write(total_eos, ion_eos, electron_eos, density, temperature)
        else:
            # For 1T data, we can only use the total EOS
            writer.write(total_eos, total_eos, total_eos, density, temperature)

    click.echo(f"Conversion complete. Output written to {output_file}")


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--input-format",
    type=click.Choice(["auto", "ionmix", "sesame"]),
    default="auto",
    help="Input file format (default: auto-detect)",
)
@click.option("--atomic-number", type=float, help="Atomic number (Z) for IONMIX files")
@click.option(
    "--atomic-mass", type=float, help="Atomic mass (A) in AMU for IONMIX files"
)
@click.option(
    "--material-id", type=int, default=1000, help="Material ID for SESAME files"
)
@click.option("--twot/--no-twot", default=True, help="Use two-temperature data")
def main(
    input_file, output_file, input_format, atomic_number, atomic_mass, material_id, twot
):
    """
    Convert between IONMIX and SESAME equation of state tables.

    This script reads an input file in either IONMIX or SESAME format and
    converts it to the other format, attempting to preserve the grid structure.

    INPUT_FILE: Path to the input file

    OUTPUT_FILE: Path to the output file
    """
    # Determine input format
    if input_format == "auto":
        input_format = detect_format(input_file)

    click.echo(f"Detected input format: {input_format}")

    # Check if we have the required parameters for IONMIX
    if input_format == "ionmix" and (atomic_number is None or atomic_mass is None):
        click.echo(
            "Error: --atomic-number and --atomic-mass are required for IONMIX input files",
            err=True,
        )
        return

    # Perform the conversion
    if input_format == "ionmix":
        ionmix_to_sesame(
            input_file, output_file, atomic_number, atomic_mass, material_id, twot
        )
    else:  # sesame
        sesame_to_ionmix(input_file, output_file, atomic_number, atomic_mass, twot)


if __name__ == "__main__":
    main()
