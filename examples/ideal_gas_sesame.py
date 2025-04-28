#!/usr/bin/env python3
"""
Script to generate a SESAME table for a given Z and A.

This script uses the pyeos library to generate a SESAME table for a material
with specified atomic number (Z) and atomic mass number (A).
"""

import os

import click

from pyeos.analytical import IdealGamma
from pyeos.modifiers import ZSplit
from pyeos.writer.sesame import SesameWriter


@click.command()
@click.option("-z", type=float, required=True, help="Atomic number (Z)")
@click.option("-a", type=float, required=True, help="Atomic mass number (A)")
@click.option(
    "--gamma", type=float, default=5 / 3, help="Adiabatic index (default: 5/3)"
)
@click.option(
    "--material-id",
    type=int,
    default=9999,
    help="Material ID for SESAME file (default: 9999)",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output file name (default: z{Z}_a{A}.sesame)",
)
def generate_sesame_table(z, a, gamma, material_id, output):
    """Generate a SESAME table for a material with atomic number Z and atomic
    mass number A."""
    # Create the output file name if not provided
    if output is None:
        output = f"z{z}_a{a}.sesame"

    # Create the EOS objects
    click.echo(f"Creating EOS for Z={z}, A={a}, gamma={gamma}")
    eos = IdealGamma(gamma, a, z)
    ion_eos, electron_eos = ZSplit(eos)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the SESAME file
    click.echo(f"Writing SESAME table to {output}")
    with SesameWriter(output, material_id) as writer:
        writer.add_comment((f"Ideal Gas SESAME table for Z={z}, A={a}, gamma={gamma}"))
        writer.write(eos, ion_eos, electron_eos)

    click.echo(f"SESAME table written to {output}")


if __name__ == "__main__":
    generate_sesame_table()
