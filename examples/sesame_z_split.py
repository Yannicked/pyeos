#!/usr/bin/env python3
"""
Example script demonstrating the use of ZSplit to create ion and electron tables
"""

import click

from pyeos.interpolated import SesameEos
from pyeos.modifiers import ZSplit
from pyeos.writer.sesame import SesameWriter


@click.command()
@click.argument("sesame_file", type=click.Path(exists=True))
@click.option(
    "--output",
    type=str,
    default="z_split.sesame",
    help="Output file name (default: z_split.sesame)",
)
def main(sesame_file, output):
    """
    This script loads a SESAME format file, splits the EOS into ion and
    electron components, and writes the result to a new SESAME file.
    """
    # Create the EOS object
    click.echo(f"Loading SESAME file: {sesame_file}")
    eos = SesameEos(sesame_file)
    click.echo(f"Loaded EOS for material with Z={eos.Z}, A={eos.A}")

    # Split the EOS into electron and ion components
    click.echo("Splitting EOS into ion and electron components...")
    ion_eos, electron_eos = ZSplit(eos)

    # Write the new SESAME file
    click.echo(f"Writing SESAME table to {output}")
    with SesameWriter(output, eos.material_id) as writer:
        writer.add_comment(
            (
                "This table was generated with pyeos by splitting "
                f"{sesame_file} into ion and electron components."
            )
        )
        writer.write(
            eos,
            ion_eos,
            electron_eos,
            # Keep the grid of the original sesame file
            density=eos.dataset.coords["density"].values,
            temperature=eos.dataset.coords["temperature"].values,
        )

    click.echo(f"SESAME table written to {output}")


if __name__ == "__main__":
    main()
