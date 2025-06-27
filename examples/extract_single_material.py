import click
from pyeos.interpolated.sesame_eos import SesameEos
from pyeos.writer.sesame import SesameWriter


@click.command()
@click.option("--matid", type=int, required=True, help="Material ID to extract")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def extract_material(matid, input_file, output_file):
    """Extracts a single material from a sesame file and writes it to a new file."""
    # Load the sesame file
    total_eos = SesameEos(input_file, "total", matid)
    ion_eos = SesameEos(input_file, "ion", matid)
    electron_eos = SesameEos(input_file, "electron", matid)

    # Create a new SesameFile object for writing
    sesame_writer = SesameWriter(output_file, matid)

    with sesame_writer as w:
        w.write(
            total_eos,
            ion_eos,
            electron_eos,
            total_eos.dataset.density,
            total_eos.dataset.temperature,
        )

    click.echo(
        f"Material {matid} extracted from {input_file} and written to {output_file}"
    )


if __name__ == "__main__":
    extract_material()
