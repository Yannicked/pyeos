"""
Tests for the SesameWriter class.
"""

import os
import tempfile

import numpy as np

from pyeos.analytical import IdealGamma
from pyeos.modifiers import ZSplit
from pyeos.writer.sesame import SesameWriter


def test_sesame_writer_initialization():
    """Test that SesameWriter initializes correctly."""
    file_name = "test.sesame"
    material_id = 9999

    writer = SesameWriter(file_name, material_id)

    assert writer.file_name == file_name
    assert writer.material_id == material_id
    assert writer.comment_table == 101
    assert writer.extra_comments == []


def test_sesame_writer_context_manager():
    """Test that SesameWriter works as a context manager."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Use the context manager
        with SesameWriter(file_name, 9999) as writer:
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


def test_add_comment():
    """Test adding comments to the writer."""
    writer = SesameWriter("test.sesame", 9999)

    # Add some comments
    writer.add_comment("Test comment 1")
    writer.add_comment("Test comment 2")

    assert len(writer.extra_comments) == 2
    assert writer.extra_comments[0] == "Test comment 1"
    assert writer.extra_comments[1] == "Test comment 2"


def test_write_full_file():
    """Test writing a complete SESAME file."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create EOS objects
        eos = IdealGamma(5 / 3, 1.008, 1)
        ion_eos, electron_eos = ZSplit(eos)

        # Write the file
        with SesameWriter(file_name, 9999) as writer:
            writer.add_comment("Test SESAME file")
            writer.write(eos, ion_eos, electron_eos)

        # Check that the file exists and has content
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0

        # Basic check of file content
        with open(file_name, "r") as f:
            content = f.read()
            # Check for expected header elements
            assert " 0  9999   101" in content  # Comment table header
            assert " 1  9999   201" in content  # Material data header
            assert " 1  9999   301" in content  # Total EOS data header
            assert " 1  9999   303" in content  # Ion EOS data header
            assert " 1  9999   304" in content  # Electron EOS data header
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)


def test_write_data_methods():
    """Test the individual data writing methods."""
    with tempfile.NamedTemporaryFile(suffix=".sesame", delete=False) as temp_file:
        file_name = temp_file.name

    try:
        # Create test data
        density = np.array([0.1, 1.0, 10.0])
        temperature = np.array([100.0, 1000.0, 10000.0])
        energy = np.ones_like(density) * 1e6
        pressure = np.ones_like(density) * 1e5
        helmholtz = np.ones_like(density) * 1e4

        # Write the file with individual methods
        with SesameWriter(file_name, 9999) as writer:
            writer.write_total_data(density, temperature, energy, pressure, helmholtz)
            writer.write_ion_data(density, temperature, energy, pressure, helmholtz)
            writer.write_electron_data(
                density, temperature, energy, pressure, helmholtz
            )
            writer.write_table_end()

        # Check that the file exists and has content
        assert os.path.exists(file_name)
        assert os.path.getsize(file_name) > 0

        # Basic check of file content
        with open(file_name, "r") as f:
            content = f.read()
            # Check for expected table headers
            assert " 1  9999   301" in content  # Total EOS data header
            assert " 1  9999   303" in content  # Ion EOS data header
            assert " 1  9999   304" in content  # Electron EOS data header
            assert " 2" in content  # Table end marker
    finally:
        # Clean up
        if os.path.exists(file_name):
            os.remove(file_name)
