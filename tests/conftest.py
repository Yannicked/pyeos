"""
Common pytest fixtures for pyeos tests.
"""

import numpy as np
import pytest

from pyeos.analytical import IdealGamma


@pytest.fixture
def ideal_gamma_eos():
    """
    Fixture providing an IdealGamma EOS instance for hydrogen.

    Returns an ideal gamma-law EOS with:
    - gamma = 5/3 (monatomic gas)
    - A = 1.008 (hydrogen atomic mass)
    - Z = 1 (hydrogen atomic number)
    """
    return IdealGamma(5 / 3, 1.008, 1)


@pytest.fixture
def density_values():
    """
    Fixture providing a range of density values for testing.

    Returns a numpy array of density values from 1e-6 to 10 g/cm³.
    """
    return np.geomspace(1e-6, 10, 10)


@pytest.fixture
def temperature_values():
    """
    Fixture providing a range of temperature values for testing.

    Returns a numpy array of temperature values from 1 to 1e6 K.
    """
    return np.geomspace(1, 1e6, 10)
