"""
Tests for the types module.
"""

import numpy as np

from pyeos.types import EOSArray, EOSReal


def test_eos_real_scalar():
    """Test that EOSReal works with scalar values."""
    value: EOSReal = 1.0
    assert isinstance(value, float)


def test_eos_real_array():
    """Test that EOSReal works with numpy arrays."""
    value: EOSReal = np.array([1.0, 2.0, 3.0])
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float64


def test_eos_array():
    """Test that EOSArray works with numpy arrays."""
    value: EOSArray = np.array([1.0, 2.0, 3.0])
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float64


def test_eos_real_operations():
    """Test operations with EOSReal values."""
    # Scalar operations
    scalar1: EOSReal = 2.0
    scalar2: EOSReal = 3.0
    assert scalar1 + scalar2 == 5.0
    assert scalar1 * scalar2 == 6.0

    # Array operations
    array1: EOSReal = np.array([1.0, 2.0, 3.0])
    array2: EOSReal = np.array([4.0, 5.0, 6.0])
    assert np.array_equal(array1 + array2, np.array([5.0, 7.0, 9.0]))
    assert np.array_equal(array1 * array2, np.array([4.0, 10.0, 18.0]))

    # Mixed operations
    scalar: EOSReal = 2.0
    array: EOSReal = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(scalar * array, np.array([2.0, 4.0, 6.0]))
    assert np.array_equal(array + scalar, np.array([3.0, 4.0, 5.0]))
