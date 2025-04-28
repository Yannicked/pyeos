"""
Type definitions for the pyeos package.

This module defines common types used throughout the package.
"""

import numpy as np
from numpy.typing import NDArray

type EOSReal = float | NDArray[np.float64]
"""Type for values that can be either scalar floats or numpy arrays of float64."""

type EOSArray = NDArray[np.float64]
"""Type for numpy arrays of float64 used in EOS calculations."""
