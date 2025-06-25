"""
Equation of state modifiers.

This subpackage provides modifiers that can be applied to existing
equation of state models to alter their behavior or combine them
in various ways.
"""

from .scaled_eos import ScaledEos
from .z_split import ZSplit
from .ramps_eos import BilinearRampEos

__all__ = ["ScaledEos", "ZSplit", "BilinearRampEos"]
