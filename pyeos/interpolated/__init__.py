"""
Interpolated equation of state module.

This module provides implementations of equation of state models
that use interpolation over tabulated data.
"""

from .ionmix_eos import IonmixEos
from .sesame_eos import SesameEos

__all__ = ["SesameEos", "IonmixEos"]
