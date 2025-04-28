"""
Analytical equation of state models.

This subpackage provides analytical equation of state implementations
that can be used for thermodynamic calculations.
"""

from .ideal_gamma import IdealGamma

__all__ = ["IdealGamma"]
