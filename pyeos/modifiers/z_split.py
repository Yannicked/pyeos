"""
Z-split equation of state modifier.

This module provides a modifier that splits an equation of state into
electron and ion components based on the Thomas-Fermi ionization model.
"""

import numpy as np

from ..eos import Eos
from ..types import EOSReal
from .scaled_eos import ScaledEos


def thomas_fermi_ionization(
    rho: EOSReal, electron_temperature: EOSReal, Z: float, A: float
) -> EOSReal:
    """
    Calculate the average ionization state using the Thomas-Fermi model.

    This function implements a parameterized Thomas-Fermi ionization model
    that estimates the average ionization state of an element as a function
    of density and temperature.

    Parameters
    ----------
    rho : EOSReal
        Density value(s)
    electron_temperature : EOSReal
        Electron temperature value(s)
    Z : float
        Atomic number
    A : float
        Atomic mass number

    Returns
    -------
    EOSReal
        Average ionization state (between 0 and Z)
    """
    # Model constants
    ALPHA = 14.3139
    BETA = 0.6624

    # Normalize density per electron-bearing nucleon
    reduced_density = rho / (A * Z)

    # Scale temperature by Z^(4/3)
    T_scaled = electron_temperature / Z ** (4.0 / 3.0)

    # Fractional temperature factor in (0,1)
    T_frac = T_scaled / (1.0 + T_scaled)

    # Precompute coefficients for finite-T branch
    a0, a1, a2, a3 = 0.003323, 0.971832, 9.26148e-5, 3.10165
    b0, b1, b2 = -1.7630, 1.43175, 0.31546
    c0, c1 = -0.366667, 0.983333

    # Zero-temperature branch: x0 = alpha * rho1^beta
    x_zero = ALPHA * reduced_density**BETA

    # Finite-temperature branch:
    A1 = a0 * T_scaled**a1 + a2 * T_scaled**a3
    B = -np.exp(b0 + b1 * T_frac + b2 * T_frac**7)
    C = c0 * T_frac + c1

    Q1 = A1 * reduced_density**B
    Q = (reduced_density**C + Q1**C) ** (1.0 / C)
    x_finite = ALPHA * Q**BETA

    # Combine branches: use zero‐T where Te==0, else finite‐T
    x = np.where(electron_temperature == 0, x_zero, x_finite)

    # Convert x to average ionization Zion, bounded between 0 and Z
    Zion = Z * x / (1.0 + x + np.sqrt(1.0 + 2.0 * x))

    return Zion


def ZSplit(eos: Eos):
    """
    Split an equation of state into electron and ion components.

    This function takes an equation of state and returns two new equation of state
    objects: one for electrons and one for ions. The splitting is based on the
    Thomas-Fermi ionization model.

    Parameters
    ----------
    eos : Eos
        The equation of state to split

    Returns
    -------
    tuple[ScaledEos, ScaledEos]
        A tuple containing (electron_eos, ion_eos)
    """

    def electron_scale(rho, electron_temperature):
        ionization = thomas_fermi_ionization(rho, electron_temperature, eos.Z, eos.A)
        return ionization / (ionization + 1)

    def ion_scale(rho, electron_temperature):
        ionization = thomas_fermi_ionization(rho, electron_temperature, eos.Z, eos.A)
        return 1 / (ionization + 1)

    return ScaledEos(eos, electron_scale), ScaledEos(eos, ion_scale)
