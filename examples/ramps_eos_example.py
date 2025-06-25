#!/usr/bin/env python3
"""
Example script demonstrating the use of the BilinearRampEos modifier with interactive sliders.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from pyeos.interpolated import SesameEos
from pyeos.modifiers.ramps_eos import BilinearRampEos
from scipy.optimize import minimize


def find_ramp_params(eos, Pc):
    T_const = 293

    r = minimize(
        lambda x: np.abs(Pc - eos.PressureFromDensityTemperature(x, T_const)), (2.7,)
    )
    # rho_low = 1e-3
    # rho_high = 1e2
    # P_low = eos.PressureFromDensityTemperature(rho_low, T_const)
    # P_high = eos.PressureFromDensityTemperature(rho_high, T_const)

    # max_iter = 100
    # tol = 1e-6

    # if np.sign(P_low) == np.sign(P_high):
    #     raise ValueError(
    #         "The pressure at rho_low and rho_high must have opposite signs. "
    #         f"P({rho_low}) = {P_low}, P({rho_high}) = {P_high}"
    #     )

    # for i in range(max_iter):
    #     rho_mid = (rho_low + rho_high) / 2.0
    #     P_mid = eos.PressureFromDensityTemperature(rho_mid, T_const)

    #     if abs(P_mid) < tol or (rho_high - rho_low) / 2.0 < tol:
    #         return rho_mid

    #     if np.sign(P_mid) == np.sign(P_low):
    #         rho_low = rho_mid
    #         P_low = P_mid
    #     else:
    #         rho_high = rho_mid

    # return (rho_low + rho_high) / 2.0
    return r.x[0]


def plot_pressure_comparison(eos, density_range, temperature_range):
    """
    Plot a comparison of pressure between a base EOS and a ramp-modified EOS,
    with sliders to adjust the ramp parameters.
    """
    rho_min, rho_max, rho_points = density_range
    temp_min, temp_max, temp_points = temperature_range

    rho = np.geomspace(rho_min, rho_max, rho_points)
    temperature = np.geomspace(temp_min, temp_max, temp_points)
    rho_grid, temp_grid = np.meshgrid(rho, temperature)

    p_base = eos.PressureFromDensityTemperature(rho_grid, temp_grid)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(bottom=0.25, hspace=0.3)
    axes = axes.flatten()

    Pc = 1e9
    r1 = find_ramp_params(eos, Pc)
    # r1 = 2.7 * 1.01
    # Pc = eos.PressureFromDg

    r0 = 1e-9
    rmid = 0.5 * (r0 + r1)
    Pe = 0.5 * Pc
    a = r0 * Pe / (rmid - r0)
    b = r0 * (Pc - Pe) / (r1 - rmid)
    c = (Pc * rmid - Pe * r1) / (r0 * (Pc - Pe))

    print(r0, a, b, c)
    # r0 = 1
    # a = 1
    # b = 0
    # c = 0

    # Initial ramp EOS
    initial_r0 = 1e-9
    initial_a = 1
    initial_b = 10
    initial_c = 0
    ramp_eos = BilinearRampEos(eos, r0=r0, a=a, b=b, c=c)
    p_ramp = ramp_eos.PressureFromDensityTemperature(rho_grid, temp_grid)
    p_diff = p_ramp - np.abs(p_base)

    # Derivatives
    dpdrho_ramp = ramp_eos.BulkModulusFromDensityTemperature(rho_grid, temp_grid)

    # Approximate dpdT
    p_ramp_plus_dt = ramp_eos.PressureFromDensityTemperature(rho_grid, temp_grid * 1.01)
    dpdt_ramp = (p_ramp_plus_dt - p_ramp) / (temp_grid * 0.01)

    # Plot base pressure
    im0 = axes[0].pcolormesh(
        rho_grid, temp_grid, np.log10(p_base + 1e-100), shading="auto"
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Density (g/cm³)")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Base EOS Pressure (log10)")
    plt.colorbar(im0, ax=axes[0])

    # Plot ramp pressure
    im1 = axes[1].pcolormesh(rho_grid, temp_grid, np.log10(p_ramp), shading="auto")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Density (g/cm³)")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_title("Ramp EOS Pressure (log10)")
    cb1 = plt.colorbar(im1, ax=axes[1])

    # Plot pressure difference
    im2 = axes[2].pcolormesh(
        rho_grid, temp_grid, np.log10(np.abs(p_diff)), shading="auto"
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Density (g/cm³)")
    axes[2].set_ylabel("Temperature (K)")
    axes[2].set_title("Pressure Difference (log10)")
    cb2 = plt.colorbar(im2, ax=axes[2])

    # Plot dP/d(rho)
    im3 = axes[3].pcolormesh(
        rho_grid, temp_grid, np.abs(dpdrho_ramp / p_ramp), shading="auto"
    )
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Density (g/cm³)")
    axes[3].set_ylabel("Temperature (K)")
    axes[3].set_title("dP/drho (log10)")
    cb3 = plt.colorbar(im3, ax=axes[3])

    # Plot dP/dT
    im4 = axes[4].pcolormesh(
        rho_grid, temp_grid, np.log10(np.abs(dpdt_ramp) + 1e-100), shading="auto"
    )
    axes[4].set_xscale("log")
    axes[4].set_yscale("log")
    axes[4].set_xlabel("Density (g/cm³)")
    axes[4].set_ylabel("Temperature (K)")
    axes[4].set_title("dP/dT (log10)")
    cb4 = plt.colorbar(im4, ax=axes[4])

    axes[5].set_visible(False)

    # Sliders
    ax_r0 = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_a = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_b = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_c = plt.axes([0.25, 0.0, 0.65, 0.03])

    s_r0 = Slider(ax_r0, "log10(r0)", 0, 1.0, valinit=initial_r0)
    s_a = Slider(ax_a, "a", 0, 20, valinit=initial_a)
    s_b = Slider(ax_b, "b", 0, 20, valinit=initial_b)
    s_c = Slider(ax_c, "c", 0, 20, valinit=initial_c)

    def update(val):
        r0 = s_r0.val
        a = s_a.val
        b = s_b.val
        c = s_c.val
        ramp_eos_updated = BilinearRampEos(eos, r0=r0, a=a, b=b, c=c)
        p_ramp_new = ramp_eos_updated.PressureFromDensityTemperature(
            rho_grid, temp_grid
        )
        p_diff_new = p_ramp_new - p_base

        dpdrho_ramp_new = (
            ramp_eos_updated.BulkModulusFromDensityTemperature(rho_grid, temp_grid)
            / p_ramp_new
        )
        p_ramp_plus_dt_new = ramp_eos_updated.PressureFromDensityTemperature(
            rho_grid, temp_grid * 1.01
        )
        dpdt_ramp_new = (p_ramp_plus_dt_new - p_ramp_new) / (temp_grid * 0.01)

        im1.set_array(np.log10(np.abs(p_ramp_new) + 1e-100).ravel())
        im2.set_array(np.log10(np.abs(p_diff_new) + 1e-100).ravel())
        im3.set_array(np.abs(dpdrho_ramp_new).ravel())
        im4.set_array(np.log10(np.abs(dpdt_ramp_new) + 1e-100).ravel())

        im1.set_clim(
            vmin=np.min(np.log10(np.abs(p_ramp_new) + 1e-100)),
            vmax=np.max(np.log10(np.abs(p_ramp_new) + 1e-100)),
        )
        im2.set_clim(
            vmin=np.min(np.log10(np.abs(p_diff_new) + 1e-100)),
            vmax=np.max(np.log10(np.abs(p_diff_new) + 1e-100)),
        )
        im3.set_clim(
            vmin=np.min(np.abs(dpdrho_ramp_new)),
            vmax=np.max(np.abs(dpdrho_ramp_new)),
        )
        im4.set_clim(
            vmin=np.min(np.log10(np.abs(dpdt_ramp_new) + 1e-100)),
            vmax=np.max(np.log10(np.abs(dpdt_ramp_new) + 1e-100)),
        )

        cb1.update_normal(im1)
        cb2.update_normal(im2)
        cb3.update_normal(im3)
        cb4.update_normal(im4)

        fig.canvas.draw_idle()

    s_r0.on_changed(update)
    s_a.on_changed(update)
    s_b.on_changed(update)
    s_c.on_changed(update)

    plt.show()


def main():
    """
    Compare a base SESAME EOS with a ramp-modified EOS.
    """
    sesame_file = "sesame_ascii_3720.ses"
    rho_min, rho_max = 1e-6, 1e2
    temp_min, temp_max = 1.0, 1e8
    points = 100

    # Create the base EOS
    eos = SesameEos(sesame_file, component="ion")

    density_range = (rho_min, rho_max, points)
    temperature_range = (temp_min, temp_max, points)

    plot_pressure_comparison(eos, density_range, temperature_range)


if __name__ == "__main__":
    main()
