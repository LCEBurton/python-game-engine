"""Jacobi pressure solver (numba backend)."""
import numpy as np
from numba import njit, prange

from .common import calculateAlpha, calculateReciprocalBeta


@njit(parallel=True, cache=True)
def _jacobi_kernel(pressure, pressure_temp, divergence, alpha, reciprocal_beta):
    height, width = pressure.shape
    for j in prange(height):
        for i in range(width):
            i_left = max(i - 1, 0)
            i_right = min(i + 1, width - 1)
            j_down = max(j - 1, 0)
            j_up = min(j + 1, height - 1)
            pressure_temp[j, i] = (pressure[j, i_left] + pressure[j, i_right] +
                                   pressure[j_down, i] + pressure[j_up, i] +
                                   alpha * divergence[j, i]) * reciprocal_beta


def jacobi_pressure_solver(pressure: np.ndarray, divergence: np.ndarray,
                            fluid_density: float, cell_size: float, dt: float,
                            iterations: int, **_ignored):
    """
    Solve for pressure using Jacobi iteration.

    Args:
        pressure: 2D array of current pressure values.
        pressure_temp: 2D array to store updated pressure values.
        divergence: 2D array of divergence values.
        fluid_density: Fluid density (kg/m^3).
        cell_size: Physical size of each cell (meters).
        dt: Time step (seconds).
        iterations: Number of Jacobi iterations to perform.
    """
    alpha = calculateAlpha(cell_size, fluid_density, dt)
    reciprocal_beta = calculateReciprocalBeta()
    pressure_temp = np.empty_like(pressure)
    src, dst = pressure, pressure_temp
    for _ in range(iterations):
        _jacobi_kernel(src, dst, divergence, alpha, reciprocal_beta)
        src, dst = dst, src
    if src is not pressure:
        pressure[:] = src
