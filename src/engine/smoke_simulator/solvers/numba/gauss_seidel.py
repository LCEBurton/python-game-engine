"""Gauss-Seidel pressure solver (numba backend, non-parallel)."""
import numpy as np
from numba import njit, prange

from .common import calculateAlpha, calculateReciprocalBeta


@njit(cache=True, fastmath=True)
def _gauss_seidel_kernel(pressure: np.ndarray, divergence: np.ndarray, alpha: float, reciprocal_beta: float,
                         iterations: int):
    height, width = pressure.shape

    for _ in range(iterations):
        for j in range(height):
            for i in range(width):
                # Compute indices for neighboring cells with clamping
                i_left = max(i - 1, 0)
                i_right = min(i + 1, width - 1)
                j_down = max(j - 1, 0)
                j_up = min(j + 1, height - 1)

                # Gauss-Seidel iteration formula
                pressure[j, i] = (pressure[j, i_left] + pressure[j, i_right] +
                                  pressure[j_down, i] + pressure[j_up, i] +
                                  alpha * divergence[j, i]) * reciprocal_beta


def gauss_seidel_pressure_solver(pressure: np.ndarray, divergence: np.ndarray, fluid_density: float, 
                                 cell_size: float, dt: float, iterations: int, **_ignored):
    """
    Solve for pressure using Gauss-Seidel iteration.

    Args:
        pressure: 2D array of current pressure values.
        divergence: 2D array of divergence values.
        fluid_density: Fluid density (kg/m^3).
        cell_size: Physical size of each cell (meters).
        dt: Time step (seconds).
        iterations: Number of Gauss-Seidel iterations to perform.
    """
    height, width = pressure.shape
    alpha = calculateAlpha(cell_size, fluid_density, dt)
    reciprocal_beta = calculateReciprocalBeta()

    _gauss_seidel_kernel(pressure, divergence, alpha, reciprocal_beta, iterations)

