"""Jacobi pressure solver (numba backend)."""
import numpy as np
from numba import njit, prange

from .common import calculateAlpha, calculateReciprocalBeta


@njit(parallel=True, cache=True)
def _jacobi_kernel(pressure, pressure_temp, divergence, is_fluid, 
                   face_mask_u, face_mask_v, alpha, reciprocal_beta):
    height, width = pressure.shape
    for j in prange(height):
        for i in range(width):
            if not is_fluid[j, i]:
                pressure_temp[j, i] = pressure[j, i]
                continue

            left  = pressure[j, max(i - 1, 0)] if face_mask_u[j, i]     else pressure[j, i]
            right = pressure[j, min(i + 1, width - 1)] if face_mask_u[j, i + 1] else pressure[j, i]
            down  = pressure[max(j - 1, 0), i] if face_mask_v[j, i]     else pressure[j, i]
            up    = pressure[min(j + 1, height - 1), i] if face_mask_v[j + 1, i] else pressure[j, i]

            pressure_temp[j, i] = (left + right + down + up +
                                   alpha * divergence[j, i]) * reciprocal_beta


def jacobi_pressure_solver(pressure: np.ndarray, divergence: np.ndarray,
                            face_mask_u: np.ndarray, face_mask_v: np.ndarray,
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
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]
    src, dst = pressure, pressure_temp
    for _ in range(iterations):
        _jacobi_kernel(src, dst, divergence, is_fluid, face_mask_u, face_mask_v, alpha, reciprocal_beta)
        src, dst = dst, src
    if src is not pressure:
        pressure[:] = src
