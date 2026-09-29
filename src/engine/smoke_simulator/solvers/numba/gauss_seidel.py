"""Gauss-Seidel pressure solver (numba backend, non-parallel)."""
import numpy as np
from numba import njit, prange

from .common import calculateAlpha, calculateReciprocalBeta


@njit(cache=True, fastmath=True)
def _gauss_seidel_kernel(pressure: np.ndarray, divergence: np.ndarray, is_fluid: np.ndarray,
                         face_mask_u: np.ndarray, face_mask_v: np.ndarray,
                         alpha: float, reciprocal_beta: float, iterations: int):
    height, width = pressure.shape

    for _ in range(iterations):
        for j in range(height):
            for i in range(width):
                if not is_fluid[j, i]:
                    continue

                left  = pressure[j, max(i - 1, 0)] if face_mask_u[j, i]     else pressure[j, i]
                right = pressure[j, min(i + 1, width - 1)] if face_mask_u[j, i + 1] else pressure[j, i]
                down  = pressure[max(j - 1, 0), i] if face_mask_v[j, i]     else pressure[j, i]
                up    = pressure[min(j + 1, height - 1), i] if face_mask_v[j + 1, i] else pressure[j, i]

                pressure[j, i] = (left + right + down + up +
                                   alpha * divergence[j, i]) * reciprocal_beta


def gauss_seidel_pressure_solver(pressure: np.ndarray, divergence: np.ndarray,  
                                 face_mask_u: np.ndarray, face_mask_v: np.ndarray, fluid_density: float,
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
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    _gauss_seidel_kernel(pressure, divergence, is_fluid, face_mask_u, face_mask_v, alpha, reciprocal_beta, iterations)

