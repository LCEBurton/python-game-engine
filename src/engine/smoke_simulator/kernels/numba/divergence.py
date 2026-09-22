import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def compute_divergence(velocity_u: np.ndarray, velocity_v: np.ndarray, divergence: np.ndarray, cell_size: float):
    """
    Compute the divergence of the velocity field.

    Args:
        velocity_u: 2D array of horizontal velocity values.
        velocity_v: 2D array of vertical velocity values.
        divergence: 2D array to store the computed divergence values.
        cell_size: Physical size of each cell (meters).
    """
    height, width = divergence.shape
    inv_cell_size = 1.0 / cell_size

    for j in prange(height):
        for i in range(width):

            # Compute divergence using central differences
            du_dx = (velocity_u[j, i + 1] - velocity_u[j, i]) * inv_cell_size
            dv_dy = (velocity_v[j + 1, i] - velocity_v[j, i]) * inv_cell_size

            divergence[j, i] = du_dx + dv_dy
