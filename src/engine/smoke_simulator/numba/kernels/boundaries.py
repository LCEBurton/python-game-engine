"""
Quick and dirty implementation of boundary conditions for the smoke simulator using Numba.
"""
import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def enforce_boundary_conditions(velocity_u: np.ndarray, velocity_v: np.ndarray, solid_mask: np.ndarray):
    """
    Enforce boundary conditions on the velocity field based on the solid cell mask.

    Args:
        velocity_u: 2D array of horizontal velocity values.
        velocity_v: 2D array of vertical velocity values.
        solid_mask: 2D array where 1 indicates a solid cell and 0 indicates a fluid cell.
    """
    height, width = solid_mask.shape

    # Enforce boundary conditions for horizontal velocity (u)
    for j in prange(height):
        for i in range(width):
            if solid_mask[j, i] == 1:
                # Set horizontal velocity to zero at solid cells
                velocity_u[j, i] = 0.0
                velocity_v[j, i] = 0.0

