"""
Quick and dirty implementation of boundary conditions for the smoke simulator using Numba.
"""
import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def enforce_boundary_conditions(velocity_u: np.ndarray, velocity_v: np.ndarray, solid_mask: np.ndarray):
    """
    Enforce boundary conditions on the velocity field based on the solid cell mask (MAC grid).

    Solid cells (including domain edges) must be explicitly marked in solid_mask.
    No boundary is implied purely from being at the edge of the domain.

    Args:
        velocity_u: 2D array of horizontal velocity at u-faces, shape (height, width + 1).
        velocity_v: 2D array of vertical velocity at v-faces, shape (height + 1, width).
        solid_mask: 2D array where 1 indicates a solid cell and 0 indicates a fluid cell, shape (height, width).
    """
    height, width = solid_mask.shape

    # Enforce boundary conditions for horizontal velocity (u-faces)
    # u[j, i] lies between cell (j, i-1) and cell (j, i)
    for j in prange(height):
        for i in range(width + 1):
            # Only the existing neighbor(s) determine solidity; domain edges are not implied solid
            left_solid = solid_mask[j, i - 1] == 1 if i > 0 else False
            right_solid = solid_mask[j, i] == 1 if i < width else False
            if left_solid or right_solid:
                velocity_u[j, i] = 0.0

    # Enforce boundary conditions for vertical velocity (v-faces)
    # v[j, i] lies between cell (j-1, i) and cell (j, i)
    for j in prange(height + 1):
        for i in range(width):
            top_solid = solid_mask[j - 1, i] == 1 if j > 0 else False
            bottom_solid = solid_mask[j, i] == 1 if j < height else False
            if top_solid or bottom_solid:
                velocity_v[j, i] = 0.0

