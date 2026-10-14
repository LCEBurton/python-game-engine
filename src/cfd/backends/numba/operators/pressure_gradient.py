import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def apply_pressure_gradient(velocity_u: np.ndarray, velocity_v: np.ndarray, pressure: np.ndarray, 
                            face_mask_u: np.ndarray, face_mask_v: np.ndarray,
                            fluid_density: float, cell_size: float, dt: float):
    """
    Apply the pressure gradient to the velocity field.

    Args:
        velocity_u: 2D array of horizontal velocity values.
        velocity_v: 2D array of vertical velocity values.
        pressure: 2D array of pressure values.
        fluid_density: Fluid density (kg/m^3).
        cell_size: Physical size of each cell (meters).
        dt: Time step (seconds).
    """

    height, width = pressure.shape
    scale = dt / (fluid_density * cell_size)

    # u-faces: shape (height, width + 1)
    for j in prange(height):
        for i in range(width + 1):
            if face_mask_u[j, i]:
                velocity_u[j, i] -= scale * (pressure[j, i] - pressure[j, i - 1])
            else:
                velocity_u[j, i] = 0.0

    # v-faces: shape (height + 1, width)
    for j in prange(height + 1):
        for i in range(width):
            if face_mask_v[j, i]:
                velocity_v[j, i] -= scale * (pressure[j, i] - pressure[j - 1, i])
            else:
                velocity_v[j, i] = 0.0
