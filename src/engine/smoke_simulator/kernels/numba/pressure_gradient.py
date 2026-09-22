import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def apply_pressure_gradient(velocity_u: np.ndarray, velocity_v: np.ndarray, pressure: np.ndarray, fluid_density: float, cell_size: float, dt: float):
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
    scale = dt / (fluid_density * cell_size * 2)

    for j in prange(height):
        for i in range(width):
            # Compute indices for neighboring cells with clamping
            i_left = max(i - 1, 0)
            i_right = min(i + 1, width - 1)
            j_down = max(j - 1, 0)
            j_up = min(j + 1, height - 1)

            # Update velocity using the pressure gradient
            velocity_u[j, i] -= scale * (pressure[j, i_right] - pressure[j, i_left])
            velocity_v[j, i] -= scale * (pressure[j_up, i] - pressure[j_down, i])
