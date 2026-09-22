import math

import numpy as np
from numba import njit, prange

@njit(inline="always")
def clamp(value: float, min: float, max: float) -> float:
    """Clamp a value between a minimum and maximum."""
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return value

@njit(inline="always")
def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between two values."""
    return a + (b - a) * t

@njit(inline="always")
def sample_scalar_field(field: np.ndarray, x: float, y: float) -> float:
    """Sample a scalar field at a given position using bilinear interpolation."""
    height, width = field.shape
    x0 = int(math.floor(x))
    x1 = int(math.ceil(x))
    y0 = int(math.floor(y))
    y1 = int(math.ceil(y))

    # Clamp coordinates to be within the field bounds
    x0 = int(clamp(x0, 0, width - 1))
    x1 = int(clamp(x1, 0, width - 1))
    y0 = int(clamp(y0, 0, height - 1))
    y1 = int(clamp(y1, 0, height - 1))

    # compute the interpolation weights
    tx = x - x0
    ty = y - y0

    # Sample the four corners of the cell
    c00 = field[y0, x0]
    c10 = field[y0, x1]
    c01 = field[y1, x0]
    c11 = field[y1, x1]

    # Perform bilinear interpolation
    c0 = lerp(c00, c10, tx)
    c1 = lerp(c01, c11, tx)
    return lerp(c0, c1, ty)

# TODO not currently using sample_vector_field, but it may be useful in the future for advecting vector fields in 3d
#@njit(inline="always")
#def sample_vector_field(field: np.ndarray, x: float, y: float) -> tuple[float, float]:
#    """Sample a vector field at a given position using bilinear interpolation."""
#    height, width, _ = field.shape
#    x0 = int(math.floor(x))
#    x1 = int(math.ceil(x))
#    y0 = int(math.floor(y))
#    y1 = int(math.ceil(y))
#
#    # Clamp coordinates to be within the field bounds
#    x0 = int(clamp(x0, 0, width - 1))
#    x1 = int(clamp(x1, 0, width - 1))
#    y0 = int(clamp(y0, 0, height - 1))
#    y1 = int(clamp(y1, 0, height - 1))
#
#    # compute the interpolation weights
#    tx = x - x0
#    ty = y - y0
#
#    # Sample the four corners of the cell
#    c00 = field[y0, x0]
#    c10 = field[y0, x1]
#    c01 = field[y1, x0]
#    c11 = field[y1, x1]
#
#    # Perform bilinear interpolation for each component
#    c0_x = lerp(c00[0], c10[0], tx)
#    c1_x = lerp(c01[0], c11[0], tx)
#    c0_y = lerp(c00[1], c10[1], tx)
#    c1_y = lerp(c01[1], c11[1], tx)
#    
#    # Perform bilinear interpolation for the final components
#    x = lerp(c0_x, c1_x, ty)
#    y = lerp(c0_y, c1_y, ty)
#
#
#    return (x, y)
   

# --- Staggered-grid coordinate mapping helpers ---
# Physical/cell-index space convention:
#   - cell (i, j) center is located at (i + 0.5, j + 0.5)
#   - u-face (i, j) is located at (i,       j + 0.5)   -> shape (height, width + 1)
#   - v-face (i, j) is located at (i + 0.5, j      )   -> shape (height + 1, width)

@njit(inline="always")
def sample_u_field(velocity_u: np.ndarray, x: float, y: float) -> float:
    """Sample the u-velocity field at a physical position (x, y)."""
    return sample_scalar_field(velocity_u, x, y - 0.5)

@njit(inline="always")
def sample_v_field(velocity_v: np.ndarray, x: float, y: float) -> float:
    """Sample the v-velocity field at a physical position (x, y)."""
    return sample_scalar_field(velocity_v, x - 0.5, y)

@njit(inline="always")
def sample_center_field(field: np.ndarray, x: float, y: float) -> float:
    """Sample a cell-centered field (e.g. density, temperature) at a physical position (x, y)."""
    return sample_scalar_field(field, x - 0.5, y - 0.5)

@njit(parallel=True, cache=True)
def advect_scalar_field(field_source: np.ndarray, field_dest: np.ndarray, velocity_u: np.ndarray, velocity_v: np.ndarray, dt: float, cell_size: float):
    """Advect a cell-centered scalar field (e.g. density, temperature) using the MAC velocity field."""
    height, width = field_source.shape
    inv_cell_size = 1.0 / cell_size
    for j in prange(height):
        for i in range(width):
            # Physical position of the cell center
            px = i + 0.5
            py = j + 0.5

            # Interpolate velocity at the cell center
            u = sample_u_field(velocity_u, px, py)
            v = sample_v_field(velocity_v, px, py)

            # Backtrace the position
            x = px - (u * dt * inv_cell_size)
            y = py - (v * dt * inv_cell_size)

            # Sample the source field at the backtraced position
            field_dest[j, i] = sample_center_field(field_source, x, y)

@njit(parallel=True, cache=True)
def advect_velocity_u(velocity_u_source: np.ndarray, velocity_u_dest: np.ndarray, velocity_v: np.ndarray, dt: float, cell_size: float):
    """Advect the u-velocity face field (self-advection) using the MAC velocity field."""
    height, width_plus_1 = velocity_u_source.shape
    inv_cell_size = 1.0 / cell_size
    for j in prange(height):
        for i in range(width_plus_1):
            # Physical position of the u-face
            px = float(i)
            py = j + 0.5

            u = velocity_u_source[j, i]
            v = sample_v_field(velocity_v, px, py)

            # Backtrace the position
            x = px - (u * dt * inv_cell_size)
            y = py - (v * dt * inv_cell_size)

            velocity_u_dest[j, i] = sample_u_field(velocity_u_source, x, y)

@njit(parallel=True, cache=True)
def advect_velocity_v(velocity_v_source: np.ndarray, velocity_v_dest: np.ndarray, velocity_u: np.ndarray, dt: float, cell_size: float):
    """Advect the v-velocity face field (self-advection) using the MAC velocity field."""
    height_plus_1, width = velocity_v_source.shape
    inv_cell_size = 1.0 / cell_size
    for j in prange(height_plus_1):
        for i in range(width):
            # Physical position of the v-face
            px = i + 0.5
            py = float(j)

            u = sample_u_field(velocity_u, px, py)
            v = velocity_v_source[j, i]

            # Backtrace the position
            x = px - (u * dt * inv_cell_size)
            y = py - (v * dt * inv_cell_size)

            velocity_v_dest[j, i] = sample_v_field(velocity_v_source, x, y)
