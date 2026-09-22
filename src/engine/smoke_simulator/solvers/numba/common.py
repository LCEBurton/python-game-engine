"""Shared coefficients used across numba pressure solver kernels."""
import math
from numba import njit


@njit(inline="always")
def calculateAlpha(cell_size: float, density: float, dt: float) -> float:
    """
    Calculate the alpha coefficient for the Jacobi iteration.

    Args:
        cell_size: Physical size of each cell (meters).
        density: Fluid density (kg/m^3).
        dt: Time step (seconds).
    return:
        Alpha coefficient for the Jacobi iteration.
    """
    return (-(cell_size * cell_size) * density) / dt

@njit(inline="always")
def calculateReciprocalBeta() -> float:
    """
    Calculate the reciprocal of the beta coefficient for the Jacobi iteration.

    return:
        Reciprocal of the beta coefficient for the Jacobi iteration.
    """
    return 0.25

@njit(inline="always")
def calculateOptimalOmega(size: int) -> float:
    """
    Calculate the optimal relaxation factor (omega) for SOR based on grid size.

    Args:
        size: Size of the grid (number of cells in one dimension).
    return 2.0 / (1.0 + math.sin(math.pi / size))
    """
    return min((2.0 / (1.0 + math.sin(math.pi / size))), 1.95) # Clamp to 1.95 to avoid overshooting due to floating point errors


