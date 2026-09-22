"""Red-Black Gauss-Seidel with SOR pressure solver (numba backend)."""
import numpy as np
from numba import njit, prange

from .common import calculateAlpha, calculateReciprocalBeta, calculateOptimalOmega


@njit(parallel=True, cache=True, fastmath=True)
def _red_black_sor_gauss_seidel_kernel(pressure, divergence, alpha, reciprocal_beta, omega, iterations):
    height, width = pressure.shape
    for _ in range(iterations):
        for color in (0, 1):
            for j in prange(height):
                start_x = color if (j % 2 == 0) else 1 - color
                for i in range(start_x, width, 2):
                    i_left = max(i - 1, 0)
                    i_right = min(i + 1, width - 1)
                    j_down = max(j - 1, 0)
                    j_up = min(j + 1, height - 1)
                    new_pressure = (pressure[j, i_left] + pressure[j, i_right] +
                                    pressure[j_down, i] + pressure[j_up, i] +
                                    alpha * divergence[j, i]) * reciprocal_beta
                    pressure[j, i] = (1.0 - omega) * pressure[j, i] + omega * new_pressure


def red_black_sor_gauss_seidel_pressure_solver(pressure: np.ndarray, divergence: np.ndarray,
                                   fluid_density: float, cell_size: float, dt: float,
                                   iterations: int, omega: float = 0.0):
    """Solve for pressure using Red-Black Gauss-Seidel with SOR."""
    height, width = pressure.shape
    alpha = calculateAlpha(cell_size, fluid_density, dt)
    reciprocal_beta = calculateReciprocalBeta()
    if omega <= 0.0 or omega > 2.0:
        omega = calculateOptimalOmega(max(height, width))
    _red_black_sor_gauss_seidel_kernel(pressure, divergence, alpha, reciprocal_beta, omega, iterations)
