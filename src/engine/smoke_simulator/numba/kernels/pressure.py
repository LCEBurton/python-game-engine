"""
Functions for computing pressure using Jacobi iteration in the smoke simulator.
"""
import numpy as np
from numba import njit, prange

import math

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

@njit(parallel=True, cache=True)
def jacobi_pressure_iteration(pressure: np.ndarray, pressure_temp: np.ndarray, divergence: np.ndarray, fluid_density: float, cell_size: float, dt: float):
    """
    Perform one Jacobi iteration to solve for pressure.

    Args:
        pressure: 2D array of current pressure values.
        pressure_temp: 2D array to store updated pressure values.
        divergence: 2D array of divergence values.
        fluid_density: Fluid density (kg/m^3).
        cell_size: Physical size of each cell (meters).
        dt: Time step (seconds).
    """
    height, width = pressure.shape
    alpha = calculateAlpha(cell_size, fluid_density, dt)
    reciprocal_beta = calculateReciprocalBeta()

    for j in prange(height):
        for i in range(width):
            # Compute indices for neighboring cells with clamping
            i_left = max(i - 1, 0)
            i_right = min(i + 1, width - 1)
            j_down = max(j - 1, 0)
            j_up = min(j + 1, height - 1)

            # Jacobi iteration formula
            pressure_temp[j, i] = (pressure[j, i_left] + pressure[j, i_right] +
                                   pressure[j_down, i] + pressure[j_up, i] +
                                   alpha * divergence[j, i]) * reciprocal_beta

def jacobi_pressure_solver(pressure: np.ndarray, pressure_temp: np.ndarray, divergence: np.ndarray, fluid_density: float, cell_size: float, dt: float, iterations: int):
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
    for _ in range(iterations):
        jacobi_pressure_iteration(pressure, pressure_temp, divergence, fluid_density, cell_size, dt)
        # Swap pressure and pressure_temp for the next iteration
        pressure, pressure_temp = pressure_temp, pressure


@njit(cache=True, fastmath=True)
def gauss_seidel_pressure_solver(pressure: np.ndarray, divergence: np.ndarray, fluid_density: float, cell_size: float, dt: float, iterations: int):
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

    for _ in range(iterations):
        for j in range(height):
            for i in range(width):
                # Compute indices for neighboring cells with clamping
                i_left = max(i - 1, 0)
                i_right = min(i + 1, width - 1)
                j_down = max(j - 1, 0)
                j_up = min(j + 1, height - 1)

                # Gauss-Seidel iteration formula
                pressure[j, i] = (pressure[j, i_left] + pressure[j, i_right] +
                                  pressure[j_down, i] + pressure[j_up, i] +
                                  alpha * divergence[j, i]) * reciprocal_beta

@njit(parallel=True, cache=True, fastmath=True)
def red_black_gauss_seidel_pressure_solver(pressure: np.ndarray, divergence: np.ndarray, fluid_density: float, cell_size: float, dt: float, iterations: int):
    """
    Solve for pressure using Red-Black Gauss-Seidel iteration.

    Args:
        pressure: 2D array of current pressure values.
        divergence: 2D array of divergence values.
        fluid_density: Fluid density (kg/m^3).
        cell_size: Physical size of each cell (meters).
        dt: Time step (seconds).
        iterations: Number of Red-Black Gauss-Seidel iterations to perform.
    """
    height, width = pressure.shape
    alpha = calculateAlpha(cell_size, fluid_density, dt)
    reciprocal_beta = calculateReciprocalBeta()

    for _ in range(iterations):
        # Red pass
        for j in prange(height):
            start_x = 1 + (j % 2)  # Start with red cells
            for i in range(start_x, width - 1, 2):
                i_left = max(i - 1, 0)
                i_right = min(i + 1, width - 1)
                j_down = max(j - 1, 0)
                j_up = min(j + 1, height - 1)

                pressure[j, i] = (pressure[j, i_left] + pressure[j, i_right] +
                                  pressure[j_down, i] + pressure[j_up, i] +
                                  alpha * divergence[j, i]) * reciprocal_beta

        # Black pass
        for j in prange(height):
            start_x = (j % 2)  # Start with black cells
            for i in range(start_x, width - 1, 2):
                i_left = max(i - 1, 0)
                i_right = min(i + 1, width - 1)
                j_down = max(j - 1, 0)
                j_up = min(j + 1, height - 1)

                pressure[j, i] = (pressure[j, i_left] + pressure[j, i_right] +
                                  pressure[j_down, i] + pressure[j_up, i] +
                                  alpha * divergence[j, i]) * reciprocal_beta



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
