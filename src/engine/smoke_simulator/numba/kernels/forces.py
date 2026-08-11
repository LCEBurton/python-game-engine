"""
Functions for computing forces in the smoke simulator.
"""
import numpy as np
from numba import njit, prange

import math

@njit(inline="always")
def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between a minimum and maximum."""
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value

@njit(inline="always")
def compute_buoyancy_force(density: np.ndarray, density_coefficient: float, gravity: float) -> np.ndarray:
    """
    Compute the buoyancy force based on the density field.

    Args:
        density: 2D array of smoke density values.
        buoyancy_coefficient: Coefficient for buoyancy force.
        gravity: Gravity value (negative for downward).

    Returns:
        2D array of buoyancy forces.
    """
    height, width = density.shape
    buoyancy_force = np.zeros((height, width), dtype=np.float32)

    for j in prange(height):
        for i in range(width):
            # Buoyancy force is proportional to the density and acts in the opposite direction of gravity
            buoyancy_force[j, i] = -density_coefficient * density[j, i] * gravity

    return buoyancy_force

@njit(inline="always")
def compute_temperature_force(temperature: np.ndarray, ambient_termperature: float, buoyancy_coefficient: float) -> np.ndarray:
    """
    Compute the temperature force based on the temperature field.

    Args:
        temperature: 2D array of temperature values.
        ambient_termperature: Ambient temperature value.
        temperature_coefficient: Coefficient for temperature force.

    Returns:
        2D array of temperature forces.
    """
    height, width = temperature.shape
    temperature_force = np.zeros((height, width), dtype=np.float32)

    for j in prange(height):
        for i in range(width):
            # Temperature force is proportional to the difference from ambient temperature
            temperature_force[j, i] = -buoyancy_coefficient * (temperature[j, i] - ambient_termperature)

    return temperature_force
    
@njit(parallel=True, cache=True)
def apply_forces(velocity_v: np.ndarray, density: np.ndarray, temperature: np.ndarray, 
                 dt: float, gravity: float, buoyancy_coefficient: float, density_coefficient: float, ambient_temperature: float):
    """
    Apply buoyancy and temperature forces to the vertical velocity field.

    Args:
        velocity_v: 2D array of vertical velocity values.
        density: 2D array of smoke density values.
        temperature: 2D array of temperature values.
        dt: Time step for the simulation.
        gravity: Gravity value (negative for downward).
        buoyancy_coefficient: Coefficient for buoyancy force.
        density_coefficient: Coefficient for density force.
        ambient_temperature: Ambient temperature value.
    """

    height, width = velocity_v.shape

    buoyancy_force = compute_buoyancy_force(density, density_coefficient, gravity)

    temperature_force = compute_temperature_force(temperature, ambient_temperature, buoyancy_coefficient)

    for j in prange(height):
        for i in range(width):
            
            # Update vertical velocity with forces
            velocity_v[j, i] += dt * (buoyancy_force[j, i] + temperature_force[j, i])
             

@njit(parallel=True, cache=True)
def cool_temperature(temperature: np.ndarray, ambient_temperature: float, cooling_rate: float, dt: float):
    """
    Cool the temperature field over time.

    Args:
        temperature: 2D array of temperature values.
        cooling_rate: Rate at which the temperature cools down.
        dt: Time step for the simulation.
    """
    height, width = temperature.shape

    decay = math.exp(-cooling_rate * dt)
    for j in prange(height):
        for i in range(width):
            temperature[j, i] = ambient_temperature + (temperature[j, i] - ambient_temperature) * decay

