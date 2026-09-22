"""
Unit tests for pressure solver methods in the fluid simulation.
"""

import pytest
import numpy as np

from engine.smoke_simulator.kernels.numba.pressure_gradient import apply_pressure_gradient


def test_constant_pressure_field(small_grid_size, fluid_density):
    """
    Test that a constant pressure field remains unchanged after applying the pressure gradient.
    """
    height, width = small_grid_size

    # Create a constant pressure field
    pressure = np.ones(small_grid_size, dtype=np.float32) * 5.0  # Constant pressure of 5 Pa

    # Create zero velocity fields
    velocity_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_v = np.zeros((height+1, width), dtype=np.float32)

    # Apply the pressure gradient
    apply_pressure_gradient(velocity_u, velocity_v, pressure, fluid_density, cell_size=1.0, dt=1.0)

    # Assert that the velocity fields remain unchanged (should still be zero)
    assert np.allclose(velocity_u, 0.0, atol=1e-6), "Velocity field u changed after applying constant pressure gradient."
    assert np.allclose(velocity_v, 0.0, atol=1e-6), "Velocity field v changed after applying constant pressure gradient."


def test_linear_pressure_field(small_grid_size, fluid_density):
    """
    Test that a linear pressure field results in a uniform velocity change after applying the pressure gradient.
    """
    height, width = small_grid_size

    # Create a linear pressure field (pressure increases with f(x, y) = 3x - 2y)
    pressure = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            pressure[j, i] = 3.0 * i - 2.0 * j  # Pressure increases with 3x - 2y

    # Create zero velocity fields
    velocity_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_v = np.zeros((height+1, width), dtype=np.float32)

    # Apply the pressure gradient
    apply_pressure_gradient(velocity_u, velocity_v, pressure, fluid_density, cell_size=1.0, dt=1.0)

    assert np.allclose(velocity_u[1:-1, 1:-1], -3.0 / fluid_density, atol=1e-6), "Velocity field u did not change correctly after applying linear pressure gradient."
    assert np.allclose(velocity_v[1:-1, 1:-1], 2.0 / fluid_density, atol=1e-6), "Velocity field v did not change correctly after applying linear pressure gradient."

    
def test_quadratic_pressure_field(small_grid_size, fluid_density):
    """
    Test that a quadratic pressure field results in a non-uniform velocity change after applying the pressure gradient.
    """
    height, width = small_grid_size

    # Create a quadratic pressure field (pressure increases with f(x, y) = x^2 + y^2)
    pressure = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            pressure[j, i] = i**2 + j**2  # Pressure increases with x^2 + y^2

    # Create zero velocity fields
    velocity_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_v = np.zeros((height+1, width), dtype=np.float32)

    expected_velocity_u = np.zeros((height, width+1), dtype=np.float32)
    expected_velocity_v = np.zeros((height+1, width), dtype=np.float32)

    for j in range(height):
        for i in range(1, width):
            expected_velocity_u[j, i] = -1.0 * (2 * i - 1) / fluid_density  # du/dx ~ 2x - 1 (discrete)

    for j in range(1, height):
        for i in range(width):
            expected_velocity_v[j, i] = -1.0 * (2 * j - 1) / fluid_density  # dv/dy ~ 2y - 1 (discrete)

    # Apply the pressure gradient
    apply_pressure_gradient(velocity_u, velocity_v, pressure, fluid_density, cell_size=1.0, dt=1.0)

    # Check that the velocity fields are not uniform and have changed
    assert np.allclose(velocity_u[:, 1:-1], expected_velocity_u[:, 1:-1], atol=1e-6), "Velocity field u did not match expected velocity after applying quadratic pressure gradient."
    assert np.allclose(velocity_v[1:-1, :], expected_velocity_v[1:-1, :], atol=1e-6), "Velocity field v did not match expected velocity after applying quadratic pressure gradient."

