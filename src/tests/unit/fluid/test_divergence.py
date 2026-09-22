"""
Unit testing for divergence calculations
"""

import pytest
import numpy as np

from engine.smoke_simulator.kernels.numba.divergence import compute_divergence

@pytest.mark.parametrize(
        "velocity",
        [
            pytest.param((0.0, 0.0), id="zero"),
            pytest.param((1.0, 0.0), id="horizontal"),
            pytest.param((0.0, 1.0), id="vertical"),
            pytest.param((3.5, -2.5), id ="arbitrary"),
        ],
)

def test_divergence_constant_velocity(small_grid_size, velocity, cell_size):
    """
    Test that the divergence of a constant velocity field is zero.
    """
    height, width = small_grid_size
    u, v = velocity

    divergence = np.zeros(small_grid_size, dtype=np.float32)

    # Create a constant velocity field
    velocity_field_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_field_v = np.zeros((height+1, width), dtype=np.float32)
    velocity_field_u[:, :] = u  # Set horizontal component
    velocity_field_v[:, :] = v  # Set vertical component

    # Compute divergence
    compute_divergence(velocity_field_u, velocity_field_v, divergence, cell_size)

    # Assert that the divergence is approximately zero everywhere
    assert np.allclose(divergence.tolist(), 0.0, atol=1e-6), f"Divergence is not zero for velocity {velocity}"


def test_divergence_linear_velocity(small_grid_size, cell_size):
    """
    Test the divergence of a linear velocity field.
    For a linear velocity field, the divergence should be constant.
    """
    height, width = small_grid_size

    divergence = np.zeros(small_grid_size, dtype=np.float32)

    # Create a linear velocity field
    height, width = small_grid_size
    velocity_field_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_field_v = np.zeros((height+1, width), dtype=np.float32)

    for j in range(height):
        for i in range(width + 1):
            velocity_field_u[j, i] = i  # u = x

    for j in range(height + 1):
        for i in range(width):
            velocity_field_v[j, i] = j  # v = y

    # Compute divergence
    compute_divergence(velocity_field_u, velocity_field_v, divergence, cell_size)

    assert np.allclose(divergence[1:-1, 1:-1].tolist(), 2.0, atol=1e-6), "Divergence is not as expected for linear velocity field"

def test_divergence_linear_horizontal_velocity(small_grid_size, cell_size):
    """
    Test the divergence of a linear horizontal velocity field.
    For a linear horizontal velocity field, the divergence should be constant.
    """
    height, width = small_grid_size

    divergence = np.zeros(small_grid_size, dtype=np.float32)

    # Create a linear horizontal velocity field
    height, width = small_grid_size
    velocity_field_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_field_v = np.zeros((height+1, width), dtype=np.float32)

    for j in range(height):
        for i in range(width + 1):
            velocity_field_u[j, i] = 3.0 * i  # u = x

    for j in range(height + 1):
        for i in range(width):
            velocity_field_v[j, i] = 0.0  # v = 0

    # Compute divergence
    compute_divergence(velocity_field_u, velocity_field_v, divergence, cell_size)

    assert np.allclose(divergence[1:-1, 1:-1].tolist(), 3.0, atol=1e-6), "Divergence is not as expected for linear horizontal velocity field"

def test_divergence_linear_vertical_velocity(small_grid_size, cell_size):
    """
    Test the divergence of a linear vertical velocity field.
    For a linear vertical velocity field, the divergence should be constant.
    """
    height, width = small_grid_size

    divergence = np.zeros(small_grid_size, dtype=np.float32)

    # Create a linear vertical velocity field
    height, width = small_grid_size
    velocity_field_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_field_v = np.zeros((height+1, width), dtype=np.float32)

    for j in range(height):
        for i in range(width + 1):
            velocity_field_u[j, i] = 0.0  # u = 0

    for j in range(height + 1):
        for i in range(width):
            velocity_field_v[j, i] = 2.0 * j  # v = y

    # Compute divergence
    compute_divergence(velocity_field_u, velocity_field_v, divergence, cell_size)

    assert np.allclose(divergence[1:-1, 1:-1].tolist(), 2.0, atol=1e-6), "Divergence is not as expected for linear vertical velocity field"


def test_divergence_rotational_field(small_grid_size, cell_size):
    """
    Test the divergence of a rotational velocity field.
    For a rotational velocity field, the divergence should be zero.
    """
    height, width = small_grid_size

    divergence = np.zeros(small_grid_size, dtype=np.float32)

    # Create a rotational velocity field
    height, width = small_grid_size
    velocity_field_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_field_v = np.zeros((height+1, width), dtype=np.float32)

    for j in range(height):
        for i in range(width + 1):
            velocity_field_u[j, i] = -j  # u = -y

    for j in range(height + 1):
        for i in range(width):
            velocity_field_v[j, i] = i   # v = x

    # Compute divergence
    compute_divergence(velocity_field_u, velocity_field_v, divergence, cell_size)

    assert np.allclose(divergence.tolist(), 0.0, atol=1e-6), "Divergence is not zero for rotational velocity field"

def test_divergence_quadratic_field(small_grid_size, cell_size):
    """
    Test the divergence of a quadratic velocity field.
    For a quadratic velocity field, the divergence should vary across the field.
    """
    height, width = small_grid_size

    divergence = np.zeros(small_grid_size, dtype=np.float32)

    # Create a quadratic velocity field
    height, width = small_grid_size
    velocity_field_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_field_v = np.zeros((height+1, width), dtype=np.float32)

    for j in range(height):
        for i in range(width + 1):
            velocity_field_u[j, i] = i**2  # u = x^2

    for j in range(height + 1):
        for i in range(width):
            velocity_field_v[j, i] = j**2  # v = y^2

    # Compute divergence
    compute_divergence(velocity_field_u, velocity_field_v, divergence, cell_size)

    # The expected divergence is 2*x + 2*y
    expected_divergence = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            expected_divergence[j, i] = 2 * i + 2 * j + 2 # The +2 accounts for the fact that the divergence is computed at cell centers, and the velocity fields are defined at faces.

    assert np.allclose(divergence[1:-1, 1:-1].tolist(), expected_divergence[1:-1, 1:-1].tolist(), atol=1e-6), "Divergence is not as expected for quadratic velocity field"
    
