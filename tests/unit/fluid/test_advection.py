"""
Unit testing for advection calculations
"""
import pytest
import numpy as np

from engine.smoke_simulator.kernels.numba.advection import advect_scalar_field


def test_zero_velocity_advection(small_grid_size, face_masks):
    """
    Test that a scalar field remains unchanged when advected with zero velocity.
    """
    face_mask_u, face_mask_v = face_masks
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]
    # Create a zero velocity field
    velocity_field_u = np.zeros(small_grid_size, dtype=np.float32)
    velocity_field_v = np.zeros(small_grid_size, dtype=np.float32)

    # Advect the random scalar field with zero velocity
    scalar_field = np.random.rand(*small_grid_size).astype(np.float32)
    scalar_destination = np.zeros(small_grid_size, dtype=np.float32)
    advected_field = np.copy(scalar_field)

    advect_scalar_field(scalar_field, scalar_destination, velocity_field_u, velocity_field_v, is_fluid, dt=0.1, cell_size=1.0)

    # Assert that the advected field is approximately equal to the original scalar field
    assert np.allclose(advected_field, scalar_field, atol=1e-6), "Scalar field changed during advection with zero velocity."

    
def test_constant_velocity_advection(small_grid_size, face_masks):
    """
    Test that a scalar field is advected correctly with a constant velocity.
    """
    height, width = small_grid_size

    face_mask_u, face_mask_v = face_masks
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    # Create a constant velocity field
    velocity_field_u = np.zeros(small_grid_size, dtype=np.float32)
    velocity_field_v = np.zeros(small_grid_size, dtype=np.float32)  

    scalar_field = np.zeros(small_grid_size, dtype=np.float32)

    for j in range(height):
        for i in range(width):
            scalar_field[j, i] = 7.0
            velocity_field_u[j, i] = 3.0  # Constant horizontal velocity
            velocity_field_v[j, i] = -2.0  # Constant vertical velocity

    scalar_destination = np.zeros(small_grid_size, dtype=np.float32)
    advect_scalar_field(scalar_field, scalar_destination, velocity_field_u, velocity_field_v, is_fluid, dt=0.1, cell_size=1.0)

    assert np.allclose(scalar_destination[1:-1, 1:-1], scalar_field[1:-1, 1:-1], atol=1e-6), "Scalar field changed during advection with constant velocity."

def test_uniform_rightward_velocity_advection(small_grid_size, face_masks):
    """
    Test that a scalar field is advected correctly with a uniform rightward velocity.
    """
    height, width = small_grid_size

    face_mask_u, face_mask_v = face_masks
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    # Create a uniform rightward velocity field
    velocity_field_u = np.ones(small_grid_size, dtype=np.float32) * 1.0  # Constant horizontal velocity to the right
    velocity_field_v = np.zeros(small_grid_size, dtype=np.float32)  # No vertical velocity

    # Create a scalar field with a gradient
    scalar_field = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            scalar_field[j, i] = float(i)  # Gradient in the x-direction

    scalar_destination = np.zeros(small_grid_size, dtype=np.float32)
    advect_scalar_field(scalar_field, scalar_destination, velocity_field_u, velocity_field_v, is_fluid, dt=1.0, cell_size=1.0)

    # The expected result is that the scalar field shifts to the right by 1 unit
    assert np.allclose(scalar_destination[1:-1, 2:-1], scalar_field[1:-1, 1:-2], atol=1e-6), "Scalar field did not advect correctly with uniform rightward velocity."

def test_uniform_upward_velocity_advection(small_grid_size, face_masks):
    """
    Test that a scalar field is advected correctly with a uniform upward velocity.
    """
    height, width = small_grid_size

    face_mask_u, face_mask_v = face_masks
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    # Create a uniform upward velocity field
    velocity_field_u = np.zeros(small_grid_size, dtype=np.float32)  # No horizontal velocity
    velocity_field_v = np.ones(small_grid_size, dtype=np.float32) * 1.0  # Constant vertical velocity upwards

    # Create a scalar field with a gradient
    scalar_field = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            scalar_field[j, i] = float(j)  # Gradient in the y-direction

    scalar_destination = np.zeros(small_grid_size, dtype=np.float32)
    advect_scalar_field(scalar_field, scalar_destination, velocity_field_u, velocity_field_v, is_fluid, dt=1.0, cell_size=1.0)

    # The expected result is that the scalar field shifts upwards by 1 unit
    assert np.allclose(scalar_destination[2:-1, 1:-1], scalar_field[1:-2, 1:-1], atol=1e-6), "Scalar field did not advect correctly with uniform upward velocity."

def test_linear_field_bilinear_interpolation(small_grid_size, face_masks):
    """
    Test that a linear scalar field is advected correctly using bilinear interpolation.
    """
    height, width = small_grid_size

    face_mask_u, face_mask_v = face_masks
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    # Create a linear scalar field
    scalar_field = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            scalar_field[j, i] = 3.0 * i + 2.0 * j  # Linear function: f(x, y) = 3x + 2y

    # Create a uniform velocity field
    velocity_field_u = np.ones(small_grid_size, dtype=np.float32) * 0.25  # Constant horizontal velocity to the right
    velocity_field_v = np.ones(small_grid_size, dtype=np.float32) * 0.5  # Constant vertical velocity upwards

    scalar_destination = np.zeros(small_grid_size, dtype=np.float32)
    advect_scalar_field(scalar_field, scalar_destination, velocity_field_u, velocity_field_v, is_fluid, dt=1.0, cell_size=1.0)

    # The expected result is that the scalar field shifts according to the velocity
    expected_field = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            x = i - 0.25  # Shift left by 0.25 due to velocity
            y = j - 0.5   # Shift down by 0.5 due to velocity
            expected_field[j, i] = 3.0 * x + 2.0 * y

    # Assert that the advected field is approximately equal to the expected field
    assert np.allclose(scalar_destination[1:-1, 1:-1], expected_field[1:-1, 1:-1], atol=1e-6), "Scalar field did not advect correctly with linear field and bilinear interpolation."

 
