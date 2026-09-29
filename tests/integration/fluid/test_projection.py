"""
Integration test for composition of laplacians.
"""

import pytest
import numpy as np

from engine.smoke_simulator.kernels.numba.pressure_gradient import apply_pressure_gradient
from engine.smoke_simulator.kernels.numba.divergence import compute_divergence

def test_divergence_of_pressure_gradient_quadratic(small_grid_size, fluid_density, face_masks):
    """
    Test that the divergence of the pressure gradient of a quadratic pressure field is zero.
    This is an integration test that combines the pressure gradient and divergence computations.
    """
    height, width = small_grid_size

    # Create a quadratic pressure field (pressure increases with f(x, y) = x^2 + y^2)
    pressure = np.zeros(small_grid_size, dtype=np.float32)
    for j in range(height):
        for i in range(width):
            pressure[j, i] = i**2 + j**2  # Quadratic pressure field

    # Create zero velocity fields
    velocity_u = np.zeros((height, width+1), dtype=np.float32)
    velocity_v = np.zeros((height+1, width), dtype=np.float32)

    # Apply the pressure gradient
    apply_pressure_gradient(velocity_u, velocity_v, pressure, face_masks[0], face_masks[1], fluid_density, cell_size=1.0, dt=1.0)

    # Compute divergence of the resulting velocity field
    divergence = np.zeros(small_grid_size, dtype=np.float32)
    compute_divergence(velocity_u, velocity_v, divergence, face_masks[0], face_masks[1], cell_size=1.0)

    # Assert that the divergence is approximately zero everywhere
    assert np.allclose(divergence[2:-2, 2:-2], -4.0 / fluid_density, atol=1e-6), "Divergence of the pressure gradient is not zero for quadratic pressure field"

def test_div_grad_matches_pressure_solver_laplacian(small_random_grid, fluid_density, cell_size, face_masks):
    """
    Test that the divergence of the pressure gradient multiplied by the change in time over fluid density matches the Laplacian of the pressure field.
    This is an integration test that combines the pressure gradient and divergence computations.
    """
    height, width = small_random_grid.shape

    # Create a random pressure field
    pressure = small_random_grid.astype(np.float32).copy()

    # Create zero velocity fields
    u = np.zeros((height, width+1), dtype=np.float32)
    v = np.zeros((height+1, width), dtype=np.float32)
    div_grad = np.zeros(small_random_grid.shape, dtype=np.float32)

    # Apply the pressure gradient
    apply_pressure_gradient(u, v, pressure, face_masks[0], face_masks[1], fluid_density, cell_size=1.0, dt=1.0)

    # Compute divergence of the resulting velocity field
    compute_divergence(u, v, div_grad, face_masks[0], face_masks[1], cell_size)

    # Compute the Laplacian of the pressure field
    laplacian = np.zeros(small_random_grid.shape, dtype=np.float32)
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            laplacian[j, i] = (pressure[j, i - 1] + pressure[j, i + 1] +
                               pressure[j - 1, i] + pressure[j + 1, i] -
                               4 * pressure[j, i]) / (cell_size ** 2)

    expected = -(1.0 / fluid_density) * laplacian

    # Assert that the divergence of the pressure gradient matches the Laplacian of the pressure field
    assert np.allclose(div_grad[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-6), "Divergence of the pressure gradient does not match the Laplacian of the pressure field"


def main_test():
    size = 11
    height, width = size, size
    pressure = np.zeros((size, size), dtype=np.float32)
    pressure[5, 5] = 1.0  # Set a single point of pressure

    u = np.zeros((size, size), dtype=np.float32)
    v = np.zeros((size, size), dtype=np.float32)
    div_grad = np.zeros((size, size), dtype=np.float32)

    solid_mask = np.zeros((height, width), dtype=bool)
    face_mask_u = np.zeros((height, width + 1), dtype=solid_mask.dtype)
    face_mask_v = np.zeros((height + 1, width), dtype=solid_mask.dtype)

    # Set the edges of the solid mask to 1 (solid)
    solid_mask[0, :] = 1 
    solid_mask[-1, :] = 1
    solid_mask[:, 0] = 1 
    solid_mask[:, -1] = 1

    # Set the face masks based on the solid mask
    face_mask_u[:, 1:-1] = ~(solid_mask[:, :-1] | solid_mask[:, 1:])
    face_mask_v[1:-1, :] = ~(solid_mask[:-1, :] | solid_mask[1:, :])

    fluid_density = 1.0
    cell_size = 1.0

    # Apply the pressure gradient
    apply_pressure_gradient(u, v, pressure, face_mask_u, face_mask_v, fluid_density, cell_size=cell_size, dt=1.0)

    print("Velocity Field u:")
    print(u)

    print("Velocity Field v:")
    print(v)

    # Compute divergence of the resulting velocity field
    compute_divergence(u, v, div_grad, face_mask_u, face_mask_v, 1.0)

    print("Pressure Field:")
    print(div_grad)

    # Compute the Laplacian of the pressure field
    laplacian = np.zeros(pressure.shape, dtype=np.float32)
    lap_stride2 = np.zeros(pressure.shape, dtype=np.float32)
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            laplacian[j, i] = (pressure[j, i - 1] + pressure[j, i + 1] +
                               pressure[j - 1, i] + pressure[j + 1, i] -
                               4 * pressure[j, i]) / (cell_size ** 2)

    for j in range(2, height - 2):
        for i in range(2, width - 2):
            lap_stride2[j, i] = (pressure[j, i - 2] + pressure[j, i + 2] +
                                 pressure[j - 2, i] + pressure[j + 2, i] -
                                 4 * pressure[j, i]) / (4 * cell_size ** 2)

    expected = -(1.0 / fluid_density) * laplacian
    expected_stride2 = -(1.0 / fluid_density) * lap_stride2

    print("Expected Laplacian:")
    print(expected)

    print("Expected Laplacian with stride 2:")
    print(expected_stride2)


if __name__ == "__main__":
    main_test()



