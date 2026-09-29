"""
Unit tests for pressure solver methods in the fluid simulation.
"""

import pytest
import numpy as np

from engine.smoke_simulator.solvers.numba.jacobi import jacobi_pressure_solver, _jacobi_kernel
from engine.smoke_simulator.solvers.numba.common import calculateAlpha, calculateReciprocalBeta


def test_pressure_solver_reduces_divergence(small_grid_size, face_masks, fluid_density):
    height, width = small_grid_size

    face_mask_u, face_mask_v = face_masks
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    # Non-trivial divergence source in the interior (away from walls)
    divergence = np.zeros((height, width), dtype=np.float32)
    divergence[height // 2, width // 2] = 1.0
    divergence[height // 2, width // 2 + 1] = -1.0

    pressure = np.zeros((height, width), dtype=np.float32)
    pressure_temp = np.zeros_like(pressure)

    alpha = calculateAlpha(cell_size=1.0, density=fluid_density, dt=1.0)  # adjust to match your solver's actual convention
    reciprocal_beta = calculateReciprocalBeta()

    def compute_residual(p):
        # Discrete Laplacian residual: (sum of neighbor pressures - 4p) - alpha * divergence
        # Using same neighbor-selection logic as the solver (mirrored at walls)
        residual = np.zeros_like(p)
        for j in range(1, height - 1):
            for i in range(1, width - 1):
                if not is_fluid[j, i]:
                    continue
                left = p[j, i - 1] if face_mask_u[j, i] else p[j, i]
                right = p[j, i + 1] if face_mask_u[j, i + 1] else p[j, i]
                down = p[j - 1, i] if face_mask_v[j, i] else p[j, i]
                up = p[j + 1, i] if face_mask_v[j + 1, i] else p[j, i]
                lap = left + right + down + up - 4 * p[j, i]
                residual[j, i] = lap + alpha * divergence[j, i]
        return np.linalg.norm(residual)

    initial_residual = compute_residual(pressure)
    print(f"Initial residual: {initial_residual:.6f}")
    assert initial_residual > 1e-6, "Test setup invalid: initial residual should be nonzero."

    iterations = 50
    for it in range(iterations):
        _jacobi_kernel(
            pressure, pressure_temp, divergence, is_fluid,
            face_mask_u, face_mask_v, alpha, reciprocal_beta
        )
        pressure, pressure_temp = pressure_temp, pressure
        r = compute_residual(pressure)
        print(f"Iteration {it + 1}/{iterations}, Residual: {r:.6f}")


    final_residual = compute_residual(pressure)

    assert final_residual < initial_residual, (
        f"Pressure solver did not reduce residual: initial={initial_residual}, final={final_residual}"
    )
    assert final_residual < initial_residual * 0.1, (
        f"Pressure solver did not converge sufficiently: initial={initial_residual}, final={final_residual}"
    )


def test_pressure_solver_zero_divergence_stays_zero(small_grid_size, face_masks, fluid_density):
    """If divergence is already zero everywhere, pressure should remain (near) zero."""
    height, width = small_grid_size

    face_mask_u, face_mask_v = face_masks

    divergence = np.zeros((height, width), dtype=np.float32)
    pressure = np.zeros((height, width), dtype=np.float32)
    pressure_temp = np.zeros_like(pressure)

    jacobi_pressure_solver(
        pressure, divergence,
        face_mask_u, face_mask_v,
        fluid_density, cell_size=1.0, dt=1.0,
        iterations=50
    )

    assert np.allclose(pressure, 0.0, atol=1e-6), "Pressure should remain zero when divergence is zero."

