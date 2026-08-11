"""
Unit tests for pressure solver methods in the fluid simulation.
"""

import pytest
import numpy as np

from engine.smoke_simulator.numba.kernels.pressure import jacobi_pressure_solver, gauss_seidel_pressure_solver, red_black_gauss_seidel_pressure_solver, apply_pressure_gradient

@pytest.fixture
def fluid_density():
    """Return a typical fluid density (e.g., water)."""
    return 1000.0  # kg/m^3



