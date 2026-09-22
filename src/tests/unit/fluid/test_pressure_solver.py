"""
Unit tests for pressure solver methods in the fluid simulation.
"""

import pytest
import numpy as np

from engine.smoke_simulator.solvers.dispatch import solve_pressure
from engine.smoke_simulator.solvers.registry import get_solver

@pytest.fixture
def fluid_density():
    """Return a typical fluid density (e.g., water)."""
    return 1000.0  # kg/m^3



