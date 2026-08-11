# tests/confest.py

import pytest

import numpy as np


@pytest.fixture
def small_zero_grid():
    """A small 3x3 grid with all zeros."""
    return np.zeros((16, 16), dtype=np.float32)

@pytest.fixture
def small_random_grid():
    """A small 3x3 grid with random values."""
    rng = np.random.default_rng(seed=42)  # For reproducibility
    return rng.random((16, 16), dtype=np.float32)

@pytest.fixture
def small_grid_size():
    """Return the size of the small grid."""
    return (16, 16)

@pytest.fixture
def cell_size():
    return 1.0

@pytest.fixture
def fluid_density():
    """Return a typical fluid density (e.g., water)."""
    return 1000.0  # kg/m^3


