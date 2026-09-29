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
def face_masks():
    """Return masks for u and v faces."""
    height, width = 16, 16

    # init
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

    return face_mask_u, face_mask_v

def make_face_masks(height, width, solid_mask):
    face_mask_u = np.zeros((height, width + 1), dtype=bool)
    face_mask_v = np.zeros((height + 1, width), dtype=bool)
    face_mask_u[:, 1:-1] = ~(solid_mask[:, :-1] | solid_mask[:, 1:])
    face_mask_v[1:-1, :] = ~(solid_mask[:-1, :] | solid_mask[1:, :])
    return face_mask_u, face_mask_v

@pytest.fixture
def domain():
    height, width = 16, 16
    solid_mask = np.zeros((height, width), dtype=bool)
    solid_mask[0, :] = True
    solid_mask[-1, :] = True
    solid_mask[:, 0] = True
    solid_mask[:, -1] = True

    face_mask_u, face_mask_v = make_face_masks(height, width, solid_mask)
    is_fluid = face_mask_u[:, :-1] | face_mask_u[:, 1:] | face_mask_v[:-1, :] | face_mask_v[1:, :]

    return {
        "height": height,
        "width": width,
        "solid_mask": solid_mask,
        "face_mask_u": face_mask_u,
        "face_mask_v": face_mask_v,
        "is_fluid": is_fluid,
    }


@pytest.fixture
def cell_size():
    return 1.0

@pytest.fixture
def fluid_density():
    """Return a typical fluid density (e.g., water)."""
    return 1.0  # used in benchmarking


