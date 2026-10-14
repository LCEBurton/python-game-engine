from typing import cast

import numpy as np

from .base import ComputeDivergenceOperator

from .registry import get_operator

def compute_divergence(
        velocity_field_u: np.ndarray,
        velocity_field_v: np.ndarray,
        divergence: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        cell_size: float,
        *,
        backend: str = "numba",
    ):
    """Dispatches the divergence computation operator to the appropriate backend implementation."""
    fn = cast(
        ComputeDivergenceOperator,
        get_operator("compute_divergence", backend),
    )

    return fn(
        velocity_field_u,
        velocity_field_v,
        divergence,
        face_mask_u,
        face_mask_v,
        cell_size,
    )
