from typing import cast

import numpy as np

from .base import ComputeResidualOperator

from .registry import get_operator

def compute_residual(
        pressure_field: np.ndarray,
        divergence_field: np.ndarray,
        is_fluid: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        alpha: float,
        reciprocal_beta: float,
        residual: np.ndarray,
        *,
        backend: str = "numba",
    ):
    """Dispatches the residual computation operator to the appropriate backend implementation."""
    fn = cast(
        ComputeResidualOperator,
        get_operator("compute_residual", backend),
    )

    return fn(
        pressure_field,
        divergence_field,
        is_fluid,
        face_mask_u,
        face_mask_v,
        alpha,
        reciprocal_beta,
        residual,
    )

