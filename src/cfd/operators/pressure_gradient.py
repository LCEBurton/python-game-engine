from typing import cast

import numpy as np

from .base import ApplyPressureGradientOperator

from .registry import get_operator

def apply_pressure_gradient(
        velocity_field_u: np.ndarray,
        velocity_field_v: np.ndarray,
        pressure_field: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        fluid_density: float,
        cell_size: float,
        dt: float,
        *,
        backend: str = "numba",
    ):
    """Dispatches the pressure gradient application operator to the appropriate backend implementation."""
    fn = cast(
        ApplyPressureGradientOperator,
        get_operator("apply_pressure_gradient", backend),
    )

    return fn(
        velocity_field_u,
        velocity_field_v,
        pressure_field,
        face_mask_u,
        face_mask_v,
        fluid_density,
        cell_size,
        dt,
    )
