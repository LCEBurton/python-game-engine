from typing import cast

import numpy as np

from .base import (
    EnforceBoundaryConditionsOperator,
    ComputeDivergenceOperator,
    ApplyForcesOperator,
    CoolTemperatureOperator,
    ApplyPressureGradientOperator,
    ComputeResidualOperator,
)

from .registry import get_operator

def enforce_boundary_conditions(
        velocity_field_u: np.ndarray,
        velocity_field_v: np.ndarray,
        solid_mask: np.ndarray,
        *,
        backend: str = "numba",
    ):
    """Dispatches the boundary condition enforcement operator to the appropriate backend implementation."""
    fn = cast(
        EnforceBoundaryConditionsOperator,
        get_operator("enforce_boundary_conditions", backend),
    )

    return fn(
        velocity_field_u,
        velocity_field_v,
        solid_mask,
    )
