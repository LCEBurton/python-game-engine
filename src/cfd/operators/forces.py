from typing import cast

import numpy as np

from .base import (
    ApplyForcesOperator,
    CoolTemperatureOperator,
)

from .registry import get_operator

def apply_forces(
        velocity_field_v: np.ndarray,
        density_field: np.ndarray,
        temperature_field: np.ndarray,
        dt: float,
        gravity: float,
        buoyancy_coefficient: float,
        density_coefficient: float,
        ambient_temperature: float,
        *,
        backend: str = "numba",
    ):
    """Dispatches the force application operator to the appropriate backend implementation."""
    fn = cast(
        ApplyForcesOperator,
        get_operator("apply_forces", backend),
    )

    return fn(
        velocity_field_v,
        density_field,
        temperature_field,
        dt,
        gravity,
        buoyancy_coefficient,
        density_coefficient,
        ambient_temperature,
    )

def cool_temperature(
        temperature_field: np.ndarray,
        ambient_temperature: float,
        cooling_rate: float,
        dt: float,
        *,
        backend: str = "numba",
    ):
    """Dispatches the temperature cooling operator to the appropriate backend implementation."""
    fn = cast(
        CoolTemperatureOperator,
        get_operator("cool_temperature", backend),
    )

    return fn(
        temperature_field,
        ambient_temperature,
        cooling_rate,
        dt,
    )



