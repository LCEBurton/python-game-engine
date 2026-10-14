from typing import cast

import numpy as np

from .base import (
    AdvectionScalarOperator,
    AdvectionVelocityUOperator,
    AdvectionVelocityVOperator,
)

from .registry import get_operator

def advect_scalar_field(
        field_src: np.ndarray,
        field_dst: np.ndarray,
        velocity_u: np.ndarray,
        velocity_v: np.ndarray,
        is_fluid: np.ndarray,
        dt: float,
        cell_size: float,
        *,
        backend: str = "numba",
    ):
    """Dispatches the scalar advection operator to the appropriate backend implementation."""
    fn = cast(
        AdvectionScalarOperator,
        get_operator("advection_scalar", backend),
    )

    return fn(
        field_src,
        field_dst,
        velocity_u,
        velocity_v,
        is_fluid,
        dt,
        cell_size,
    )

def advect_velocity_u(
        velocity_u_src: np.ndarray,
        velocity_u_dst: np.ndarray,
        velocity_v: np.ndarray,
        face_mask_u: np.ndarray,
        dt: float,
        cell_size: float,
        *,
        backend: str = "numba",
    ):
    """Dispatches the u-velocity advection operator to the appropriate backend implementation."""
    fn = cast(
        AdvectionVelocityUOperator,
        get_operator("advection_velocity_u", backend),
    )

    return fn(
        velocity_u_src,
        velocity_u_dst,
        velocity_v,
        face_mask_u,
        dt,
        cell_size,
    )

def advect_velocity_v(
        velocity_v_src: np.ndarray,
        velocity_v_dst: np.ndarray,
        velocity_u: np.ndarray,
        face_mask_v: np.ndarray,
        dt: float,
        cell_size: float,
        *,
        backend: str = "numba",
    ):
    """Dispatches the v-velocity advection operator to the appropriate backend implementation."""
    fn = cast(
        AdvectionVelocityVOperator,
        get_operator("advection_velocity_v", backend),
    )

    return fn(
        velocity_v_src,
        velocity_v_dst,
        velocity_u,
        face_mask_v,
        dt,
        cell_size,
    )
