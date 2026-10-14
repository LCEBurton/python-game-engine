from typing import Protocol
import numpy as np

class AdvectionScalarOperator(Protocol):
    """Common call signature for all scalar advection operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        scalar_field: np.ndarray,
        scalar_destination: np.ndarray,
        velocity_field_u: np.ndarray,
        velocity_field_v: np.ndarray,
        is_fluid: np.ndarray,
        dt: float,
        cell_size: float,
        **kwargs,
    ) -> None:
        ...

class AdvectionVelocityUOperator(Protocol):
    """Common call signature for all u-velocity advection operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        velocity_field_u: np.ndarray,
        velocity_destination_u: np.ndarray,
        velocity_field_v: np.ndarray,
        face_mask_u: np.ndarray,
        dt: float,
        cell_size: float,
        **kwargs,
    ) -> None:
        ...

class AdvectionVelocityVOperator(Protocol):
    """Common call signature for all v-velocity advection operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        velocity_field_v: np.ndarray,
        velocity_destination_v: np.ndarray,
        velocity_field_u: np.ndarray,
        face_mask_v: np.ndarray,
        dt: float,
        cell_size: float,
        **kwargs,
    ) -> None:
        ...

class EnforceBoundaryConditionsOperator(Protocol):
    """Common call signature for all boundary condition enforcement operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        velocity_field_u: np.ndarray,
        velocity_field_v: np.ndarray,
        solid_mask: np.ndarray,
        **kwargs,
    ) -> None:
        ...

class ComputeDivergenceOperator(Protocol):
    """Common call signature for all divergence computation operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        velocity_field_u: np.ndarray,
        velocity_field_v: np.ndarray,
        divergence: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        cell_size: float,
        **kwargs,
    ) -> None:
        ...

class ApplyForcesOperator(Protocol):
    """Common call signature for all force application operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        velocity_field_v: np.ndarray,
        density: np.ndarray,
        temperature: np.ndarray,
        dt: float,
        gravity: float,
        buoyancy_coefficient: float,
        density_coefficient: float,
        ambient_temperature: float,
        **kwargs,
    ) -> None:
        ...

class CoolTemperatureOperator(Protocol):
    """Common call signature for all temperature cooling operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        temperature: np.ndarray,
        ambient_temperature: float,
        cooling_rate: float,
        dt: float,
        **kwargs,
    ) -> None:
        ...

class ApplyPressureGradientOperator(Protocol):
    """Common call signature for all pressure gradient application operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        velocity_field_u: np.ndarray,
        velocity_field_v: np.ndarray,
        pressure: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        fluid_density: float,
        cell_size: float,
        dt: float,
        **kwargs,
    ) -> None:
        ...

class ComputeResidualOperator(Protocol):
    """Common call signature for all residual computation operator backends (numba, C++, etc.)."""

    def __call__(
        self,
        pressure: np.ndarray,
        divergence: np.ndarray,
        is_fluid: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        alpha: float,
        reciprocal_beta: float,
        residual: np.ndarray,
        **kwargs,
    ) -> None:
        ...


