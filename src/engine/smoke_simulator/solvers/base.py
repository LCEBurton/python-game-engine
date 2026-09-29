from typing import Protocol
import numpy as np


class PressureSolver(Protocol):
    """Common call signature for all pressure solver backends (numba, C++, etc.)."""

    def __call__(
        self,
        pressure: np.ndarray,
        divergence: np.ndarray,
        face_mask_u: np.ndarray,
        face_mask_v: np.ndarray,
        fluid_density: float,
        cell_size: float,
        dt: float,
        iterations: int,
        **solver_kwargs,
    ) -> None:
        ...
