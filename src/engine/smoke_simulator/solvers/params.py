from dataclasses import dataclass
from typing import Union


@dataclass
class JacobiParams:
    """No extra tunable parameters currently."""
    pass


@dataclass
class GaussSeidelParams:
    """No extra tunable parameters currently."""
    pass


@dataclass
class SORParams:
    omega: float = 0.0  # 0.0 triggers auto-computed optimal omega in the solver


@dataclass
class MultigridParams:
    num_levels: int = 4
    smoother_iterations: int = 2


SolverParams = Union[JacobiParams, GaussSeidelParams, SORParams, MultigridParams]
