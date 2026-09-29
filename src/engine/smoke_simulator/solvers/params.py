from dataclasses import dataclass
from typing import Union


@dataclass
class JacobiParams:
    """No extra tunable parameters currently."""
    def string(self):
        return "JacobiParams"


@dataclass
class GaussSeidelParams:
    """No extra tunable parameters currently."""
    def string(self):
        return "GaussSeidelParams"


@dataclass
class SORParams:
    omega: float = 0.0  # 0.0 triggers auto-computed optimal omega in the solver

    def string(self):
        return f"SORParams_omega_{self.omega}"


@dataclass
class MultigridParams:
    num_levels: int = 4
    smoother_iterations: int = 2

    def string(self):
        return f"MultigridParams_num_levels_{self.num_levels}_smoother_iterations_{self.smoother_iterations}"


SolverParams = Union[JacobiParams, GaussSeidelParams, SORParams, MultigridParams]
