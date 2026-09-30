from dataclasses import dataclass
from typing import Union


@dataclass
class JacobiParams:
    iterations: int = 40

    def string(self):
        return f"JacobiParams_iters_{self.iterations}"

    def to_fn(self):
        return f"i_{self.iterations}"


@dataclass
class GaussSeidelParams:
    iterations: int = 40

    def string(self):
        return f"GaussSeidelParams_iters_{self.iterations}"

    def to_fn(self):
        return f"i_{self.iterations}"


@dataclass
class SORParams:
    iterations: int = 40
    omega: float = 0.0  # 0.0 triggers auto-computed optimal omega in the solver

    def string(self):
        return f"SORParams_iters_{self.iterations}_omega_{self.omega}"

    def to_fn(self):
        return f"i_{self.iterations}_o_{self.omega}"


@dataclass
class MultigridParams:
    smoother_type: str = "jacobi"
    num_levels: int = 4
    v_cycles: int = 2
    smoother_iterations: int = 2
    coarsest_iterations: int = 30
    rebuild_masks: bool = False

    def string(self):
        return (f"MultigridParams_smoother_{self.smoother_type}"
                f"_vcycles_{self.v_cycles}"
                f"_levels_{self.num_levels}"
                f"_smoother_iters_{self.smoother_iterations}"
                f"_coarsest_{self.coarsest_iterations}")

    def to_fn(self):
        return (f"st_{self.smoother_type}_v{self.v_cycles}_l{self.num_levels}"
                f"_si{self.smoother_iterations}_ci{self.coarsest_iterations}")


DEFAULT_SOLVER_PARAMS = {
    "jacobi": JacobiParams(),
    "gauss_seidel": GaussSeidelParams(),
    "rb_sor_gauss_seidel": SORParams(),
    "multigrid": MultigridParams(),
}

SolverParams = Union[JacobiParams, GaussSeidelParams, SORParams, MultigridParams]
