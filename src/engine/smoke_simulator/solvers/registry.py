from .numba.jacobi import jacobi_pressure_solver
from .numba.red_black_sor_gauss_seidel import red_black_sor_gauss_seidel_pressure_solver
from .numba.gauss_seidel import gauss_seidel_pressure_solver
from .numba.multigrid_solver import MultigridSolver
from .registry_core import register_solver, get_solver, SOLVER_REGISTRY

register_solver("jacobi", jacobi_pressure_solver)
register_solver("gauss_seidel", gauss_seidel_pressure_solver)
register_solver("rb_sor_gauss_seidel", red_black_sor_gauss_seidel_pressure_solver)
register_solver("multigrid", MultigridSolver())
# register_solver("sor_cpp", red_black_sor_pressure_solver_cpp)

__all__ = ["get_solver", "SOLVER_REGISTRY"]

