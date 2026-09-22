from .numba.jacobi import jacobi_pressure_solver
from .numba.red_black_sor_gauss_seidel import red_black_sor_gauss_seidel_pressure_solver
from .numba.gauss_seidel import gauss_seidel_pressure_solver

# Populated with C++ variants once available, e.g.:
# from .cpp import red_black_sor_pressure_solver as red_black_sor_pressure_solver_cpp

SOLVER_REGISTRY = {
    "jacobi": jacobi_pressure_solver,
    "gauss_seidel": gauss_seidel_pressure_solver,
    "rb_sor_gauss_seidel": red_black_sor_gauss_seidel_pressure_solver,
    # "sor_cpp": red_black_sor_pressure_solver_cpp,
}


def get_solver(name: str):
    try:
        return SOLVER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown pressure solver: {name!r}. Available: {list(SOLVER_REGISTRY)}")
