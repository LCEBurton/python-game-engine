from .registry import register_solver

_initalized = False


def register_solvers():
    global _initalized
    
    if _initalized:
        return

    from cfd.backends.numba.solvers.jacobi import jacobi_pressure_solver
    from cfd.backends.numba.solvers.red_black_sor_gauss_seidel import red_black_sor_gauss_seidel_pressure_solver
    from cfd.backends.numba.solvers.gauss_seidel import gauss_seidel_pressure_solver
    from cfd.backends.numba.solvers.multigrid.solver import MultigridSolver

    register_solver("jacobi", "numba", jacobi_pressure_solver)
    register_solver("gauss_seidel", "numba", gauss_seidel_pressure_solver)
    register_solver("rb_sor_gauss_seidel", "numba", red_black_sor_gauss_seidel_pressure_solver)
    register_solver("multigrid", "numba", MultigridSolver())
    # register_solver("sor_cpp", "cpp", red_black_sor_pressure_solver_cpp)

    _initalized = True
