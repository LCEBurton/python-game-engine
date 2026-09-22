from dataclasses import asdict

from .registry import get_solver


def solve_pressure(pressure, divergence, fluid_density, cell_size, dt, params):
    solver_fn = get_solver(params.pressure_solver_method)
    solver_kwargs = asdict(params.solver_params) if params.solver_params else {}
    solver_fn(pressure, divergence, fluid_density, cell_size, dt,
              params.pressure_iterations, **solver_kwargs)
