from dataclasses import asdict

from .registry import get_solver

def solve_pressure(pressure, divergence, face_mask_u, face_mask_v, fluid_density, cell_size, dt, params):
    solver_fn = get_solver(params.pressure_solver_method)
    solver_kwargs = asdict(params.solver_params)
    solver_fn(pressure, divergence, face_mask_u, face_mask_v, fluid_density, cell_size, dt,
              profiler=params.profiler, **solver_kwargs)
