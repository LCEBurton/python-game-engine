from .registry import register_operator 

_initalized = False


def register_operators():
    global _initalized
    
    if _initalized:
        return

    from cfd.backends.numba.operators.advection import advect_scalar_field, advect_velocity_u, advect_velocity_v


    from cfd.backends.numba.operators.boundaries import enforce_boundary_conditions
    from cfd.backends.numba.operators.divergence import compute_divergence
    from cfd.backends.numba.operators.forces import apply_forces, cool_temperature
    from cfd.backends.numba.operators.pressure_gradient import apply_pressure_gradient
    from cfd.backends.numba.operators.residual import compute_residual

    register_operator("advection_scalar", "numba", advect_scalar_field)
    register_operator("advection_velocity_u", "numba", advect_velocity_u)
    register_operator("advection_velocity_v", "numba", advect_velocity_v)
    register_operator("enforce_boundary_conditions", "numba", enforce_boundary_conditions)
    register_operator("compute_divergence", "numba", compute_divergence)
    register_operator("apply_forces", "numba", apply_forces)
    register_operator("cool_temperature", "numba", cool_temperature)
    register_operator("apply_pressure_gradient", "numba", apply_pressure_gradient)
    register_operator("compute_residual", "numba", compute_residual)

    _initalized = True
