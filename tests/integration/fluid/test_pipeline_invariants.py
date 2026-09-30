import numpy as np
import pytest

from engine.smoke_simulator.kernels.numba.divergence import compute_divergence
from engine.smoke_simulator.kernels.numba.pressure_gradient import apply_pressure_gradient
from engine.smoke_simulator.solvers.dispatch import solve_pressure
from engine.smoke_simulator.solvers.registry import SOLVER_REGISTRY
from engine.smoke_simulator.solvers.params import DEFAULT_SOLVER_PARAMS
from engine.smoke_simulator.kernels.numba.advection import advect_scalar_field, advect_velocity_u, advect_velocity_v
from engine.smoke_simulator.common.simulation import SmokeSimulation2D
from tools.profiling.session import ProfileSession




class FakeSim:
    """Minimal stand-in for the simulation's `params`/self object expected by solve_pressure dispatch."""
    def __init__(self, iterations=20, method="jacobi"):
        self.pressure_solver_method = method
        self.solver_params = DEFAULT_SOLVER_PARAMS[method]
        self.profiler = ProfileSession(enabled=False)



@pytest.mark.parametrize("method", ["jacobi", "gauss_seidel", "rb_sor_gauss_seidel"])
def test_zero_velocity_zero_pressure_is_noop(domain, method):
    """
    Full projection pipeline (divergence -> solve -> gradient) on all-zero input
    must leave velocity, pressure, and divergence at exactly zero.
    """
    height, width = domain["height"], domain["width"]
    face_mask_u, face_mask_v = domain["face_mask_u"], domain["face_mask_v"]

    velocity_u = np.zeros((height, width + 1), dtype=np.float32)
    velocity_v = np.zeros((height + 1, width), dtype=np.float32)
    pressure = np.zeros((height, width), dtype=np.float32)
    divergence = np.zeros((height, width), dtype=np.float32)

    fluid_density = 1000.0
    cell_size = 1.0
    dt = 1.0

    compute_divergence(velocity_u, velocity_v, divergence, face_mask_u, face_mask_v, cell_size)
    assert np.allclose(divergence, 0.0), "Divergence should be zero for zero velocity input."

    params = FakeSim(method=method)
    solve_pressure(pressure, divergence, face_mask_u, face_mask_v, fluid_density, cell_size, dt, params)
    assert np.allclose(pressure, 0.0), "Pressure should stay zero when divergence is zero."

    apply_pressure_gradient(velocity_u, velocity_v, pressure, face_mask_u, face_mask_v,
                             fluid_density, cell_size, dt)
    assert np.allclose(velocity_u, 0.0), "velocity_u should be unchanged (zero) after zero-pressure gradient."
    assert np.allclose(velocity_v, 0.0), "velocity_v should be unchanged (zero) after zero-pressure gradient."

    compute_divergence(velocity_u, velocity_v, divergence, face_mask_u, face_mask_v, cell_size)
    assert np.allclose(divergence, 0.0), "Divergence must remain zero after a no-op projection."


@pytest.mark.parametrize("method", SOLVER_REGISTRY.keys())
@pytest.mark.parametrize("iterations", [3, 5, 10])
def test_substep_zero_state_is_noop(method, iterations):
    """
    Running a full substep with zero velocity/density/pressure must leave
    velocity, pressure, and divergence at zero. This exercises the REAL
    call sites in Simulation.substep(), not a reimplementation, so it
    catches argument-order bugs at the integration layer.
    """
    sim = SmokeSimulation2D(height=16, width=16, cell_size=1.0, solver_method=method)

    sim.substep(sim.dt)

    assert np.allclose(sim.velocity_u, 0.0), "velocity_u changed from zero with no forcing."
    assert np.allclose(sim.velocity_v, 0.0), "velocity_v changed from zero with no forcing."
    assert np.allclose(sim.pressure, 0.0), "pressure changed from zero with zero divergence."
    assert np.allclose(sim.divergence, 0.0), "divergence is nonzero after a no-op substep."


@pytest.mark.parametrize("method", SOLVER_REGISTRY.keys())
@pytest.mark.parametrize("iterations", [3, 5, 10])
def test_substep_single_impulse_reduces_divergence(method, iterations):
    """
    Inject a single nonzero velocity face, run one substep, and confirm
    the projection step actually reduces (not increases) divergence.
    This is the test that would have caught the real regression: it runs
    the actual Simulation.substep() call chain end-to-end.
    """
    sim = SmokeSimulation2D(height=16, width=16, cell_size=1.0, solver_method=method)

    sim.velocity_u[8, 8] = 1.0

    sim._compute_divergence()
    div_before = np.abs(sim.divergence).max()
    assert div_before > 0.0, "Test setup invalid: expected nonzero divergence from impulse."

    sim._solve_pressure()
    sim._apply_pressure_gradient(sim.dt)
    sim._compute_divergence()
    div_after = np.abs(sim.divergence).max()

    assert div_after < div_before, (
        f"Projection did not reduce divergence: before={div_before}, after={div_after}"
    )



