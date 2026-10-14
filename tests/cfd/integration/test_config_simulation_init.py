import numpy as np
import pytest

from cfd.common.config import (
    SimulationConfig,
    GridConfig,
    PressureSolverConfig,
    OperatorBackendConfig,
    Backend,
)
from cfd.common.simulation import SmokeSimulation2D
from cfd.solvers.params import JacobiParams, SORParams


def make_config(width=16, height=16, cell_size=1.0, method="jacobi", params=None, debug=False):
    return SimulationConfig(
        grid=GridConfig(size=(width, height), cell_size=cell_size),
        pressure_solver=PressureSolverConfig(
            method=method,
            backend=Backend.NUMBA,
            params=params if params is not None else JacobiParams(),
        ),
        operators=OperatorBackendConfig(),
        debug=debug,
    )


class TestSimulationInitFromConfig:
    def test_grid_dimensions_match_config(self):
        config = make_config(width=20, height=12, cell_size=0.5)
        sim = SmokeSimulation2D(config)

        assert sim.width == 20
        assert sim.height == 12
        assert sim.cell_size == 0.5
        assert sim.domain_width == pytest.approx(10.0)
        assert sim.domain_height == pytest.approx(6.0)

    def test_field_shapes_match_grid(self):
        config = make_config(width=16, height=10)
        sim = SmokeSimulation2D(config)

        assert sim.density.shape == (10, 16)
        assert sim.pressure.shape == (10, 16)
        assert sim.temperature.shape == (10, 16)
        assert sim.velocity_u.shape == (10, 17)
        assert sim.velocity_v.shape == (11, 16)

    def test_fields_initialized_to_zero(self):
        config = make_config()
        sim = SmokeSimulation2D(config)

        assert np.allclose(sim.density, 0.0)
        assert np.allclose(sim.velocity_u, 0.0)
        assert np.allclose(sim.velocity_v, 0.0)
        assert np.allclose(sim.pressure, 0.0)
        assert np.allclose(sim.temperature, 0.0)

    def test_boundary_cells_marked_solid(self):
        config = make_config(width=16, height=16)
        sim = SmokeSimulation2D(config)
        mask = sim.get_solid_mask()

        assert mask[0, :].all()
        assert mask[-1, :].all()
        assert mask[:, 0].all()
        assert mask[:, -1].all()
        assert not mask[1:-1, 1:-1].any()

    def test_pressure_solver_settings_propagated(self):
        config = make_config(method="rb_sor_gauss_seidel", params=SORParams(iterations=10, omega=1.5))
        sim = SmokeSimulation2D(config)

        assert sim.pressure_solver_method == "rb_sor_gauss_seidel"
        assert sim.pressure_solver_backend == Backend.NUMBA
        assert sim.solver_params.iterations == 10 # type: ignore
        assert sim.solver_params.omega == 1.5 # type: ignore

    def test_debug_flag_propagated(self):
        config = make_config(debug=True)
        sim = SmokeSimulation2D(config)
        assert sim.debug is True

    def test_config_stored_on_simulation(self):
        config = make_config()
        sim = SmokeSimulation2D(config)
        assert sim.config is config

    def test_invalid_config_raises_before_use(self):
        config = make_config(width=-1, height=16)
        with pytest.raises(ValueError):
            config.validate_config()

    def test_substep_runs_without_error_from_config(self):
        """Smoke test: a simulation built purely from a validated config
        should be able to run a substep without raising."""
        config = make_config(width=16, height=16)
        config.validate_config()
        sim = SmokeSimulation2D(config)
        sim.substep(sim.dt)  # should not raise
