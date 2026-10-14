import pytest
import json

from cfd.common.config import (
    SimulationConfig,
    GridConfig,
    PressureSolverConfig,
    OperatorBackendConfig,
    Backend,
)
from cfd.solvers.params import JacobiParams, SORParams, MultigridParams


def make_config(**overrides):
    defaults = dict(
        grid=GridConfig(size=(16, 16), cell_size=1.0),
        pressure_solver=PressureSolverConfig(),
        operators=OperatorBackendConfig(),
        debug=False,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults) # type: ignore


class TestInitialization:
    def test_default_pressure_solver_and_operators(self):
        config = make_config()
        assert config.pressure_solver.method == "jacobi"
        assert config.pressure_solver.backend == Backend.NUMBA
        assert isinstance(config.pressure_solver.params, JacobiParams)
        assert config.operators.advection == Backend.NUMBA
        assert config.debug is False

    def test_grid_values_are_set(self):
        config = make_config(grid=GridConfig(size=(32, 24), cell_size=0.5))
        assert config.grid.size == (32, 24)
        assert config.grid.cell_size == 0.5

    def test_custom_operator_backends(self):
        operators = OperatorBackendConfig(
            advection=Backend.CPP,
            boundaries=Backend.CUDA,
        )
        config = make_config(operators=operators)
        assert config.operators.advection == Backend.CPP
        assert config.operators.boundaries == Backend.CUDA
        assert config.operators.divergence == Backend.NUMBA  # untouched default


class TestValidation:
    def test_valid_config_does_not_raise(self):
        config = make_config()
        config.validate_config()  # should not raise

    @pytest.mark.parametrize("size", [(0, 16), (16, 0), (-1, 16)])
    def test_invalid_grid_dimensions_raise(self, size):
        config = make_config(grid=GridConfig(size=size, cell_size=1.0))
        with pytest.raises(ValueError, match="Grid dimensions"):
            config.validate_config()

    def test_grid_size_not_tuple_raises(self):
        config = make_config(grid=GridConfig(size=[16, 16], cell_size=1.0)) # type: ignore
        with pytest.raises(ValueError, match="Grid size must be a tuple"):
            config.validate_config()

    @pytest.mark.parametrize("cell_size", [0, -1.0])
    def test_invalid_cell_size_raises(self, cell_size):
        config = make_config(grid=GridConfig(size=(16, 16), cell_size=cell_size))
        with pytest.raises(ValueError, match="Cell size must be a positive number"):
            config.validate_config()

    def test_unsupported_pressure_solver_method_raises(self):
        config = make_config(
            pressure_solver=PressureSolverConfig(method="not_a_method", params=JacobiParams())
        )
        with pytest.raises(ValueError, match="Unsupported pressure solver method"):
            config.validate_config()

    def test_mismatched_solver_params_type_raises(self):
        # jacobi method with SOR params should fail the isinstance check
        config = make_config(
            pressure_solver=PressureSolverConfig(method="jacobi", params=SORParams())
        )
        with pytest.raises(ValueError, match="Pressure solver params must be of type"):
            config.validate_config()

    def test_invalid_backend_type_raises(self):
        config = make_config()
        config.pressure_solver.backend = "numba"  # plain string, not Backend enum # type: ignore
        with pytest.raises(ValueError, match="Unsupported pressure solver backend"):
            config.validate_config()

    def test_invalid_debug_flag_raises(self):
        config = make_config()
        config.debug = "yes" # type: ignore
        with pytest.raises(ValueError, match="Debug flag must be a boolean"):
            config.validate_config()


class TestSerialization:
    def test_to_dict_round_trip(self):
        config = make_config()
        data = config.to_dict()
        restored = SimulationConfig.from_dict(data)

        assert restored.grid.size == config.grid.size
        assert restored.grid.cell_size == config.grid.cell_size
        assert restored.pressure_solver.method == config.pressure_solver.method
        assert restored.pressure_solver.backend == config.pressure_solver.backend
        assert restored.pressure_solver.params == config.pressure_solver.params
        assert restored.operators == config.operators

    def test_from_dict_with_multigrid_params(self):
        data = {
            "grid": {"size": (8, 8), "cell_size": 2.0},
            "pressure_solver": {
                "method": "multigrid",
                "backend": "cpp",
                "params": {"num_levels": 3, "v_cycles": 1},
            },
            "operators": {"advection": "cuda"},
        }
        config = SimulationConfig.from_dict(data)

        assert isinstance(config.pressure_solver.params, MultigridParams)
        assert config.pressure_solver.params.num_levels == 3
        assert config.pressure_solver.backend == Backend.CPP
        assert config.operators.advection == Backend.CUDA
        config.validate_config()  # should be a fully valid config

    def test_from_dict_defaults_when_missing_sections(self):
        config = SimulationConfig.from_dict({})
        assert config.grid.size == (16, 16)
        assert config.grid.cell_size == 1.0
        assert config.pressure_solver.method == "jacobi"
        config.validate_config()

    def test_json_round_trip(self, tmp_path):
        config = make_config()
        path = tmp_path / "config.json"
        config.to_json(str(path))

        loaded = SimulationConfig.from_json(str(path))
        assert loaded.grid.size == config.grid.size
        assert loaded.pressure_solver.method == config.pressure_solver.method

        with open(path) as f:
            raw = json.load(f)
        assert raw["grid"]["size"] == list(config.grid.size)
