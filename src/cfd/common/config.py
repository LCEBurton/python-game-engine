from dataclasses import dataclass, field
import json
from enum import StrEnum
from cfd.solvers.params import SolverParams, JacobiParams, SOLVER_PARAM_TYPES


class Backend(StrEnum):
    """Backend options for CFD simulations."""
    NUMBA = "numba"
    CPP = "cpp"
    CUDA = "cuda"


@dataclass
class GridConfig:
    size: tuple[int, int]
    cell_size: float


@dataclass
class OperatorBackendConfig:
    advection: Backend = Backend.NUMBA
    boundaries: Backend = Backend.NUMBA
    divergence: Backend = Backend.NUMBA
    forces: Backend = Backend.NUMBA
    pressure_gradient: Backend = Backend.NUMBA
    residual: Backend = Backend.NUMBA


@dataclass
class PressureSolverConfig:
    method: str = "jacobi"
    backend: Backend = Backend.NUMBA
    params: SolverParams = field(default_factory=JacobiParams)


@dataclass
class SimulationConfig:
    grid: GridConfig
    pressure_solver: PressureSolverConfig = field(default_factory=PressureSolverConfig)
    operators: OperatorBackendConfig = field(default_factory=OperatorBackendConfig)
    debug: bool = False

    def to_dict(self) -> dict:
        """
        Convert the SimulationConfig to a dictionary.

        Returns:
            dict: Dictionary representation of the simulation configuration.
        """
        return {
            "grid": {
                "size": self.grid.size,
                "cell_size": self.grid.cell_size
            },
            "pressure_solver": {
                "method": self.pressure_solver.method,
                "backend": self.pressure_solver.backend.value,
                "params": self.pressure_solver.params.__dict__
            },
            "operators": {
                name: backend.value
                for name, backend in vars(self.operators).items()
            },
        }

    def to_json(self, path: str) -> None:
        """
        Save the SimulationConfig to a JSON file.

        Args:
            path (str): Path to the JSON file.
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """
        Load a SimulationConfig from a dict.

        Args:
            data (dict): Dictionary of config options.

        Returns:
            SimulationConfig: The loaded simulation configuration.
        """
        # Convert the dictionary into a SimulationConfig dataclass
        grid_data = data.get('grid', {})
        grid_config = GridConfig(
            size=tuple(grid_data.get('size', (16, 16))),
            cell_size=grid_data.get('cell_size', 1.0)
        )

        solver_data = data.get('pressure_solver', {})
        solver_method = solver_data.get('method', 'jacobi')
        solver_backend = solver_data.get('backend', 'numba')
        solver_params_type = SOLVER_PARAM_TYPES.get(solver_method, JacobiParams)

        return cls(
            grid=grid_config,
            pressure_solver=PressureSolverConfig(
                method=solver_method,
                backend=Backend(solver_backend),
                params=solver_params_type(**solver_data.get('params', {}))
            ),
            operators=OperatorBackendConfig(
                **{
                    name: Backend(value)
                    for name, value in data.get('operators', {}).items()
                }
            ),
        )

    @classmethod
    def from_json(cls, path: str) -> "SimulationConfig":
        """
        Load a SimulationConfig from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            SimulationConfig: The loaded simulation configuration.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate_config(self) -> None:
        """
        Validate the simulation configuration.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        # Validate grid size
        if not (isinstance(self.grid.size, tuple) and len(self.grid.size) == 2):
            raise ValueError("Grid size must be a tuple of two integers.")
        if not all(isinstance(dim, int) and dim > 0 for dim in self.grid.size):
            raise ValueError("Grid dimensions must be positive integers.")

        # Validate cell size
        if not isinstance(self.grid.cell_size, (int, float)) or self.grid.cell_size <= 0:
            raise ValueError("Cell size must be a positive number.")

        # Validate pressure solver method
        if self.pressure_solver.method not in SOLVER_PARAM_TYPES.keys():
            raise ValueError(f"Unsupported pressure solver method: {self.pressure_solver.method}")
        
        if not isinstance(self.pressure_solver.backend, Backend):
            raise ValueError(f"Unsupported pressure solver backend: {self.pressure_solver.backend}")

        if not isinstance(self.pressure_solver.params, SOLVER_PARAM_TYPES[self.pressure_solver.method]):
            raise ValueError(f"Pressure solver params must be of type {SOLVER_PARAM_TYPES[self.pressure_solver.method].__name__} for method {self.pressure_solver.method}")

        if not isinstance(self.operators, OperatorBackendConfig):
            raise ValueError("Operators must be an instance of OperatorBackendConfig.")
        if not all(isinstance(getattr(self.operators, op), Backend) for op in vars(self.operators)):
            raise ValueError("All operator backends must be instances of Backend.")

        if not isinstance(self.debug, bool):
            raise ValueError("Debug flag must be a boolean.")


        




