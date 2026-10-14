import json
import re
import hashlib

from dataclasses import dataclass, field, asdict
from typing import Any
from itertools import product


from cfd.common.config import SimulationConfig, Backend, OperatorBackendConfig
from cfd.common.emitters import EMITTER_APPROACHES
from cfd.profiling.config import ProfileConfig
from cfd.solvers.params import SOLVER_PARAM_TYPES
from cfd.solvers.registry import get_solver
from cfd.operators.registry import get_operator, OPERATOR_REGISTRY


SCHEMA_VERSION = "0.1"

def validate_emitters(emitters: list[dict]) -> None:
    if not emitters or not isinstance(emitters, list):
        raise ValueError("Emitters must be provided and must be a list of dictionaries.")

    # Validate emitters
    for emitter in emitters:
        if not isinstance(emitter, dict):
            raise ValueError("Each emitter must be a dictionary.")
        if "position" not in emitter or "radius" not in emitter:
            raise ValueError("Each emitter must have 'position' and 'radius' keys.")
        if not isinstance(emitter["position"], (list, tuple)) or len(emitter["position"]) != 2:
            raise ValueError("Emitter 'position' must be a list or tuple of length 2.")
        if not isinstance(emitter["radius"], (int, float)) or emitter["radius"] <= 0:
            raise ValueError("Emitter 'radius' must be a positive number.")
        if "density_value" not in emitter or not isinstance(emitter["density_value"], (int, float)):
            raise ValueError("Emitter 'density_value' must be a number.")
        if "temperature_value" not in emitter or not isinstance(emitter["temperature_value"], (int, float)):
            raise ValueError("Emitter 'temperature_value' must be a number.")
        if "velocity_value" not in emitter or not isinstance(emitter["velocity_value"], (list, tuple)):
            raise ValueError("Emitter 'velocity_value' must be a list or tuple.")
        if (any(not isinstance(v, (int, float)) for v in emitter["velocity_value"]) 
            or len(emitter["velocity_value"]) != 2):
            raise ValueError("Emitter 'velocity_value' must contain only numbers and be length 2.")
        if "approach" not in emitter or not isinstance(emitter["approach"], str):
            raise ValueError("Emitter 'approach' must be a string.")
        if emitter.get("approach") not in EMITTER_APPROACHES:
            raise ValueError(f"Emitter 'approach' must be one of {EMITTER_APPROACHES}.")
        if "numba_optimized" not in emitter or not isinstance(emitter["numba_optimized"], bool):
            raise ValueError("Emitter 'numba_optimized' must be a boolean.")


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""
    frames: int
    warmup_frames: int
    repetitions: int = 1


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmarking run."""
    name: str
    simulation: SimulationConfig
    run: RunConfig
    profiling: ProfileConfig = field(default_factory=ProfileConfig)
    emitters: list[dict] = field(default_factory=list)


    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """
        Load a BenchmarkConfig from a dict.

        Args:
            data (dict): Dictionary of config options.

        Returns:
            BenchmarkConfig: The loaded benchmark configuration.

        """
        # Convert the dictionary into a BenchmarkConfig dataclass
        profile_config = ProfileConfig.from_dict(data.get('profiling', {}))
        simulation_config = SimulationConfig.from_dict(data['simulation'])
        run_config = RunConfig(**data['run'])
        

        return cls(
            name=data['name'],
            simulation=simulation_config,
            run=run_config,
            profiling=profile_config,
            emitters=data.get('emitters', [])
        )

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkConfig":
        """
        Load a BenchmarkConfig from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            BenchmarkConfig: The loaded benchmark configuration.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def validate_config(self) -> None:
        """
        Validate the benchmark configuration.

        Returns:
            ValueError: If any part of the configuration is invalid.
        """
        # Validate repetitions
        if self.run.repetitions <= 0:
            raise ValueError("Repetitions must be a positive integer.")

        # Validate simulation config
        try:
            self.simulation.validate_config()
        except ValueError as e:
            raise ValueError(f"Simulation config validation failed: {e}")

        # Validate run config
        if self.run.frames <= 0 or self.run.warmup_frames < 0:
            raise ValueError("Run config validation failed: frames and warmup_frames must be positive.")

        # Validate profiling config
        try:
            self.profiling.validate_config()
        except ValueError as e:
            raise ValueError(f"Profiling config validation failed: {e}")

        try:
            validate_emitters(self.emitters)
        except ValueError as e:
            raise ValueError(f"Emitters config validation failed: {e}")

    def to_dict(self) -> dict:
        """
        Convert the BenchmarkConfig to a dictionary.

        Returns:
            dict: Dictionary representation of the BenchmarkConfig.
        """
        return {
            "name": self.name,
            "schema_version": SCHEMA_VERSION,
            "type": "benchmark",
            "simulation": self.simulation.to_dict(),
            "emitters": self.emitters,
            "run": {
                "frames": self.run.frames,
                "warmup_frames": self.run.warmup_frames,
                "repetitions": self.run.repetitions,
            },
            "profiling": self.profiling.to_dict()
        }

        
def make_benchmark_id(
    index: int,
    config: BenchmarkConfig,
) -> str:
    readable = [
        config.simulation.pressure_solver.method,
        config.simulation.pressure_solver.backend.value,
        f"{config.simulation.grid.size[0]}x"
        f"{config.simulation.grid.size[1]}",
    ]

    digest = _config_hash(config)

    return "-".join([
        f"{index:03d}",
        *(_slugify(x) for x in readable),
        digest,
    ])


def _config_hash(config: BenchmarkConfig) -> str:
    encoded = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(encoded).hexdigest()[:8]


def _slugify(value: object) -> str:
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")

@dataclass
class SuiteConfig:
    name: str
    benchmark_definitions: dict[str, Any]  # List of benchmark definitions as dictionaries

    def _validate_simulation(self, simulation: dict | None) -> None:
        if not simulation:
            raise ValueError("Simulation configuration must be provided.")

        # Validate grid configuration
        grid = simulation["grid"]
        if not grid:
            raise ValueError("Grid configuration must be provided.")

        grid_size = grid["size"]
        if not grid_size or any(not isinstance(size, (list, tuple)) or len(size) != 2 for size in grid_size):
            raise ValueError("Grid size must be a list of 2D tuples.")

        for size in grid_size:
            if any(not isinstance(dim, int) or dim <= 0 for dim in size):
                raise ValueError("Grid dimensions must be positive integers.")

        cell_size = grid.get("cell_size")
        if not cell_size or not(isinstance(cell_size, list)):
            raise ValueError("Grid cell size must be a list of positive numbers.")

        if any(not isinstance(size, (int, float)) or size <= 0 for size in cell_size):
            raise ValueError(f"Cell sizes must be all positive number.")

        # Validate pressure solver configuration
        pressure_solver = simulation["pressure_solver"]
        if not pressure_solver:
            raise ValueError("Pressure solver configuration must be provided.")

        methods = pressure_solver["method"]
        backends = pressure_solver["backend"]

        if not methods or not isinstance(methods, dict) or any(not isinstance(m, str) for m in methods.keys()):
            raise ValueError("Pressure solver methods must be a dictionary with string keys.")

        if not backends or not isinstance(backends, list) or any(not isinstance(b, str) for b in backends):
            raise ValueError("Pressure solver backend must be a list of strings.")

        if len(backends) != len(set(backends)):
            raise ValueError("Pressure solver backends must be unique.")

        for method, method_config in methods.items():
            if method not in SOLVER_PARAM_TYPES.keys():
                raise ValueError(f"Unsupported pressure solver method: {method}. Supported methods are: {list(SOLVER_PARAM_TYPES.keys())}")

            if not isinstance(method_config, dict):
                raise ValueError(f"Pressure solver method configuration for {method} must be a dictionary.")

            for backend in backends:
                if backend not in [backend.value for backend in Backend]:
                    raise ValueError(f"Unsupported pressure solver backend: {backend}. Supported backends are: {[backend.value for backend in Backend]}")

                try:
                    get_solver(method, backend)
                except ValueError as e:
                    raise ValueError(f"Suite pressure solver validation failed: {e}")

            if "params" not in method_config:
                raise ValueError(f"Pressure solver params for method {method} must be provided.")

            param_configs = method_config["params"]

            if not param_configs or not isinstance(param_configs, list):
                raise ValueError(f"Pressure solver params for method {method} must be a list of dictionaries.")

            param_type = SOLVER_PARAM_TYPES[method]

            for params in method_config["params"]:
                if not isinstance(params, dict):
                    raise ValueError(f"Pressure solver params for method {method} must be a dictionary.")
                try:
                    param_type(**params)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                            f"Invalid parameters for pressure solver method {method}: {e}"
                    ) from e

        # Validate operators configuration
        operators = simulation["operators"]
        if not operators or not isinstance(operators, dict):
            raise ValueError("Operators configuration must be provided and must be a dictionary.")

        required_operators = {op for op in asdict(OperatorBackendConfig()).keys()}

        missing_operators = required_operators - set(operators.keys())

        if missing_operators:
            raise ValueError(f"Missing required operators: {missing_operators}. Required operators are: {required_operators}")

        supported_backends = {backend.value for backend in Backend}

        for operator_name, operator_backends in operators.items():
            if not isinstance(operator_name, str):
                raise ValueError("Operator names must be strings.")

            if not isinstance(operator_backends, list) or not operator_backends:
                raise ValueError(
                    f"Backends for operator '{operator_name}' must be "
                    f"a non-empty list of strings."
                )

            if any(not isinstance(backend, str) for backend in operator_backends):
                raise ValueError(
                    f"Backends for operator '{operator_name}' must be strings."
                )

            if len(operator_backends) != len(set(operator_backends)):
                raise ValueError(
                    f"Backends for operator '{operator_name}' must be unique."
                )


            # TODO(operator-dispatch)

             # Currently operators are not being switched by backends in Simulation class
             # Need to figure out naming and registry for operators but this validation will fail
#            for backend in operator_backends:
#                if backend not in supported_backends:
#                    raise ValueError(
#                        f"Unsupported backend '{backend}' for operator "
#                        f"'{operator_name}'. Supported backends are: "
#                        f"{sorted(supported_backends)}"
#                    )
#
#                try:
#                    get_operator(operator_name, backend)
#                except ValueError as e:
#                    raise ValueError(
#                        f"Operator '{operator_name}' is not available "
#                        f"for backend '{backend}': {e}"
#                    ) from e

    def validate(self) -> None:
        
        if not self.name:
            raise ValueError("Suite name must be provided.")

        if not self.benchmark_definitions:
            raise ValueError("Benchmark definitions must be provided.")

        # Validate simulation configuration
        simulation = self.benchmark_definitions.get("simulation")
        self._validate_simulation(simulation)

        emitters = self.benchmark_definitions.get("emitters", [])
        validate_emitters(emitters)

        # Validate run configuration
        run = self.benchmark_definitions.get("run")
        if not run or not isinstance(run, dict):
            raise ValueError("Run configuration must be provided and must be a dictionary.")

        frames = run.get("frames")
        warmup_frames = run.get("warmup_frames")
        repetitions = run.get("repetitions")

        if not isinstance(frames, list) or not frames:
            raise ValueError("Frames must be a non-empty list of positive integers.")

        if any(not isinstance(frame, int) or frame <= 0 for frame in frames):
            raise ValueError("Frames must be a list of positive integers.")

        if not isinstance(warmup_frames, list) or not warmup_frames:
            raise ValueError("Warmup frames must be a non-empty list of positive integers.")

        if any(not isinstance(warmup_frame, int) or warmup_frame < 0 for warmup_frame in warmup_frames):
            raise ValueError("Warmup frames must be a list of positive integers.")

        if not isinstance(repetitions, list) or not repetitions:
            raise ValueError("Repetitions must be a non-empty list of positive integers.")

        if any(not isinstance(rep, int) or rep <= 0 for rep in repetitions):
            raise ValueError("Repetitions must be a list of positive integers.")

        # Validate profiling configuration
        try:
            profile_config = ProfileConfig.from_dict(self.benchmark_definitions.get('profiling', {}))
            profile_config.validate_config()
        except ValueError as e:
            raise ValueError(f"Suite config validation failed for profiler: {e}")
    
    def _expand_pressure_solvers(self, pressure_solver: dict) -> list[dict]:
        """
        Expand the pressure solver configurations into a list of simulation configurations.

        Args:
            simulation (dict): The simulation configuration dictionary.

        Returns:
            list[dict]: List of expanded simulation configurations.
        """
        expanded_solvers = []

        for method, method_config in pressure_solver["method"].items():
            for backend in pressure_solver["backend"]:
                for params in method_config["params"]:
                    expanded_solvers.append({
                        "method": method,
                        "backend": backend,
                        "params": params
                    })

        return expanded_solvers

    def _expand_operators(self, operators: dict) -> list[dict[str, str]]:
        """
        Expand the operator configurations into a list of simulation configurations.

        Args:
            operators (dict): The operator configuration dictionary.

        Returns:
            list[dict]: List of expanded operator configurations.
        """
        names = list(operators.keys())
        backends_list = [operators[name] for name in names]

        return [ 
                dict(zip(names, backends)) 
                for backends in product(*backends_list) 
        ]

    def _expand_simulation(self, 
                           simulation: dict, 
                           solver_configs: list[dict],
                           operator_configs) -> list[dict]:
        """
        Expand the simulation configurations into a list of simulation configurations.

        Args:
            simulation (dict): The simulation configuration dictionary.
            solver_configs (list[dict]): List of expanded pressure solver configurations.

        Returns:
            list[dict]: List of expanded simulation configurations.
        """
        expanded_simulations = []

        grid = simulation["grid"]

        for (
            size,
            cell_size,
            pressure_solver,
            operator_config,
        ) in product(
            grid["size"],
            grid["cell_size"],
            solver_configs,
            operator_configs,
        ):
            expanded_simulations.append({
                "grid": {
                    "size": size,
                    "cell_size": cell_size
                },
                "pressure_solver": pressure_solver,
                "operators": operator_config
            })

        return expanded_simulations            

    def expand_suite(self) -> list[BenchmarkConfig]:
        """
        Expand the suite configuration into a list of BenchmarkConfig instances.

        Returns:
            list[BenchmarkConfig]: List of expanded benchmark configurations.
        """

        definition = self.benchmark_definitions

        simulation = definition.get("simulation", {})
        emitters = definition.get("emitters", [])
        run = definition.get("run", {})
        profiling = definition.get("profiling")

        solver_configs = self._expand_pressure_solvers(simulation["pressure_solver"])
        operator_configs = self._expand_operators(simulation["operators"])

        expanded_simulations = self._expand_simulation(simulation, solver_configs, operator_configs)

        benchmarks = []

        for (
            simulation_config,
            frames,
            warmup_frames,
            repetition_count,
        ) in product(
            expanded_simulations,
            run["frames"],
            run["warmup_frames"],
            run["repetitions"],
        ):
            benchmark_config = BenchmarkConfig.from_dict({
                "name": "test",
                "simulation": simulation_config,
                "emitters": emitters,
                "run": {
                    "frames": frames,
                    "warmup_frames": warmup_frames,
                    "repetitions": repetition_count
                },
                "profiling": profiling,
            })
            name = make_benchmark_id(len(benchmarks), benchmark_config)
            benchmark_config.name = name
            benchmarks.append(benchmark_config)


        return benchmarks

    def to_dict(self) -> dict:
        """
        Convert the SuiteConfig to a dictionary.

        Returns:
            dict: Dictionary representation of the SuiteConfig.
        """
        return {
            "name": self.name,
            "schema_version": SCHEMA_VERSION,
            "type": "suite",
            "benchmark": self.benchmark_definitions
        } 

    @classmethod
    def from_dict(cls, data: dict) -> "SuiteConfig":
        """
        Load a SuiteConfig from a dict.

        Args:
            data (dict): Dictionary of config options.

        Returns:
            SuiteConfig: The loaded suite configuration.
        """
        return cls(
            name=data['name'],
            benchmark_definitions=data['benchmark'],
        )

    @classmethod
    def from_json(cls, path: str) -> "SuiteConfig":
        """
        Load a SuiteConfig from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            SuiteConfig: The loaded suite configuration.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        if data["type"] == "suite":
            return cls.from_dict(data)
        if data["type"] == "benchmark":
            return cls.from_benchmark_config(BenchmarkConfig.from_dict(data))
        else:
            raise ValueError(f"Invalid config type: {data['type']}. Expected 'suite' or 'benchmark'.")

    @classmethod
    def from_benchmark_config(cls, benchmark_config: BenchmarkConfig) -> "SuiteConfig":
        """
        Create a SuiteConfig from a BenchmarkConfig.

        Args:
            benchmark_config (BenchmarkConfig): The benchmark configuration.

        Returns:
            SuiteConfig: The created suite configuration.
        """
        simulation_dict = dict()
        simulation_dict["pressure_solver"] = {
            "method": {
                benchmark_config.simulation.pressure_solver.method: {
                    "params": [asdict(benchmark_config.simulation.pressure_solver.params)]
                }
            },
            "backend": [benchmark_config.simulation.pressure_solver.backend.value]
        }

        simulation_dict["operators"] = {
            name: [backend.value]
            for name, backend in asdict(benchmark_config.simulation.operators).items()
        }
        simulation_dict["grid"] = dict()
        simulation_dict["grid"]["size"] = [benchmark_config.simulation.grid.size]
        simulation_dict["grid"]["cell_size"] = [benchmark_config.simulation.grid.cell_size]

        emitters_dict = benchmark_config.emitters if benchmark_config.emitters else []

        run_dict = {
            "frames": [benchmark_config.run.frames],
            "warmup_frames": [benchmark_config.run.warmup_frames],
            "repetitions": [benchmark_config.run.repetitions]
        }


        return cls(
                name=benchmark_config.name,
                benchmark_definitions={
                    "simulation": simulation_dict,
                    "emitters": emitters_dict,
                    "run": run_dict,
                    "profiling": benchmark_config.profiling.to_dict()
                }
        )

