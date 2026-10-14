import json
import os

from .config import SuiteConfig
from .result import BenchmarkResult, SuiteResult

class ResultWriter:
    def __init__(self, root: str, write_raw: bool = True):
        self.root = root
        self.write_raw_flag = write_raw
        self.execution_id = None
        self.execution_dir = None

    def set_execution_id(self, execution_id: str) -> None:
        self.execution_id = execution_id

    def initialize_execution(self, suite_name: str) -> None:
        if self.execution_id is None:
            raise ValueError("Execution ID must be set before initializing execution.")
        
        self.execution_dir = os.path.join(self.root, f"{self.execution_id}")
        os.mkdir(self.execution_dir)
        os.mkdir(os.path.join(self.execution_dir, "results"))
        os.mkdir(os.path.join(self.execution_dir, "raw"))

    def write_config(self, config: SuiteConfig) -> None:
        if self.execution_dir is None:
            raise ValueError("Execution directory must be set before writing config.")

        config_path = os.path.join(self.execution_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=4)

    def write_raw(self, benchmark_id: str, run_index: int, raw_data: dict) -> None:
        if not self.write_raw_flag:
            return

        if self.execution_dir is None:
            raise ValueError("Execution directory must be set before writing raw data.")

        raw_path = os.path.join(self.execution_dir, "raw", benchmark_id, f"run_{run_index}.json")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump(raw_data, f, indent=4)

    def write_benchmark(self, result: BenchmarkResult) -> None:
        if self.execution_dir is None:
            raise ValueError("Execution directory must be set before writing benchmark results.")

        benchmark_path = os.path.join(self.execution_dir, "results", f"{result.benchmark_id}.json")
        with open(benchmark_path, "w") as f:
            json.dump(result.to_summary(), f, indent=4)

    def write_suite_result(self, result: SuiteResult) -> None:
        if self.execution_dir is None:
            raise ValueError("Execution directory must be set before writing suite results.")

        suite_path = os.path.join(self.execution_dir, "summary.json")
        with open(suite_path, "w") as f:
            json.dump(result.to_summary(), f, indent=4)



