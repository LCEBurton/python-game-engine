from dataclasses import dataclass, field
from typing import Any
import os

from cfd.profiling.result import ProfilerResult
from .config import BenchmarkConfig

@dataclass
class RunResult:
    run_index: int
    summary_result: dict[str, Any]
    duration_s: float
    status: str = "success"
    error: str | None = None

    def to_dict(self) -> dict:
        """Return a summary dictionary of the run result."""
        return {
            "run_index": self.run_index,
            "duration_s": self.duration_s,
            "status": self.status,
            "error": self.error,
            "profiler_summary": self.summary_result
        }

    def __str__(self):
        return f"RunResult(run_index={self.run_index}, status={self.status}, duration_s={self.duration_s}, error={self.error})"

@dataclass
class BenchmarkResult:
    benchmark_id: str
    config: BenchmarkConfig
    runs: list[RunResult]
    status: str = "success"
    error: str | None = None

    def to_summary(self) -> dict:
        """Return a summary dictionary of the benchmark result."""
        durations = [run.duration_s for run in self.runs if run.duration_s is not None]
        duration_avg_s = (sum(durations) / len(self.runs)) if self.runs else -1.0
        duration_total_s = sum(durations) if self.runs else -1.0
        duration_std_s = (sum((d - duration_avg_s) ** 2 for d in durations) / len(self.runs)) ** 0.5 if self.runs else -1.0

        return {
            "benchmark_id": self.benchmark_id,
            "status": "success" if all(run.status == "success" for run in self.runs) else "partial_failure" if any(run.status == "success" for run in self.runs) else "failure",
            "error": self.error,
            "config": self.config.to_dict(),
            "summary": {
                "total_runs": len(self.runs),
                "successful_runs": sum(1 for run in self.runs if run.status == "success"),
                "failed_runs": sum(1 for run in self.runs if run.status != "success"),
                "duration_avg_s": duration_avg_s,
                "duration_total_s": duration_total_s,
                "duration_std_s": duration_std_s,
            }
        }

    def to_dict(self) -> dict:
        """Return a dictionary representation of the benchmark result."""
        return {
            "benchmark_id": self.benchmark_id,
            "config": self.config.to_dict(),
            "status": self.status,
            "error": self.error,
            "runs": [run.to_dict() for run in self.runs]
        }

    def __str__(self):
        return f"BenchmarkResult(name={self.benchmark_id}, config={self.config}, runs={self.runs})"

@dataclass
class SuiteResult:
    execution_id: str
    name: str
    benchmark_results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a dictionary representation of the suite result."""
        return {
            "execution_id": self.execution_id,
            "name": self.name,
            "benchmark_results": [benchmark.to_dict() for benchmark in self.benchmark_results]
        }

    def to_summary(self) -> dict:
        """Return a summary dictionary of the suite result."""
        total_benchmarks = len(self.benchmark_results)
        successful_benchmarks = sum(1 for benchmark in self.benchmark_results if benchmark.status == "success")
        failed_benchmarks = sum(1 for benchmark in self.benchmark_results if benchmark.status != "success")
        duration_s = sum(run.duration_s for benchmark in self.benchmark_results for run in benchmark.runs if run.duration_s is not None)
        benchmarks = []
        for benchmark in self.benchmark_results:
            benchmark_summary = benchmark.to_summary()
            benchmarks.append({
                "benchmark_id": benchmark.benchmark_id,
                "status": benchmark_summary["status"],
                "result": os.path.join("results", f"{benchmark.benchmark_id}.json"),
                "duration_avg_s": benchmark_summary["summary"]["duration_avg_s"],
                "duration_total_s": benchmark_summary["summary"]["duration_total_s"],
                "duration_std_s": benchmark_summary["summary"]["duration_std_s"],
            })

        return {
            "execution_id": self.execution_id,
            "name": self.name,
            "total_benchmarks": total_benchmarks,
            "successful_benchmarks": successful_benchmarks,
            "failed_benchmarks": failed_benchmarks,
            "duration_s": duration_s,
            "benchmarks": benchmarks
        }

            

        
