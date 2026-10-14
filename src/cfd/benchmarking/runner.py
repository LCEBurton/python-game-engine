from time import perf_counter_ns
import traceback

from cfd.common.config import SimulationConfig
from cfd.common.simulation import SmokeSimulation2D as Simulation
from cfd.common.emitters import Emitter2D
from cfd.profiling.session import ProfileSession

from .config import BenchmarkConfig
from .result import BenchmarkResult, RunResult
from .writer import ResultWriter


class BenchmarkRunner:
    """Class to run benchmarks based on a BenchmarkConfig."""
    def __init__(self, writer: ResultWriter):
        self.writer = writer
        
        

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        runs = []

        for run_index in range(config.run.repetitions):
           results, raw_results = self._run_once(config, run_index)
           runs.append(results)

           if self.writer is not None:
               self.writer.write_raw(
                    benchmark_id=config.name,
                    run_index=run_index,
                    raw_data=raw_results
                )

        result = BenchmarkResult(
                benchmark_id=config.name,
                config=config,
                runs=runs
            )
        if self.writer is not None:
            self.writer.write_benchmark(result=result)

        return result

    def _run_once(self, config: BenchmarkConfig, run_index: int) -> tuple[RunResult, dict]:
        error = None
        status = "success"
        profiler = ProfileSession(config=config.profiling, grid_shape=config.simulation.grid.size)
        sim = Simulation(config=config.simulation, profiler=profiler)
        for e in config.emitters:
            emitter = Emitter2D.from_dict(e)
            sim.add_emitter(emitter)


        start = perf_counter_ns()
        try:
            for _ in range(config.run.warmup_frames):
                sim.substep(0.016)  # Warmup step with a fixed timestep

            sim.profiler.reset()  # Reset profiler before actual benchmarking

            for _ in range(config.run.frames):
                sim.substep(0.016)  # Benchmark step with a fixed timestep


        except Exception as e:
            status = "error"
            error = str(traceback.format_exc())
        finally:
            duration_s = (perf_counter_ns() - start) / 1_000_000_000

        result = RunResult(
            run_index=run_index,
            summary_result=sim.profiler.to_result().to_dict(),
            duration_s=duration_s,
            status=status,
            error=error
        )
        raw = sim.profiler.to_dict()

        return result, raw

        

