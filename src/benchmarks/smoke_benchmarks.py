"""
Benchmarks for different smoke simulation methods.
"""

from engine.smoke_simulator.common.simulation import SmokeSimulation2D
from engine.smoke_simulator.common.emitters import Emitter2D
from engine.smoke_simulator.solvers.params import (
        SolverParams, 
        DEFAULT_SOLVER_PARAMS, 
        SORParams, 
        MultigridParams, 
        JacobiParams, 
        GaussSeidelParams
)

import tools.profiling.profiler_plotter as profiler_plotter

import numpy as np
import os

from datetime import datetime

import pprint

results = {} # (name with params, iterations) -> list of benchmark results

field_snapshots = {} # (name, iterations) -> field snapshot at frame 500


def run_benchmark(simulation_method: str, num_frames: int, solver_params: SolverParams):
    """
    Run a benchmark for the specified smoke simulation method.

    Args:
        simulation_method (str): The simulation method to benchmark.
        pressure_iterations (int): Number of pressure solver iterations per frame.
        num_frames (int): Number of frames to simulate.
        solver_params: Optional solver-specific params (e.g. SORParams(omega=1.9)).
    """
    if solver_params is None:
        solver_params = DEFAULT_SOLVER_PARAMS[simulation_method]

    sim = SmokeSimulation2D(width=512, height=512, cell_size=1.0,
                             solver_method=simulation_method, 
                             solver_params=solver_params, debug=True)

    emitter = Emitter2D(position=(sim.domain_width / 2, sim.domain_height / 2),
                         radius=15.0,
                         shape=(sim.height, sim.width),
                         density_value=1.0,
                         temperature_value=15.0,
                         velocity_value=(0.0, -5.0),  # Upward velocity
                         approach='gaussian',
                         numba_optimized=True)

    sim.add_emitter(emitter)

    # Register a snapshot of peak divergence near the emitter at a fixed frame,
    # so results are directly comparable across different pressure_iterations.
    snapshot_frame = num_frames // 2
    window = int(emitter.radius)
    emitter_j, emitter_i = int(emitter.position[0]), int(emitter.position[1])

    def get_peak_divergence_near_emitter():
        region = sim.div_after[emitter_j - window:emitter_j + window, emitter_i - window:emitter_i + window]
        return np.max(np.abs(region))

    def get_divergence():
        return sim.divergence

    sim.profiler.snapshots.register_snapshot("peak_divergence_at_frame", snapshot_frame, get_peak_divergence_near_emitter)

    sim.profiler.snapshots.register_field_snapshot("divergence_after_pressure", snapshot_frame, get_divergence)


    max_div = (0.0, (-1, -1))

    print(f"Running benchmark: {simulation_method} ({solver_params.to_fn()} )...")

    for frame in range(num_frames):
        sim.update(0.016)  # ~60 fps
        sim.profiler.snapshots.tick(frame)

        m_div = np.max(sim.divergence)
        if m_div > max_div[0]:
            max_div = (m_div, tuple(map(int, np.unravel_index(sim.divergence.argmax(), sim.divergence.shape))))

    simulation_name = f"{simulation_method}_{solver_params.to_fn()}"

    if simulation_name not in results:
        results[simulation_name] = []

    results[simulation_name].append({
        "params": solver_params.to_fn(),
        "max_divergence": max_div[0],
        "max_divergence_index": max_div[1],
        "peak_divergence_at_frame": sim.profiler.snapshots.get_value("peak_divergence_at_frame"),
        "mean_divergence_before": sim.profiler.metrics.get_avg("mean_divergence_before"),
        "mean_divergence_after": sim.profiler.metrics.get_avg("mean_divergence_after"),
        "RMS_divergence_before": sim.profiler.metrics.get_avg("RMS_divergence_before"),
        "RMS_divergence_after": sim.profiler.metrics.get_avg("RMS_divergence_after"),
        "avg_pressure_solve_ms": sim.profiler.timing.get_avg_ms("pressure_solve"),
        "avg_total_frame_ms": sum(
            sim.profiler.timing.get_avg_ms(name) for name in (
                "emitters", "forces", "advect_velocity", "advect_density",
                "advect_temperature", "cooling", "dissipation", "divergence",
                "pressure_solve", "pressure_gradient", "divergence_after_pressure",
                "boundaries",
            )
        ),
    })

    field_snapshots[(simulation_name, solver_params.to_fn())] = sim.profiler.snapshots.get_field("divergence_after_pressure")

    if not os.path.exists(os.path.join(os.getcwd(), "benchmarks", f"{datetime.now().strftime('%Y-%m-%d')}")):
        os.makedirs(os.path.join(os.getcwd(), "benchmarks", f"{datetime.now().strftime('%Y-%m-%d')}"))

    sim.profiler.export_json(os.path.join(os.getcwd(), "benchmarks", f"{datetime.now().strftime('%Y-%m-%d')}",
                                          f"benchmark_{simulation_name}_{solver_params.to_fn()}.json"))


def print_results_table():
    """
    Print a formatted table summarizing benchmark results across all simulation methods.
    """
    headers = [
        "Method", "Params", "Max Div", "Max Div Idx",
        "Max Div at frame 500", "Mean Div (Before)", 
        "Mean Div (After)", "RMS Div (Before)", "RMS Div (After)",
        "Avg Pressure Solve (ms)", "Avg Frame (ms)"
    ]

    rows = []
    for method, runs in results.items():
        for run in runs:
            rows.append([
                method,
                run["params"],
                f"{run['max_divergence']:.6f}",
                str(run["max_divergence_index"]),
                f"{run['peak_divergence_at_frame']:.6f}",
                f"{run['mean_divergence_before']:.6f}",
                f"{run['mean_divergence_after']:.6f}",
                f"{run['RMS_divergence_before']:.6f}",
                f"{run['RMS_divergence_after']:.6f}",
                f"{run['avg_pressure_solve_ms']:.3f}",
                f"{run['avg_total_frame_ms']:.3f}",
            ])

    col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(len(headers))]

    def format_row(row):
        return " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))

    separator = "-+-".join("-" * w for w in col_widths)

    print("\nBenchmark Results:")
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))


def main():
    simulation_methods = [
        ("jacobi", JacobiParams(iterations=10)),
        ("jacobi", JacobiParams(iterations=20)),
        ("jacobi", JacobiParams(iterations=40)),
        
    #    ("gauss_seidel", 10),
    #    ("gauss_seidel", 20),
    #    ("rb_sor_gauss_seidel", 20, SORParams(omega=1.0)),
    #    ("rb_sor_gauss_seidel", 40, SORParams(omega=1.0)),
    #    ("rb_sor_gauss_seidel", 80, SORParams(omega=1.0)),
        ("rb_sor_gauss_seidel", SORParams(omega=1.7, iterations=20)),
        ("rb_sor_gauss_seidel", SORParams(omega=1.7, iterations=40)),
        ("rb_sor_gauss_seidel", SORParams(omega=1.7, iterations=80)),
    #    ("rb_sor_gauss_seidel", 20, SORParams(omega=0.0)),  # Test with default omega
    #    ("rb_sor_gauss_seidel", 40, SORParams(omega=0.0)),  # Test with default omega
    ("multigrid", MultigridParams(smoother_type="jacobi", num_levels=4, v_cycles=1, 
                                  smoother_iterations=2, coarsest_iterations=30)),
    ("multigrid", MultigridParams(smoother_type="jacobi", num_levels=4, v_cycles=2, 
                                  smoother_iterations=2, coarsest_iterations=30)),
    ("multigrid", MultigridParams(smoother_type="jacobi", num_levels=4, v_cycles=3, 
                                  smoother_iterations=2, coarsest_iterations=30)),
    #    ("rb_sor_gauss_seidel", 80, SORParams(omega=0.0)),  # Test with default omega
    ]

    for method, params in simulation_methods:
        run_benchmark(method, num_frames=1000, solver_params=params)

    print_results_table()

    profiler_plotter.plot_field_comparison(
        {f"{name} ({params_str})": field_snapshots[(name, params_str)]
         for name, params_str in field_snapshots.keys()},
        title="Divergence After Pressure Solve at Frame 500")


if __name__ == "__main__":
    main()

