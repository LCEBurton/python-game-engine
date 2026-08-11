"""
Benchmarks for different smoke simulation methods.
"""

from engine.smoke_simulator.numba.simulation import SmokeSimulation2D
from engine.smoke_simulator.common.emitters import Emitter2D
from tools.profiler import Profiler

import numpy as np

def run_benchmark(simulation_method: str, pressure_iterations: int, num_frames: int = 1000):
    """
    Run a benchmark for the specified smoke simulation method.

    Args:
        simulation_method (str): The simulation method to benchmark.
        num_frames (int): Number of frames to simulate.
    """
    # Create a smoke simulation instance
    sim = SmokeSimulation2D(width=512, height=512, cell_size=1.0, 
                            pressure_solver_method=simulation_method, pressure_iterations=pressure_iterations)
    
    # Create an emitter at the center of the domain
    emitter = Emitter2D(position=(sim.domain_width / 2, sim.domain_height / 2), 
                      radius=15.0, 
                      shape=(sim.height, sim.width),
                      density_value=1.0,
                      temperature_value=15.0,
                      velocity_value=(0.0, -5.0),  # Upward velocity
                      approach='gaussian',
                      numba_optimized=True)

    sim.add_emitter(emitter)

    profiler = Profiler()
    profiler.start()
    max_div = (0.0, (-1, -1))

    print(f"Benchmark for {simulation_method} with {pressure_iterations} iterations:")

    for frame in range(num_frames):
        sim.update(0.016)  # ~60 fps
        m_div = np.max(sim.divergence)
        if m_div > max_div[0]:
            max_div = (m_div, list(map(lambda x: int(x), np.unravel_index(sim.divergence.argmax(), sim.divergence.shape))))


    sim.profiler.report()
    print(f"Max divergence: {max_div[0]} at index {max_div[1][0]}, {max_div[1][1]}")
    print("\n")

def main():
    # Define the simulation methods and their corresponding pressure iterations to benchmark
    simulation_methods = [
        ("jacobi", 20),
        ("jacobi", 40),
        ("jacobi", 80),
        ("gauss_seidel", 10),
        ("gauss_seidel", 20), 
        ("red_black_gauss_seidel", 10),
        ("red_black_gauss_seidel", 20)
    ]

    for method, iterations in simulation_methods:
        run_benchmark(method, iterations)
    
if __name__ == "__main__":
    main()
