import numpy as np

from engine.smoke_simulator.numba.simulation import SmokeSimulation2D
from engine.smoke_simulator.common.emitters import Emitter2D
from tools.profiler import Profiler

import pygame



def main():
    # Create a smoke simulation instance
    sim = SmokeSimulation2D(width=512, height=512, cell_size=1.0, pressure_solver_method='jacobi', pressure_iterations=60)
    
    # Create an emitter at the center of the domain
    emitter = Emitter2D(position=(sim.domain_width / 2, sim.domain_height / 2), 
                      radius=15.0, 
                      shape=(sim.height, sim.width),
                      density_value=3.0,
                      temperature_value=15.0,
                      velocity_value=(0.0, -5.0),  # Upward velocity
                      approach='gaussian',
                      numba_optimized=True)

    sim.add_emitter(emitter)
    
    pygame.init()
    width, height = sim.domain_width, sim.domain_height  # match your grid size
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 36)
    i = 0
    profiler = Profiler()
    profiler.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        profiler.checkpoint("frame_start")
        sim.update(0.016)  # ~60 fps
        density = sim.get_density_field()
        profiler.checkpoint("sim_update")
        
        # Convert density to RGB (0-255 range)
        density_rgb = (density.T * 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(density_rgb)
        
        screen.blit(surf, (0, 0))
        profiler.checkpoint("render")

        # Display FPS
        fps = clock.get_fps()
        fps_text = font.render(f'FPS: {fps:.1f}', True, (255, 0, 0))
        screen.blit(fps_text, (10, 10))
        profiler.checkpoint("fps_render")

        pygame.display.flip()
        clock.tick(60)
        profiler.checkpoint("frame_end")

    print("Sim profiling report:")
    sim.profiler.report()
    print("\nOverall profiling report:")
    profiler.report()


    
if __name__ == "__main__":
    main()
