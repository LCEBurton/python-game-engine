import numpy as np

from engine.smoke_simulator.common.simulation import SmokeSimulation2D
from engine.smoke_simulator.common.emitters import Emitter2D
from tools.profiling import ProfileSession

import pygame


def main():
    # Create a smoke simulation instance
    sim = SmokeSimulation2D(width=512, height=512, cell_size=1.0, pressure_solver_method='jacobi', pressure_iterations=40, debug=True)
    
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
    loop_profiler = ProfileSession(enabled=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with loop_profiler.timing.scope("sim_update"):
            sim.update(0.016)  # ~60 fps
            density = sim.get_density_field()
        
        # Convert density to RGB (0-255 range)
        with loop_profiler.timing.scope("render"):
            density_rgb = (density.T * 255).astype(np.uint8)
            surf = pygame.surfarray.make_surface(density_rgb)
            
            screen.blit(surf, (0, 0))

        # Display FPS
        with loop_profiler.timing.scope("fps_display"):
            fps = clock.get_fps()
            fps_text = font.render(f'FPS: {fps:.1f}', True, (255, 0, 0))
            screen.blit(fps_text, (10, 10))
            pygame.display.flip()


        clock.tick(60)

    print("Sim profiling report:")
    sim.profiler.report()
    print("\nOverall profiling report:")
    loop_profiler.report()


    
if __name__ == "__main__":
    main()
