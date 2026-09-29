import numpy as np

from ..kernels.numba.advection import advect_scalar_field
from ..kernels.numba.divergence import compute_divergence
from ..kernels.numba.forces import apply_forces, cool_temperature
from ..kernels.numba.boundaries import enforce_boundary_conditions
from ..kernels.numba.pressure_gradient import apply_pressure_gradient

from tools.profiling import ProfileSession

class SmokeSimulation3D:
    """
    Real-time 3D smoke simulator using semi-Lagrangian advection
    and pressure projection for incompressibility.
    """
    
    def __init__(self, width=128, height=128, depth=128, cell_size=1.0):
        """
        Initialize the 3D smoke simulation grid.
        
        Args:
            width: Number of grid cells in x direction
            height: Number of grid cells in y direction
            depth: Number of grid cells in z direction
            cell_size: Physical size of each cell (meters)
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.cell_size = cell_size
        
        # Physical domain size
        self.domain_width = width * cell_size
        self.domain_height = height * cell_size
        self.domain_depth = depth * cell_size
        
        # Simulation parameters
        self.dt = 0.016  # Fixed timestep (60 FPS)
        self.accumulator = 0.0
        self.max_substeps = 4
        
        # Solver parameters
        self.pressure_iterations = 40
        self.density_dissipation = 0.999  # Slight fade per substep
        self.buoyancy = 5.0
        self.gravity = -0.5
        self.fluid_density = 1.0  # For pressure solver
        
        # Allocate simulation arrays (all float32, C-contiguous)
        shape = (height, width, depth)
        
        # Density field
        self.density = np.zeros(shape, dtype=np.float32)
        self.density_temp = np.zeros(shape, dtype=np.float32)
        
        # Velocity field (staggered or cell-centered)
        # Starting with cell-centered for simplicity
        self.velocity_u = np.zeros(shape, dtype=np.float32)  # x-component
        self.velocity_v = np.zeros(shape, dtype=np.float32)  # y-component
        self.velocity_w = np.zeros(shape, dtype=np.float32)  # z-component
        self.velocity_u_temp = np.zeros(shape, dtype=np.float32)
        self.velocity_v_temp = np.zeros(shape, dtype=np.float32)
        self.velocity_w_temp = np.zeros(shape, dtype=np.float32)
        
        # Pressure field
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.pressure_temp = np.zeros(shape, dtype=np.float32)

        # Temperature field (optional, for buoyancy)
        self.temperature = np.zeros(shape, dtype=np.float32)
        self.temperature_temp = np.zeros(shape, dtype=np.float32)
        self.ambient_temperature = 0.0  # Ambient temperature for cooling
        self.cooling_rate = 0.5
        
        # Divergence field
        self.divergence = np.zeros(shape, dtype=np.float32)
        
        # Solid cell mask (1 = solid, 0 = fluid)
        self.solid_mask = np.zeros(shape, dtype=np.uint8)
        
        # Create boundary walls
        self._init_boundaries()
        
        # Emitters (position, radius, density_rate, velocity)
        self.emitters = []
        self.debug = False

        self.profiler = ProfileSession()
        
        print(f"Initialized 3D smoke sim: {width}×{height}x{depth} grid, {self.domain_width:.1f}×{self.domain_height:.1f}x{self.domain_depth:.1f}m domain")
    
    def _init_boundaries(self):
        """Mark boundary cells as solid."""
        self.solid_mask[0, :, :] = 1   # Bottom
        self.solid_mask[-1, :, :] = 1  # Top
        self.solid_mask[:, 0, :] = 1   # Left
        self.solid_mask[:, -1, :] = 1  # Right
        self.solid_mask[:, :, 0] = 1   # Front
        self.solid_mask[:, :, -1] = 1  # Back
    
    def add_emitter(self, emitter):
        """
        Add a smoke emitter at grid coordinates.
        
        Args:
            x, y: Grid cell position
            radius: Emission radius in cells
            density_rate: Density added per second
            vel_x, vel_y: Initial velocity injection
        """
        self.emitters.append(emitter)
    
    def update(self, frame_dt):
        """
        Update simulation with variable frame time.
        Uses fixed internal timesteps with accumulator.
        
        Args:
            frame_dt: Time elapsed since last frame (seconds)
        """
        self.accumulator += frame_dt
        
        substeps = 0
        while self.accumulator >= self.dt and substeps < self.max_substeps:
            self.substep(self.dt)
            self.accumulator -= self.dt
            substeps += 1
        
        # Clamp accumulator to prevent spiral of death
        if substeps >= self.max_substeps:
            self.accumulator = 0.0
    
    def substep(self, dt):
        """
        Perform one simulation substep.
        
        Args:
            dt: Simulation timestep (seconds)
        """

        # 1. Apply emitters
        with self.profiler.timing.scope("emitters"):
            self._apply_emitters(dt)
        
        # 2. Apply forces (buoyancy, gravity)
        with self.profiler.timing.scope("forces"):
            self._apply_forces(dt)
        
        # 3. Advect velocity (self-advection)
        with self.profiler.timing.scope("advect_velocity"):
            self._advect_velocity(dt)
        
        # 4. Advect density
        with self.profiler.timing.scope("advect_density"):
            self._advect_density(dt)

        # 5. Advect temperature (if used)
        with self.profiler.timing.scope("advect_temperature"):
            self._advect_temperature(dt)
        
        with self.profiler.timing.scope("cooling"):
            self._cool_temperature(dt)

        # 5. Apply dissipation
        with self.profiler.timing.scope("dissipation"):
            self.density *= self.density_dissipation
        
        # 6. Pressure projection (make velocity divergence-free)
        with self.profiler.timing.scope("divergence"):
            self._compute_divergence()

        if self.debug:
            self.div_before = self.divergence.copy()

        with self.profiler.timing.scope("pressure_solve"):
            self._solve_pressure()

        with self.profiler.timing.scope("pressure_gradient"):
            self._apply_pressure_gradient(dt)

        with self.profiler.timing.scope("divergence_after_pressure"):
            self._compute_divergence()

        if self.debug:
            with self.profiler.timing.scope("divergence_stats"):
                self.div_after = self.divergence.copy()
                self.profiler.metrics.store("max_divergence_before", np.max(np.abs(self.div_before)))
                self.profiler.metrics.store("max_divergence_after", np.max(np.abs(self.div_after)))
                self.profiler.metrics.store("max_divergence_interior_before", np.max(np.abs(self.div_before[1:-1, 1:-1])))
                self.profiler.metrics.store("max_divergence_interior_after", np.max(np.abs(self.div_after[1:-1, 1:-1])))
                self.profiler.metrics.store("mean_divergence_before", np.mean(np.abs(self.div_before)))
                self.profiler.metrics.store("mean_divergence_after", np.mean(np.abs(self.div_after)))
                self.profiler.metrics.store("mean_divergence_interior_before", np.mean(np.abs(self.div_before[1:-1, 1:-1])))
                self.profiler.metrics.store("mean_divergence_interior_after", np.mean(np.abs(self.div_after[1:-1, 1:-1])))
                self.profiler.metrics.store("RMS_divergence_before", np.sqrt(np.mean(self.div_before**2)))
                self.profiler.metrics.store("RMS_divergence_interior_before", np.sqrt(np.mean(self.div_before[1:-1, 1:-1]**2)))
                self.profiler.metrics.store("RMS_divergence_after", np.sqrt(np.mean(self.div_after**2)))
                self.profiler.metrics.store("RMS_divergence_interior_after", np.sqrt(np.mean(self.div_after[1:-1, 1:-1]**2)))
        
        # 7. Enforce boundary conditions
        with self.profiler.timing.scope("boundaries"):
            self._enforce_boundaries()
    
    def _apply_emitters(self, dt):
        """Inject density and velocity from active emitters."""
        for e in self.emitters:
            e.emit(self.density, self.velocity_u, self.velocity_v, self.velocity_w, self.temperature, dt)
    
    def _apply_forces(self, dt):
        """Apply buoyancy and gravity to velocity field."""
        apply_forces(self.velocity_v, self.density, self.temperature, dt,
                     gravity=self.gravity, buoyancy_coefficient=self.buoyancy,
                     density_coefficient=1.0, ambient_temperature=self.ambient_temperature)

    
    def _advect_velocity(self, dt):
        """Semi-Lagrangian advection of velocity field."""
        advect_scalar_field(self.velocity_u, self.velocity_u_temp,
                            self.velocity_u, self.velocity_v, dt, self.cell_size)
        advect_scalar_field(self.velocity_v, self.velocity_v_temp,
                            self.velocity_u, self.velocity_v, dt, self.cell_size)
        # Swap velocity fields
        self.velocity_u, self.velocity_u_temp = self.velocity_u_temp, self.velocity_u
        self.velocity_v, self.velocity_v_temp = self.velocity_v_temp, self.velocity_v
    
    def _advect_density(self, dt):
        """Semi-Lagrangian advection of density field."""
        advect_scalar_field(self.density, self.density_temp,
                            self.velocity_u, self.velocity_v, dt, self.cell_size)
        # Swap density fields
        self.density, self.density_temp = self.density_temp, self.density

    def _advect_temperature(self, dt):
        """Semi-Lagrangian advection of temperature field."""
        advect_scalar_field(self.temperature, self.temperature_temp,
                            self.velocity_u, self.velocity_v, dt, self.cell_size)
        # Swap temperature fields
        self.temperature, self.temperature_temp = self.temperature_temp, self.temperature
    
    def _compute_divergence(self):
        """Compute divergence of velocity field."""
        #compute_divergence(self.velocity_u, self.velocity_v, self.divergence, self.solid_mask, self.cell_size)

    def _cool_temperature(self, dt):
        """Cool the temperature field over time."""
        cool_temperature(self.temperature, self.ambient_temperature, self.cooling_rate, dt)
    
    def _solve_pressure(self):
        """Solve for pressure using Jacobi iteration."""
        self.pressure.fill(0.0)  # Reset pressure field
        #jacobi_pressure_solver(self.pressure, self.pressure_temp,
        #                       self.divergence, self.fluid_density,
        #                       self.cell_size, self.dt, self.pressure_iterations)
        #gauss_seidel_pressure_solver(self.pressure, self.divergence,
        #                                self.fluid_density, self.cell_size,
        #                                self.dt, self.pressure_iterations)
        #red_black_gauss_seidel_pressure_solver(self.pressure, self.divergence,
        #                                        self.fluid_density, self.cell_size,
        #                                        self.dt, self.pressure_iterations)
 
    def _apply_pressure_gradient(self, dt):
        """Subtract pressure gradient from velocity (projection step)."""
#        apply_pressure_gradient(self.velocity_u, self.velocity_v, 
#                                self.pressure, self.fluid_density, self.cell_size, dt)
    
    def _enforce_boundaries(self):
        """Enforce boundary conditions on velocity and density."""
        enforce_boundary_conditions(self.velocity_u, self.velocity_v, self.solid_mask)
    
    def get_density_field(self):
        """Return current density field for rendering."""
        return self.density
    
    def reset(self):
        """Clear all fields and reset simulation state."""
        self.density.fill(0.0)
        self.velocity_u.fill(0.0)
        self.velocity_v.fill(0.0)
        self.temperature.fill(0.0)
        self.pressure.fill(0.0)
        self.divergence.fill(0.0)
        self.accumulator = 0.0


