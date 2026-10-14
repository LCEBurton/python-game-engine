import numpy as np


from .config import SimulationConfig

from cfd.operators.advection import advect_scalar_field, advect_velocity_u, advect_velocity_v
from cfd.operators.divergence import compute_divergence
from cfd.operators.forces import apply_forces, cool_temperature
from cfd.operators.boundaries import enforce_boundary_conditions
from cfd.operators.pressure_gradient import apply_pressure_gradient
from cfd.operators.residual import compute_residual
from cfd.solvers.dispatch import solve_pressure
from cfd.solvers.params import DEFAULT_SOLVER_PARAMS, SolverParams
from cfd.backends.numba.solvers.common import calculateAlpha, calculateReciprocalBeta
from cfd.profiling import ProfileSession
from cfd.profiling.config import ProfileConfig


class SmokeSimulation2D:
    """
    Real-time 2D smoke simulator using semi-Lagrangian advection
    and pressure projection for incompressibility.
    """
    solver_params: SolverParams

    def __init__(
        self,
        config: SimulationConfig,
        profiler: ProfileSession | None = None,
    ) -> None:
        self.config = config
        self._initalise_from_config(config, profiler)

    def _initalise_from_config(self, config: SimulationConfig, profiler: ProfileSession | None):
        """
        Initialize the 2D smoke simulation grid.

        Args:
            config: Simulation configuration.
            profiler: Optional ProfileSession; if not provided, an empty
                (fully disabled) profiling session is created, incurring
                no measurable overhead.
        """

        self.width, self.height = config.grid.size
        self.cell_size = config.grid.cell_size
        self.pressure_solver_method = config.pressure_solver.method
        self.pressure_solver_backend = config.pressure_solver.backend

        self.solver_params = config.pressure_solver.params if config.pressure_solver.params is not None else DEFAULT_SOLVER_PARAMS[self.pressure_solver_method]

        self.operators_backend = config.operators
        
        # Physical domain size
        self.domain_width = self.width * self.cell_size
        self.domain_height = self.height * self.cell_size
        
        # Simulation parameters
        self.dt = 0.016  # Fixed timestep (60 FPS)
        self.accumulator = 0.0
        self.max_substeps = 4
        
        # Solver parameters
        self.density_dissipation = 0.999  # Slight fade per substep
        self.buoyancy = 5.0
        self.gravity = -0.5
        self.fluid_density = 1.0  # For pressure solver
        
        # Allocate simulation arrays (all float32, C-contiguous)
        shape = (self.height, self.width)
        
        # Density field
        self.density = np.zeros(shape, dtype=np.float32)
        self.density_temp = np.zeros(shape, dtype=np.float32)
        
        # Velocity fields staggered using MAC grid convention (u, v components) 
        # Physical/cell-index space convention:
        #   - cell (i, j) center is located at (i + 0.5, j + 0.5)
        #   - u-face (i, j) is located at (i, j + 0.5)   -> shape (height, width + 1)
        #   - v-face (i, j) is located at (i + 0.5, j)   -> shape (height + 1, width)
        self.velocity_u = np.zeros((self.height, self.width + 1), dtype=np.float32)  # x-component
        self.velocity_v = np.zeros((self.height + 1, self.width), dtype=np.float32)  # y-component
        self.velocity_u_temp = np.zeros((self.height, self.width + 1), dtype=np.float32)
        self.velocity_v_temp = np.zeros((self.height + 1, self.width), dtype=np.float32)
        
        # Pressure field
        self.pressure = np.zeros(shape, dtype=np.float32)

        # Temperature field (optional, for buoyancy)
        self.temperature = np.zeros(shape, dtype=np.float32)
        self.temperature_temp = np.zeros(shape, dtype=np.float32)
        self.ambient_temperature = 0.0  # Ambient temperature for cooling
        self.cooling_rate = 0.5
        
        # Divergence field
        self.divergence = np.zeros(shape, dtype=np.float32)

        # Pressure residual field (debugging/capture only, not used by solver)
        self.pressure_residual = np.zeros(shape, dtype=np.float32)
        
        # Solid cell mask (1 = solid, 0 = fluid)
        self._solid_mask = np.zeros(shape, dtype=bool)
        self._face_mask_u = np.zeros((self.height, self.width + 1), dtype=self._solid_mask.dtype)
        self._face_mask_v = np.zeros((self.height + 1, self.width), dtype=self._solid_mask.dtype)

        self._face_mask_u[:, 1:-1] = ~(self._solid_mask[:, :-1] | self._solid_mask[:, 1:])
        self._face_mask_v[1:-1, :] = ~(self._solid_mask[:-1, :] | self._solid_mask[1:, :])
        self.is_fluid = self._face_mask_u[:, :-1] | self._face_mask_u[:, 1:] | self._face_mask_v[:-1, :] | self._face_mask_v[1:, :]

        self._mask_dirty = False
        
        # Create boundary walls
        self._init_boundaries()
        
        # Emitters (position, radius, density_rate, velocity)
        self.emitters = []

        self.debug = config.debug
        self.profiler = profiler if profiler is not None else ProfileSession(
            config=ProfileConfig(), grid_shape=shape
        )

        print(f"Initialized 2D smoke sim: {self.width}×{self.height} grid, "
              f"{self.domain_width:.1f}×{self.domain_height:.1f}m domain")
    
    def _init_boundaries(self):
        """Mark boundary cells as solid."""
        self._solid_mask[0, :] = 1   # Bottom
        self._solid_mask[-1, :] = 1  # Top
        self._solid_mask[:, 0] = 1   # Left
        self._solid_mask[:, -1] = 1  # Right

    def set_solid_mask(self, solid_mask):
        """
        Set the solid cell mask for the simulation.
        
        Args:
            solid_mask: 2D array of shape (height, width) where 1 indicates solid and 0 indicates fluid.
        """
        if solid_mask.shape != (self.height, self.width):
            raise ValueError(f"Solid mask must have shape ({self.height}, {self.width})")
        self._solid_mask = solid_mask.astype(bool)
        self._mask_dirty = True  # Mark mask as dirty for reprocessing

    def get_solid_mask(self):
        """
        Get the current solid cell mask.
        
        Returns:
            2D array of shape (height, width) where 1 indicates solid and 0 indicates fluid.
        """
        return self._solid_mask.copy()
    
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
        self.profiler.tick()

        # Update face masks if solid mask has changed
        # Face_masks have convention 0: solid, 1: fluid to make calculations easier (e.g., for divergence and pressure solve)
        # Face masks is solid if either adjacent cell is solid
        if self._mask_dirty:
            # interior faces
            self._face_mask_u[:, 1:-1] = ~(self._solid_mask[:, :-1] | self._solid_mask[:, 1:])
            self._face_mask_v[1:-1, :] = ~(self._solid_mask[:-1, :] | self._solid_mask[1:, :])
            self.is_fluid = self._face_mask_u[:, :-1] | self._face_mask_u[:, 1:] | self._face_mask_v[:-1, :] | self._face_mask_v[1:, :]

            self._mask_dirty = False

        with self.profiler.timing.scope("simulation.substep"):
            # 1. Apply emitters
            with self.profiler.timing.scope("simulation.emitters"):
                self._apply_emitters(dt)

            # 2. Apply forces (buoyancy, gravity)
            with self.profiler.timing.scope("simulation.forces"):
                self._apply_forces(dt)

            # 3. Advect velocity (self-advection)
            with self.profiler.timing.scope("simulation.advection.velocity"):
                self._advect_velocity(dt)

            # 4. Advect density
            with self.profiler.timing.scope("simulation.advection.density"):
                self._advect_density(dt)

            # 5. Advect temperature (if used)
            with self.profiler.timing.scope("simulation.advection.temperature"):
                self._advect_temperature(dt)

            with self.profiler.timing.scope("simulation.cooling"):
                self._cool_temperature(dt)

            # 5. Apply dissipation
            with self.profiler.timing.scope("simulation.dissipation"):
                self.density *= self.density_dissipation

            # 6. Pressure projection (make velocity divergence-free)
            with self.profiler.timing.scope("simulation.divergence"):
                self._compute_divergence()

            with self.profiler.timing.scope("profiling.capture"):
                self.profiler.metrics.capture("divergence", "before_projection", self.divergence)

            with self.profiler.timing.scope("simulation.pressure.solve"):
                self._solve_pressure()

            with self.profiler.timing.scope("simulation.pressure.apply"):
                self._apply_pressure_gradient(dt)

            # 7. Enforce boundary conditions
            with self.profiler.timing.scope("simulation.boundaries"):
                self._enforce_boundaries()

        with self.profiler.timing.scope("profiling.metrics"):
            self._compute_divergence()
            self.profiler.metrics.capture("divergence", "after_projection", self.divergence)
            self.profiler.metrics.capture("pressure", "final", self.pressure)

            alpha = calculateAlpha(self.cell_size, self.fluid_density, dt)
            reciprocal_beta = calculateReciprocalBeta()
            compute_residual(
                self.pressure, self.divergence, self.is_fluid,
                self._face_mask_u, self._face_mask_v,
                alpha, reciprocal_beta, self.pressure_residual,
            )
            self.profiler.metrics.capture("pressure_residual", "final", self.pressure_residual)

            self.profiler.metrics.process()

    
    def _apply_emitters(self, dt):
        """Inject density and velocity from active emitters."""
        for e in self.emitters:
            e.emit(self.density, self.velocity_u, self.velocity_v, self.temperature, dt)
    
    def _apply_forces(self, dt):
        """Apply buoyancy and gravity to velocity field."""
        apply_forces(self.velocity_v, self.density, self.temperature, dt,
                     gravity=self.gravity, buoyancy_coefficient=self.buoyancy,
                     density_coefficient=1.0, ambient_temperature=self.ambient_temperature)

    
    def _advect_velocity(self, dt):
        """Semi-Lagrangian advection of velocity field."""
        advect_velocity_u(self.velocity_u, self.velocity_u_temp, self.velocity_v, self._face_mask_u, dt, self.cell_size)
        advect_velocity_v(self.velocity_v, self.velocity_v_temp, self.velocity_u, self._face_mask_v, dt, self.cell_size)
        # Swap velocity fields
        self.velocity_u, self.velocity_u_temp = self.velocity_u_temp, self.velocity_u
        self.velocity_v, self.velocity_v_temp = self.velocity_v_temp, self.velocity_v
    
    def _advect_density(self, dt):
        """Semi-Lagrangian advection of density field."""
        advect_scalar_field(self.density, self.density_temp, self.velocity_u, self.velocity_v, self.is_fluid, dt, self.cell_size)
        # Swap density fields
        self.density, self.density_temp = self.density_temp, self.density

    def _advect_temperature(self, dt):
        """Semi-Lagrangian advection of temperature field."""
        advect_scalar_field(self.temperature, self.temperature_temp, self.velocity_u, self.velocity_v, self.is_fluid, dt, self.cell_size)
        # Swap temperature fields
        self.temperature, self.temperature_temp = self.temperature_temp, self.temperature
    
    def _compute_divergence(self):
        """Compute divergence of velocity field."""
        compute_divergence(self.velocity_u, self.velocity_v, self.divergence, self._face_mask_u, self._face_mask_v, self.cell_size)

    def _cool_temperature(self, dt):
        """Cool the temperature field over time."""
        cool_temperature(self.temperature, self.ambient_temperature, self.cooling_rate, dt)
    
    def _solve_pressure(self):
        """Solve for pressure using the configured solver method."""
        self.pressure.fill(0.0)
        solve_pressure(self.pressure, self.divergence, self._face_mask_u, self._face_mask_v,
                       self.fluid_density, self.cell_size, self.dt, self)
 
    def _apply_pressure_gradient(self, dt):
        """Subtract pressure gradient from velocity (projection step)."""
        apply_pressure_gradient(self.velocity_u, self.velocity_v, self.pressure, self._face_mask_u, 
                                self._face_mask_v, self.fluid_density, self.cell_size, dt)
    
    def _enforce_boundaries(self):
        """Enforce boundary conditions on velocity and density."""
        enforce_boundary_conditions(self.velocity_u, self.velocity_v, self._solid_mask)
    
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



