import numpy as np
from numba import njit, prange

# Numba-compiled falloff functions
@njit(cache=True)
def _calculate_falloff_numba(distance, radius, approach_type):
    """
    Calculate falloff based on distance and approach type.
    
    Args:
        distance: Distance from emitter center
        radius: Emitter radius
        approach_type: String indicating approach ('linear', 'gaussian', etc.)
    
    Returns:
        Falloff value (0 to 1)
    """
    if approach_type == 'hard_cutoff':
        return 1.0 if distance < radius else 0.0
    elif approach_type == 'linear':
        if distance >= radius:
            return 0.0
        return 1.0 - (distance / radius)
    elif approach_type == 'quadratic':
        if distance >= radius:
            return 0.0
        return 1.0 - (distance / radius) ** 2
    elif approach_type == 'gaussian':
        if distance >= radius:
            return 0.0
        sigma = radius / 3.0
        return np.exp(-0.5 * (distance / sigma) ** 2)
    else:
        return 0.0

@njit(parallel=True, cache=True, fastmath=True)
def _apply_scalar_field_numba(field, value, dt, pos_x, pos_y, radius, approach_type):
    """
    Apply scalar emission to field using parallel Numba.
    
    Args:
        field: 2D array to modify
        value: Emission value
        dt: Time step
        pos_x, pos_y: Emitter position
        radius: Emitter radius
        approach_type: Falloff approach
    """
    height, width = field.shape
    
    # Calculate bounding box to avoid processing entire grid
    x_min = max(0, int(pos_x - radius - 1))
    x_max = min(width, int(pos_x + radius + 2))
    y_min = max(0, int(pos_y - radius - 1))
    y_max = min(height, int(pos_y + radius + 2))
    
    for y in prange(y_min, y_max):
        for x in range(x_min, x_max):
            dx = x - pos_x
            dy = y - pos_y
            distance = np.sqrt(dx * dx + dy * dy)
            
            if distance < radius:
                falloff = _calculate_falloff_numba(distance, radius, approach_type)
                field[y, x] += value * falloff * dt


@njit(parallel=True, cache=True, fastmath=True)
def _apply_vector_field_numba(field, vx, vy, dt, pos_x, pos_y, radius, approach_type):
    """
    Apply vector emission to field using parallel Numba.
    
    Args:
        field: 3D array (height, width, 2) to modify
        vx, vy: Velocity components
        dt: Time step
        pos_x, pos_y: Emitter position
        radius: Emitter radius
        approach_type: Falloff approach
    """
    height, width = field.shape[0], field.shape[1]
    
    # Calculate bounding box
    x_min = max(0, int(pos_x - radius - 1))
    x_max = min(width, int(pos_x + radius + 2))
    y_min = max(0, int(pos_y - radius - 1))
    y_max = min(height, int(pos_y + radius + 2))
    
    for y in prange(y_min, y_max):
        for x in range(x_min, x_max):
            dx = x - pos_x
            dy = y - pos_y
            distance = np.sqrt(dx * dx + dy * dy)
            
            if distance < radius:
                falloff = _calculate_falloff_numba(distance, radius, approach_type)
                field[y, x, 0] += vx * falloff * dt
                field[y, x, 1] += vy * falloff * dt

class Emitter2D:
    """
    Smoke emitter class that can inject density, velocity and temperature into the simulation fields. 
    Can take approach arg to change method used for emittance (Linear, Gaussian, etc.).
    """
    def __init__(self, 
                 position, 
                 radius, 
                 shape,
                 density_value=1.0,
                 temperature_value=0.0,
                 velocity_value=(0.0, 0.0),
                 approach='linear',
                 numba_optimized=False):
        """
        Initialize the emitter.

        Args:
            position: Tuple (x, y) indicating the emitter's position in grid coordinates.
            radius: Radius of the emitter's influence in grid cells.
            shape: Tuple (height, width) of the simulation grid.
            density_value: Scalar value to add to the density field.
            temperature_value: Scalar value to add to the temperature field.
            velocity_value: Tuple (vx, vy) representing the velocity vector to add to the velocity field.
            approach: String indicating the emission approach ('linear', 'gaussian', etc.).
            numba_optimized: Boolean indicating whether to use Numba-optimized methods for emission.
        """
        self.position = position
        self.radius = radius
        self.shape = shape
        self.approach = approach
        self.emission_approach = self._get_emission_approach(approach)
        self.density_value = density_value
        self.temperature_value = temperature_value
        self.velocity_value = velocity_value
        self.enabled = True
        self.numba_optimized = numba_optimized

    def _get_emission_approach(self, approach):
        """
        Get the emission approach class based on the provided approach string.

        Args:
            approach: String indicating the emission approach ('linear', 'gaussian', etc.).

        Returns:
            An instance of the corresponding EmitterApproach subclass.
        """
        if approach == 'linear':
            return LinearEmitterApproach(self)
        elif approach == 'gaussian':
            return GaussianEmitterApproach(self)
        elif approach == 'quadratic':
            return QuadraticEmitterApproach(self)
        elif approach == 'hard_cutoff':
            return HardCutoffEmitterApproach(self)
        else:
            raise ValueError(f"Unknown emission approach: {approach}")

    def _apply_scalar_field(self, field, value, dt):
        """
        Apply a scalar value to the field based on the emitter's position and radius.

        Args:
            field: 2D numpy array representing the scalar field (density or temperature).
            value: Scalar value to add to the field.
            dt: Time step for scaling the emission.
        """

        if self.numba_optimized:
            _apply_scalar_field_numba(field, value, dt, self.position[0], self.position[1], self.radius, self.approach)
            return

        height, width = self.shape
        for y in range(height):
            for x in range(width):
                dx = x - self.position[0]
                dy = y - self.position[1]
                distance = np.sqrt(dx**2 + dy**2)
                falloff = self.emission_approach.calculateFaloff(distance)
                field[y, x] += value * falloff * dt

    def _apply_vector_field(self, field, value, dt):
        """
        Apply a vector value to the field based on the emitter's position and radius.

        Args:
            field: 3D numpy array representing the vector field (height, width, 2).
            value: Tuple (vx, vy) representing the vector value to add to the field.
            dt: Time step for scaling the emission.
        """
        if self.numba_optimized:
            _apply_vector_field_numba(field, value[0], value[1], dt, self.position[0], self.position[1], self.radius, self.approach)

        height, width, _ = self.shape
        for y in range(height):
            for x in range(width):
                dx = x - self.position[0]
                dy = y - self.position[1]
                distance = np.sqrt(dx**2 + dy**2)
                falloff = self.emission_approach.calculateFaloff(distance)
                field[y, x, 0] += value[0] * falloff * dt
                field[y, x, 1] += value[1] * falloff * dt

    def emit(self, density_field, velocity_u_field, velocity_v_field, temperature_field, dt):
        """
        Emit smoke into the simulation fields.

        Args:
            density_field: 2D numpy array representing the density field.
            velocity_u_field: 2D numpy array representing the x-component of the velocity field.
            velocity_v_field: 2D numpy array representing the y-component of the velocity field.
            temperature_field: 2D numpy array representing the temperature field.
            dt: Time step for scaling the emission.
        """
        if self.enabled:
            if density_field is not None and self.density_value != 0.0:
                self._apply_scalar_field(density_field, self.density_value, dt)
            if temperature_field is not None and self.temperature_value != 0.0:
                self._apply_scalar_field(temperature_field, self.temperature_value, dt)
            if velocity_u_field is not None and self.velocity_value[0] != 0.0:
                self._apply_scalar_field(velocity_u_field, self.velocity_value[0], dt)
            if velocity_v_field is not None and self.velocity_value[1] != 0.0:
                self._apply_scalar_field(velocity_v_field, self.velocity_value[1], dt)

    def set_enabled(self, enabled):
        """
        Enable or disable the emitter.

        Args:
            enabled: Boolean indicating whether the emitter should be enabled.
        """
        self.enabled = enabled

    def status(self):
        """
        Get the current status of the emitter.

        Returns:
            Dictionary containing the emitter's position, radius, and enabled state.
        """
        return {
            'position': self.position,
            'radius': self.radius,
            'shape': self.shape,
            'emission_approach': self.emission_approach,
            'density_value': self.density_value,
            'temperature_value': self.temperature_value,
            'velocity_value': self.velocity_value,
            'enabled': self.enabled
        }

    def set_position(self, position):
        """
        Set the position of the emitter.

        Args:
            position: Tuple (x, y) indicating the new position in grid coordinates.
        """
        self.position = position

    def set_radius(self, radius):
        """
        Set the radius of the emitter.

        Args:
            radius: New radius of the emitter's influence in grid cells.
        """
        self.radius = radius

    def set_density_value(self, density_value):
        """
        Set the density value to be emitted.

        Args:
            density_value: Scalar value to add to the density field.
        """
        self.density_value = density_value

    def set_temperature_value(self, temperature_value):
        """
        Set the temperature value to be emitted.

        Args:
            temperature_value: Scalar value to add to the temperature field.
        """
        self.temperature_value = temperature_value

    def set_velocity_value(self, velocity_value):
        """
        Set the velocity value to be emitted.

        Args:
            velocity_value: Tuple (vx, vy) representing the velocity vector to add to the velocity field.
        """
        self.velocity_value = velocity_value


            


class EmitterApproach:
    """
    Base class for different emitter approaches (e.g., Linear, Gaussian).
    """
    def __init__(self, emitter):
        self.emitter = emitter

    def calculateFaloff(self, distance):
        """
        Calculate the falloff based on distance from the emitter's center.

        Args:
            distance: Distance from the emitter's center.

        Returns:
            Falloff value (0 to 1).
        """
        raise NotImplementedError("calculateFaloff method must be implemented by subclasses.")


class HardCutoffEmitterApproach(EmitterApproach):
    """
    Hard cutoff emitter approach.
    """
    def calculateFaloff(self, distance):
        """
        Calculate hard cutoff falloff based on distance.

        Args:
            distance: Distance from the emitter's center.

        Returns:
            Falloff value (0 or 1).
        """
        return 1.0 if distance < self.emitter.radius else 0.0


class LinearEmitterApproach(EmitterApproach):
    """
    Linear falloff emitter approach.
    """
    def calculateFaloff(self, distance):
        """
        Calculate linear falloff based on distance.

        Args:
            distance: Distance from the emitter's center.

        Returns:
            Falloff value (0 to 1).
        """
        if distance >= self.emitter.radius:
            return 0.0
        return 1.0 - (distance / self.emitter.radius)


class QuadraticEmitterApproach(EmitterApproach):
    """
    Quadratic falloff emitter approach.
    """
    def calculateFaloff(self, distance):
        """
        Calculate quadratic falloff based on distance.

        Args:
            distance: Distance from the emitter's center.

        Returns:
            Falloff value (0 to 1).
        """
        if distance >= self.emitter.radius:
            return 0.0
        return 1.0 - (distance / self.emitter.radius) ** 2


class GaussianEmitterApproach(EmitterApproach):
    """
    Gaussian falloff emitter approach.
    """
    def calculateFaloff(self, distance):
        """
        Calculate Gaussian falloff based on distance.

        Args:
            distance: Distance from the emitter's center.

        Returns:
            Falloff value (0 to 1).
        """
        if distance >= self.emitter.radius:
            return 0.0
        sigma = self.emitter.radius / 3.0  # Standard deviation
        return np.exp(-0.5 * (distance / sigma) ** 2)


















