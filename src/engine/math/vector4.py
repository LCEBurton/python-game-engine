"""
4D Vector class implementation.
"""

import math
from typing import List, Tuple

from .vector3 import Vector3


class Vector4:
    """4D vector with x, y, z, w components.
    
    Used for homogeneous coordinates in 3D graphics transformations.
    """
    
    __slots__ = ('x', 'y', 'z', 'w')
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0):
        """Initialize a 4D vector.
        
        Args:
            x: X component (default: 0.0)
            y: Y component (default: 0.0)
            z: Z component (default: 0.0)
            w: W component (default: 1.0)
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
    
    def __add__(self, other: 'Vector4') -> 'Vector4':
        """Add two vectors component-wise."""
        if not isinstance(other, Vector4):
            raise TypeError(f"Cannot add Vector4 and {type(other).__name__}")
        return Vector4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
    
    def __sub__(self, other: 'Vector4') -> 'Vector4':
        """Subtract two vectors component-wise."""
        if not isinstance(other, Vector4):
            raise TypeError(f"Cannot subtract {type(other).__name__} from Vector4")
        return Vector4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
    
    def __mul__(self, scalar: float) -> 'Vector4':
        """Multiply vector by a scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Cannot multiply Vector4 by {type(scalar).__name__}")
        return Vector4(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector4':
        """Right multiply (scalar * vector)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector4':
        """Divide vector by a scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Cannot divide Vector4 by {type(scalar).__name__}")
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Vector4(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)
    
    def __neg__(self) -> 'Vector4':
        """Negate the vector."""
        return Vector4(-self.x, -self.y, -self.z, -self.w)
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another vector."""
        if not isinstance(other, Vector4):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w
    
    def __str__(self) -> str:
        """String representation of the vector."""
        return f"Vector4({self.x}, {self.y}, {self.z}, {self.w})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Vector4(x={self.x}, y={self.y}, z={self.z}, w={self.w})"
    
    def dot(self, other: 'Vector4') -> float:
        """Calculate dot product with another vector."""
        if not isinstance(other, Vector4):
            raise TypeError(f"Cannot compute dot product with {type(other).__name__}")
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    
    def magnitude(self) -> float:
        """Calculate the length/magnitude of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)
    
    def magnitude_squared(self) -> float:
        """Calculate squared magnitude (faster, avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    
    def normalized(self) -> 'Vector4':
        """Return a normalized copy of this vector."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero-length vector")
        return self / mag
    
    def normalize(self) -> None:
        """Normalize this vector in-place."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero-length vector")
        self.x /= mag
        self.y /= mag
        self.z /= mag
        self.w /= mag
    
    def to_vector3(self) -> Vector3:
        """Convert to Vector3 by dropping w component."""
        return Vector3(self.x, self.y, self.z)
    
    def to_list(self) -> List[float]:
        """Convert to list [x, y, z, w]."""
        return [self.x, self.y, self.z, self.w]
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple (x, y, z, w)."""
        return (self.x, self.y, self.z, self.w)
    
    def copy(self) -> 'Vector4':
        """Create a copy of this vector."""
        return Vector4(self.x, self.y, self.z, self.w)
