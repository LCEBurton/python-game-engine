"""
3D Vector class implementation.
"""
import math
from typing import List, Tuple


class Vector3:
    """3D vector with x, y, z components.
    
    Supports standard vector operations including addition, subtraction,
    scalar multiplication, dot product, cross product, and normalization.
    """
    
    __slots__ = ('x', 'y', 'z')
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialize a 3D vector.
        
        Args:
            x: X component (default: 0.0)
            y: Y component (default: 0.0)
            z: Z component (default: 0.0)
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __add__(self, other: 'Vector3') -> 'Vector3':
        """Add two vectors component-wise."""
        if not isinstance(other, Vector3):
            raise TypeError(f"Cannot add Vector3 and {type(other).__name__}")
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        """Subtract two vectors component-wise."""
        if not isinstance(other, Vector3):
            raise TypeError(f"Cannot subtract {type(other).__name__} from Vector3")
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3':
        """Multiply vector by a scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Cannot multiply Vector3 by {type(scalar).__name__}")
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector3':
        """Right multiply (scalar * vector)."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3':
        """Divide vector by a scalar."""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Cannot divide Vector3 by {type(scalar).__name__}")
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self) -> 'Vector3':
        """Negate the vector."""
        return Vector3(-self.x, -self.y, -self.z)
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another vector."""
        if not isinstance(other, Vector3):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __str__(self) -> str:
        """String representation of the vector."""
        return f"Vector3({self.x}, {self.y}, {self.z})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"
    
    def dot(self, other: 'Vector3') -> float:
        """Calculate dot product with another vector.
        
        Args:
            other: Another Vector3
            
        Returns:
            Dot product (scalar)
        """
        if not isinstance(other, Vector3):
            raise TypeError(f"Cannot compute dot product with {type(other).__name__}")
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3') -> 'Vector3':
        """Calculate cross product with another vector.
        
        Args:
            other: Another Vector3
            
        Returns:
            Cross product (Vector3 perpendicular to both)
        """
        if not isinstance(other, Vector3):
            raise TypeError(f"Cannot compute cross product with {type(other).__name__}")
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self) -> float:
        """Calculate the length/magnitude of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def magnitude_squared(self) -> float:
        """Calculate squared magnitude (faster, avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z
    
    def normalized(self) -> 'Vector3':
        """Return a normalized copy of this vector (unit length).
        
        Returns:
            Normalized vector
            
        Raises:
            ValueError: If vector is zero-length
        """
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
    
    def to_list(self) -> List[float]:
        """Convert to list [x, y, z]."""
        return [self.x, self.y, self.z]
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple (x, y, z)."""
        return (self.x, self.y, self.z)
    
    def copy(self) -> 'Vector3':
        """Create a copy of this vector."""
        return Vector3(self.x, self.y, self.z)
