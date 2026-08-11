"""
Vector2 class for 2D operations (UI, texture coordinates, etc.)
"""
import math


class Vector2:
    """2D vector with x and y components."""
    
    __slots__ = ('x', 'y')
    
    def __init__(self, x=0.0, y=0.0):
        """Initialize a 2D vector.
        
        Args:
            x: X component (default: 0.0)
            y: Y component (default: 0.0)
        """
        self.x = float(x)
        self.y = float(y)
    
    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"
    
    def __str__(self):
        return f"({self.x:.3f}, {self.y:.3f})"
    
    def __eq__(self, other):
        if not isinstance(other, Vector2):
            return False
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        """Add two vectors."""
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    def __sub__(self, other):
        """Subtract two vectors."""
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __mul__(self, scalar):
        """Multiply vector by scalar."""
        if isinstance(scalar, (int, float)):
            return Vector2(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    def __rmul__(self, scalar):
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """Divide vector by scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError("Cannot divide vector by zero")
            return Vector2(self.x / scalar, self.y / scalar)
        return NotImplemented
    
    def __neg__(self):
        """Negate vector."""
        return Vector2(-self.x, -self.y)
        
    def copy(self):
        """Return a copy of this vector."""
        return Vector2(self.x, self.y)
    
    def dot(self, other):
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y
    
    def length_squared(self):
        """Return the squared length of the vector."""
        return self.x * self.x + self.y * self.y
    
    def length(self):
        """Return the length (magnitude) of the vector."""
        return math.sqrt(self.length_squared())
    
    def normalized(self):
        """Return a normalized copy of this vector."""
        length = self.length()
        if length == 0:
            return Vector2(0, 0)
        return self / length
    
    def normalize(self):
        """Normalize this vector in place."""
        length = self.length()
        if length > 0:
            self.x /= length
            self.y /= length
        return self
    
    def distance_to(self, other):
        """Calculate distance to another vector."""
        return (self - other).length()
    
    def distance_squared_to(self, other):
        """Calculate squared distance to another vector."""
        return (self - other).length_squared()
    
    def lerp(self, other, t):
        """Linear interpolation between this vector and another.
        
        Args:
            other: Target vector
            t: Interpolation factor [0, 1]
        
        Returns:
            Interpolated vector
        """
        return Vector2(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t
        )
    
    def angle_to(self, other):
        """Calculate angle in radians to another vector."""
        dot_product = self.dot(other)
        lengths = self.length() * other.length()
        if lengths == 0:
            return 0.0
        cos_angle = dot_product / lengths
        # Clamp to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)
    
    @staticmethod
    def zero():
        """Return a zero vector."""
        return Vector2(0, 0)
    
    @staticmethod
    def one():
        """Return a vector with all components set to 1."""
        return Vector2(1, 1)
