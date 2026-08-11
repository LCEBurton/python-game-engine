"""
Quaternion class for smooth 3D rotations.
"""
import math

from .vector3 import Vector3
from .matrix4 import Matrix4


class Quaternion:
    """Quaternion for representing 3D rotations.
    
    Represented as q = w + xi + yj + zk where w is the scalar part
    and (x, y, z) is the vector part.
    """
    
    __slots__ = ('w', 'x', 'y', 'z')
    
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        """Initialize a quaternion.
        
        Args:
            w: Scalar (real) component (default: 1.0 for identity)
            x: X component of vector part
            y: Y component of vector part
            z: Z component of vector part
        """
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def __str__(self):
        return f"({self.w:.3f} + {self.x:.3f}i + {self.y:.3f}j + {self.z:.3f}k)"
    
    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return False
        return (self.w == other.w and self.x == other.x and 
                self.y == other.y and self.z == other.z)
    
    def __mul__(self, other):
        """Multiply two quaternions (Hamilton product) or scale by scalar."""
        if isinstance(other, Quaternion):
            # Quaternion multiplication
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return Quaternion(self.w * other, self.x * other, 
                            self.y * other, self.z * other)
        return NotImplemented
    
    def __rmul__(self, scalar):
        """Right multiplication by scalar."""
        if isinstance(scalar, (int, float)):
            return self.__mul__(scalar)
        return NotImplemented
    
    def __add__(self, other):
        """Add two quaternions."""
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x,
                            self.y + other.y, self.z + other.z)
        return NotImplemented
    
    def __sub__(self, other):
        """Subtract two quaternions."""
        if isinstance(other, Quaternion):
            return Quaternion(self.w - other.w, self.x - other.x,
                            self.y - other.y, self.z - other.z)
        return NotImplemented
    
    def copy(self):
        """Return a copy of this quaternion."""
        return Quaternion(self.w, self.x, self.y, self.z)
    
    def conjugate(self):
        """Return the conjugate of this quaternion."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def length_squared(self):
        """Return the squared length of the quaternion."""
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    
    def length(self):
        """Return the length (magnitude) of the quaternion."""
        return math.sqrt(self.length_squared())
    
    def normalized(self):
        """Return a normalized copy of this quaternion."""
        length = self.length()
        if length == 0:
            return Quaternion.identity()
        return Quaternion(self.w / length, self.x / length, 
                         self.y / length, self.z / length)
    
    def normalize(self):
        """Normalize this quaternion in place."""
        length = self.length()
        if length > 0:
            self.w /= length
            self.x /= length
            self.y /= length
            self.z /= length
        return self
    
    def inverse(self):
        """Return the inverse of this quaternion."""
        conj = self.conjugate()
        length_sq = self.length_squared()
        if length_sq == 0:
            return Quaternion.identity()
        return Quaternion(conj.w / length_sq, conj.x / length_sq,
                         conj.y / length_sq, conj.z / length_sq)
    
    def dot(self, other):
        """Calculate dot product with another quaternion."""
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    
    def rotate_vector(self, v):
        """Rotate a Vector3 by this quaternion.
        
        Args:
            v: Vector3 to rotate
        
        Returns:
            Rotated Vector3
        """
        # Convert vector to quaternion: q_v = (0, v)
        q_v = Quaternion(0, v.x, v.y, v.z)
        
        # Perform rotation: q * q_v * q^(-1)
        result = self * q_v * self.conjugate()
        
        return Vector3(result.x, result.y, result.z)
    
    def to_matrix4(self):
        """Convert quaternion to a 4x4 rotation matrix.
        
        Returns:
            Matrix4 representing the same rotation
        """
        # Normalize first
        q = self.normalized()
        
        w, x, y, z = q.w, q.x, q.y, q.z
        
        # Calculate matrix elements
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        
        yy = y * y
        yz = y * z
        yw = y * w
        
        zz = z * z
        zw = z * w
        
        matrix_data = [
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw), 0],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw), 0],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy), 0],
            [0, 0, 0, 1]
        ]
        
        return Matrix4(matrix_data)
    
    @staticmethod
    def from_axis_angle(axis, angle):
        """Create a quaternion from an axis and angle.
        
        Args:
            axis: Vector3 representing rotation axis (will be normalized)
            angle: Rotation angle in radians
        
        Returns:
            Quaternion representing the rotation
        """
        axis_normalized = axis.normalized()
        half_angle = angle / 2.0
        sin_half = math.sin(half_angle)
        
        return Quaternion(
            math.cos(half_angle),
            axis_normalized.x * sin_half,
            axis_normalized.y * sin_half,
            axis_normalized.z * sin_half
        )
    
    @staticmethod
    def from_euler(pitch, yaw, roll):
        """Create a quaternion from Euler angles (in radians).
        
        Args:
            pitch: Rotation around X axis (radians)
            yaw: Rotation around Y axis (radians)
            roll: Rotation around Z axis (radians)
        
        Returns:
            Quaternion representing the rotation
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return Quaternion(w, x, y, z)
    
    @staticmethod
    def from_matrix4(matrix):
        """Create a quaternion from a rotation matrix.
        
        Args:
            matrix: Matrix4 containing rotation
        
        Returns:
            Quaternion representing the rotation
        """
        m = matrix.data
        trace = m[0][0] + m[1][1] + m[2][2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2][1] - m[1][2]) * s
            y = (m[0][2] - m[2][0]) * s
            z = (m[1][0] - m[0][1]) * s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s
        
        return Quaternion(w, x, y, z)
    
    @staticmethod
    def slerp(q1, q2, t):
        """Spherical linear interpolation between two quaternions.
        
        Args:
            q1: Start quaternion
            q2: End quaternion
            t: Interpolation factor [0, 1]
        
        Returns:
            Interpolated quaternion
        """
        # Normalize inputs
        q1 = q1.normalized()
        q2 = q2.normalized()
        
        dot = q1.dot(q2)
        
        # If dot is negative, negate q2 to take shorter path
        if dot < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            dot = -dot
        
        # Clamp dot to avoid numerical issues
        dot = max(-1.0, min(1.0, dot))
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + (q2 - q1) * t
            return result.normalized()
        
        # Calculate angle between quaternions
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        
        w1 = math.sin((1 - t) * theta) / sin_theta
        w2 = math.sin(t * theta) / sin_theta
        
        return q1 * w1 + q2 * w2
    
    @staticmethod
    def identity():
        """Return the identity quaternion (no rotation)."""
        return Quaternion(1, 0, 0, 0)
