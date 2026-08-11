"""
3x3 Matrix class for 2D transformations and rotations.
"""

from typing import List, Union
import math

from .vector3 import Vector3


class Matrix3:
    """3x3 matrix for 2D transformations and rotations.
    
    Stored in row-major order as a flat list of 9 elements.
    """
    
    __slots__ = ('elements',)
    
    def __init__(self, elements: Union[List[List[float]], List[float], None] = None):
        """Initialize a 3x3 matrix.
        
        Args:
            elements: Can be:
                - None: Creates identity matrix
                - List of 9 floats: Row-major flat array
                - List of 3 lists of 3 floats: 2D array
        """
        if elements is None:
            # Identity matrix
            self.elements = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
        elif isinstance(elements, list):
            if len(elements) == 9 and all(isinstance(e, (int, float)) for e in elements):
                # Flat list of 9 elements
                self.elements = [float(e) for e in elements] # type: ignore
            elif len(elements) == 3 and all(isinstance(row, list) and len(row) == 3 for row in elements):
                # 2D list (3x3)
                self.elements = [float(elements[i][j]) for i in range(3) for j in range(3)] # type: ignore
            else:
                raise ValueError("Matrix3 requires 9 elements as flat list or 3x3 nested list")
        else:
            raise TypeError(f"Cannot initialize Matrix3 from {type(elements).__name__}")
    
    def __mul__(self, other: Union['Matrix3', Vector3, float]) -> Union['Matrix3', Vector3]:
        """Multiply matrix by another matrix, vector, or scalar."""
        if isinstance(other, Matrix3):
            # Matrix-matrix multiplication
            result = [0.0] * 9
            for i in range(3):
                for j in range(3):
                    result[i * 3 + j] = sum(
                        self.elements[i * 3 + k] * other.elements[k * 3 + j]
                        for k in range(3)
                    )
            return Matrix3(result)
        elif isinstance(other, Vector3):
            # Matrix-vector multiplication
            return Vector3(
                self.elements[0] * other.x + self.elements[1] * other.y + self.elements[2] * other.z,
                self.elements[3] * other.x + self.elements[4] * other.y + self.elements[5] * other.z,
                self.elements[6] * other.x + self.elements[7] * other.y + self.elements[8] * other.z
            )
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return Matrix3([e * other for e in self.elements])
        else:
            raise TypeError(f"Cannot multiply Matrix3 by {type(other).__name__}")
    
    def __rmul__(self, scalar: float) -> 'Matrix3':
        """Right multiply (scalar * matrix)."""
        if isinstance(scalar, (int, float)):
            return Matrix3([e * scalar for e in self.elements])
        raise TypeError(f"Cannot multiply {type(scalar).__name__} by Matrix3")
    
    def __add__(self, other: 'Matrix3') -> 'Matrix3':
        """Add two matrices element-wise."""
        if not isinstance(other, Matrix3):
            raise TypeError(f"Cannot add Matrix3 and {type(other).__name__}")
        return Matrix3([a + b for a, b in zip(self.elements, other.elements)])
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another matrix."""
        if not isinstance(other, Matrix3):
            return False
        return self.elements == other.elements
    
    def __str__(self) -> str:
        """String representation of the matrix."""
        return (f"Matrix3(\n"
                f"  [{self.elements[0]:.3f}, {self.elements[1]:.3f}, {self.elements[2]:.3f}]\n"
                f"  [{self.elements[3]:.3f}, {self.elements[4]:.3f}, {self.elements[5]:.3f}]\n"
                f"  [{self.elements[6]:.3f}, {self.elements[7]:.3f}, {self.elements[8]:.3f}]\n"
                f")")
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Matrix3({self.elements})"
    
    def transpose(self) -> 'Matrix3':
        """Return the transpose of this matrix."""
        return Matrix3([
            self.elements[0], self.elements[3], self.elements[6],
            self.elements[1], self.elements[4], self.elements[7],
            self.elements[2], self.elements[5], self.elements[8]
        ])
    
    def determinant(self) -> float:
        """Calculate the determinant of this matrix."""
        e = self.elements
        return (e[0] * (e[4] * e[8] - e[5] * e[7]) -
                e[1] * (e[3] * e[8] - e[5] * e[6]) +
                e[2] * (e[3] * e[7] - e[4] * e[6]))
    
    def inverse(self) -> 'Matrix3':
        """Calculate the inverse of this matrix.
        
        Returns:
            Inverse matrix
            
        Raises:
            ValueError: If matrix is singular (determinant is zero)
        """
        e = self.elements
        
        # Calculate cofactors
        c00 = e[4] * e[8] - e[5] * e[7]
        c01 = -(e[3] * e[8] - e[5] * e[6])
        c02 = e[3] * e[7] - e[4] * e[6]
        
        c10 = -(e[1] * e[8] - e[2] * e[7])
        c11 = e[0] * e[8] - e[2] * e[6]
        c12 = -(e[0] * e[7] - e[1] * e[6])
        
        c20 = e[1] * e[5] - e[2] * e[4]
        c21 = -(e[0] * e[5] - e[2] * e[3])
        c22 = e[0] * e[4] - e[1] * e[3]
        
        # Calculate determinant
        det = e[0] * c00 + e[1] * c01 + e[2] * c02
        
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted (determinant is zero)")
        
        # Adjugate matrix (transpose of cofactor matrix) divided by determinant
        inv_det = 1.0 / det
        
        return Matrix3([
            c00 * inv_det, c10 * inv_det, c20 * inv_det,
            c01 * inv_det, c11 * inv_det, c21 * inv_det,
            c02 * inv_det, c12 * inv_det, c22 * inv_det
        ])

    @staticmethod
    def rotation_x(angle: float) -> 'Matrix3':
        """Create a rotation matrix around the X axis.
        
        Args:
            angle: Rotation angle in radians
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Matrix3([
            1.0, 0.0, 0.0,
            0.0, c, -s,
            0.0, s, c
        ])
    
    @staticmethod
    def rotation_y(angle: float) -> 'Matrix3':
        """Create a rotation matrix around the Y axis.
        
        Args:
            angle: Rotation angle in radians
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Matrix3([
            c, 0.0, s,
            0.0, 1.0, 0.0,
            -s, 0.0, c
        ])
    
    @staticmethod
    def rotation_z(angle: float) -> 'Matrix3':
        """Create a rotation matrix around the Z axis.
        
        Args:
            angle: Rotation angle in radians
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Matrix3([
            c, -s, 0.0,
            s, c, 0.0,
            0.0, 0.0, 1.0
        ])

    def to_list(self) -> List[float]:
        """Convert to flat list."""
        return self.elements.copy()
    
    def to_2d_list(self) -> List[List[float]]:
        """Convert to 2D list (3x3)."""
        return [
            self.elements[0:3],
            self.elements[3:6],
            self.elements[6:9]
        ]
    
    def copy(self) -> 'Matrix3':
        """Create a copy of this matrix."""
        return Matrix3(self.elements.copy())
    
    @staticmethod
    def identity() -> 'Matrix3':
        """Create an identity matrix."""
        return Matrix3()
