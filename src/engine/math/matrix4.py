"""
4x4 Matrix class for 3D transformations.
"""

from typing import List, Union, cast
import math

from .vector3 import Vector3
from .vector4 import Vector4
from .matrix3 import Matrix3

class Matrix4:
    """4x4 matrix for 3D transformations.
    
    Stored in row-major order as a flat list of 16 elements.
    Used for model, view, and projection transformations.
    """
    
    __slots__ = ('elements',)
    
    def __init__(self, elements: Union[List[List[float]], List[float], None] = None):
        """Initialize a 4x4 matrix.
        
        Args:
            elements: Can be:
                - None: Creates identity matrix
                - List of 16 floats: Row-major flat array
                - List of 4 lists of 4 floats: 2D array
                - List of 4 Vector4: Row vectors
        """
        if elements is None:
            # Identity matrix
            self.elements = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ]
        elif isinstance(elements, list):
            if len(elements) == 16 and all(isinstance(e, (int, float)) for e in elements):
                # Flat list of 16 elements
                self.elements = [float(e) for e in elements] # type: ignore
            elif len(elements) == 4:
                if all(isinstance(row, Vector4) for row in elements):
                    # List of 4 Vector4
                    self.elements = []
                    vec4_list = cast(List[Vector4], elements)
                    for vec in vec4_list:
                        self.elements.extend([vec.x, vec.y, vec.z, vec.w])
                elif all(isinstance(row, list) and len(row) == 4 for row in elements):
                    # 2D list (4x4)
                    self.elements = [float(elements[i][j]) for i in range(4) for j in range(4)] # type: ignore
                else:
                    raise ValueError("Matrix4 requires 4 rows of 4 elements or 4 Vector4 objects")
            else:
                raise ValueError("Matrix4 requires 16 elements as flat list or 4x4 nested list")
        else:
            raise TypeError(f"Cannot initialize Matrix4 from {type(elements).__name__}")
    
    def __mul__(self, other: Union['Matrix4', Vector4, Vector3, float]) -> Union['Matrix4', Vector4, Vector3]:
        """Multiply matrix by another matrix, vector, or scalar."""
        if isinstance(other, Matrix4):
            # Matrix-matrix multiplication
            result = [0.0] * 16
            for i in range(4):
                for j in range(4):
                    result[i * 4 + j] = sum(
                        self.elements[i * 4 + k] * other.elements[k * 4 + j]
                        for k in range(4)
                    )
            return Matrix4(result)
        elif isinstance(other, Vector4):
            # Matrix-vector multiplication (4D)
            return Vector4(
                self.elements[0] * other.x + self.elements[1] * other.y + self.elements[2] * other.z + self.elements[3] * other.w,
                self.elements[4] * other.x + self.elements[5] * other.y + self.elements[6] * other.z + self.elements[7] * other.w,
                self.elements[8] * other.x + self.elements[9] * other.y + self.elements[10] * other.z + self.elements[11] * other.w,
                self.elements[12] * other.x + self.elements[13] * other.y + self.elements[14] * other.z + self.elements[15] * other.w
            )
        elif isinstance(other, Vector3):
            # Matrix-vector multiplication (3D, assuming w=1 for position)
            vec4 = Vector4(other.x, other.y, other.z, 1.0)
            result = self * vec4
            result_vec4 = cast(Vector4, result)
            # Perspective divide if needed
            if result_vec4.w != 0 and result_vec4.w != 1.0:
                return Vector3(result_vec4.x / result_vec4.w, result_vec4.y / result_vec4.w, result_vec4.z / result_vec4.w)
            return result_vec4.to_vector3()
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return Matrix4([e * other for e in self.elements])
        else:
            raise TypeError(f"Cannot multiply Matrix4 by {type(other).__name__}")
    
    def __rmul__(self, scalar: float) -> 'Matrix4':
        """Right multiply (scalar * matrix)."""
        if isinstance(scalar, (int, float)):
            return Matrix4([e * scalar for e in self.elements])
        raise TypeError(f"Cannot multiply {type(scalar).__name__} by Matrix4")
    
    def __add__(self, other: 'Matrix4') -> 'Matrix4':
        """Add two matrices element-wise."""
        if not isinstance(other, Matrix4):
            raise TypeError(f"Cannot add Matrix4 and {type(other).__name__}")
        return Matrix4([a + b for a, b in zip(self.elements, other.elements)])
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another matrix."""
        if not isinstance(other, Matrix4):
            return False
        return self.elements == other.elements
    
    def __str__(self) -> str:
        """String representation of the matrix."""
        e = self.elements
        return (f"Matrix4(\n"
                f"  [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}, {e[3]:.3f}]\n"
                f"  [{e[4]:.3f}, {e[5]:.3f}, {e[6]:.3f}, {e[7]:.3f}]\n"
                f"  [{e[8]:.3f}, {e[9]:.3f}, {e[10]:.3f}, {e[11]:.3f}]\n"
                f"  [{e[12]:.3f}, {e[13]:.3f}, {e[14]:.3f}, {e[15]:.3f}]\n"
                f")")
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"Matrix4({self.elements})"
    
    def transpose(self) -> 'Matrix4':
        """Return the transpose of this matrix."""
        return Matrix4([
            self.elements[0], self.elements[4], self.elements[8], self.elements[12],
            self.elements[1], self.elements[5], self.elements[9], self.elements[13],
            self.elements[2], self.elements[6], self.elements[10], self.elements[14],
            self.elements[3], self.elements[7], self.elements[11], self.elements[15]
        ])
    
    def _get_submatrix3(self, skip_row: int, skip_col: int) -> Matrix3:
        """Extract a 3x3 submatrix by removing one row and column.
        
        Args:
            skip_row: Row index to skip (0-3)
            skip_col: Column index to skip (0-3)
            
        Returns:
            3x3 submatrix with specified row and column removed
        """
        e = self.elements
        result = []
        
        for i in range(4):
            if i == skip_row:
                continue
            for j in range(4):
                if j == skip_col:
                    continue
                result.append(e[i * 4 + j])
        
        return Matrix3(result)
    
    def determinant(self) -> float:
        """Calculate the determinant of this matrix using cofactor expansion."""
        e = self.elements
        
        # Cofactor expansion along first row
        # det(M) = m00*C00 - m01*C01 + m02*C02 - m03*C03
        c00 = self._get_submatrix3(0, 0).determinant()
        c01 = self._get_submatrix3(0, 1).determinant()
        c02 = self._get_submatrix3(0, 2).determinant()
        c03 = self._get_submatrix3(0, 3).determinant()
        
        return e[0] * c00 - e[1] * c01 + e[2] * c02 - e[3] * c03
    
    def inverse(self) -> 'Matrix4':
        """Calculate the inverse of this matrix.
        
        Returns:
            Inverse matrix
            
        Raises:
            ValueError: If matrix is singular (determinant is zero)
        """
        e = self.elements
        
        # Calculate all 16 cofactors using 3x3 submatrix determinants
        # Cofactor Cij = (-1)^(i+j) * det(submatrix_ij)
        cofactors = []
        for i in range(4):
            for j in range(4):
                submat_det = self._get_submatrix3(i, j).determinant()
                # Apply checkerboard sign pattern
                sign = 1 if (i + j) % 2 == 0 else -1
                cofactors.append(sign * submat_det)
        
        # Calculate determinant using first row cofactors
        det = e[0] * cofactors[0] + e[1] * cofactors[1] + e[2] * cofactors[2] + e[3] * cofactors[3]
        
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted (determinant is zero)")
        
        inv_det = 1.0 / det
        
        # Adjugate is transpose of cofactor matrix, then divide by determinant
        # Transpose by accessing cofactors in column-major order
        return Matrix4([
            cofactors[0] * inv_det, cofactors[4] * inv_det, cofactors[8] * inv_det, cofactors[12] * inv_det,
            cofactors[1] * inv_det, cofactors[5] * inv_det, cofactors[9] * inv_det, cofactors[13] * inv_det,
            cofactors[2] * inv_det, cofactors[6] * inv_det, cofactors[10] * inv_det, cofactors[14] * inv_det,
            cofactors[3] * inv_det, cofactors[7] * inv_det, cofactors[11] * inv_det, cofactors[15] * inv_det
        ])
    
    def to_list(self) -> List[float]:
        """Convert to flat list."""
        return self.elements.copy()
    
    def to_2d_list(self) -> List[List[float]]:
        """Convert to 2D list (4x4)."""
        return [
            self.elements[0:4],
            self.elements[4:8],
            self.elements[8:12],
            self.elements[12:16]
        ]
    
    def copy(self) -> 'Matrix4':
        """Create a copy of this matrix."""
        return Matrix4(self.elements.copy())
    
    @staticmethod
    def identity() -> 'Matrix4':
        """Create an identity matrix."""
        return Matrix4()
    
    @staticmethod
    def translation(x: float, y: float, z: float) -> 'Matrix4':
        """Create a translation matrix.
        
        Args:
            x, y, z: Translation amounts
            
        Returns:
            Translation matrix
        """
        return Matrix4([
            1.0, 0.0, 0.0, x,
            0.0, 1.0, 0.0, y,
            0.0, 0.0, 1.0, z,
            0.0, 0.0, 0.0, 1.0
        ])
    
    @staticmethod
    def scale(x: float, y: float, z: float) -> 'Matrix4':
        """Create a scale matrix.
        
        Args:
            x, y, z: Scale factors
            
        Returns:
            Scale matrix
        """
        return Matrix4([
            x, 0.0, 0.0, 0.0,
            0.0, y, 0.0, 0.0,
            0.0, 0.0, z, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])

    @staticmethod
    def rotation_x(angle: float) -> 'Matrix4':
        """Create a rotation matrix around the X axis.
        
        Args:
            angle: Rotation angle in radians
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Matrix4([
            1.0, 0.0, 0.0, 0.0,
            0.0, c, -s, 0.0,
            0.0, s, c, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])
    
    @staticmethod
    def rotation_y(angle: float) -> 'Matrix4':
        """Create a rotation matrix around the Y axis.
        
        Args:
            angle: Rotation angle in radians
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Matrix4([
            c, 0.0, s, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -s, 0.0, c, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])
    
    @staticmethod
    def rotation_z(angle: float) -> 'Matrix4':
        """Create a rotation matrix around the Z axis.
        
        Args:
            angle: Rotation angle in radians
        """
        c = math.cos(angle)
        s = math.sin(angle)
        return Matrix4([
            c, -s, 0.0, 0.0,
            s, c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])
    
    @staticmethod
    def rotation(axis: Vector3, angle: float) -> 'Matrix4':
        """Create a rotation matrix around an arbitrary axis (Rodrigues' rotation formula).
        
        Args:
            axis: Rotation axis (will be normalized)
            angle: Rotation angle in radians
        """
        axis = axis.normalized()
        c = math.cos(angle)
        s = math.sin(angle)
        t = 1.0 - c
        
        x, y, z = axis.x, axis.y, axis.z
        
        return Matrix4([
            t*x*x + c,    t*x*y - s*z,  t*x*z + s*y,  0.0,
            t*x*y + s*z,  t*y*y + c,    t*y*z - s*x,  0.0,
            t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,    0.0,
            0.0,          0.0,          0.0,          1.0
        ])
    
    @staticmethod
    def look_at(eye: Vector3, target: Vector3, up: Vector3) -> 'Matrix4':
        """Create a view matrix using the look-at method.
        
        Args:
            eye: Camera position
            target: Point to look at
            up: Up direction (usually (0, 1, 0))
            
        Returns:
            View matrix
        """
        if not all(isinstance(v, Vector3) for v in [eye, target, up]):
            raise TypeError("eye, target, and up must be Vector3")
        
        # Calculate forward (from eye to target)
        forward = (target - eye).normalized()
        
        # Calculate right (perpendicular to forward and up)
        right = forward.cross(up).normalized()
        
        # Recalculate up (perpendicular to right and forward)
        new_up = right.cross(forward)
        
        # Create view matrix
        return Matrix4([
            right.x, right.y, right.z, -right.dot(eye),
            new_up.x, new_up.y, new_up.z, -new_up.dot(eye),
            -forward.x, -forward.y, -forward.z, forward.dot(eye),
            0.0, 0.0, 0.0, 1.0
        ])
    
    @staticmethod
    def perspective(fov: float, aspect: float, near: float, far: float) -> 'Matrix4':
        """Create a perspective projection matrix.
        
        Args:
            fov: Field of view in degrees
            aspect: Aspect ratio (width/height)
            near: Near clipping plane distance
            far: Far clipping plane distance
            
        Returns:
            Perspective projection matrix
        """
        if near <= 0 or far <= near:
            raise ValueError("Invalid near/far plane values: near must be > 0 and far > near")
        if aspect <= 0:
            raise ValueError("Aspect ratio must be positive")
        if fov <= 0 or fov >= 180:
            raise ValueError("Field of view must be between 0 and 180 degrees")
        
        tan_half_fov = math.tan(math.radians(fov) / 2.0)
        
        return Matrix4([
            1.0 / (aspect * tan_half_fov), 0.0, 0.0, 0.0,
            0.0, 1.0 / tan_half_fov, 0.0, 0.0,
            0.0, 0.0, -(far + near) / (far - near), -(2.0 * far * near) / (far - near),
            0.0, 0.0, -1.0, 0.0
        ])
    
    @staticmethod
    def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> 'Matrix4':
        """Create an orthographic projection matrix.
        
        Args:
            left, right: Left and right clipping planes
            bottom, top: Bottom and top clipping planes
            near, far: Near and far clipping planes
            
        Returns:
            Orthographic projection matrix
        """
        if right == left or top == bottom or far == near:
            raise ValueError("Invalid orthographic projection bounds")
        
        return Matrix4([
            2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left),
            0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom),
            0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near),
            0.0, 0.0, 0.0, 1.0
        ])

