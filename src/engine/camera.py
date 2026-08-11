"""
Camera class for 3D view and projection transforms.
"""
from engine.math.matrix4 import Matrix4
from engine.math.vector3 import Vector3
from typing import Tuple


class Camera:
    """
    Camera with perspective projection and view transform.
    """
    
    def __init__(
        self,
        fov: float = 60.0,
        aspect: float = 16.0/9.0,
        near: float = 0.1,
        far: float = 100.0,
        position: Tuple[float, float, float] = (0.0, 0.0, 5.0),
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    ):
        """
        Create a camera.
        
        Args:
            fov: Field of view in degrees
            aspect: Aspect ratio (width/height)
            near: Near clipping plane
            far: Far clipping plane
            position: Camera position (x, y, z)
            target: Point camera is looking at (x, y, z)
            up: Up vector (x, y, z)
        """
        # TODO: Store projection parameters
        # TODO: Store view parameters (position, target, up as Vector3)
        # TODO: Calculate projection matrix using Matrix4.perspective()
        # TODO: Calculate view matrix using Matrix4.look_at()
        self._fov = fov
        self._aspect = aspect
        self._near = near
        self._far = far
        self._position = Vector3(*position)
        self._target = Vector3(*target)
        self._up = Vector3(*up)

        self._projection_matrix = Matrix4.perspective(fov, aspect, near, far)
        self._view_matrix = Matrix4.look_at(self._position, self._target, self._up)
        self._view_projection_matrix = self._projection_matrix * self._view_matrix
    
    @property
    def projection(self) -> Matrix4:
        """Get the projection matrix."""
        return self._projection_matrix
    
    @property
    def view(self) -> Matrix4:
        """Get the view matrix."""
        return self._view_matrix
    
    def set_position(self, position: Tuple[float, float, float]) -> None:
        """
        Update camera position and recalculate view matrix.
        
        Args:
            position: New camera position (x, y, z)
        """
        self._position = Vector3(*position)
        self._view_matrix = Matrix4.look_at(self._position, self._target, self._up)
        self._view_projection_matrix = self._projection_matrix * self._view_matrix
    
    def set_target(self, target: Tuple[float, float, float]) -> None:
        """
        Update camera target and recalculate view matrix.
        
        Args:
            target: New target position (x, y, z)
        """
        self._target = Vector3(*target)
        self._view_matrix = Matrix4.look_at(self._position, self._target, self._up)
        self._view_projection_matrix = self._projection_matrix * self._view_matrix
    
    def set_aspect_ratio(self, aspect: float) -> None:
        """
        Update aspect ratio and recalculate projection matrix.
        
        Args:
            aspect: New aspect ratio (width/height)
        """
        self._aspect = aspect
        self._projection_matrix = Matrix4.perspective(self._fov, self._aspect, self._near, self._far)
        self._view_projection_matrix = self._projection_matrix * self._view_matrix

