"""
Abstract base class for renderer implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

from engine.mesh import Mesh
from engine.rendering.shader import Shader


class Renderer(ABC):
    """
    Abstract renderer interface. Backend implementations (ModernGL, etc.)
    should inherit from this class.
    """
    
    @abstractmethod
    def clear(self, color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """
        Clear the screen with the specified color.
        
        Args:
            color: RGBA color tuple (values 0.0-1.0)
        """
        pass
    
    @abstractmethod
    def upload(self, mesh: 'Mesh') -> None:
        """
        Upload mesh data to GPU. Creates and caches vertex/index buffers.
        
        Args:
            mesh: The mesh to upload
        """
        pass
    
    @abstractmethod
    def draw(self, mesh: 'Mesh', shader: 'Shader', uniforms: Dict[str, Any]) -> None:
        """
        Draw a mesh using the specified shader and uniforms.
        
        Args:
            mesh: The mesh to draw (must be uploaded first)
            shader: The shader program to use
            uniforms: Dictionary of uniform name -> value
        """
        pass
    
    @abstractmethod
    def create_shader(self, vertex_path: str, fragment_path: str) -> 'Shader':
        """
        Create a shader program from vertex and fragment shader files.
        
        Args:
            vertex_path: Path to vertex shader file
            fragment_path: Path to fragment shader file
            
        Returns:
            Shader object
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up GPU resources. Should be called before shutdown.
        """
        pass
