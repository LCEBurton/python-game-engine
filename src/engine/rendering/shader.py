"""
Abstract base class for shader programs.
"""
from abc import ABC, abstractmethod
from typing import Any


class Shader(ABC):
    """
    Abstract shader program interface.
    """
    
    @abstractmethod
    def use(self) -> None:
        """
        Bind this shader program for rendering.
        """
        pass
    
    @abstractmethod
    def set_uniform(self, name: str, value: Any) -> None:
        """
        Set a uniform variable in the shader.
        
        Args:
            name: Uniform variable name
            value: Value to set (int, float, Matrix4, etc.)
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release shader resources.
        """
        pass
