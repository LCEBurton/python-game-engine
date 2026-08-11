"""
ModernGL implementation of Shader.
"""
import moderngl
from pathlib import Path
from typing import Any
from ..shader import Shader

from engine.math.matrix4 import Matrix4


class ModernGLShader(Shader):
    """
    ModernGL shader program implementation.
    """
    _program: moderngl.Program
    _vertex_path: str
    _fragment_path: str
    
    
    def __init__(self, ctx: moderngl.Context, vertex_path: str, fragment_path: str):
        """
        Create a shader program from vertex and fragment shader files.
        
        Args:
            ctx: ModernGL context
            vertex_path: Path to vertex shader file
            fragment_path: Path to fragment shader file
        """
        vertex_source = Path(vertex_path).read_text()
        fragment_source = Path(fragment_path).read_text()
        self._program = ctx.program(vertex_shader=vertex_source, fragment_shader=fragment_source)
        self._vertex_path = vertex_path
        self._fragment_path = fragment_path

    
    def use(self) -> None:
        """
        Bind this shader program for rendering.
        """
        # TODO: ModernGL doesn't require explicit use() call - it's automatic
        # You can leave this empty or store state if needed
        pass
    
    def set_uniform(self, name: str, value: Any) -> None:
        """
        Set a uniform variable in the shader.
        
        Args:
            name: Uniform variable name
            value: Value to set (int, float, Matrix4, tuple, etc.)
        """
        # TODO: Handle different value types:
        # - Matrix4: convert to tuple/list and flatten
        # - tuple/list: set directly
        # - int/float: set directly
        # Use: self._program[name] = value
        if name not in self._program:
            raise ValueError(f"Uniform '{name}' not found in shader program.")
        if isinstance(value, (int, float)):
            self._program[name] = value
        elif isinstance(value, (list, tuple)):
            self._program[name] = tuple(value)
        elif isinstance(value, Matrix4):
            self._program[name] = tuple(value.transpose().to_list())
        else:
            raise TypeError(f"Unsupported uniform type for '{name}': {type(value).__name__}")
    
    def cleanup(self) -> None:
        """
        Release shader resources.
        """
        # ModernGL objects have .release() method
        self._program.release()
