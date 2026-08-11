"""
ModernGL implementation of Renderer.
"""
import moderngl
import struct
from typing import Dict, Any, Tuple
from engine.rendering.renderer import Renderer
from engine.rendering.shader import Shader
from engine.rendering.backends.moderngl_shader import ModernGLShader
from engine.mesh import Mesh, ATTRIBUTE_SIZES


class ModernGLRenderer(Renderer):
    """
    ModernGL renderer implementation. Handles mesh uploading, buffer caching, and drawing.
    """
    _ctx: moderngl.Context
    
    
    
    def __init__(self, ctx: moderngl.Context):
        """
        Create a ModernGL renderer.
        
        Args:
            ctx: ModernGL context from Window
        """
        self._ctx = ctx
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)
        # Cache stores: {mesh_id -> {vbo, ibo, format, attributes, vao_cache}}
        # vao_cache: {shader_id -> vao} (VAOs are shader-specific)
        self._mesh_buffers: Dict[int, Dict[str, Any]] = {}

    def clear(self, color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> None:
        """
        Clear the screen with the specified color.
        
        Args:
            color: RGBA color tuple (values 0.0-1.0)
        """
        self._ctx.clear(*color, depth=1.0)
    
    def upload(self, mesh: Mesh) -> None:
        """
        Upload mesh data to GPU. Creates and caches vertex/index buffers.
        
        Args:
            mesh: The mesh to upload
        """
        mesh_id = id(mesh)
        
        # Skip if already uploaded
        if mesh_id in self._mesh_buffers:
            return
        
        # Convert vertex data to bytes
        vertices = mesh.vertices
        vertex_bytes = struct.pack(f'{len(vertices)}f', *vertices)
        
        # Convert index data to bytes
        indices = mesh.indices
        index_bytes = struct.pack(f'{len(indices)}I', *indices)
        
        # Create GPU buffers
        vbo = self._ctx.buffer(vertex_bytes)
        ibo = self._ctx.buffer(index_bytes)
        
        # Build format string from attributes
        # e.g., ['position', 'color'] -> "3f 3f"
        format_parts = [f"{ATTRIBUTE_SIZES[attr]}f" for attr in mesh.attributes]
        format_string = " ".join(format_parts)
        
        # Store buffer data (VAO will be created lazily on first draw)
        self._mesh_buffers[mesh_id] = {
            'vbo': vbo,
            'ibo': ibo,
            'format': format_string,
            'attributes': mesh.attributes,
            'vao_cache': {}  # shader_id -> vao
        }
    
    def draw(self, mesh: Mesh, shader: Shader, uniforms: Dict[str, Any]) -> None:
        """
        Draw a mesh using the specified shader and uniforms.
        
        Args:
            mesh: The mesh to draw (must be uploaded first)
            shader: The shader program to use
            uniforms: Dictionary of uniform name -> value
        """
        mesh_id = id(mesh)
        
        # Check if mesh is uploaded
        if mesh_id not in self._mesh_buffers:
            raise RuntimeError(f"Mesh must be uploaded before drawing. Call renderer.upload(mesh) first.")
        
        buffer_data = self._mesh_buffers[mesh_id]
        shader_id = id(shader)
        
        # Get or create VAO for this mesh+shader combination
        if shader_id not in buffer_data['vao_cache']:
            # Create VAO binding this mesh's buffers to this shader's attributes
            vao = self._ctx.vertex_array(
                shader._program, # type: ignore
                [(buffer_data['vbo'], buffer_data['format'], *buffer_data['attributes'])],
                index_buffer=buffer_data['ibo']
            )
            buffer_data['vao_cache'][shader_id] = vao
        
        vao = buffer_data['vao_cache'][shader_id]
        
        # Set uniforms
        for name, value in uniforms.items():
            shader.set_uniform(name, value)
        
        # Render
        vao.render(moderngl.TRIANGLES)
    
    def create_shader(self, vertex_path: str, fragment_path: str) -> Shader:
        """
        Create a shader program from vertex and fragment shader files.
        
        Args:
            vertex_path: Path to vertex shader file
            fragment_path: Path to fragment shader file
            
        Returns:
            ModernGLShader object
        """
        return ModernGLShader(self._ctx, vertex_path, fragment_path)
    
    def cleanup(self) -> None:
        """
        Clean up GPU resources.
        """
        for buffer_data in self._mesh_buffers.values():
            # Release all VAOs for this mesh
            for vao in buffer_data['vao_cache'].values():
                vao.release()
            # Release buffers
            buffer_data['vbo'].release()
            buffer_data['ibo'].release()
        
        self._mesh_buffers.clear()
    
