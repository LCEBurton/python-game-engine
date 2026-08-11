"""
Rotating cube example - demonstrates the rendering pipeline.
"""
import sys
import math
import time
from pathlib import Path

# Add parent directory to path to import engine modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.window import Window
from engine.mesh import Mesh
from engine.camera import Camera
from engine.math.matrix4 import Matrix4
from engine.rendering.backends.moderngl_renderer import ModernGLRenderer


def create_cube_mesh() -> Mesh:
    """
    Create a colored cube mesh.
    
    Returns:
        Mesh with position and color attributes
    """
    # Cube vertices: 8 corners with positions (x,y,z) and colors (r,g,b)
    vertices = [
        # position          color
        -1.0, -1.0, -1.0,  1.0, 0.0, 0.0, 1.0,  # 0: back-bottom-left (red)
         1.0, -1.0, -1.0,  0.0, 1.0, 0.0, 1.0,  # 1: back-bottom-right (green)
         1.0,  1.0, -1.0,  0.0, 0.0, 1.0, 1.0,  # 2: back-top-right (blue)
        -1.0,  1.0, -1.0,  1.0, 1.0, 0.0, 1.0,  # 3: back-top-left (yellow)
        -1.0, -1.0,  1.0,  1.0, 0.0, 1.0, 1.0,  # 4: front-bottom-left (magenta)
         1.0, -1.0,  1.0,  0.0, 1.0, 1.0, 1.0,  # 5: front-bottom-right (cyan)
         1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0,  # 6: front-top-right (white)
        -1.0,  1.0,  1.0,  0.5, 0.5, 0.5, 1.0,  # 7: front-top-left (gray)
    ]
    
    # Cube indices: 12 triangles (2 per face, 6 faces)
    indices = [
        # Back face
        0, 1, 2,  0, 2, 3,
        # Front face
        4, 6, 5,  4, 7, 6,
        # Left face
        0, 3, 7,  0, 7, 4,
        # Right face
        1, 5, 6,  1, 6, 2,
        # Bottom face
        0, 4, 5,  0, 5, 1,
        # Top face
        3, 2, 6,  3, 6, 7,
    ]
    
    return Mesh(vertices, indices, ['position', 'color'])


def main():
    """Main application loop."""
    # Create window
    window = Window(800, 600, "Rotating Cube")
    
    # Create renderer
    renderer = ModernGLRenderer(window.get_context())
    
    # Create camera
    camera = Camera(
        fov=60.0,
        aspect=window.aspect_ratio,
        near=0.1,
        far=100.0,
        position=(0.0, 0.0, 5.0),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0)
    )
    
    # Create cube mesh and upload to GPU
    cube = create_cube_mesh()
    renderer.upload(cube)
    
    # Load shader
    shader = renderer.create_shader('src/shaders/basic.vert', 'src/shaders/basic.frag')
    
    # Animation state
    angle = 0.0
    
    # Main loop
    running = True
    while running:
        # Process events
        running = window.poll_events()
        
        # Update rotation
        angle += 0.01
        
        # Create model matrix (rotation around Y and X axes)
        model = Matrix4.rotation_y(angle) * Matrix4.rotation_x(angle * 0.5)
        
        # Calculate MVP matrix
        mvp = camera.projection * camera.view * model # type: ignore
        
        # Render
        renderer.clear((0.1, 0.1, 0.15, 1.0))
        renderer.draw(cube, shader, {'mvp': mvp})
        
        # Display
        window.swap_buffers()

        time.sleep(0.01)  # Limit frame rate to ~100 FPS
    
    # Cleanup
    renderer.cleanup()
    window.cleanup()


if __name__ == '__main__':
    main()
