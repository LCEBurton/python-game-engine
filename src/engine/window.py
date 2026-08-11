"""
Window management using pygame and OpenGL.
"""
import pygame
import moderngl
from typing import Tuple


class Window:
    """
    Manages window creation, OpenGL context, and basic window operations.
    """

    _width: int
    _height: int
    _ctx: moderngl.Context
    _title: str
    _screen: pygame.Surface
    
    def __init__(self, width: int = 800, height: int = 600, title: str = "Game Engine"):
        """
        Create a window with OpenGL context.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            title: Window title
        """
        # TODO: Initialize pygame
        # TODO: Set OpenGL attributes (version 3.3 core profile)
        # TODO: Create pygame display with OPENGL and DOUBLEBUF flags
        # TODO: Set window title
        # TODO: Create ModernGL context
        # TODO: Store width, height
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        self._screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

        self._title = title
        self._width = width
        self._height = height
        self._ctx = moderngl.create_context()
         
    def get_context(self) -> moderngl.Context:
        """
        Get the ModernGL context.
        
        Returns:
            ModernGL context object
        """
        return self._ctx
    
    def swap_buffers(self) -> None:
        """
        Swap the front and back buffers (display rendered frame).
        """
        pygame.display.flip()
    
    def poll_events(self) -> bool:
        """
        Process window events.
        
        Returns:
            False if window should close, True otherwise
        """
        # TODO: Process pygame events
        # TODO: Check for QUIT event
        # TODO: Return False if should quit, True otherwise
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    
    @property
    def width(self) -> int:
        """Get window width."""
        return self._width
    
    @property
    def height(self) -> int:
        """Get window height."""
        return self._height
    
    @property
    def aspect_ratio(self) -> float:
        """Get window aspect ratio (width/height)."""
        return self._width / self._height if self._height != 0 else 1.0
    
    def cleanup(self) -> None:
        """
        Clean up window resources.
        """
        self._ctx.release()
        pygame.quit()
