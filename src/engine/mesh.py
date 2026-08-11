"""
Mesh class for storing geometry data (vertices, indices, vertex layout).
"""
from typing import List, Tuple

ATTRIBUTE_SIZES = {
    'position': 3,  # x, y, z
    'normal': 3,    # nx, ny, nz
    'color': 4,     # r, g, b, a
    'texcoord': 2   # u, v
}


class Mesh:
    """
    Represents a 3D mesh with vertex data, indices, and attribute layout.
    Immutable after creation. Stores data as pure Python lists.
    """

    __slots__ = ['_vertices', '_indices', '_attributes', '_stride']
    
    def __init__(
        self,
        vertices: List[float],
        indices: List[int],
        attributes: List[str]
    ):
        """
        Create a new mesh.
        
        Args:
            vertices: Flattened list of vertex data [x,y,z,r,g,b, x,y,z,r,g,b, ...]
            indices: List of indices referencing vertices
            attributes: List of attribute names (e.g., ['position', 'color'])
        """
        # TODO: Store vertices, indices, attributes as immutable
        # TODO: Validate that data makes sense
        # TODO: Calculate vertex_count and index_count

        if len(attributes) == 0:
            raise ValueError("Attribute list cannot be empty")
        elif len(vertices) == 0:
            raise ValueError("Vertex list cannot be empty")
        elif len(indices) == 0:
            raise ValueError("Index list cannot be empty")
        elif any(attr not in ATTRIBUTE_SIZES for attr in attributes):
            raise ValueError(f"Invalid attribute in {attributes}. Valid attributes: {list(ATTRIBUTE_SIZES.keys())}")
        elif any(not isinstance(i, int) or i < 0 for i in indices):
            raise ValueError("Indices must be non-negative integers")
        elif any(not isinstance(v, (int, float)) for v in vertices):
            raise ValueError("Vertices must be a list of floats")

        # Validate that the vertex data length is consistent with attributes
        expected_vertex_size = sum(ATTRIBUTE_SIZES[attr] for attr in attributes)
        if len(vertices) % expected_vertex_size != 0:
            raise ValueError(f"Vertex data length {len(vertices)} is not a multiple of expected vertex size {expected_vertex_size} based on attributes {attributes}")
        
        vertices = [float(v) for v in vertices]  # Ensure all vertices are floats

        self._vertices = tuple(vertices)
        self._indices = tuple(indices)
        self._attributes = tuple(attributes)
        self._stride = expected_vertex_size  # Number of floats per vertex
    
    @property
    def vertices(self) -> Tuple[float, ...]:
        """Return vertex data."""
        return self._vertices
    
    @property
    def indices(self) -> Tuple[int, ...]:
        """Return index data."""
        return self._indices
    
    @property
    def attributes(self) -> Tuple[str, ...]:
        """Return attribute layout."""
        return self._attributes
    
    @property
    def vertex_count(self) -> int:
        """Return number of vertices."""
        return len(self._vertices) // self._stride
    
    @property
    def index_count(self) -> int:
        """Return number of indices."""
        return len(self._indices)

