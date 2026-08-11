"""
Ray class for picking and intersection tests.
"""
from .vector3 import Vector3


class Ray:
    """3D ray represented by an origin point and direction vector.
    
    A ray extends infinitely from the origin in the direction.
    """
    
    __slots__ = ('origin', 'direction')
    
    def __init__(self, origin=None, direction=None):
        """Initialize a ray.
        
        Args:
            origin: Vector3 origin point (default: zero)
            direction: Vector3 direction (default: forward, will be normalized)
        """
        self.origin = origin if origin is not None else Vector3.zero()
        dir_vec = direction if direction is not None else Vector3(0, 0, -1)
        self.direction = dir_vec.normalized()
    
    def __repr__(self):
        return f"Ray(origin={self.origin}, direction={self.direction})"
    
    def __str__(self):
        return f"Ray(O={self.origin}, D={self.direction})"
    
    def __eq__(self, other):
        if not isinstance(other, Ray):
            return False
        return self.origin == other.origin and self.direction == other.direction
    
    def copy(self):
        """Return a copy of this ray."""
        return Ray(self.origin.copy(), self.direction.copy())
    
    def point_at(self, t):
        """Get a point at distance t along the ray.
        
        Args:
            t: Distance along ray
        
        Returns:
            Vector3 point at origin + t * direction
        """
        return self.origin + self.direction * t
    
    def closest_point_to(self, point):
        """Find the closest point on the ray to a given point.
        
        Args:
            point: Vector3 point
        
        Returns:
            Tuple of (closest_point: Vector3, t: float)
        """
        to_point = point - self.origin
        t = max(0, to_point.dot(self.direction))
        closest = self.point_at(t)
        return (closest, t)
    
    def distance_to_point(self, point):
        """Calculate distance from ray to a point.
        
        Args:
            point: Vector3 point
        
        Returns:
            Minimum distance to the point
        """
        closest, _ = self.closest_point_to(point)
        return (point - closest).length()
    
    def intersect_sphere(self, center, radius):
        """Test intersection with a sphere.
        
        Args:
            center: Vector3 center of sphere
            radius: Radius of sphere
        
        Returns:
            Tuple of (intersects: bool, t_near: float or None, t_far: float or None)
            where t values are distances along the ray
        """
        oc = self.origin - center
        a = self.direction.dot(self.direction)
        b = 2.0 * oc.dot(self.direction)
        c = oc.dot(oc) - radius * radius
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return (False, None, None)
        
        sqrt_discriminant = discriminant ** 0.5
        t_near = (-b - sqrt_discriminant) / (2.0 * a)
        t_far = (-b + sqrt_discriminant) / (2.0 * a)
        
        # If both are negative, sphere is behind ray
        if t_far < 0:
            return (False, None, None)
        
        # If near is negative but far is positive, we're inside
        if t_near < 0:
            t_near = 0
        
        return (True, t_near, t_far)
    
    def intersect_plane(self, plane):
        """Test intersection with a plane.
        
        Args:
            plane: Plane object
        
        Returns:
            Tuple of (intersects: bool, point: Vector3 or None, t: float or None)
        """
        return plane.intersect_ray(self.origin, self.direction)
    
    def intersect_triangle(self, v0, v1, v2):
        """Test intersection with a triangle using Möller-Trumbore algorithm.
        
        Args:
            v0, v1, v2: Vector3 vertices of the triangle
        
        Returns:
            Tuple of (intersects: bool, point: Vector3 or None, t: float or None)
        """
        epsilon = 1e-6
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = self.direction.cross(edge2)
        a = edge1.dot(h)
        
        if abs(a) < epsilon:
            return (False, None, None)  # Ray is parallel to triangle
        
        f = 1.0 / a
        s = self.origin - v0
        u = f * s.dot(h)
        
        if u < 0.0 or u > 1.0:
            return (False, None, None)
        
        q = s.cross(edge1)
        v = f * self.direction.dot(q)
        
        if v < 0.0 or u + v > 1.0:
            return (False, None, None)
        
        t = f * edge2.dot(q)
        
        if t > epsilon:  # Ray intersection
            intersection_point = self.point_at(t)
            return (True, intersection_point, t)
        
        return (False, None, None)
