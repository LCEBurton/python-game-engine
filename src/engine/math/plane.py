"""
Plane class for frustum culling and collision detection.
"""
import math

from .vector3 import Vector3


class Plane:
    """3D plane represented by a normal and distance from origin.
    
    The plane equation is: normal.dot(point) + distance = 0
    """
    
    __slots__ = ('normal', 'distance')
    
    def __init__(self, normal=None, distance=0.0):
        """Initialize a plane.
        
        Args:
            normal: Vector3 normal to the plane (default: up vector)
            distance: Distance from origin along normal (default: 0.0)
        """
        self.normal = normal if normal is not None else Vector3(0, 1, 0)
        self.distance = float(distance)
    
    def __repr__(self):
        return f"Plane(normal={self.normal}, distance={self.distance})"
    
    def __str__(self):
        return f"Plane({self.normal}, d={self.distance:.3f})"
    
    def __eq__(self, other):
        if not isinstance(other, Plane):
            return False
        return self.normal == other.normal and self.distance == other.distance
    
    def copy(self):
        """Return a copy of this plane."""
        return Plane(self.normal.copy(), self.distance)
    
    def normalize(self):
        """Normalize the plane equation in place."""
        length = self.normal.length()
        if length > 0:
            self.normal = self.normal / length
            self.distance /= length
        return self
    
    def normalized(self):
        """Return a normalized copy of this plane."""
        plane = self.copy()
        plane.normalize()
        return plane
    
    def distance_to_point(self, point):
        """Calculate signed distance from plane to a point.
        
        Positive distance means point is on the side the normal points to.
        
        Args:
            point: Vector3 point
        
        Returns:
            Signed distance to point
        """
        return self.normal.dot(point) + self.distance
    
    def is_point_in_front(self, point):
        """Check if a point is in front of (on positive side of) the plane.
        
        Args:
            point: Vector3 point
        
        Returns:
            True if point is in front, False otherwise
        """
        return self.distance_to_point(point) > 0
    
    def project_point(self, point):
        """Project a point onto the plane.
        
        Args:
            point: Vector3 point to project
        
        Returns:
            Vector3 projected point on the plane
        """
        dist = self.distance_to_point(point)
        return point - self.normal * dist
    
    def flip(self):
        """Flip the plane (reverse normal and distance) in place."""
        self.normal = -self.normal
        self.distance = -self.distance
        return self
    
    def flipped(self):
        """Return a flipped copy of this plane."""
        return Plane(-self.normal, -self.distance)
    
    @staticmethod
    def from_points(p1, p2, p3):
        """Create a plane from three points.
        
        Args:
            p1, p2, p3: Vector3 points (should not be collinear)
        
        Returns:
            Plane passing through the three points
        """
        # Calculate two edge vectors
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Normal is perpendicular to both edges
        normal = v1.cross(v2).normalized()
        
        # Distance can be calculated from any point on the plane
        distance = -normal.dot(p1)
        
        return Plane(normal, distance)
    
    @staticmethod
    def from_normal_and_point(normal, point):
        """Create a plane from a normal and a point on the plane.
        
        Args:
            normal: Vector3 normal to the plane
            point: Vector3 point on the plane
        
        Returns:
            Plane with given normal passing through the point
        """
        normalized_normal = normal.normalized()
        distance = -normalized_normal.dot(point)
        return Plane(normalized_normal, distance)
    
    def intersect_line(self, line_start, line_end):
        """Find intersection point of a line segment with the plane.
        
        Args:
            line_start: Vector3 start of line segment
            line_end: Vector3 end of line segment
        
        Returns:
            Tuple of (intersects: bool, point: Vector3 or None, t: float or None)
            where t is the parameter [0, 1] along the line
        """
        direction = line_end - line_start
        denominator = self.normal.dot(direction)
        
        # Line is parallel to plane
        if abs(denominator) < 1e-6:
            return (False, None, None)
        
        t = -(self.normal.dot(line_start) + self.distance) / denominator
        
        # Check if intersection is within line segment
        if t < 0 or t > 1:
            return (False, None, None)
        
        intersection_point = line_start + direction * t
        return (True, intersection_point, t)
    
    def intersect_ray(self, ray_origin, ray_direction):
        """Find intersection point of a ray with the plane.
        
        Args:
            ray_origin: Vector3 origin of ray
            ray_direction: Vector3 direction of ray (should be normalized)
        
        Returns:
            Tuple of (intersects: bool, point: Vector3 or None, t: float or None)
            where t is the distance along the ray
        """
        denominator = self.normal.dot(ray_direction)
        
        # Ray is parallel to plane
        if abs(denominator) < 1e-6:
            return (False, None, None)
        
        t = -(self.normal.dot(ray_origin) + self.distance) / denominator
        
        # Ray points away from plane
        if t < 0:
            return (False, None, None)
        
        intersection_point = ray_origin + ray_direction * t
        return (True, intersection_point, t)
