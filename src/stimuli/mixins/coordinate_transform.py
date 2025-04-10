"""
Coordinate transformation mixin for visual stimuli.

This module implements the spherical coordinate system as described in the MMC1 document:
- The eye is the origin of the coordinate system
- The perpendicular bisector from the eye to the monitor is the (0°,0°) axis
- The y and z axes are defined as the horizontal and vertical dimensions on the monitor
- x-dimension is constant on the monitor surface (x₀)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Protocol
from abc import abstractmethod


class TransformationStrategy(Protocol):
    """Protocol defining the interface for coordinate transformation strategies."""

    @abstractmethod
    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray],
        parameters: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform coordinates using a specific strategy.

        Args:
            x: X coordinates array in degrees
            y: Y coordinates array in degrees
            z: Z coordinates (optional, calculated if None)
            parameters: Configuration parameters for the transformation

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed coordinates
        """
        pass


class SphericalTransformation:
    """
    Implements spherical coordinate transformation as described in MMC1.

    For spherical coordinates:
    θ = π/2 - cos⁻¹(z/√(x₀²+y²+z²))  (altitude)
    φ = tan⁻¹(-y/x₀)                 (azimuth)
    """

    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray],
        parameters: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform coordinates using spherical transformation.

        Args:
            x: X coordinates array in degrees
            y: Y coordinates array in degrees
            z: Z coordinates (optional, calculated if None)
            parameters: Configuration parameters for the transformation

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed coordinates
        """
        # Get screen distance (x₀) - this is the constant x value for all points on the monitor
        screen_distance = parameters.get(
            "screen_distance", 10.0
        )  # cm from eye to screen
        x0 = (
            screen_distance  # This is the constant x-coordinate for the monitor surface
        )

        # Convert input coordinates from degrees to cm on screen
        # Using the small angle approximation: tan(θ) ≈ θ for small θ in radians
        y_cm = np.tan(np.radians(y)) * screen_distance

        # If z is not provided, calculate based on the screen position
        if z is None:
            pixels_per_degree = parameters.get("_pixels_per_degree")
            if pixels_per_degree:
                # Convert from degrees to cm
                z_deg = x  # Using x input as z in degrees for simplicity
                z_cm = np.tan(np.radians(z_deg)) * screen_distance
            else:
                # Default z is zero (flat screen at constant distance)
                z_cm = np.zeros_like(y)
        else:
            # Use provided z values
            z_cm = z

        # For a flat screen at distance x₀, each point has coordinates (x₀, y, z)
        # Calculate radius from eye to each point
        r = np.sqrt(x0**2 + y_cm**2 + z_cm**2)

        # Calculate spherical coordinates exactly as in the equations
        # θ = π/2 - cos⁻¹(z/√(x₀²+y²+z²))
        theta = np.pi / 2 - np.arccos(z_cm / r)

        # φ = tan⁻¹(-y/x₀)
        phi = np.arctan2(-y_cm, x0)

        # Get sweep direction to determine which transformation to apply
        sweep_direction = parameters.get("sweep_direction", "left-to-right")

        # Initialize transformed coordinates
        x_transformed = np.copy(x)
        y_transformed = np.copy(y)

        # For orientation changes, apply rotation around x-axis as described in MMC1
        orientation = parameters.get("orientation", 0.0)
        if orientation != 0:
            # Convert orientation to radians
            orientation_rad = np.radians(-orientation)

            # Apply rotation around x-axis by redefining y and z:
            # y' = z*sin(-φ) + y*cos(-φ)
            # z' = z*cos(-φ) - y*sin(-φ)
            y_rot = z_cm * np.sin(orientation_rad) + y_cm * np.cos(orientation_rad)
            z_rot = z_cm * np.cos(orientation_rad) - y_cm * np.sin(orientation_rad)

            # Recalculate spherical coordinates with rotated values
            r_rot = np.sqrt(x0**2 + y_rot**2 + z_rot**2)
            theta = np.pi / 2 - np.arccos(z_rot / r_rot)
            phi = np.arctan2(-y_rot, x0)

        # Convert back to screen coordinates based on sweep direction
        if sweep_direction in ["left-to-right", "right-to-left"]:
            # For horizontal drift, maintain constant altitude (θ)
            # Convert back to degrees for rendering
            y_transformed = np.degrees(np.arctan2(-y_cm, x0))

        elif sweep_direction in ["top-to-bottom", "bottom-to-top"]:
            # For vertical drift, use the altitude (θ) for vertical positioning
            # Convert altitude (θ) to visual degrees from center
            y_transformed = np.degrees(theta - np.pi / 2)  # Adjust to make 0 at horizon

        # Horizontal meridian offset (if specified)
        horizontal_meridian_offset = parameters.get("horizontal_meridian_offset", 0.0)
        if horizontal_meridian_offset != 0:
            y_transformed += horizontal_meridian_offset

        return x_transformed, y_transformed


class HyperbolicTransformation:
    """
    Implements hyperbolic coordinate transformation.

    This transformation applies a distance-based correction that increases
    with eccentricity according to a quadratic function.
    """

    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray],
        parameters: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform coordinates using hyperbolic transformation.

        Args:
            x: X coordinates array in degrees
            y: Y coordinates array in degrees
            z: Z coordinates (optional, calculated if None)
            parameters: Configuration parameters for the transformation

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed coordinates
        """
        # Normalize coordinates to range [-1, 1]
        x_norm = x / 90.0  # Assuming maximum 90 degrees field of view
        y_norm = y / 90.0

        # Calculate distance from center (eccentricity)
        distance = np.sqrt(x_norm**2 + y_norm**2)

        # Get correction strength
        strength = parameters.get("correction_strength", 1.0)

        # Apply hyperbolic transformation
        correction = strength * distance**2
        scaling_factor = 1.0 / (1.0 + correction)

        # Apply correction
        x_transformed = x_norm * scaling_factor * 90.0
        y_transformed = y_norm * scaling_factor * 90.0

        return x_transformed, y_transformed


class CoordinateTransformMixin:
    """
    Mixin class for coordinate transformations in visual stimuli.

    This implements the spherical coordinate system as described in the MMC1 document
    for accurate visual representation across a wide field of view.
    """

    def __init__(self):
        """Initialize the transformation registry."""
        self._transformation_registry = {
            "spherical": SphericalTransformation(),
            "hyperbolic": HyperbolicTransformation(),
            "none": None,
        }

    def _apply_transformation(
        self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply coordinate transformation based on the selected type.

        Args:
            x: X coordinates array in degrees
            y: Y coordinates array in degrees
            z: Z coordinates (optional, calculated if None)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed coordinates
        """
        # Check if parameters attribute exists (should be provided by the parent class)
        if not hasattr(self, "parameters"):
            raise AttributeError(
                "CoordinateTransformMixin requires 'parameters' attribute"
            )

        # Get transformation type
        transformation_type = self.parameters.get("transformation", "none")
        if transformation_type == "none":
            return x, y

        # Get the appropriate transformation strategy
        transformation = self._transformation_registry.get(transformation_type)
        if transformation is None:
            return x, y

        # Apply the transformation strategy
        return transformation.transform(x, y, z, self.parameters)
