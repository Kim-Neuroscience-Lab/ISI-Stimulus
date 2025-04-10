# src/stimuli/spherical_correction.py
"""
Spherical stimulus correction as described in MMC1.

This module implements the spherical coordinate transformation as described in the MMC1 document:
- For a drifting grating, S = cos(2πfsθ - tft)
- For a drifting bar, the correction maintains constant spatial and temporal frequency

The screen is positioned at an angle of 20° relative to the animal to properly stimulate
the vertical meridian, and also tilted to match the tilt of the animal.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
from pathlib import Path
import math


class SphericalCorrection:
    """
    Implements spherical stimulus correction following MMC1 procedure.

    As described in MMC1:
    "Spherical corrections were applied to all stimuli in order to account for the distortions
    created by displaying stimuli to the animal on a flat monitor... Drifting bar and grating
    stimuli were generated through custom routines in Psychtoolbox and Matlab and then
    transformed with spherical projection."
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the spherical correction with the given parameters.

        Args:
            parameters: Dictionary containing:
                - screen_distance: Distance from eye to screen in cm
                - screen_angle: Angle of screen relative to animal (default 20°)
                - x_size: Horizontal visual field in degrees
                - y_size: Vertical visual field in degrees
                - horizontal_meridian_offset: Offset for horizontal meridian (default 20°)
        """
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
        self._setup_transformation_params()

        # Initialize cached grid coordinates
        self._cached_grid = None
        self._cached_grid_shape = None

    def _setup_transformation_params(self):
        """Set up the transformation parameters and precompute constants."""
        # Get screen parameters
        self.screen_distance = self.parameters.get("screen_distance", 10.0)  # cm
        self.screen_angle = self.parameters.get("screen_angle", 20.0)  # degrees

        # Visual field parameters
        self.x_size = self.parameters.get("x_size", 147.0)  # degrees horizontal
        self.y_size = self.parameters.get("y_size", 153.0)  # degrees vertical

        # In the spherical model, the horizontal_meridian_offset determines where the
        # equator of the sphere is positioned. This value represents the number of degrees
        # above the center of the screen where the equator is located.
        # Since the equator should be centered on the screen, this is now used to
        # adjust y coordinates before transformation.
        self.horizontal_meridian_offset = self.parameters.get(
            "horizontal_meridian_offset", 20.0
        )

        self.logger.info(
            f"Spherical correction: Equator centered on screen at {self.horizontal_meridian_offset}° offset"
        )

        # Convert screen angle to radians
        self.screen_angle_rad = np.radians(self.screen_angle)

        # Precompute sin/cos values for the screen angle
        self.sin_screen_angle = np.sin(self.screen_angle_rad)
        self.cos_screen_angle = np.cos(self.screen_angle_rad)

        # Store commonly used constants
        self.pi_half = np.pi / 2

    def transform_coordinates(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        z_norm: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform normalized screen coordinates using spherical correction.

        This implements the transformation from Cartesian coordinates to spherical coordinates:
        θ (altitude) = π/2 - cos⁻¹(z/√(x₀² + y² + z²))
        φ (azimuth) = tan⁻¹(-y/x₀)

        Args:
            x_norm: Normalized x coordinates (-1 to 1)
            y_norm: Normalized y coordinates (-1 to 1)
            z_norm: Optional normalized z coordinates (if None, calculated based on x,y)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed x, y coordinates
        """
        # Convert normalized coordinates to visual degrees
        x_deg = x_norm * (self.x_size / 2)
        y_deg = y_norm * (self.y_size / 2)

        # Apply the transformation as described in MMC1
        x_transformed, y_transformed = self._apply_spherical_transformation(
            x_deg, y_deg
        )

        # Convert back to normalized coordinates
        x_norm_transformed = x_transformed / (self.x_size / 2)
        y_norm_transformed = y_transformed / (self.y_size / 2)

        return x_norm_transformed, y_norm_transformed

    def _apply_spherical_transformation(
        self, x_deg: np.ndarray, y_deg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the spherical transformation to visual degree coordinates.

        Implements the mathematical transformation:
        θ (altitude) = π/2 - cos⁻¹(z/√(x₀² + y² + z²))
        φ (azimuth) = tan⁻¹(-y/x₀)

        Where:
        - x₀ is the distance from eye to screen
        - y is the horizontal position on screen
        - z is the vertical position on screen

        Args:
            x_deg: X coordinates in visual degrees (horizontal on screen)
            y_deg: Y coordinates in visual degrees (vertical on screen)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed coordinates in visual degrees
        """
        # First apply horizontal meridian offset to y coordinates
        y_deg_adjusted = y_deg - self.horizontal_meridian_offset

        # Get the correction axis and direction
        correct_axis = self.parameters.get("correct_axis", "azimuth")
        sweep_direction = self.parameters.get("sweep_direction", "right-to-left")

        # We need to map the degrees to actual distances
        # Convert eye-to-screen distance to the same units
        x0 = self.screen_distance  # Distance from eye to screen (cm)

        # Convert degrees to distances on screen
        # Using small angle approximation: tan(θ) ≈ θ (in radians)
        # d = x0 * tan(θ)
        y_rad = np.radians(x_deg)  # Horizontal on screen (azimuth)
        z_rad = np.radians(y_deg_adjusted)  # Vertical on screen (altitude)

        # Convert to distances on screen (relative to center)
        y = x0 * np.tan(y_rad)  # Horizontal distance on screen
        z = x0 * np.tan(z_rad)  # Vertical distance on screen

        # Compute the spherical coordinates using the formulas
        # θ (altitude) = π/2 - cos⁻¹(z/√(x₀² + y² + z²))
        # φ (azimuth) = tan⁻¹(-y/x₀)

        # Calculate the square root term
        sqrt_term = np.sqrt(x0**2 + y**2 + z**2)

        # Calculate altitude (θ)
        # Handle potential division by zero or arguments outside [-1, 1]
        z_over_sqrt = np.zeros_like(z)
        valid_indices = sqrt_term > 0
        z_over_sqrt[valid_indices] = z[valid_indices] / sqrt_term[valid_indices]
        # Clip to handle numerical errors
        z_over_sqrt = np.clip(z_over_sqrt, -1.0, 1.0)

        # Calculate altitude using the formula θ = π/2 - cos⁻¹(z/√(x₀² + y² + z²))
        theta = np.pi / 2 - np.arccos(z_over_sqrt)

        # Calculate azimuth (φ)
        # Handle potential division by zero
        phi = np.zeros_like(y)
        nonzero_indices = np.abs(x0) > 1e-10
        phi[nonzero_indices] = np.arctan2(-y[nonzero_indices], x0)

        # Convert back to degrees
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)

        # Apply appropriate transformation based on correction axis
        if correct_axis == "azimuth":
            # For horizontal sweep, apply azimuth correction
            x_transformed = phi_deg
            y_transformed = theta_deg
        elif correct_axis == "altitude":
            # For vertical sweep, apply altitude correction
            x_transformed = phi_deg
            y_transformed = theta_deg
        else:
            raise ValueError(f"Unknown correction axis: {correct_axis}")

        # Apply the offset back to center properly
        y_transformed = y_transformed + self.horizontal_meridian_offset

        return x_transformed, y_transformed

    def create_transformation_maps(
        self, width: int, height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create transformation maps for the entire frame for use with OpenCV.

        This method is optimized for performance and creates maps suitable
        for use with cv2.remap() for efficient transformation of frames.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            Tuple[np.ndarray, np.ndarray]: (map_x, map_y) for use with cv2.remap()
        """
        self.logger.info(f"Creating transformation maps ({width}x{height})...")

        # Create coordinate grid (vectorized)
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Normalize coordinates to range [-1, 1]
        x_norm = x_coords / (width / 2) - 1.0
        y_norm = y_coords / (height / 2) - 1.0

        # Apply spherical transformation
        x_transformed, y_transformed = self.transform_coordinates(x_norm, y_norm)

        # Convert back to pixel coordinates (OpenCV requires float32)
        map_x = ((x_transformed + 1.0) * (width / 2)).astype(np.float32)
        map_y = ((y_transformed + 1.0) * (height / 2)).astype(np.float32)

        self.logger.info(f"Transformation maps created successfully")
        return map_x, map_y

    def apply_transformation_to_frame(
        self, frame: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
    ) -> np.ndarray:
        """
        Apply precomputed transformation maps to a frame.

        Args:
            frame: Input frame
            map_x: X coordinate mapping
            map_y: Y coordinate mapping

        Returns:
            np.ndarray: Transformed frame
        """
        # Apply transformation using OpenCV's remap
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def transform_drifting_grating(
        self,
        coords: np.ndarray,
        spatial_freq: float,
        orientation_deg: float,
        temporal_freq: float,
        time: float,
    ) -> np.ndarray:
        """
        Transform a drifting grating using spherical correction.

        As per MMC1: S = cos(2πfsθ - tft)
        This maintains constant spatial and temporal frequency throughout the visual field.

        Args:
            coords: Coordinate array shape (height, width, 2) for x,y
            spatial_freq: Spatial frequency in cycles/degree
            orientation_deg: Orientation in degrees
            temporal_freq: Temporal frequency in Hz
            time: Current time in seconds

        Returns:
            np.ndarray: Transformed grating pattern
        """
        # Extract coordinates
        y_coords, x_coords = np.mgrid[0 : coords.shape[0], 0 : coords.shape[1]]
        x_norm = x_coords / (coords.shape[1] / 2) - 1.0
        y_norm = y_coords / (coords.shape[0] / 2) - 1.0

        # Apply spherical transformation
        x_transformed, y_transformed = self.transform_coordinates(x_norm, y_norm)

        # Convert orientation to radians
        orientation_rad = np.radians(orientation_deg)

        # Precompute sin/cos values
        cos_orientation = np.cos(orientation_rad)
        sin_orientation = np.sin(orientation_rad)

        # Calculate grating (vectorized)
        # For a drifting grating defined as cos(2πfsθ - tft) where:
        # fs is spatial frequency, θ is position, ft is temporal frequency, t is time
        phase = (
            2
            * np.pi
            * spatial_freq
            * (x_transformed * cos_orientation + y_transformed * sin_orientation)
            - 2 * np.pi * temporal_freq * time
        )

        # Create grating (vectorized)
        grating = np.cos(phase)

        # Scale to [0, 1] range
        grating = (grating + 1) / 2

        return grating

    @property
    def coordinates_grid_deg(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get coordinate grids in visual degrees based on current screen dimensions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: y and x coordinate grids in visual degrees
        """
        # Get current shape
        shape = (
            self.parameters.get("height", 1024),
            self.parameters.get("width", 1280),
        )

        # Check if we need to recompute the grid
        if self._cached_grid is None or self._cached_grid_shape != shape:
            height, width = shape

            # Create normalized coordinate grids
            y_norm, x_norm = np.mgrid[
                -1 : 1 : complex(0, height), -1 : 1 : complex(0, width)
            ]

            # Convert to visual degrees
            y_grid_deg = y_norm * (self.y_size / 2)
            x_grid_deg = x_norm * (self.x_size / 2)

            # Cache the result
            self._cached_grid = (y_grid_deg, x_grid_deg)
            self._cached_grid_shape = shape

        return self._cached_grid

    def transform_drifting_bar(
        self, pos_deg: float, sweep_direction: str, size_deg: float
    ) -> np.ndarray:
        """
        Create mask for the drifting bar with static checkerboard texture.
        The bar shape is properly transformed according to spherical coordinates.

        Args:
            pos_deg: Position of drifting bar in visual degrees (angle in the sweep direction)
            sweep_direction: Direction in which the bar is drifting
            size_deg: Size of the bar in visual degrees

        Returns:
            np.ndarray: Binary mask for drifting bar with static checkerboard pattern
        """
        # Store sweep direction to use in transformation
        self.parameters["sweep_direction"] = sweep_direction

        # Determine correction axis based on sweep direction
        if sweep_direction in ["left-to-right", "right-to-left"]:
            self.parameters["correct_axis"] = "azimuth"
        elif sweep_direction in ["top-to-bottom", "bottom-to-top"]:
            self.parameters["correct_axis"] = "altitude"
        else:
            raise ValueError(f"Invalid sweep direction: {sweep_direction}")

        # Get image dimensions
        height, width = self.parameters.get("height", 1024), self.parameters.get(
            "width", 1280
        )

        # Create coordinate arrays - ensure these match the actual image dimensions
        x = np.arange(width)
        y = np.arange(height)

        # Create meshgrid
        xx, yy = np.meshgrid(x, y)

        # Normalize coordinates to range [-1, 1]
        x_norm = 2 * (xx / max(1, width - 1)) - 1
        y_norm = 2 * (yy / max(1, height - 1)) - 1

        # Convert to visual degrees
        x_deg = x_norm * (self.x_size / 2)
        y_deg = y_norm * (self.y_size / 2)

        # Calculate the spherical coordinates directly for all pixels
        # First apply horizontal meridian offset to y coordinates
        y_deg_adjusted = y_deg - self.horizontal_meridian_offset

        # We need to map the degrees to actual distances
        # Convert eye-to-screen distance to the same units
        x0 = self.screen_distance  # Distance from eye to screen (cm)

        # Convert degrees to distances on screen
        # Using small angle approximation: tan(θ) ≈ θ (in radians)
        # d = x0 * tan(θ)
        azimuth_rad = np.radians(x_deg)  # Horizontal on screen (azimuth)
        altitude_rad = np.radians(y_deg_adjusted)  # Vertical on screen (altitude)

        # Convert to distances on screen (relative to center)
        y_screen = x0 * np.tan(azimuth_rad)  # Horizontal distance on screen
        z_screen = x0 * np.tan(altitude_rad)  # Vertical distance on screen

        # Calculate the square root term
        sqrt_term = np.sqrt(x0**2 + y_screen**2 + z_screen**2)

        # Calculate altitude (θ)
        # Handle potential division by zero or arguments outside [-1, 1]
        z_over_sqrt = np.zeros_like(z_screen)
        valid_indices = sqrt_term > 0
        z_over_sqrt[valid_indices] = z_screen[valid_indices] / sqrt_term[valid_indices]
        # Clip to handle numerical errors
        z_over_sqrt = np.clip(z_over_sqrt, -1.0, 1.0)

        # Calculate altitude using the formula θ = π/2 - cos⁻¹(z/√(x₀² + y² + z²))
        theta = np.pi / 2 - np.arccos(z_over_sqrt)

        # Calculate azimuth (φ)
        # Handle potential division by zero
        phi = np.zeros_like(y_screen)
        nonzero_indices = np.abs(x0) > 1e-10
        phi[nonzero_indices] = np.arctan2(-y_screen[nonzero_indices], x0)

        # Convert back to degrees
        theta_deg = np.degrees(theta)  # altitude angle
        phi_deg = np.degrees(phi)  # azimuth angle

        # Create the bar mask directly in spherical coordinate space
        # This ensures the bar follows the correct spherical curvature
        # Initialize empty mask
        bar_mask = np.zeros((height, width), dtype=bool)

        if sweep_direction in ["left-to-right", "right-to-left"]:
            # For horizontal sweep, define the bar as a region of constant azimuth
            # The bar is defined by phi (azimuth) values within a range
            mask_condition = (phi_deg > (pos_deg - size_deg / 2)) & (
                phi_deg < (pos_deg + size_deg / 2)
            )
        else:
            # For vertical sweep, define the bar as a region of constant altitude
            # The bar is defined by theta (altitude) values within a range
            adjusted_theta_deg = theta_deg + self.horizontal_meridian_offset
            mask_condition = (adjusted_theta_deg > (pos_deg - size_deg / 2)) & (
                adjusted_theta_deg < (pos_deg + size_deg / 2)
            )

        # Apply the mask condition, ensuring proper dimensions
        bar_mask = mask_condition

        # Ensure mask has correct dimensions
        assert bar_mask.shape == (
            height,
            width,
        ), f"Mask shape {bar_mask.shape} doesn't match expected dimensions {(height, width)}"

        return bar_mask
