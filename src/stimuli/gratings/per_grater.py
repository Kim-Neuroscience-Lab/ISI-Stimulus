# src/stimuli/gratings/per_grater.py
"""Periodic grating stimulus module."""

import numpy as np
from typing import Dict, Any, Optional


def register_module(parameter_config):
    """
    Register this module with the parameter configuration.

    Args:
        parameter_config: The parameter configuration object
    """
    parameter_config.register_module_factory("PG", create_default_parameters)


def create_default_parameters() -> Dict[str, Any]:
    """
    Create default parameters for the periodic grating stimulus.

    Returns:
        Dict[str, Any]: The default parameters
    """
    return {
        # Grating parameters
        "orientation": 0.0,  # degrees
        "spatial_freq": 0.1,  # cycles/degree
        "temporal_freq": 2.0,  # Hz
        "phase": 0.0,  # degrees
        "contrast": 100.0,  # percent
        # Size and position
        "x_size": 30.0,  # degrees
        "y_size": 30.0,  # degrees
        "x_pos": 0.0,  # degrees from center
        "y_pos": 0.0,  # degrees from center
        # Background
        "background": 127,  # 0-255
        # Mask parameters
        "mask_type": "none",  # 'none', 'gaussian', 'circle'
        "mask_radius": 15.0,  # degrees
        "mask_sigma": 5.0,  # degrees (for gaussian)
        # Timing
        "duration": 1.0,  # seconds
        # Additional parameters
        "reverse": 0,  # 0 = no, 1 = yes (reverse grating direction)
        "square_wave": 0,  # 0 = no (sinusoidal), 1 = yes (square wave)
    }


class PeriodicGratingStimulus:
    """
    Periodic grating stimulus generator.

    This class generates periodic (drifting) grating stimuli
    for visual neuroscience experiments.
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the grating stimulus generator.

        Args:
            parameters: The stimulus parameters
        """
        self.parameters = parameters

        # Set default values for required parameters if not provided
        if "refresh_rate" not in self.parameters:
            self.parameters["refresh_rate"] = 60

        # Calculated properties
        self._calculate_derived_values()

    def _calculate_derived_values(self):
        """Calculate derived values based on current parameters."""
        # Calculate pixels per degree (simple approximation)
        # In a real implementation, this would be more precise
        # For a simple approximation: pixels = tan(angle_in_rad) * distance_in_pixels
        # For small angles (in degrees): pixels ≈ (angle/57.3) * distance_in_pixels

        # Use screen distance in cm
        distance_cm = self.parameters.get("screen_distance", 25.0)

        # For simplicity, assume 1920x1080 screen of width 50cm by default
        # but use provided resolution if available
        if "resolution" in self.parameters:
            resolution = self.parameters["resolution"]
            if isinstance(resolution, str) and "x" in resolution:
                width_px, height_px = map(int, resolution.split("x"))
            elif isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
                width_px, height_px = resolution[0], resolution[1]
            else:
                width_px, height_px = 1920, 1080
        else:
            width_px, height_px = 1920, 1080

        # Screen physical size (default 50cm width)
        width_cm = self.parameters.get("screen_width_cm", 50.0)

        # Calculate pixels per cm
        pixels_per_cm = width_px / width_cm

        # Calculate pixels per degree
        # At distance D, 1 degree spans approximately D*tan(1°) ≈ D*0.01745 cm
        cm_per_degree = distance_cm * 0.01745
        self._pixels_per_degree = pixels_per_cm * cm_per_degree

        # Calculate stimulus dimensions in pixels
        self._size_x_px = int(self.parameters["x_size"] * self._pixels_per_degree)
        self._size_y_px = int(self.parameters["y_size"] * self._pixels_per_degree)

        # Ensure dimensions are at least 1 pixel
        self._size_x_px = max(1, self._size_x_px)
        self._size_y_px = max(1, self._size_y_px)

        # Calculate position in pixels (from center)
        self._position_x_px = int(self.parameters["x_pos"] * self._pixels_per_degree)
        self._position_y_px = int(self.parameters["y_pos"] * self._pixels_per_degree)

        # Calculate total number of frames
        refresh_rate = float(self.parameters["refresh_rate"])
        self._total_frames = max(1, int(self.parameters["duration"] * refresh_rate))

        # Initialize grating texture to None
        self._grating_texture = None

    def _create_grating_texture(self, phase_offset: float = 0.0):
        """
        Create the grating texture.

        Args:
            phase_offset: Additional phase offset in degrees
        """
        # Get parameters
        sf = self.parameters["spatial_freq"]
        contrast = self.parameters["contrast"] / 100.0
        bg = self.parameters["background"] / 255.0
        orientation_deg = self.parameters["orientation"]
        phase_deg = (self.parameters["phase"] + phase_offset) % 360.0
        square = self.parameters["square_wave"] == 1

        # Convert to radians
        orientation = np.radians(orientation_deg)
        phase = np.radians(phase_deg)

        # Create coordinate grid
        x = np.linspace(-self._size_x_px / 2, self._size_x_px / 2, self._size_x_px)
        y = np.linspace(-self._size_y_px / 2, self._size_y_px / 2, self._size_y_px)
        X, Y = np.meshgrid(x, y)

        # Rotate coordinates
        X_rot = X * np.cos(orientation) - Y * np.sin(orientation)

        # Calculate grating
        # Convert spatial frequency from cycles/degree to cycles/pixel
        sf_pixels = sf / self._pixels_per_degree

        if square:
            # Square wave grating
            grating = np.sign(np.sin(2 * np.pi * sf_pixels * X_rot + phase))
        else:
            # Sinusoidal grating
            grating = np.sin(2 * np.pi * sf_pixels * X_rot + phase)

        # Apply contrast and shift to range [0, 1]
        grating = bg + (contrast * grating / 2)

        # Apply mask if needed
        mask_type = self.parameters["mask_type"]
        if mask_type != "none":
            # Distance from center for each pixel
            distance = np.sqrt(X**2 + Y**2)

            # Convert mask radius to pixels
            mask_radius_px = self.parameters["mask_radius"] * self._pixels_per_degree

            if mask_type == "circle":
                # Circle mask (1 inside radius, 0 outside)
                mask = np.where(distance <= mask_radius_px, 1.0, 0.0)

            elif mask_type == "gaussian":
                # Gaussian mask
                sigma_px = self.parameters["mask_sigma"] * self._pixels_per_degree
                mask = np.exp(-(distance**2) / (2 * sigma_px**2))
            else:
                # Default to no mask
                mask = np.ones_like(grating)

            # Apply mask
            grating = bg + (grating - bg) * mask

        # Convert to 8-bit range [0, 255]
        return (grating * 255).astype(np.uint8)

    def get_frame(self, frame_number: int) -> np.ndarray:
        """
        Get a specific frame of the stimulus.

        Args:
            frame_number: The frame number to generate

        Returns:
            np.ndarray: The stimulus frame image
        """
        # Get parameters for temporal component
        tf = self.parameters["temporal_freq"]
        reverse = self.parameters["reverse"] == 1
        refresh_rate = float(self.parameters["refresh_rate"])

        # Calculate phase for this frame
        time = frame_number / refresh_rate
        phase_offset_deg = 360.0 * tf * time  # In degrees

        if reverse:
            phase_offset_deg = -phase_offset_deg

        # Create texture with new phase
        return self._create_grating_texture(phase_offset_deg)

    def get_all_frames(self) -> np.ndarray:
        """
        Get all frames of the stimulus.

        Returns:
            np.ndarray: All stimulus frames in shape [frames, height, width]
        """
        # Initialize array for all frames - using uint8 for memory efficiency
        frames = np.zeros(
            (self._total_frames, self._size_y_px, self._size_x_px), dtype=np.uint8
        )

        # Generate each frame
        for i in range(self._total_frames):
            frames[i] = self.get_frame(i)

        return frames
