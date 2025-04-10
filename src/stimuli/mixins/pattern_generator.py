"""Pattern generation mixin for stimuli."""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class PatternGeneratorMixin:
    """Mixin class for pattern generation."""

    def _create_pattern_texture(self, phase_offset: float = 0.0) -> np.ndarray:
        """
        Create a texture for the pattern.

        Args:
            phase_offset: Phase offset in degrees (for moving patterns) or position in degrees (for bar pattern)

        Returns:
            np.ndarray: Pattern texture
        """
        # Initialize variables
        pattern_type = self.parameters.get("pattern", "checkerboard")
        grid_spacing_x = self.parameters.get("grid_spacing_x", 10.0)
        grid_spacing_y = self.parameters.get("grid_spacing_y", 10.0)
        line_width = self.parameters.get("line_width", 0.5)
        x_size = self.parameters.get("x_size", 120.0)
        y_size = self.parameters.get("y_size", 120.0)
        inverse = self.parameters.get("inverse", False)
        bar_width = self.parameters.get("bar_width", 20.0)
        transformation = self.parameters.get("transformation", "none")
        sweep_direction = self.parameters.get("sweep_direction", "left-to-right")

        # Get resolution from parameters with fallback to calculated values
        resolution = self.parameters.get("resolution")
        if resolution is None:
            # Use the calculated resolution from _calculate_derived_values
            res_x = self._size_x_px
            res_y = self._size_y_px
        else:
            res_x, res_y = resolution

        self.logger.debug(f"Creating pattern with resolution: {res_x}x{res_y}")

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(res_x), np.arange(res_y))

        # Convert to degrees
        x_deg = (x_coords - res_x / 2) / self._pixels_per_degree
        y_deg = (y_coords - res_y / 2) / self._pixels_per_degree

        # Apply coordinate transformation if needed
        if transformation != "none":
            x_deg, y_deg = self._apply_transformation(x_deg, y_deg)

        # Initialize pattern as a boolean array
        pattern = np.zeros((res_y, res_x), dtype=bool)

        # Create pattern
        if pattern_type == "checkerboard":
            # For checkerboard, phase offset is interpreted as degrees (0-360)
            phase_rad = np.deg2rad(phase_offset)
            x_idx = np.floor(
                (x_deg + phase_rad * grid_spacing_x / (2 * np.pi)) / grid_spacing_x
            )
            y_idx = np.floor(y_deg / grid_spacing_y)
            pattern = (x_idx + y_idx) % 2 == 0

        elif pattern_type == "grid":
            # For grid, phase offset is interpreted as degrees (0-360)
            phase_rad = np.deg2rad(phase_offset)
            x_mod = np.mod(
                x_deg + phase_rad * grid_spacing_x / (2 * np.pi), grid_spacing_x
            )
            y_mod = np.mod(y_deg, grid_spacing_y)
            x_lines = x_mod < line_width
            y_lines = y_mod < line_width
            pattern = x_lines | y_lines

        elif pattern_type == "bars":
            # For bars, phase_offset is interpreted as the position of the bar center in degrees
            if sweep_direction in ["left-to-right", "right-to-left"]:
                # Bar moves horizontally
                pattern = np.abs(x_deg - phase_offset) <= bar_width / 2
                self.logger.debug(
                    f"Creating horizontal bar at position {phase_offset}° with width {bar_width}°"
                )
            else:  # ["top-to-bottom", "bottom-to-top"]
                # Bar moves vertically
                pattern = np.abs(y_deg - phase_offset) <= bar_width / 2
                self.logger.debug(
                    f"Creating vertical bar at position {phase_offset}° with width {bar_width}°"
                )

        # Convert pattern to image
        image = np.zeros((res_y, res_x), dtype=np.uint8)
        image[pattern] = 255 if not inverse else 0
        image[~pattern] = 0 if not inverse else 255

        # Log pattern statistics
        self.logger.debug(
            f"Pattern statistics: min={image.min()}, max={image.max()}, mean={image.mean():.2f}"
        )

        return image
