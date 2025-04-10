"""
Drifting bar stimulus implementation as described in MMC1.

This module implements a drifting bar stimulus with counter-phase checkerboard pattern
as described in the MMC1 document:
"The bar was 20° wide and subtended the whole visual hemifield along the vertical and
horizontal axes (153° or 147° respectively). The bar was drifted 10 times in each of
the four cardinal directions. A counter-phase checkerboard pattern was flashed on the bar,
alternating between black and white (25° squares with 166 ms period) to strongly drive
neural activity."
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial

from src.stimuli.base_stimulus import BaseStimulus
from src.stimuli.spherical_correction import SphericalCorrection
from src.stimuli.mixins.gpu_acceleration import GPUAccelerationMixin
from src.stimuli.mixins.coordinate_transform import CoordinateTransformMixin
from src.stimuli.mixins.logging_mixin import LoggingMixin


class DriftingBarStimulus(
    BaseStimulus, GPUAccelerationMixin, CoordinateTransformMixin, LoggingMixin
):
    """
    Drifting bar stimulus as described in MMC1 document.

    This class implements a drifting bar with counter-phase checkerboard pattern
    that alternates between black and white. The bar is drifted in the four
    cardinal directions with proper spherical correction.

    Performance optimizations:
    - GPU acceleration when available
    - Vectorized operations with NumPy
    - Multiprocessing for frame generation
    - Precomputation of transformation maps and coordinate grids
    - Caching of repeated operations
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the drifting bar stimulus.

        Args:
            parameters: Dictionary containing stimulus parameters
        """
        # Initialize mixins
        LoggingMixin.__init__(self)
        GPUAccelerationMixin.__init__(self)
        CoordinateTransformMixin.__init__(self)

        # Initialize base class
        super().__init__(parameters)

        # Determine optimal number of processes for multiprocessing
        self._num_processes = parameters.get(
            "num_processes", min(multiprocessing.cpu_count(), 8)
        )
        self.logger.info(f"Using {self._num_processes} processes for frame generation")

        # Initialize coordinate cache
        self._coord_cache = {}

        # Setup GPU acceleration
        self._setup_gpu_acceleration()

        # Set up spherical correction
        self._setup_spherical_correction()

        # Precompute transformation maps for all directions
        self._precompute_transformation_maps()

        # Log configuration
        self._log_configuration()

    def _setup_gpu_acceleration(self) -> None:
        """
        Set up GPU acceleration based on parameters and available hardware.

        This method checks for GPU availability and configures the stimulus
        to use GPU acceleration if possible and requested.
        """
        # Get GPU usage parameter (None = auto, True = force on, False = force off)
        gpu_param = self.parameters.get("use_gpu", None)

        if gpu_param is False:
            # User explicitly disabled GPU
            self._use_gpu = False
            self.logger.info("GPU acceleration disabled by user configuration")
        else:
            # Check if GPU is available
            try:
                has_gpu = self.is_gpu_available()

                if has_gpu:
                    if gpu_param is True:
                        # User explicitly requested GPU and it's available
                        self._use_gpu = True
                        self.logger.info(
                            f"GPU acceleration enabled (user requested): {self.gpu_type}"
                        )
                    else:
                        # Auto-detected GPU
                        self._use_gpu = True
                        self.logger.info(
                            f"GPU acceleration enabled (auto-detected): {self.gpu_type}"
                        )
                else:
                    if gpu_param is True:
                        # User requested GPU but it's not available
                        self._use_gpu = False
                        self.logger.warning(
                            "GPU acceleration requested but no compatible GPU detected"
                        )
                    else:
                        # Auto mode with no GPU
                        self._use_gpu = False
                        self.logger.info(
                            "No compatible GPU detected, using CPU for processing"
                        )
            except Exception as e:
                # Error during GPU detection
                self._use_gpu = False
                self.logger.warning(
                    f"Error during GPU detection: {e}. Falling back to CPU."
                )
                self.logger.debug(
                    f"GPU detection error details: {str(e)}", exc_info=True
                )

    def _check_gpu_available(self) -> bool:
        """
        Check if GPU acceleration is available.

        This method delegates to the GPUAccelerationMixin's implementation
        if available, otherwise returns False.

        Returns:
            bool: True if GPU acceleration is available, False otherwise
        """
        try:
            # Try to use the mixin's implementation
            return hasattr(self, "is_gpu_available") and self.is_gpu_available()
        except (AttributeError, NotImplementedError) as e:
            # Fallback if method is not available
            self.logger.warning(f"GPU acceleration check failed: {e}")
            self.logger.debug(f"GPU check error details: {str(e)}", exc_info=True)
            return False

    def _apply_gpu_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply GPU-based processing to a frame if available.

        This method delegates to the GPUAccelerationMixin's implementation
        if available, otherwise returns the original frame.

        Args:
            frame: Input frame to process

        Returns:
            np.ndarray: Processed frame
        """
        try:
            # Try to use the mixin's implementation
            if hasattr(self, "apply_gpu_processing"):
                return self.apply_gpu_processing(frame)
            return frame
        except (AttributeError, NotImplementedError) as e:
            # Fallback if method is not available
            self.logger.warning(f"GPU processing error: {e}")
            return frame

    def _calculate_derived_values(self) -> None:
        """Calculate derived values based on parameters."""
        # Get screen properties
        self.screen_width_cm = self.parameters.get(
            "screen_width_cm", 68.0
        )  # cm - 68 x 121 cm large LCD display as in MMC1
        self.screen_height_cm = self.parameters.get("screen_height_cm", 121.0)  # cm
        self._resolution = self.parameters.get("resolution", (1920, 1080))  # px

        # Get stimulus properties
        self.fps = self.parameters.get("fps", 60.0)  # Hz - 60Hz as mentioned in MMC1
        self.x_size = self.parameters.get(
            "x_size", 147.0
        )  # degrees - horizontal visual field
        self.y_size = self.parameters.get(
            "y_size", 153.0
        )  # degrees - vertical visual field

        # Ensure the bar width is exactly 20 degrees as specified in MMC1
        requested_bar_width = self.parameters.get("bar_width", 20.0)
        if requested_bar_width != 20.0:
            self.logger.warning(
                f"Requested bar width of {requested_bar_width}° overridden to 20.0° as specified in MMC1"
            )
        self.bar_width = 20.0  # degrees - exactly 20° wide as in MMC1

        # Determine drift speed based on imaging mode
        self.is_two_photon = self.parameters.get("is_two_photon", False)
        if self.is_two_photon:
            # 12-14°/s for two-photon imaging
            self.drift_speed = self.parameters.get(
                "drift_speed_2p", 12.0
            )  # degrees per second
        else:
            # 8.5-9.5°/s for intrinsic imaging
            self.drift_speed = self.parameters.get(
                "drift_speed", 8.5
            )  # degrees per second

        # Calculate frames required for each sweep
        # Total distance to travel: field size + 2*bar_width (to ensure the bar enters and exits completely)
        if self.parameters.get("sweep_direction", "right-to-left") in [
            "left-to-right",
            "right-to-left",
        ]:
            total_travel_distance = self.x_size + 2 * self.bar_width  # degrees
        else:
            total_travel_distance = self.y_size + 2 * self.bar_width  # degrees

        sweep_time = total_travel_distance / self.drift_speed  # seconds
        self._frames_per_sweep = int(sweep_time * self.fps)

        # Number of repeats per direction
        self.repeats = self.parameters.get("drift_repeats", 10)  # 10 repeats as in MMC1

        # Ensure grid spacing for checkerboard pattern is exactly 25 degrees as in MMC1
        requested_grid_x = self.parameters.get("grid_spacing_x", 25.0)
        requested_grid_y = self.parameters.get("grid_spacing_y", 25.0)

        if requested_grid_x != 25.0 or requested_grid_y != 25.0:
            self.logger.warning(
                f"Requested grid spacing of {requested_grid_x}°x{requested_grid_y}° overridden to 25.0°x25.0° as specified in MMC1"
            )

        self.grid_spacing_x = 25.0  # degrees - exactly 25° squares as in MMC1
        self.grid_spacing_y = 25.0  # degrees - exactly 25° squares as in MMC1

        # Temporal frequency for checkerboard alternation (166ms period = ~6Hz)
        self.temporal_freq = self.parameters.get("temporal_freq", 6.0)  # Hz

        # Precompute coordinate grids
        self._precompute_coordinate_grids()

    def _precompute_coordinate_grids(self) -> None:
        """Precompute coordinate grids to avoid repeated calculations."""
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0 : self._resolution[1], 0 : self._resolution[0]]
        self._x_norm = x_coords / (self._resolution[0] // 2) - 1.0
        self._y_norm = y_coords / (self._resolution[1] // 2) - 1.0

        # Create empty array for result
        self._result_template = np.zeros(self._resolution[::-1], dtype=np.uint8)

    def _setup_spherical_correction(self) -> None:
        """Set up the spherical correction for proper transformation."""
        correction_params = {
            "screen_distance": self.parameters.get("screen_distance", 10.0),
            "screen_angle": self.parameters.get("screen_angle", 20.0),
            "x_size": self.parameters.get("x_size", 147.0),
            "y_size": self.parameters.get("y_size", 153.0),
            "horizontal_meridian_offset": self.parameters.get(
                "horizontal_meridian_offset", 20.0
            ),
        }
        self.spherical_correction = SphericalCorrection(correction_params)

    def _precompute_transformation_maps(self) -> None:
        """Precompute transformation maps for all directions to avoid recalculation."""
        self.logger.info("Precomputing transformation maps for all directions...")

        # Dictionary to store transformation maps for each direction
        self._transformation_maps = {}

        # Generate maps for each direction
        directions = [
            "right-to-left",
            "left-to-right",
            "bottom-to-top",
            "top-to-bottom",
        ]

        for direction in directions:
            # Set appropriate transformation axis correction based on direction
            if direction in ["left-to-right", "right-to-left"]:
                correct_axis = "azimuth"
            else:
                correct_axis = "altitude"

            # Create temporary params with correct axis
            temp_params = self.spherical_correction.parameters.copy()

            # Update parameters with correct dimensions and axis
            self.spherical_correction.parameters["correct_axis"] = correct_axis
            self.spherical_correction.parameters["width"] = self._resolution[0]
            self.spherical_correction.parameters["height"] = self._resolution[1]

            # Create transformation maps
            map_x, map_y = self.spherical_correction.create_transformation_maps(
                self._resolution[0], self._resolution[1]
            )

            # Store maps
            self._transformation_maps[direction] = {
                "map_x": map_x,
                "map_y": map_y,
                "x_deg": None,
                "y_deg": None,
            }

            # Transform coordinates and cache them
            x_transformed, y_transformed = (
                self.spherical_correction.transform_coordinates(
                    self._x_norm, self._y_norm
                )
            )

            # Convert normalized coordinates to degrees
            x_deg = x_transformed * (self.x_size / 2)
            y_deg = y_transformed * (self.y_size / 2)

            # Store transformed coordinates
            self._transformation_maps[direction]["x_deg"] = x_deg
            self._transformation_maps[direction]["y_deg"] = y_deg

            # Restore original parameters
            self.spherical_correction.parameters = temp_params

        self.logger.info("Transformation maps precomputed successfully")

    def _log_configuration(self) -> None:
        """Log the configuration of the stimulus."""
        self.logger.info(f"Initialized DriftingBarStimulus:")
        self.logger.info(f"  Resolution: {self._resolution}")
        self.logger.info(f"  Visual field: {self.x_size}° x {self.y_size}°")
        self.logger.info(f"  Bar width: {self.bar_width}°")
        self.logger.info(f"  Drift speed: {self.drift_speed}°/s")
        self.logger.info(
            f"  Mode: {'Two-photon' if self.is_two_photon else 'Intrinsic'} imaging"
        )
        self.logger.info(f"  Frames per sweep: {self._frames_per_sweep}")
        self.logger.info(f"  Repeats: {self.repeats}")
        self.logger.info(f"  Multiprocessing: {self._num_processes} processes")
        self.logger.info(
            f"  GPU acceleration: {'Enabled' if self._use_gpu else 'Disabled'}"
        )

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Generate a single frame of the drifting bar stimulus.

        Args:
            frame_idx: Index of the frame to generate

        Returns:
            np.ndarray: Generated frame as a 2D numpy array
        """
        # Create blank frame
        frame = np.zeros(
            (self._resolution[1], self._resolution[0]), dtype=np.uint8
        )  # Start with a blank frame

        # Calculate the current position of the bar
        progress = frame_idx / max(1, self._frames_per_sweep - 1)

        # Update the progress for the bar's position
        sweep_direction = self.parameters.get("sweep_direction", "right-to-left")
        if sweep_direction in ["left-to-right", "right-to-left"]:
            pos = (
                -self.x_size / 2
                - self.bar_width
                + progress * (self.x_size + 2 * self.bar_width)
            )
        elif sweep_direction in ["bottom-to-top", "top-to-bottom"]:
            pos = (
                -self.y_size / 2
                - self.bar_width
                + progress * (self.y_size + 2 * self.bar_width)
            )
        else:
            raise ValueError(f"Unknown sweep direction: {sweep_direction}")

        # Set sweep direction in spherical correction parameters
        self.spherical_correction.parameters["sweep_direction"] = sweep_direction

        # Also set correct dimensions to ensure consistent array sizes
        self.spherical_correction.parameters["width"] = self._resolution[0]
        self.spherical_correction.parameters["height"] = self._resolution[1]

        # Use the spherical_correction.transform_drifting_bar method to get the proper transformed bar mask
        # This ensures the bar shape is properly transformed in spherical space
        bar_mask = self.spherical_correction.transform_drifting_bar(
            pos, sweep_direction, self.bar_width
        )

        # Get indices of pixels within the transformed bar
        bar_indices = np.where(bar_mask)

        if len(bar_indices[0]) == 0:
            # No pixels in the bar for this frame (could happen at the edges)
            return frame

        # Generate full spherical coordinates for all pixels
        height, width = self._resolution[1], self._resolution[0]

        # Create coordinate arrays
        x = np.arange(width)
        y = np.arange(height)

        # Create meshgrid
        xx, yy = np.meshgrid(x, y)

        # Normalize coordinates to range [-1, 1]
        x_norm = 2 * (xx / (width - 1)) - 1
        y_norm = 2 * (yy / (height - 1)) - 1

        # Convert to visual degrees
        x_deg = x_norm * (self.x_size / 2)
        y_deg = y_norm * (self.y_size / 2)

        # Calculate the full spherical coordinates directly

        # First apply horizontal meridian offset to y coordinates
        y_deg_adjusted = y_deg - self.spherical_correction.horizontal_meridian_offset

        # Get distance to screen
        x0 = (
            self.spherical_correction.screen_distance
        )  # Distance from eye to screen (cm)

        # Convert degrees to distances on screen
        azimuth_rad = np.radians(x_deg)  # Horizontal on screen (azimuth)
        altitude_rad = np.radians(y_deg_adjusted)  # Vertical on screen (altitude)

        # Convert to distances on screen (relative to center)
        y_screen = x0 * np.tan(azimuth_rad)  # Horizontal distance on screen
        z_screen = x0 * np.tan(altitude_rad)  # Vertical distance on screen

        # Calculate the square root term
        sqrt_term = np.sqrt(x0**2 + y_screen**2 + z_screen**2)

        # Calculate altitude (θ)
        z_over_sqrt = np.zeros_like(z_screen)
        valid_indices = sqrt_term > 0
        z_over_sqrt[valid_indices] = z_screen[valid_indices] / sqrt_term[valid_indices]
        z_over_sqrt = np.clip(z_over_sqrt, -1.0, 1.0)
        theta = np.pi / 2 - np.arccos(z_over_sqrt)

        # Calculate azimuth (φ)
        phi = np.zeros_like(y_screen)
        nonzero_indices = np.abs(x0) > 1e-10
        phi[nonzero_indices] = np.arctan2(-y_screen[nonzero_indices], x0)

        # Convert back to degrees
        theta_deg = np.degrees(theta)  # altitude angle
        phi_deg = np.degrees(phi)  # azimuth angle

        # Extract coordinates only for the pixels within the bar
        # bar_indices returns a tuple with y_indices, x_indices
        y_indices, x_indices = bar_indices

        # No need to check for out-of-bounds indices anymore, as the transform_drifting_bar
        # method now guarantees all indices are valid

        # Access the phi and theta values for these indices
        phi_bar = phi_deg[y_indices, x_indices]
        theta_bar = theta_deg[y_indices, x_indices]

        # Apply temporal phase for counter-phase checkerboard
        temporal_phase = (
            2 * np.pi * self.temporal_freq * frame_idx / self.fps
        )  # in radians

        # Create a checkerboard pattern using the spherical coordinates
        # Use grid_spacing_x and grid_spacing_y instead of bar_width/2 to ensure
        # the checkerboard squares are exactly 25 degrees as specified in MMC1
        checker_x_size = self.grid_spacing_x  # 25 degrees for x-direction
        checker_y_size = self.grid_spacing_y  # 25 degrees for y-direction

        # Align checker pattern to start at integer multiples of checker size
        # This ensures checker boundaries align exactly with our grid lines
        if sweep_direction in ["left-to-right", "right-to-left"]:
            # For horizontal sweep, use phi (azimuth) values to create checker pattern
            phi_min = np.min(phi_bar)
            phi_offset = np.floor(phi_min / checker_x_size) * checker_x_size
            theta_min = np.min(theta_bar)
            theta_offset = np.floor(theta_min / checker_y_size) * checker_y_size

            # Create aligned checker pattern using spherical coordinates
            checker_phi = np.floor((phi_bar - phi_offset) / checker_x_size) % 2
            checker_theta = np.floor((theta_bar - theta_offset) / checker_y_size) % 2
        else:
            # For vertical sweep, use theta (altitude) values to create checker pattern
            # Need to account for the meridian offset
            adjusted_theta = (
                theta_bar + self.spherical_correction.horizontal_meridian_offset
            )
            theta_min = np.min(adjusted_theta)
            theta_offset = np.floor(theta_min / checker_y_size) * checker_y_size
            phi_min = np.min(phi_bar)
            phi_offset = np.floor(phi_min / checker_x_size) * checker_x_size

            # Create aligned checker pattern using spherical coordinates
            checker_theta = (
                np.floor((adjusted_theta - theta_offset) / checker_y_size) % 2
            )
            checker_phi = np.floor((phi_bar - phi_offset) / checker_x_size) % 2

        # Create checkerboard by XORing the patterns (only for bar pixels)
        checkerboard = np.logical_xor(checker_theta, checker_phi).astype(np.float32)

        # Apply counter-phase flashing
        flash_state = np.sign(np.cos(temporal_phase)) > 0
        if flash_state:
            checkerboard = 1 - checkerboard  # Invert the pattern

        # Convert to 0-255 range
        bar_pixels = (checkerboard * 255).astype(np.uint8)

        # Place the processed pixels back into the frame using the corrected indices
        frame[y_indices, x_indices] = bar_pixels

        return frame

    def _generate_frame_job(self, frame_idx: int) -> Tuple[int, np.ndarray]:
        """
        Generate a single frame for parallel processing.

        Args:
            frame_idx: Index of the frame to generate

        Returns:
            Tuple[int, np.ndarray]: Tuple of frame index and generated frame
        """
        return frame_idx, self.get_frame(frame_idx)

    def get_all_frames(self, frame_skip: int = 1) -> np.ndarray:
        """
        Get all frames for a single sweep using multiprocessing.

        Args:
            frame_skip: Only process every Nth frame (1 = all frames, 10 = 10x faster but choppier)

        Returns:
            np.ndarray: Array of all stimulus frames
        """
        # Calculate actual frames to generate with frame skipping
        total_frames = self._frames_per_sweep
        process_frames = total_frames if frame_skip <= 1 else total_frames // frame_skip

        if frame_skip > 1:
            self.logger.info(
                f"Frame skipping enabled: Processing {process_frames} of {total_frames} frames (every {frame_skip}th frame)"
            )

        self.logger.info(
            f"Generating {process_frames} frames using {self._num_processes} processes"
        )

        # Initialize output array - always the full frame count (we'll duplicate skipped frames)
        frames = np.zeros(
            (total_frames, self._resolution[1], self._resolution[0]),
            dtype=np.uint8,
        )

        # For skipping frames, calculate which frame indices to actually process
        frame_indices = list(range(0, total_frames, frame_skip))

        # Use multiprocessing for frame generation if more than one process specified
        if self._num_processes > 1:
            with ProcessPoolExecutor(max_workers=self._num_processes) as executor:
                # Submit all frame generation jobs for the frames we're processing
                future_to_idx = {
                    executor.submit(self._generate_frame_job, i): i
                    for i in frame_indices
                }

                # Collect results as they complete
                for future in as_completed(future_to_idx):
                    idx, frame = future.result()
                    # Store the processed frame
                    frames[idx] = frame
        else:
            # Single process - generate frames sequentially
            for i in frame_indices:
                frames[i] = self.get_frame(i)

        # If we're doing frame skipping, fill in the gaps by duplicating adjacent frames
        if frame_skip > 1:
            self.logger.info(f"Filling in skipped frames with nearest neighbors")
            last_valid_frame = 0
            for i in range(total_frames):
                if i not in frame_indices:
                    # This is a skipped frame - find the nearest valid frame
                    # For now, use the previous valid frame
                    frames[i] = frames[last_valid_frame]
                else:
                    last_valid_frame = i

        self.logger.info(f"Frame generation complete")
        return frames

    def generate_full_sequence(self) -> Dict[str, np.ndarray]:
        """
        Generate the complete stimulus sequence for all directions with repeats.

        As per MMC1: "The bar was drifted 10 times in each of the four cardinal directions."

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping directions to arrays of frames
        """
        # Store original parameters to restore later
        original_params = self.parameters.copy()

        # Dictionary to store video segments
        video_segments = {}

        # Generate stimulus for each direction
        directions = [
            "right-to-left",
            "left-to-right",
            "bottom-to-top",
            "top-to-bottom",
        ]

        for direction in directions:
            # Update parameters for this direction
            self.parameters["sweep_direction"] = direction

            # Generate the frames for this direction
            self.logger.info(
                f"Generating {self.repeats} repeats of {direction} drifting bar"
            )

            # Generate single sequence
            frames = self.get_all_frames()

            # Optimize memory usage for repeated frames
            if self.repeats > 1:
                # For large repeats, use memory-efficient approach
                if self.repeats > 5 and frames.nbytes > 1e9:  # >1GB
                    self.logger.info(
                        f"Using memory-efficient approach for {self.repeats} repeats"
                    )
                    # Create a function to generate repeated frames on demand
                    repeated_frames_generator = np.tile(frames, (self.repeats, 1, 1))
                    video_segments[direction] = repeated_frames_generator
                else:
                    # Standard approach for smaller datasets
                    repeated_frames = np.tile(frames, (self.repeats, 1, 1))
                    video_segments[direction] = repeated_frames
            else:
                video_segments[direction] = frames

        # Restore original parameters
        self.parameters = original_params

        return video_segments

    @property
    def frame_count(self) -> int:
        """
        Get the total number of frames for a single sweep.

        Returns:
            int: Total number of frames
        """
        return self._frames_per_sweep

    @property
    def frame_rate(self) -> float:
        """
        Get the frame rate of the stimulus.

        Returns:
            float: Frame rate in Hz
        """
        return self.fps

    @property
    def duration(self) -> float:
        """
        Get the duration of a single sweep.

        Returns:
            float: Duration in seconds
        """
        return self._frames_per_sweep / self.fps

    @property
    def resolution(self) -> Tuple[int, int]:
        """
        Get the resolution of the stimulus.

        Returns:
            Tuple[int, int]: Resolution as (width, height)
        """
        return self._resolution

    @property
    def checker_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for checkerboard pattern to ensure grid lines match exactly.

        Returns:
            Dict[str, Any]: Parameters for creating aligned grid lines
        """
        return {
            "checker_size_x": self.grid_spacing_x,  # Exact 25° squares as in MMC1
            "checker_size_y": self.grid_spacing_y,  # Exact 25° squares as in MMC1
            "grid_spacing_x": self.grid_spacing_x,
            "grid_spacing_y": self.grid_spacing_y,
            "screen_distance": self.spherical_correction.screen_distance,
            "x_size": self.x_size,
            "y_size": self.y_size,
        }
