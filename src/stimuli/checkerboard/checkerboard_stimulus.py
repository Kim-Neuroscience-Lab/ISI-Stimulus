# src/stimuli/checkerboard/checkerboard_stimulus.py
"""
Checkerboard pattern stimulus module.

This module implements the drifting bar and drifting grating stimuli as described
in the MMC1 document, with appropriate spherical corrections.
"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional, Callable
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.log_config import setup_logging
from src.stimuli.base_stimulus import BaseStimulus
from src.stimuli.mixins.gpu_acceleration import GPUAccelerationMixin
from src.stimuli.mixins.coordinate_transform import CoordinateTransformMixin
from src.stimuli.mixins.pattern_generator import PatternGeneratorMixin
from src.stimuli.mixins.logging_mixin import LoggingMixin


def register_module(parameter_config):
    """
    Register this module with the parameter configuration.

    Args:
        parameter_config: The parameter configuration object
    """
    # Register CHECKERBOARD as a module type
    parameter_config.register_module_factory("CHECKERBOARD", create_default_parameters)


def create_default_parameters() -> Dict[str, Any]:
    """
    Create a set of default parameters for the checkerboard stimulus based on MMC1 specifications.
    """
    return {
        # Screen setup parameters as described in MMC1
        "screen_distance": 10.0,  # cm from eye to screen
        "screen_width_cm": 68.0,  # cm - 68 x 121 cm large LCD display as in MMC1
        "screen_height_cm": 121.0,  # cm
        "resolution": (1920, 1080),  # px - adjust as needed for actual display
        "downscale": 1,  # downscale factor for testing
        # Visual field parameters from MMC1
        "x_size": 147.0,  # degrees - horizontal visual field
        "y_size": 153.0,  # degrees - vertical visual field
        # Timing parameters
        "duration": 10.0,  # seconds
        "fps": 60.0,  # frames per second - MMC1 mentions 60Hz stimulus
        # Stimulus parameters
        "pattern": "checkerboard",  # "checkerboard" or "bars"
        "temporal_freq": 6.0,  # Hz - period mentioned in MMC1 is 166 ms (6Hz)
        "transformation": "spherical",  # "spherical" or "hyperbolic" or None
        # Checkerboard pattern parameters - from MMC1
        "grid_spacing_x": 25.0,  # degrees - 25° squares mentioned in MMC1
        "grid_spacing_y": 25.0,  # degrees - 25° squares mentioned in MMC1
        "progressive": False,  # whether to fade in the pattern
        # Bar stimulus parameters - from MMC1
        "bar_width": 20.0,  # degrees - 20° wide as in MMC1
        "drift_speed": 8.5,  # degrees per second - 8.5-9.5°/s for intrinsic imaging
        "drift_speed_2p": 12.0,  # degrees per second - 12-14°/s for two-photon imaging
        "drift_directions": [
            "right-to-left",
            "left-to-right",
            "bottom-to-top",
            "top-to-bottom",
        ],  # All 4 cardinal directions as in MMC1
        "drift_repeats": 10,  # MMC1 specifies 10 times in each direction
        # Screen positioning parameters from MMC1
        "screen_angle": 20.0,  # degrees - screen placed at 20° angle relative to animal
        "horizontal_meridian_offset": 20.0,  # degrees - MMC1 estimates horizontal meridian at ~20° altitude
        # Transformation parameters
        "eye_radius": 1.75,  # mm - mouse eye radius
        "correction_strength": 1.0,  # strength of spherical correction (0.0-1.0)
        # Sweep direction
        "sweep_direction": "right-to-left",  # Default direction, will be overridden for different directions
        # Output parameters
        "output": "output",  # output directory
    }


# Strategy pattern for different pattern types
class PatternStrategy:
    """Base class for pattern generation strategies."""

    def generate(
        self,
        frame_idx: int,
        total_frames: int,
        parameters: Dict[str, Any],
        transform_func: Callable,
    ) -> np.ndarray:
        """
        Generate a pattern frame.

        Args:
            frame_idx: Current frame index
            total_frames: Total number of frames
            parameters: Stimulus parameters
            transform_func: Coordinate transformation function

        Returns:
            np.ndarray: Generated frame
        """
        raise NotImplementedError("Subclasses must implement this method")


class CheckerboardPatternStrategy(PatternStrategy):
    """Strategy for generating checkerboard patterns."""

    def generate(
        self,
        frame_idx: int,
        total_frames: int,
        parameters: Dict[str, Any],
        transform_func: Callable,
    ) -> np.ndarray:
        """Generate a checkerboard pattern frame."""
        # Calculate phase (0-360 degrees)
        if total_frames > 1:
            phase = (frame_idx / total_frames) * 360.0
        else:
            phase = 0

        # Calculate progress for progressive patterns
        progress = frame_idx / max(1, total_frames - 1)

        # Get pattern dimensions
        width_px, height_px = parameters.get("_effective_resolution", (1920, 1080))

        # Create coordinate grids
        y, x = np.mgrid[0:height_px, 0:width_px]

        # Center coordinates (origin at center)
        x = x - width_px // 2
        y = y - height_px // 2

        # Normalize coordinates to range [-1, 1]
        x_norm = x / (width_px // 2)
        y_norm = y / (height_px // 2)

        # Apply coordinate transformation if specified
        if callable(transform_func):
            x_norm, y_norm = transform_func(x_norm, y_norm)

        # Convert normalized coordinates to degrees
        pixels_per_degree = parameters.get("_pixels_per_degree", 10.0)
        x_deg = x_norm * (width_px // 2) / pixels_per_degree
        y_deg = y_norm * (height_px // 2) / pixels_per_degree

        # Calculate grid cells based on specified spacing
        grid_spacing_x = parameters.get("grid_spacing_x", 25.0)
        grid_spacing_y = parameters.get("grid_spacing_y", 25.0)

        cell_x = np.floor(x_deg / grid_spacing_x)
        cell_y = np.floor(y_deg / grid_spacing_y)

        # Create alternating pattern with time-based phase
        temporal_freq = parameters.get("temporal_freq", 6.0)  # Hz
        frame_rate = parameters.get("fps", 60.0)  # frames per second

        # Calculate phase offset based on frame index
        phase_offset = (frame_idx * temporal_freq / frame_rate) % 1.0

        # Create checkerboard pattern with alternating phase
        checkerboard = np.mod(cell_x + cell_y + np.floor(phase_offset * 2), 2)

        # Handle progressive appearance if enabled
        if parameters.get("progressive", False):
            # Get sweep direction
            sweep_direction = parameters.get("sweep_direction", "right-to-left")

            # Create sweep mask
            mask = np.zeros_like(checkerboard)

            if sweep_direction == "left-to-right":
                sweep_pos = -1.0 + progress * 2.0  # From -1 to 1
                mask = np.where(x_norm <= sweep_pos, 1.0, 0.0)
            elif sweep_direction == "right-to-left":
                sweep_pos = 1.0 - progress * 2.0  # From 1 to -1
                mask = np.where(x_norm >= sweep_pos, 1.0, 0.0)
            elif sweep_direction == "top-to-bottom":
                sweep_pos = -1.0 + progress * 2.0  # From -1 to 1
                mask = np.where(y_norm <= sweep_pos, 1.0, 0.0)
            elif sweep_direction == "bottom-to-top":
                sweep_pos = 1.0 - progress * 2.0  # From 1 to -1
                mask = np.where(y_norm >= sweep_pos, 1.0, 0.0)

            # Apply mask to checkerboard
            checkerboard = checkerboard * mask

        # Convert to 8-bit grayscale image
        result = (checkerboard * 255).astype(np.uint8)

        return result


class DriftingBarPatternStrategy(PatternStrategy):
    """Strategy for generating drifting bar patterns as described in MMC1."""

    def generate(
        self,
        frame_idx: int,
        total_frames: int,
        parameters: Dict[str, Any],
        transform_func: Callable,
    ) -> np.ndarray:
        """Generate a drifting bar pattern frame."""
        # Get pattern dimensions
        width_px, height_px = parameters.get("_effective_resolution", (1920, 1080))

        # Create coordinate grids
        y, x = np.mgrid[0:height_px, 0:width_px]

        # Center coordinates (origin at center)
        x = x - width_px // 2
        y = y - height_px // 2

        # Normalize coordinates to range [-1, 1]
        x_norm = x / (width_px // 2)
        y_norm = y / (height_px // 2)

        # Apply coordinate transformation if specified
        if callable(transform_func):
            x_norm, y_norm = transform_func(x_norm, y_norm)

        # Calculate progress (0 to 1)
        progress = frame_idx / max(1, total_frames - 1)

        # Get sweep direction
        sweep_direction = parameters.get("sweep_direction", "right-to-left")

        # Convert normalized coordinates to degrees
        pixels_per_degree = parameters.get("_pixels_per_degree", 10.0)
        x_deg = x_norm * (width_px // 2) / pixels_per_degree
        y_deg = y_norm * (height_px // 2) / pixels_per_degree

        # Get bar width in degrees
        bar_width = parameters.get("bar_width", 20.0)

        # Calculate bar position based on sweep direction and progress
        if sweep_direction in ["left-to-right", "right-to-left"]:
            # Horizontal sweep
            x_size_degrees = parameters.get("x_size", 147.0)
            bar_start = -x_size_degrees / 2 - bar_width
            bar_end = x_size_degrees / 2 + bar_width
            travel_distance = bar_end - bar_start

            if sweep_direction == "left-to-right":
                bar_center = bar_start + progress * travel_distance
            else:  # right-to-left
                bar_center = bar_end - progress * travel_distance

            # Create bar mask (1 inside bar, 0 outside)
            bar_mask = np.abs(x_deg - bar_center) < (bar_width / 2)

        else:  # vertical sweep
            # Vertical sweep
            y_size_degrees = parameters.get("y_size", 153.0)
            bar_start = -y_size_degrees / 2 - bar_width
            bar_end = y_size_degrees / 2 + bar_width
            travel_distance = bar_end - bar_start

            if sweep_direction == "top-to-bottom":
                bar_center = bar_start + progress * travel_distance
            else:  # bottom-to-top
                bar_center = bar_end - progress * travel_distance

            # Create bar mask (1 inside bar, 0 outside)
            bar_mask = np.abs(y_deg - bar_center) < (bar_width / 2)

        # Create checkerboard pattern within the bar as described in MMC1
        # "A counter-phase checkerboard pattern was flashed on the bar, alternating between
        # black and white (25° squares with 166 ms period)"
        grid_spacing = parameters.get("grid_spacing_x", 25.0)

        # Create coordinate grid for checkerboard
        if sweep_direction in ["left-to-right", "right-to-left"]:
            cell_x = np.zeros_like(x_deg)  # Not used for horizontal drift
            cell_y = np.floor(y_deg / grid_spacing)
        else:
            cell_x = np.floor(x_deg / grid_spacing)
            cell_y = np.zeros_like(y_deg)  # Not used for vertical drift

        # Create temporal alternation
        # MMC1 specifies 166ms period which is approximately 6Hz
        temporal_freq = parameters.get("temporal_freq", 6.0)  # Hz
        frame_rate = parameters.get("fps", 60.0)  # frames per second

        # Calculate phase offset based on frame index
        phase_offset = (frame_idx * temporal_freq / frame_rate) % 1.0

        # Create checkerboard pattern with alternating phase
        checkerboard = np.mod(cell_x + cell_y + np.floor(phase_offset * 2), 2)

        # Apply bar mask to checkerboard
        result = (checkerboard * bar_mask * 255).astype(np.uint8)

        return result


class CheckerboardStimulus(
    BaseStimulus,
    GPUAccelerationMixin,
    CoordinateTransformMixin,
    PatternGeneratorMixin,
    LoggingMixin,
):
    """
    Checkerboard pattern stimulus generator following MMC1 specifications.

    This class generates visual stimuli for neuroscience experiments as described
    in the MMC1 document, with proper spherical correction.
    """

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the checkerboard stimulus.

        Args:
            parameters: Dictionary of stimulus parameters
        """
        # Initialize mixins
        LoggingMixin.__init__(self)
        GPUAccelerationMixin.__init__(self)
        CoordinateTransformMixin.__init__(self)

        # Initialize base class
        super().__init__(parameters)

        # Initialize pattern strategies
        self._pattern_strategies = {
            "checkerboard": CheckerboardPatternStrategy(),
            "bars": DriftingBarPatternStrategy(),
        }

        # Log configuration
        self._log_configuration()

    def _log_configuration(self):
        """Log the current configuration."""
        # Check if we're in preview mode for faster generation
        self.preview_mode = self.parameters.get("preview", False)
        if self.preview_mode:
            self.logger.info(
                "Preview mode enabled - using reduced resolution for faster generation"
            )

        # Log if transformation is enabled
        transformation = self.parameters.get("transformation", "none")
        if transformation != "none":
            self.logger.info(
                "%s transformation enabled with eye radius: %s mm, distance: %s cm, strength: %s",
                transformation.capitalize() if transformation else "Unknown",
                self.parameters.get("eye_radius", 1.75),
                self.parameters.get("screen_distance", 10.0),
                self.parameters.get("correction_strength", 1.0),
            )

        # Log stimulus type and parameters
        pattern = self.parameters.get("pattern", "checkerboard")
        self.logger.info(
            "Generating %s stimulus with %s transformation, field of view: %s x %s degrees",
            pattern,
            transformation,
            self.parameters.get("x_size", 147.0),
            self.parameters.get("y_size", 153.0),
        )

    def _calculate_derived_values(self) -> None:
        """Calculate derived values based on parameters."""
        # Get resolution and apply downscale first for efficiency
        resolution = self.parameters.get("resolution")
        downscale = self.parameters.get("downscale", 1)

        # Parse and validate resolution
        if resolution is None:
            width_px, height_px = 1920, 1080
        elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            width_px, height_px = resolution
        elif isinstance(resolution, str) and "x" in resolution:
            try:
                width_px, height_px = map(int, resolution.split("x"))
            except ValueError:
                self.logger.warning(
                    f"Invalid resolution string format: {resolution}, using default 1920x1080"
                )
                width_px, height_px = 1920, 1080
        else:
            # If resolution is not in the expected format, use default values
            self.logger.warning(
                f"Invalid resolution format: {resolution}, using default 1920x1080"
            )
            width_px, height_px = 1920, 1080

        # Apply downscale factor immediately to reduce computation
        if downscale > 1:
            # Round dimensions to ensure even division
            width_px = width_px // downscale
            height_px = height_px // downscale
            self.logger.info(
                f"Downscaled resolution to {width_px}x{height_px} (factor {downscale})"
            )

        # Get physical parameters
        distance_cm = self.parameters.get("screen_distance", 10.0)
        width_cm = self.parameters.get("screen_width_cm", 68.0)

        # Calculate pixels per cm
        pixels_per_cm = width_px / width_cm

        # Calculate pixels per degree
        # From MMC1: r = d×tan(θ) where d is distance and θ is angle in radians
        cm_per_degree = distance_cm * np.tan(np.radians(1.0))
        self._pixels_per_degree = pixels_per_cm * cm_per_degree

        # Store pixels_per_degree in parameters for use by pattern strategies
        self.parameters["_pixels_per_degree"] = self._pixels_per_degree

        # Calculate stimulus dimensions in pixels
        self._size_x_px = int(self.parameters["x_size"] * self._pixels_per_degree)
        self._size_y_px = int(self.parameters["y_size"] * self._pixels_per_degree)

        # Ensure dimensions are at least 1 pixel
        self._size_x_px = max(1, self._size_x_px)
        self._size_y_px = max(1, self._size_y_px)

        # Calculate total number of frames
        fps = float(self.parameters.get("fps", 60.0))
        self._total_frames = max(1, int(self.parameters["duration"] * fps))

        # Store the processed resolution for reference
        self._effective_resolution = (width_px, height_px)
        self.parameters["_effective_resolution"] = self._effective_resolution

    @property
    def frame_count(self) -> int:
        """Get the total number of frames."""
        return self._total_frames

    @property
    def frame_rate(self) -> float:
        """Get the frame rate of the stimulus."""
        return float(self.parameters.get("fps", 60.0))

    @property
    def duration(self) -> float:
        """Get the duration of the stimulus."""
        return float(self.parameters.get("duration", 10.0))

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the resolution of the stimulus."""
        return (self._size_x_px, self._size_y_px)

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a specific frame by index.

        Args:
            frame_idx: Frame index

        Returns:
            np.ndarray: The frame at the specified index
        """
        # Select the appropriate pattern strategy
        pattern_type = self.parameters.get("pattern", "checkerboard")
        strategy = self._pattern_strategies.get(
            pattern_type, self._pattern_strategies["checkerboard"]
        )

        # Generate the frame using the selected strategy
        frame = strategy.generate(
            frame_idx,
            self._total_frames,
            self.parameters,
            lambda x, y: self._apply_transformation(x, y),
        )

        # Convert to 3-channel if not already
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return frame

    def get_all_frames(self) -> np.ndarray:
        """
        Generate all frames for the stimulus.

        Returns:
            np.ndarray: Array of all frames, shape (frames, height, width, channels)
        """
        # Determine processing strategy (parallel vs sequential)
        use_parallel = self._should_use_parallel_processing()

        # Generate all frames
        if use_parallel:
            return self._generate_frames_parallel()
        else:
            return self._generate_frames_sequential()

    def _should_use_parallel_processing(self) -> bool:
        """Determine if parallel processing should be used."""
        # Always use parallel in non-preview mode unless explicitly disabled
        if not self.preview_mode:
            return not self.parameters.get("disable_parallel", False)

        # In preview mode, only use parallel for high resolution/long duration
        resolution = self._effective_resolution
        frame_count = self._total_frames

        # Heuristic: use parallel if generating more than 100 frames at high resolution
        return (
            frame_count > 100
            and resolution[0] * resolution[1] > 640 * 480
            and not self.parameters.get("disable_parallel", False)
        )

    def _generate_frames_parallel(self) -> np.ndarray:
        """Generate frames using parallel processing."""
        # Determine number of frames
        frame_count = self._total_frames

        # Set up parallel processing
        cpu_count = multiprocessing.cpu_count()
        worker_count = max(
            1, min(cpu_count - 1, 8)
        )  # Use at most 8 cores, leave 1 for system

        # Determine batch size based on frame count
        batch_size = max(1, frame_count // worker_count)

        # Create batches
        batches = []
        for i in range(0, frame_count, batch_size):
            batch_end = min(i + batch_size, frame_count)
            batches.append((i, batch_end))

        # Process batches in parallel
        all_frames = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for batch_start, batch_end in batches:
                future = executor.submit(
                    self._process_frame_batch, batch_start, batch_end
                )
                futures.append(future)

            # Gather results as they complete
            for future in as_completed(futures):
                batch_frames = future.result()
                all_frames.extend(batch_frames)

        # Convert to numpy array and ensure correct order
        all_frames = sorted(all_frames, key=lambda x: x[0])
        frames = np.array([frame for _, frame in all_frames])

        return frames

    def _generate_frames_sequential(self) -> np.ndarray:
        """Generate frames sequentially."""
        # Determine number of frames
        frame_count = self._total_frames

        # Generate each frame
        frames = []
        for i in range(frame_count):
            frame = self.get_frame(i)
            frames.append(frame)

        return np.array(frames)

    def _process_frame_batch(
        self, batch_start: int, batch_end: int
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Process a batch of frames.

        Args:
            batch_start: Starting frame index
            batch_end: Ending frame index (exclusive)

        Returns:
            List[Tuple[int, np.ndarray]]: List of (frame_idx, frame) pairs
        """
        batch_frames = []
        for i in range(batch_start, batch_end):
            frame = self.get_frame(i)
            batch_frames.append((i, frame))

        return batch_frames

    def generate_stimulus_sequence(self) -> List[np.ndarray]:
        """
        Generate the complete stimulus sequence as described in MMC1.

        As per MMC1: "The bar was drifted 10 times in each of the four cardinal directions."

        Returns:
            List[np.ndarray]: List of video segments for each direction
        """
        # Store original parameters to restore later
        original_params = self.parameters.copy()

        # List to store video segments
        video_segments = []

        # Generate stimulus for each direction
        directions = self.parameters.get(
            "drift_directions",
            ["right-to-left", "left-to-right", "bottom-to-top", "top-to-bottom"],
        )
        repeats = self.parameters.get("drift_repeats", 10)

        for direction in directions:
            # Update parameters for this direction
            self.parameters["sweep_direction"] = direction
            self.parameters["pattern"] = "bars"

            # Set appropriate transformation axis correction based on direction
            if direction in ["left-to-right", "right-to-left"]:
                self.parameters["correct_axis"] = "azimuth"
            else:
                self.parameters["correct_axis"] = "altitude"

            # Generate the frames for this direction
            self.logger.info(
                f"Generating {repeats} repeats of {direction} drifting bar"
            )

            # Generate single sequence
            frames = self.get_all_frames()

            # Repeat the sequence as specified
            repeated_frames = np.tile(frames, (repeats, 1, 1, 1))

            # Add to list of segments
            video_segments.append(repeated_frames)

        # Restore original parameters
        self.parameters = original_params

        return video_segments

    def _apply_transformation(
        self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply coordinate transformation based on the selected type.

        This implements the spherical coordinate system as described in the paper:
        - The eye is the origin of the coordinate system
        - The perpendicular bisector from the eye to the monitor is the (0°,0°) axis
        - Altitude (θ) increases above the origin
        - Azimuth (φ) increases anterior to the origin
        - The screen is placed at an angle of 20° relative to the animal's dorsoventral axis
        - The horizontal meridian is approximated to be at 20° altitude
        - The screen is 10cm from the eye

        For spherical coordinates:
        θ = π/2 - cos⁻¹(z/√(x²+y²+z²))  (altitude)
        φ = tan⁻¹(-y/x₀)                (azimuth)

        Args:
            x: X coordinates array in degrees
            y: Y coordinates array in degrees
            z: Z coordinates (optional, will be calculated if None)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed coordinates
        """
        transformation = self.parameters.get("transformation", "none")
        if transformation == "none":
            return x, y

        # Get transformation parameters
        eye_radius = self.parameters.get("eye_radius", 0.17)  # cm
        eye_distance = self.parameters.get(
            "screen_distance", 10.0
        )  # cm (distance from eye to screen)
        correction_strength = self.parameters.get("correction_strength", 0.1)

        # Get screen placement parameters
        screen_angle_h = np.radians(
            self.parameters.get("screen_angle_horizontal", 20.0)
        )
        screen_angle_v = np.radians(self.parameters.get("screen_angle_vertical", 20.0))
        horizontal_meridian_offset = self.parameters.get(
            "horizontal_meridian_offset", 20.0
        )

        # Get sweep direction to determine which coordinate to transform
        sweep_direction = self.parameters.get("sweep_direction", "left-to-right")

        # Initialize transformed coordinates
        x_transformed = np.copy(x)
        y_transformed = np.copy(y)

        # Apply meridian offset to account for eye rotation in skull
        # The horizontal meridian is approximated to be at 20° altitude
        y_adjusted = y - horizontal_meridian_offset

        # Convert degrees to distance on screen (cm)
        # Using small angle approximation: tan(θ) ≈ θ for small θ in radians
        # Convert degrees to radians first
        x_rad = np.radians(x)
        y_rad = np.radians(y_adjusted)

        # Apply screen tilt correction (screen is angled 20° inward toward nose)
        # This affects the x-coordinate mapping
        x_corrected = x_rad * np.cos(screen_angle_h)

        # Apply screen vertical tilt correction
        # This affects the y-coordinate mapping
        y_corrected = y_rad * np.cos(screen_angle_v)

        # Convert to distances on screen (cm)
        x_cm = np.tan(x_corrected) * eye_distance
        y_cm = np.tan(y_corrected) * eye_distance

        # Default z coordinate is the screen distance (constant for flat screen)
        if z is None:
            # The z varies across the screen due to the tilt
            z_cm = np.ones_like(x) * eye_distance
            # Apply screen angle adjustments to z
            x_offset = np.tan(x_corrected) * eye_distance
            y_offset = np.tan(y_corrected) * eye_distance
            # Adjust z based on screen angles
            z_cm = (
                z_cm
                - x_offset * np.sin(screen_angle_h)
                - y_offset * np.sin(screen_angle_v)
            )
        else:
            z_cm = z

        # Calculate radius from eye to each point on screen
        r = np.sqrt(x_cm**2 + y_cm**2 + z_cm**2)

        if transformation == "spherical":
            # Calculate spherical coordinates
            # Altitude (θ): angle from XY plane to point
            theta = np.pi / 2 - np.arccos(z_cm / r)
            # Azimuth (φ): angle in XY plane (from X-axis)
            phi = np.arctan2(-y_cm, x_cm)

            # Apply transformation based on sweep direction
            if sweep_direction in ["left-to-right", "right-to-left"]:
                # For horizontal drift, apply azimuth correction
                # This maintains constant angular width in azimuth
                azimuth_correction = correction_strength * np.abs(phi)
                phi_corrected = phi * (1 + azimuth_correction)

                # Convert back to Cartesian coordinates
                y_transformed = -np.tan(phi_corrected) * x_cm

            if sweep_direction in ["top-to-bottom", "bottom-to-top"]:
                # For vertical drift, apply altitude correction
                # This maintains constant angular height in altitude
                altitude_correction = correction_strength * np.abs(theta)
                theta_corrected = theta * (1 + altitude_correction)

                # Convert back to Cartesian coordinates
                z_prime = r * np.sin(theta_corrected)
                xy_dist = r * np.cos(theta_corrected)
                x_prime = xy_dist * np.cos(phi)
                y_prime = -xy_dist * np.sin(phi)

                # Only apply the altitude correction to y-coordinate
                y_transformed = np.degrees(np.arctan2(y_prime, z_prime))

        # Convert back to degrees for rendering
        x_transformed = np.degrees(np.arctan2(x_transformed, eye_distance))

        # For y, we need to adjust by the horizontal meridian offset to match retinotopic mapping
        if sweep_direction in ["top-to-bottom", "bottom-to-top"]:
            y_transformed = (
                np.degrees(np.arctan2(y_transformed, eye_distance))
                + horizontal_meridian_offset
            )
        else:
            y_transformed = (
                np.degrees(np.arctan2(y_cm, eye_distance)) + horizontal_meridian_offset
            )

        self.logger.debug(
            f"Transformed coordinates shape: {x_transformed.shape}, "
            f"transformation={transformation}, sweep_direction={sweep_direction}"
        )

        return x_transformed, y_transformed

    # Add transformation map caching
    def _create_transformation_maps(self):
        """
        Create and cache transformation maps for faster rendering.
        """
        if hasattr(self, "_cached_map_x") and hasattr(self, "_cached_map_y"):
            return self._cached_map_x, self._cached_map_y

        transformation = self.parameters.get("transformation", "none")
        if transformation == "none":
            self._cached_map_x = None
            self._cached_map_y = None
            return None, None

        # Create coordinate grid (in pixels)
        y, x = np.mgrid[0 : self._size_y_px, 0 : self._size_x_px]

        # Center coordinates and normalize to range [-1, 1]
        x = (x - self._size_x_px // 2) / (self._size_x_px // 2)
        y = (y - self._size_y_px // 2) / (self._size_y_px // 2)

        # Initialize z as a zero array for 2D transformation
        z = np.zeros_like(x)

        # Apply transformation directly (vectorized)
        x_transformed, y_transformed = self._apply_transformation(x, y, z)

        # Convert back to pixel coordinates
        x_map = (x_transformed + 1) * (self._size_x_px // 2)
        y_map = (y_transformed + 1) * (self._size_y_px // 2)

        # Create the transformation map for OpenCV remap
        self._cached_map_x = x_map.astype(np.float32)
        self._cached_map_y = y_map.astype(np.float32)

        return self._cached_map_x, self._cached_map_y

    def _create_pattern_texture(self, phase_offset: float = 0.0) -> np.ndarray:
        """
        Create a texture for the pattern based on spherical coordinates when needed.

        This implements the stimulus generation described in MMC1:
        For spherical coordinates:
        S(θ(x₀,y,z), φ(x₀,y,z)) = S(π/2 - cos⁻¹(z/√(x₀²+y²+z²)), tan⁻¹(-y/x₀))

        For drifting gratings with contours along altitude lines:
        S = cos(2πfsθ - tft)

        Where:
        - fs is spatial frequency (cycles/radian)
        - ft is temporal frequency (radians/second)
        - t is time

        Args:
            phase_offset: Phase offset in degrees (for moving patterns) or position in degrees (for bar pattern)

        Returns:
            np.ndarray: Pattern texture
        """
        # Initialize variables
        pattern_type = self.parameters.get("pattern", "checkerboard")
        grid_spacing_x = self.parameters.get("grid_spacing_x", 25.0)  # degrees
        grid_spacing_y = self.parameters.get("grid_spacing_y", 25.0)  # degrees
        line_width = self.parameters.get("line_width", 0.0)  # degrees
        x_size = self.parameters.get("x_size", 147.0)  # degrees
        y_size = self.parameters.get("y_size", 153.0)  # degrees
        inverse = self.parameters.get("inverse", False)
        bar_width = self.parameters.get("bar_width", 20.0)  # degrees
        transformation = self.parameters.get("transformation", "none")
        sweep_direction = self.parameters.get("sweep_direction", "left-to-right")
        screen_distance = self.parameters.get(
            "screen_distance", 10.0
        )  # cm from eye to screen

        # Get resolution
        resolution = self.parameters.get("resolution")
        if resolution is None:
            res_x = self._size_x_px
            res_y = self._size_y_px
        elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            res_x, res_y = resolution
        elif isinstance(resolution, str) and "x" in resolution:
            try:
                res_x, res_y = map(int, resolution.split("x"))
            except ValueError:
                self.logger.warning(
                    f"Invalid resolution string format: {resolution}, using calculated values"
                )
                res_x = self._size_x_px
                res_y = self._size_y_px
        else:
            self.logger.warning(
                f"Invalid resolution format: {resolution}, using calculated values"
            )
            res_x = self._size_x_px
            res_y = self._size_y_px

        self.logger.debug(f"Creating pattern with resolution: {res_x}x{res_y}")

        # Create coordinate grids (centered at screen center)
        x_coords, y_coords = np.meshgrid(np.arange(res_x), np.arange(res_y))

        # Convert to centered coordinates (0,0 at center of screen)
        x_centered = x_coords - res_x / 2
        y_centered = y_coords - res_y / 2

        # Convert to degrees from center
        x_deg = x_centered / self._pixels_per_degree
        y_deg = y_centered / self._pixels_per_degree

        # Apply spherical transformation if needed
        if transformation == "spherical":
            # Convert screen coordinates to 3D coordinates (x₀, y, z)
            # x₀ is constant (distance to screen)
            x0 = screen_distance  # cm

            # Convert degrees to cm on screen using small angle approximation
            y_cm = np.tan(np.radians(y_deg)) * x0
            z_cm = (
                np.tan(np.radians(x_deg)) * x0
            )  # x_deg is used for vertical dimension

            # Calculate radius from eye to each point
            r = np.sqrt(x0**2 + y_cm**2 + z_cm**2)

            # Calculate spherical coordinates according to MMC1
            # θ = π/2 - cos⁻¹(z/√(x₀²+y²+z²))  (altitude)
            theta = np.pi / 2 - np.arccos(z_cm / r)

            # φ = tan⁻¹(-y/x₀)  (azimuth)
            phi = np.arctan2(-y_cm, x0)

            # For rotations to change orientation, redefine y and z as:
            # y' = z*sin(-φ) + y*cos(-φ)
            # z' = z*cos(-φ) - y*sin(-φ)
            orientation = self.parameters.get("orientation", 0.0)
            if orientation != 0:
                orientation_rad = np.radians(-orientation)
                y_rot = z_cm * np.sin(orientation_rad) + y_cm * np.cos(orientation_rad)
                z_rot = z_cm * np.cos(orientation_rad) - y_cm * np.sin(orientation_rad)

                # Recalculate spherical coordinates
                r_rot = np.sqrt(x0**2 + y_rot**2 + z_rot**2)
                theta = np.pi / 2 - np.arccos(z_rot / r_rot)
                phi = np.arctan2(-y_rot, x0)

            # Initialize pattern as a boolean array
            pattern = np.zeros((res_y, res_x), dtype=bool)

            # Apply pattern based on spherical coordinates
            if pattern_type == "checkerboard":
                # For spherical checkerboard, use altitude (θ) and azimuth (φ)
                # Convert phase_offset to radians
                phase_rad = np.deg2rad(phase_offset)

                # Spatial frequency in cycles/radian
                fs_theta = 1.0 / np.deg2rad(
                    grid_spacing_y
                )  # Convert from degrees to cycles/radian
                fs_phi = 1.0 / np.deg2rad(
                    grid_spacing_x
                )  # Convert from degrees to cycles/radian

                # Initialize index arrays
                theta_idx = np.zeros_like(theta)
                phi_idx = np.zeros_like(phi)

                # For a drifting grating that is a function of altitude (θ) and constant in azimuth (φ)
                # S = cos(2πfsθ - tft)
                if sweep_direction in ["top-to-bottom", "bottom-to-top"]:
                    # Use altitude (θ) for vertical bars
                    # Scale θ to get correct grid spacing
                    theta_idx = np.floor(
                        (theta + phase_rad / (2 * np.pi * fs_theta))
                        * fs_theta
                        / (0.5 * np.pi)
                    )
                    phi_idx = np.floor(phi * fs_phi / (0.5 * np.pi))
                elif sweep_direction in ["left-to-right", "right-to-left"]:
                    # Use azimuth (φ) for horizontal bars
                    # Scale φ to get correct grid spacing
                    theta_idx = np.floor(theta * fs_theta / (0.5 * np.pi))
                    phi_idx = np.floor(
                        (phi + phase_rad / (2 * np.pi * fs_phi))
                        * fs_phi
                        / (0.5 * np.pi)
                    )

                # Create checkerboard pattern
                pattern = (theta_idx + phi_idx) % 2 == 0

            elif pattern_type == "grid":
                # For spherical grid, use altitude (θ) and azimuth (φ)
                # Convert parameters to radians
                line_width_rad = np.deg2rad(line_width)
                grid_spacing_theta_rad = np.deg2rad(grid_spacing_y)
                grid_spacing_phi_rad = np.deg2rad(grid_spacing_x)
                phase_rad = np.deg2rad(phase_offset)

                # Calculate modulo positions in spherical coordinates
                theta_mod = np.mod(
                    theta + phase_rad * grid_spacing_theta_rad / (2 * np.pi),
                    grid_spacing_theta_rad,
                )
                phi_mod = np.mod(phi, grid_spacing_phi_rad)

                # Draw grid lines
                theta_lines = theta_mod < line_width_rad
                phi_lines = phi_mod < line_width_rad
                pattern = theta_lines | phi_lines

            elif pattern_type == "bars":
                # For bars, phase_offset is the position of the bar center in degrees
                # Convert to radians
                bar_width_rad = np.deg2rad(bar_width)
                position_rad = np.deg2rad(phase_offset)

                if sweep_direction in ["left-to-right", "right-to-left"]:
                    # Horizontal bar (constant altitude)
                    # Use azimuth (φ) for horizontal positioning
                    pattern = np.abs(phi - position_rad) <= bar_width_rad / 2
                    self.logger.debug(
                        f"Creating horizontal bar at position {phase_offset}° with width {bar_width}°"
                    )
                else:  # ["top-to-bottom", "bottom-to-top"]
                    # Vertical bar (constant azimuth)
                    # Use altitude (θ) for vertical positioning
                    pattern = np.abs(theta - position_rad) <= bar_width_rad / 2
                    self.logger.debug(
                        f"Creating vertical bar at position {phase_offset}° with width {bar_width}°"
                    )
        else:
            # Without transformation, use regular Cartesian coordinates
            # Initialize pattern
            pattern = np.zeros((res_y, res_x), dtype=bool)

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
        self.logger.debug(f"Pattern shape: {image.shape}, dtype: {image.dtype}")

        return image

    def _create_checkerboard_texture_gpu(
        self,
        phase_offset: float,
        contrast: float,
        bg: float,
        orientation_deg: float,
        progressive: bool,
        sweep_direction: str,
    ) -> np.ndarray:
        """
        Create a checkerboard pattern texture using GPU acceleration.

        Args:
            phase_offset: Phase offset for moving patterns (in degrees)
            contrast: Contrast value (0-1)
            bg: Background value (0-1)
            orientation_deg: Orientation in degrees
            progressive: Whether to use progressive appearance
            sweep_direction: Direction for progressive sweep

        Returns:
            np.ndarray: The texture as a numpy array
        """
        # GPU acceleration might not be fully implemented or optimized for complex
        # transformations like hyperbolic checkerboard. Use CPU implementation instead.
        return self._create_pattern_texture(phase_offset)
