"""
Factory module for creating stimulus instances.

This module provides a factory function to create the appropriate
stimulus object based on the specified parameters.
"""

from typing import Dict, Any, Optional, Type

from src.stimuli.base_stimulus import BaseStimulus
from src.stimuli.checkerboard.checkerboard_stimulus import CheckerboardStimulus
from src.stimuli.drifting_bar import DriftingBarStimulus


def create_stimulus(parameters: Dict[str, Any]) -> BaseStimulus:
    """
    Create a stimulus instance based on the specified parameters.

    Args:
        parameters: Dictionary containing stimulus parameters

    Returns:
        BaseStimulus: The created stimulus instance
    """
    # Determine stimulus type
    stimulus_type = parameters.get("type", "checkerboard").lower()

    # Create appropriate stimulus instance
    if stimulus_type == "drifting_bar" or (
        stimulus_type == "checkerboard"
        and parameters.get("pattern", "checkerboard") == "bars"
    ):
        # Special case for drifting bar stimulus as described in MMC1
        return DriftingBarStimulus(parameters)
    elif stimulus_type == "checkerboard":
        # Use the general-purpose checkerboard stimulus
        return CheckerboardStimulus(parameters)
    else:
        # Default to checkerboard stimulus
        return CheckerboardStimulus(parameters)


def create_mmc1_stimulus(
    stimulus_type: str = "drifting_bar",
    is_two_photon: bool = False,
    additional_params: Optional[Dict[str, Any]] = None,
) -> BaseStimulus:
    """
    Create a stimulus as specified in the MMC1 document.

    This is a convenience function to quickly create a stimulus with parameters
    pre-configured to match those described in the MMC1 document.

    Args:
        stimulus_type: Type of stimulus ("drifting_bar" or "drifting_grating")
        is_two_photon: Whether to use two-photon imaging speeds (True) or
                       intrinsic imaging speeds (False)
        additional_params: Additional parameters to override defaults

    Returns:
        BaseStimulus: The created stimulus instance
    """
    # Set default MMC1 parameters
    parameters = {
        # Screen setup parameters as described in MMC1
        "screen_distance": 10.0,  # cm from eye to screen
        "screen_width_cm": 68.0,  # cm - 68 x 121 cm large LCD display as in MMC1
        "screen_height_cm": 121.0,  # cm
        "resolution": (1920, 1080),  # px - adjust as needed for actual display
        # Visual field parameters
        "x_size": 147.0,  # degrees - horizontal visual field
        "y_size": 153.0,  # degrees - vertical visual field
        # Screen positioning parameters
        "screen_angle": 20.0,  # degrees - screen placed at 20° angle relative to animal
        "horizontal_meridian_offset": 20.0,  # degrees - estimated horizontal meridian at ~20° altitude
        # Transformation
        "transformation": "spherical",  # spherical transformation as described in MMC1
        # Timing parameters
        "fps": 60.0,  # frames per second - 60Hz as mentioned in MMC1
        # Two-photon vs intrinsic imaging mode
        "is_two_photon": is_two_photon,
        # Performance optimization defaults
        "num_processes": 4,  # Default to 4 processes for multiprocessing
        "use_gpu": None,  # Auto-detect GPU availability
        "precompute_all": True,  # Precompute all transformations by default
    }

    # Override with additional parameters if provided
    if additional_params:
        parameters.update(additional_params)

    if stimulus_type == "drifting_bar":
        # Configure for drifting bar stimulus
        parameters.update(
            {
                "type": "drifting_bar",
                "bar_width": 20.0,  # degrees - 20° wide as in MMC1
                "drift_speed": 8.5,  # degrees per second - 8.5-9.5°/s for intrinsic imaging
                "drift_speed_2p": 12.0,  # degrees per second - 12-14°/s for two-photon imaging
                "grid_spacing_x": 25.0,  # degrees - 25° squares as in MMC1
                "grid_spacing_y": 25.0,  # degrees
                "temporal_freq": 6.0,  # Hz - corresponds to 166ms period mentioned in MMC1
                "drift_repeats": 1,  # Default to 1 repeat per direction for testing (original MMC1: 10)
            }
        )
        return DriftingBarStimulus(parameters)

    elif stimulus_type == "drifting_grating":
        # Configure for drifting grating stimulus
        parameters.update(
            {
                "type": "checkerboard",
                "pattern": "grating",
                "spatial_freq": 0.04,  # cycles per degree (typical value from MMC1)
                "temporal_freq": 1.0,  # Hz (from MMC1 experiments)
                "orientation": 0.0,  # degrees (can be changed as needed)
            }
        )
        return CheckerboardStimulus(parameters)

    else:
        # Default to drifting bar stimulus
        parameters["type"] = "drifting_bar"
        return DriftingBarStimulus(parameters)
