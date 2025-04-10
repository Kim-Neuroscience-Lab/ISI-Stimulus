"""Base class for all stimulus types."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseStimulus(ABC):
    """Abstract base class for all stimulus types."""

    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the stimulus with parameters.

        Args:
            parameters: Dictionary of stimulus parameters
        """
        self.parameters = parameters
        self._calculate_derived_values()

    @abstractmethod
    def _calculate_derived_values(self) -> None:
        """Calculate derived values based on parameters."""
        pass

    @abstractmethod
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a specific frame of the stimulus.

        Args:
            frame_idx: Index of the frame to generate

        Returns:
            np.ndarray: The stimulus frame
        """
        pass

    @abstractmethod
    def get_all_frames(self) -> np.ndarray:
        """
        Get all frames of the stimulus.

        Returns:
            np.ndarray: Array of all stimulus frames
        """
        pass

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """
        Get the total number of frames.

        Returns:
            int: Total number of frames
        """
        pass

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        """
        Get the frame rate of the stimulus.

        Returns:
            float: Frame rate in Hz
        """
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Get the duration of the stimulus.

        Returns:
            float: Duration in seconds
        """
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """
        Get the resolution of the stimulus.

        Returns:
            Tuple[int, int]: Resolution as (width, height)
        """
        pass
