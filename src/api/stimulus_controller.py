"""API for external control of stimulus generation."""

import threading
import queue
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union


class StimulusController:
    """
    Controller for external stimulus control.

    This class provides an API for other programs to interact with
    the stimulus generation process, including frame control and
    synchronization.

    Examples:
        >>> controller = StimulusController()
        >>> controller.load_stimulus("PG", {"orientation": 0.0, "duration": 2.0})
        >>> frame_info = controller.get_current_frame()
        >>> controller.advance_frame()
    """

    def __init__(self):
        """Initialize the stimulus controller."""
        self.logger = logging.getLogger("StimulusController")

        # Configure basic logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        self._current_stimulus = None
        self._current_frame = 0
        self._total_frames = 0
        self._is_running = False
        self._is_paused = False
        self._command_queue = queue.Queue()
        self._frame_ready_callbacks: List[Callable] = []
        self._frame_request_callbacks: List[Callable] = []
        self._stimulus_info: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def load_stimulus(self, stimulus_type: str, parameters: Dict[str, Any]) -> bool:
        """
        Load a stimulus for presentation.

        Args:
            stimulus_type: Type of stimulus (e.g., 'PG')
            parameters: Parameters for the stimulus

        Returns:
            bool: True if successful
        """
        try:
            # Create the stimulus
            self._create_stimulus(stimulus_type, parameters)

            with self._lock:
                self._current_frame = 0
                self._stimulus_info = {
                    "type": stimulus_type,
                    "parameters": parameters.copy(),
                    "current_frame": 0,
                    "total_frames": self._total_frames,
                }

            self.logger.info(
                f"Loaded {stimulus_type} stimulus with {self._total_frames} frames"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error loading stimulus: {str(e)}")
            return False

    def _create_stimulus(self, stimulus_type: str, parameters: Dict[str, Any]):
        """Create the stimulus object based on type."""
        if stimulus_type == "PG":
            from src.stimuli.gratings.per_grater import PeriodicGratingStimulus

            self._current_stimulus = PeriodicGratingStimulus(parameters)
            self._total_frames = self._current_stimulus._total_frames
        # Add other stimulus types as needed
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")

    def get_current_frame(self) -> Dict[str, Any]:
        """
        Get the current frame and its metadata.

        Returns:
            Dict with frame data and metadata
        """
        if self._current_stimulus is None:
            return {"error": "No stimulus loaded"}

        with self._lock:
            frame = self._current_stimulus.get_frame(self._current_frame)
            info = self._stimulus_info.copy()
            info["current_frame"] = self._current_frame

            # Add calculated information about this frame
            current_params = self._calculate_current_frame_parameters()

            return {
                "frame": frame,
                "frame_number": self._current_frame,
                "total_frames": self._total_frames,
                "frame_parameters": current_params,
                "stimulus_info": info,
            }

    def _calculate_current_frame_parameters(self) -> Dict[str, Any]:
        """Calculate dynamic parameters for the current frame."""
        if not self._current_stimulus or not hasattr(
            self._current_stimulus, "parameters"
        ):
            return {}

        params = {}

        try:
            # For grating stimulus, calculate current phase based on frame number
            base_params = self._current_stimulus.parameters
            tf = base_params.get("temporal_freq", 0)
            reverse = base_params.get("reverse", 0) == 1
            refresh_rate = float(base_params.get("refresh_rate", 60))

            # Calculate phase for this frame
            time = self._current_frame / refresh_rate
            phase_offset = 360 * tf * time  # In degrees

            if reverse:
                phase_offset = -phase_offset

            # Add original phase
            orig_phase = base_params.get("phase", 0)
            current_phase = (orig_phase + phase_offset) % 360

            params["current_phase"] = current_phase
            params["current_time"] = time
        except Exception as e:
            self.logger.error(f"Error calculating frame parameters: {str(e)}")

        return params

    def advance_frame(self) -> bool:
        """
        Advance to the next frame.

        Returns:
            bool: True if successful, False if at the end
        """
        if self._current_stimulus is None:
            return False

        with self._lock:
            if self._current_frame < self._total_frames - 1:
                self._current_frame += 1
                self._stimulus_info["current_frame"] = self._current_frame

                # Notify observers
                self._notify_frame_ready()
                return True
            else:
                return False

    def set_frame(self, frame_number: int) -> bool:
        """
        Set the current frame number.

        Args:
            frame_number: Frame number to set

        Returns:
            bool: True if successful
        """
        if self._current_stimulus is None:
            return False

        with self._lock:
            if 0 <= frame_number < self._total_frames:
                self._current_frame = frame_number
                self._stimulus_info["current_frame"] = frame_number

                # Notify observers
                self._notify_frame_ready()
                return True
            else:
                return False

    def add_frame_ready_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Add a callback to be notified when a new frame is ready.

        Args:
            callback: Function to call with frame information
        """
        self._frame_ready_callbacks.append(callback)

    def add_frame_request_callback(self, callback: Callable[[], None]):
        """
        Add a callback to be notified when an external system requests a frame.

        Args:
            callback: Function to call when frame is requested
        """
        self._frame_request_callbacks.append(callback)

    def _notify_frame_ready(self):
        """Notify callbacks that a new frame is ready."""
        frame_info = self.get_current_frame()

        for callback in self._frame_ready_callbacks:
            try:
                callback(frame_info)
            except Exception as e:
                self.logger.error(f"Error in frame ready callback: {str(e)}")

    def request_next_frame(self):
        """
        Request the next frame from external systems.

        This method is called when the stimulus system is waiting for
        an external trigger to advance to the next frame.
        """
        for callback in self._frame_request_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in frame request callback: {str(e)}")

    def get_stimulus_info(self) -> Dict[str, Any]:
        """
        Get information about the current stimulus.

        Returns:
            Dict with stimulus information
        """
        return self._stimulus_info.copy()

    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress information.

        Returns:
            Dict with progress information
        """
        if not self._current_stimulus:
            return {"progress": 0, "current_frame": 0, "total_frames": 0}

        progress = 0
        if self._total_frames > 0:
            progress = (self._current_frame / self._total_frames) * 100

        return {
            "progress": progress,
            "current_frame": self._current_frame,
            "total_frames": self._total_frames,
        }


# Global instance for easy access
_default_controller = None


def get_controller() -> StimulusController:
    """
    Get the default StimulusController instance.

    Returns:
        StimulusController: The default controller instance
    """
    global _default_controller
    if _default_controller is None:
        _default_controller = StimulusController()
    return _default_controller
