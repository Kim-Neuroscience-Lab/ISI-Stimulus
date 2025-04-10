# examples/api_example.py
"""
Example of using the ISI Stimulus API to integrate with other software.

This example shows how to:
1. Load a stimulus
2. Control frame advance
3. Get frame data and metadata
4. Use callbacks for synchronization
"""

import os
import sys
import time
import numpy as np
import cv2

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the API
from src.api.stimulus_controller import StimulusController


def on_frame_ready(frame_info):
    """
    Callback function called when a new frame is ready.

    Args:
        frame_info: Dictionary with frame data and metadata
    """
    frame_num = frame_info["frame_number"]
    total = frame_info["total_frames"]
    progress = (frame_num / total) * 100 if total > 0 else 0

    print(f"Frame {frame_num}/{total} ready ({progress:.1f}%)")
    if "frame_parameters" in frame_info:
        if "current_phase" in frame_info["frame_parameters"]:
            phase = frame_info["frame_parameters"]["current_phase"]
            print(f"  Current phase: {phase:.2f}°")


def display_frame(frame, window_name="Stimulus"):
    """
    Display a frame using OpenCV.

    Args:
        frame: The frame to display
        window_name: The window name
    """
    # Resize if too large
    height, width = frame.shape[:2]
    max_size = 800
    if width > max_size or height > max_size:
        scale = min(max_size / width, max_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))

    cv2.imshow(window_name, frame)
    cv2.waitKey(1)  # Update the window


def main():
    """Run the example."""
    print("ISI Stimulus API Example")
    print("=======================")

    # Create the controller
    controller = StimulusController()

    # Add a callback to be notified when frames are ready
    controller.add_frame_ready_callback(on_frame_ready)

    # Define stimulus parameters
    parameters = {
        "spatial_freq": 0.1,  # cycles/degree
        "temporal_freq": 0.5,  # Hz
        "contrast": 100.0,  # percent
        "orientation": 45.0,  # degrees
        "x_size": 20.0,  # degrees
        "y_size": 20.0,  # degrees
        "refresh_rate": 60,  # Hz
        "duration": 2.0,  # seconds
        "screen_distance": 30.0,  # cm
        "background": 127,  # 0-255
        "mask_type": "none",  # none, circle, gaussian
    }

    print("\nLoading stimulus...")
    if not controller.load_stimulus("PG", parameters):
        print("Failed to load stimulus")
        return 1

    print("\nCreating display window...")
    cv2.namedWindow("Stimulus", cv2.WINDOW_NORMAL)

    print("\nPlaying stimulus (press 'q' to stop)...")
    try:
        # Main loop
        while True:
            # Get current frame
            frame_info = controller.get_current_frame()
            frame = frame_info["frame"]

            # Display the frame
            display_frame(frame)

            # Advance to next frame
            if not controller.advance_frame():
                print("Reached end of stimulus")
                # Loop back to beginning
                controller.set_frame(0)

            # Check for key press
            key = cv2.waitKey(16)  # ~60fps
            if key == ord("q") or key == 27:  # q or ESC
                break

            # Sleep to maintain approximately the specified refresh rate
            time.sleep(1.0 / parameters["refresh_rate"])

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()

    print("\nExample completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
