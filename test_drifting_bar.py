# test_drifting_bar.py
"""
Test script for the updated drifting bar stimulus with spherical correction.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.stimuli.drifting_bar import DriftingBarStimulus

# Create output directory if it doesn't exist
os.makedirs("test_output", exist_ok=True)

# Set up parameters
params = {
    "resolution": (800, 600),
    "width": 800,
    "height": 600,
    "sweep_direction": "left-to-right",
    "bar_width": 20.0,
    "grid_spacing_x": 25.0,
    "grid_spacing_y": 25.0,
    "drift_speed": 15.0,  # faster for testing
    "temporal_freq": 6.0,
    "horizontal_meridian_offset": 0.0,  # Ensure 0,0 is at the exact center
}


def test_horizontal_bar():
    """Test horizontal bar sweeping left to right"""
    # Create the stimulus
    params["sweep_direction"] = "left-to-right"
    stimulus = DriftingBarStimulus(params)

    print(
        f"Generating frames for {stimulus.duration:.2f} seconds at {stimulus.fps} fps"
    )

    # Generate 10 sample frames from different points in the sequence
    total_frames = stimulus.frame_count
    sample_indices = [int(total_frames * i / 9) for i in range(10)]

    # Generate and save sample frames
    for i, idx in enumerate(sample_indices):
        frame = stimulus.get_frame(idx)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame, cmap="gray")
        plt.title(f"Frame {idx}/{total_frames} (Progress: {idx/total_frames:.2f})")
        plt.colorbar()
        plt.savefig(f"test_output/frame_{i:02d}_left_to_right.png")
        plt.close()

    print("Horizontal bar test complete.")


def test_vertical_bar():
    """Test vertical bar sweeping bottom to top"""
    # Create the stimulus
    params["sweep_direction"] = "bottom-to-top"
    stimulus = DriftingBarStimulus(params)

    # Generate 10 sample frames from different points in the sequence
    total_frames = stimulus.frame_count
    sample_indices = [int(total_frames * i / 9) for i in range(10)]

    # Generate and save sample frames
    for i, idx in enumerate(sample_indices):
        frame = stimulus.get_frame(idx)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame, cmap="gray")
        plt.title(f"Frame {idx}/{total_frames} (Progress: {idx/total_frames:.2f})")
        plt.colorbar()
        plt.savefig(f"test_output/frame_{i:02d}_bottom_to_top.png")
        plt.close()

    print("Vertical bar test complete.")


if __name__ == "__main__":
    print("Testing drifting bar with spherical correction...")
    test_horizontal_bar()
    test_vertical_bar()
    print("Test complete. Check the test_output directory for sample frames.")
