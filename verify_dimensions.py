# verify_dimensions.py
"""
Script to verify the dimensions of the drifting bar and checkerboard pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from src.stimuli.drifting_bar import DriftingBarStimulus

# Set up parameters for visualization
params = {
    "resolution": (800, 600),
    "width": 800,
    "height": 600,
    "sweep_direction": "right-to-left",
    "bar_width": 20.0,  # Should be exactly 20 degrees
    "grid_spacing_x": 25.0,  # Should be exactly 25 degrees
    "grid_spacing_y": 25.0,  # Should be exactly 25 degrees
    "drift_speed": 8.5,  # Intrinsic imaging
    "temporal_freq": 6.0,
    "horizontal_meridian_offset": 0.0,  # Center on screen
    "fps": 60.0,
}


def verify_dimensions():
    """Generate a frame and overlay grid lines to verify dimensions."""
    # Create the stimulus
    stimulus = DriftingBarStimulus(params)

    print(f"Stimulus initialized with:")
    print(f"  Bar width: {stimulus.bar_width}°")
    print(f"  Checker size (x): {stimulus.grid_spacing_x}°")
    print(f"  Checker size (y): {stimulus.grid_spacing_y}°")

    # Generate a frame with the bar in the center of the visual field
    # The center frame index
    center_frame_idx = stimulus.frame_count // 2

    # Generate the frame
    frame = stimulus.get_frame(center_frame_idx)

    # Get visual field extents
    x_min, x_max = -stimulus.x_size / 2, stimulus.x_size / 2
    y_min, y_max = -stimulus.y_size / 2, stimulus.y_size / 2

    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(12, 9))

    # Display the frame
    ax.imshow(
        frame,
        cmap="gray",
        extent=(float(x_min), float(x_max), float(y_min), float(y_max)),
    )

    # Add grid lines at 25 degree intervals (the checker size)
    checker_x_size = stimulus.grid_spacing_x
    checker_y_size = stimulus.grid_spacing_y

    # Vertical grid lines (x axis)
    x_ticks = np.arange(
        np.ceil(x_min / checker_x_size) * checker_x_size,
        np.floor(x_max / checker_x_size) * checker_x_size + 1,
        checker_x_size,
    )
    for x in x_ticks:
        ax.axvline(x=float(x), color="blue", linestyle="--", alpha=0.7, lw=1)

    # Horizontal grid lines (y axis)
    y_ticks = np.arange(
        np.ceil(y_min / checker_y_size) * checker_y_size,
        np.floor(y_max / checker_y_size) * checker_y_size + 1,
        checker_y_size,
    )
    for y in y_ticks:
        ax.axhline(y=float(y), color="blue", linestyle="--", alpha=0.7, lw=1)

    # Add a rectangle showing the theoretical 20 degree bar width
    # Bar is centered, so the x position should be at the center
    bar_start_pos = 0 - stimulus.bar_width / 2
    rect = patches.Rectangle(
        (bar_start_pos, y_min),
        stimulus.bar_width,
        y_max - y_min,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)

    # Add labels and title
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Altitude (degrees)")
    ax.set_title(
        f"Drifting Bar Visualization\nBar Width: {stimulus.bar_width}°, Checker Size: {checker_x_size}°x{checker_y_size}°"
    )

    # Add legend
    ax.legend(
        [
            patches.Patch(edgecolor="r", facecolor="none"),
            mlines.Line2D([0], [0], color="blue", linestyle="--"),
        ],
        ["20° Bar Width", "25° Grid Lines"],
        loc="upper right",
    )

    # Save the figure
    plt.savefig("verify_dimensions.png", dpi=150, bbox_inches="tight")
    print("Verification image saved as 'verify_dimensions.png'")

    # Show the figure
    plt.show()


if __name__ == "__main__":
    verify_dimensions()
