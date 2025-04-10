# compare_results.py
"""
Script to display a grid of the generated frames for comparison, labeled with angular position.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from typing import Dict, Any


def create_composite_view():
    # Import the DriftingBarStimulus class here to generate frames directly
    from src.stimuli.drifting_bar import DriftingBarStimulus
    from src.stimuli.spherical_correction import SphericalCorrection

    # Get screen resolution parameters
    width, height = 800, 600

    # Use the pixel aspect ratio for the plots to match the video output
    pixel_aspect_ratio = width / height

    # Set up parameters for the stimulus - USE EXACTLY THE SAME PARAMETERS FOR BOTH
    # DRIFTING BAR STIMULUS AND SPHERICAL CORRECTION
    params = {
        "resolution": (800, 600),
        "width": 800,
        "height": 600,
        "bar_width": 20.0,  # Exactly 20° as in MMC1
        "grid_spacing_x": 25.0,  # Exactly 25° as in MMC1
        "grid_spacing_y": 25.0,  # Exactly 25° as in MMC1
        "drift_speed": 9.0,  # Exactly 9° per second as requested
        "temporal_freq": 6.0,  # 6Hz counterphase frequency
        "screen_distance": 10.0,
        "screen_angle": 20.0,
        "horizontal_meridian_offset": 0.0,  # Set to 0 for visualization to match grid
        "x_size": 147.0,  # degrees horizontal
        "y_size": 153.0,  # degrees vertical
    }

    # Get visual field parameters
    x_size = params["x_size"]
    y_size = params["y_size"]
    bar_width = params["bar_width"]
    checker_size_x = params["grid_spacing_x"]
    checker_size_y = params["grid_spacing_y"]

    # Define progress points to sample (0.0 to 1.0)
    progress_points = [0.0, 0.22, 0.44, 0.67, 0.89]

    # Define angular extent for the plots
    azimuth_min, azimuth_max = -x_size / 2, x_size / 2
    altitude_min, altitude_max = -y_size / 2, y_size / 2

    # Calculate extent for the plots
    extent = (
        float(azimuth_min),
        float(azimuth_max),
        float(altitude_min),
        float(altitude_max),
    )

    # Create grid lines at exactly 25-degree intervals to match the checkerboard pattern
    azimuth_lines = np.arange(
        np.ceil(azimuth_min / checker_size_x) * checker_size_x,
        np.floor(azimuth_max / checker_size_x) * checker_size_x + 1,
        checker_size_x,
    )

    altitude_lines = np.arange(
        np.ceil(altitude_min / checker_size_y) * checker_size_y,
        np.floor(altitude_max / checker_size_y) * checker_size_y + 1,
        checker_size_y,
    )

    # Create a SphericalCorrection instance for generating transformed coordinates
    spherical_correction = SphericalCorrection(params)

    # Function that exactly mimics the transformation in transform_drifting_bar
    def apply_exact_spherical_transform(x_deg, y_deg):
        """
        Apply the spherical transformation in a way that ensures the equator is a straight horizontal line.
        """
        # Get distance to screen
        x0 = spherical_correction.screen_distance  # Distance from eye to screen (cm)

        # Keep track of the exact center of the screen coordinate
        center_y_deg = 0.0  # Center of screen is at altitude 0°

        # Convert degrees to distances on screen using tangent
        azimuth_rad = np.radians(x_deg)  # Horizontal on screen (azimuth)
        altitude_rad = np.radians(y_deg)  # Vertical on screen (altitude)

        # Convert to distances on screen (relative to center)
        y_screen = x0 * np.tan(azimuth_rad)  # Horizontal distance on screen
        z_screen = x0 * np.tan(altitude_rad)  # Vertical distance on screen

        # Calculate the square root term
        sqrt_term = np.sqrt(x0**2 + y_screen**2 + z_screen**2)

        # Calculate altitude (θ)
        # Handle potential division by zero or arguments outside [-1, 1]
        z_over_sqrt = np.zeros_like(z_screen)
        valid_indices = sqrt_term > 0
        z_over_sqrt[valid_indices] = z_screen[valid_indices] / sqrt_term[valid_indices]
        # Clip to handle numerical errors
        z_over_sqrt = np.clip(z_over_sqrt, -1.0, 1.0)

        # Calculate altitude using the formula θ = π/2 - cos⁻¹(z/√(x₀² + y² + z²))
        theta = np.pi / 2 - np.arccos(z_over_sqrt)

        # Calculate azimuth (φ)
        # Handle potential division by zero
        phi = np.zeros_like(y_screen)
        nonzero_indices = np.abs(x0) > 1e-10
        phi[nonzero_indices] = np.arctan2(-y_screen[nonzero_indices], x0)

        # Convert back to degrees
        theta_deg = np.degrees(theta)  # altitude angle
        phi_deg = np.degrees(phi)  # azimuth angle

        # Apply meridian offset differently to keep equator as a straight line
        # Instead of modifying input coordinates, we shift the output
        adjusted_theta_deg = theta_deg

        return phi_deg, adjusted_theta_deg

    # Function to map spherical coordinates to pixel coordinates
    def spherical_to_pixel(phi_deg, theta_deg):
        """Convert spherical coordinates (phi, theta) to pixel coordinates."""
        # Create coordinate grid for the frame
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        x_norm = x_coords / (width / 2) - 1.0
        y_norm = y_coords / (height / 2) - 1.0

        # Convert normalized coordinates to visual degrees
        x_deg = x_norm * (x_size / 2)
        y_deg = y_norm * (y_size / 2)

        # Transform them to spherical coordinates
        all_phi, all_theta = apply_exact_spherical_transform(x_deg, y_deg)

        # Create a grid of requested phi, theta values
        phi_deg = np.asarray(phi_deg)
        theta_deg = np.asarray(theta_deg)

        # Initialize the result arrays
        x_pixels = np.zeros_like(phi_deg, dtype=np.int32)
        y_pixels = np.zeros_like(theta_deg, dtype=np.int32)

        # For each requested (phi, theta) pair, find closest point in the transformed grid
        for i in range(len(phi_deg)):
            # Calculate distance to each point in the grid
            dist = (all_phi - phi_deg[i]) ** 2 + (all_theta - theta_deg[i]) ** 2

            # Find index of minimum distance
            min_idx = np.argmin(dist)
            y_idx, x_idx = np.unravel_index(min_idx, dist.shape)

            # Get corresponding pixel coordinates
            x_pixels[i] = x_idx
            y_pixels[i] = y_idx

        return x_pixels, y_pixels

    # Create DriftingBarStimulus instances for each direction
    direction_stimuli: Dict[str, DriftingBarStimulus] = {}
    directions = ["left-to-right", "bottom-to-top"]

    for direction in directions:
        # Create a separate stimulus instance for each direction to ensure proper setup
        direction_params = params.copy()
        direction_params["sweep_direction"] = direction
        direction_stimuli[direction] = DriftingBarStimulus(direction_params)

    # Store position angles for labeling
    horizontal_angles = []
    vertical_angles = []

    # Calculate positions for each direction
    for progress in progress_points:
        # Calculate horizontal (left-to-right) position
        h_pos = -x_size / 2 - bar_width + progress * (x_size + 2 * bar_width)
        horizontal_angles.append(h_pos)

        # Calculate vertical (bottom-to-top) position
        v_pos = -y_size / 2 - bar_width + progress * (y_size + 2 * bar_width)
        vertical_angles.append(v_pos)

    # Function to create a figure with the specified frames
    def create_figure(use_overlays=False):
        # Create figure with 2 rows (directions) and 5 columns (selected frames)
        # Use the pixel ratio for the figure to match the video output
        fig_width = 20
        fig_height = fig_width / pixel_aspect_ratio * 2 / 5  # Adjust for 2 rows, 5 cols
        fig, axs = plt.subplots(2, 5, figsize=(fig_width, fig_height))

        # Set the figure title
        mode = "Debug View (with Grid Overlay)" if use_overlays else "Black and White"
        fig.suptitle(f"Drifting Bar with Spherical Correction - {mode}", fontsize=16)

        # Generate frames for each direction and progress point
        for row, direction in enumerate(directions):
            # Get the stimulus for this direction
            stimulus = direction_stimuli[direction]

            # Update the spherical correction parameters to match
            spherical_correction.parameters["sweep_direction"] = direction
            spherical_correction.parameters["correct_axis"] = (
                "azimuth"
                if direction in ["left-to-right", "right-to-left"]
                else "altitude"
            )
            spherical_correction.parameters["width"] = width
            spherical_correction.parameters["height"] = height

            # Create transformation maps for this direction
            map_x, map_y = spherical_correction.create_transformation_maps(
                width, height
            )

            for col, progress in enumerate(progress_points):
                # Calculate frame index from progress
                frame_idx = int(progress * (stimulus.frame_count - 1))

                # Generate the frame with properly transformed bar
                frame = stimulus.get_frame(frame_idx)

                # Get current bar position
                if direction == "left-to-right":
                    bar_pos = horizontal_angles[col]
                else:
                    bar_pos = vertical_angles[col]

                # If overlay mode, create grid and bar outlines
                if use_overlays:
                    # Convert to RGB for drawing colored lines
                    frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)

                    # Create separate images for grid and bar outline
                    grid_img = np.zeros((height, width, 3), dtype=np.uint8)
                    bar_img = np.zeros((height, width, 3), dtype=np.uint8)

                    # Create a full coordinate grid for the entire frame
                    y_coords, x_coords = np.mgrid[0:height, 0:width]
                    x_norm = x_coords / (width / 2) - 1.0
                    y_norm = y_coords / (height / 2) - 1.0

                    # Convert to visual degrees
                    x_deg = x_norm * (x_size / 2)
                    y_deg = y_norm * (y_size / 2)

                    # Transform to get spherical coordinates (phi and theta)
                    phi_deg, theta_deg = apply_exact_spherical_transform(x_deg, y_deg)

                    # Calculate the visual center of the screen in spherical coordinates
                    screen_center_y = height // 2

                    # Draw a horizontal line at the exact center of the screen as the equator
                    grid_img[screen_center_y, :] = [0, 255, 0]  # Green equator line

                    # Make a copy of theta_deg for grid alignment
                    # Shift all theta values so 0° is exactly at the center row of the screen
                    center_theta = theta_deg[screen_center_y, width // 2]
                    adjusted_theta_deg = theta_deg - center_theta

                    # Find the grid line positions in phi/theta space
                    phi_grid_lines = np.arange(
                        np.floor(np.min(phi_deg) / checker_size_x) * checker_size_x,
                        np.ceil(np.max(phi_deg) / checker_size_x) * checker_size_x + 1,
                        checker_size_x,
                    )

                    theta_grid_lines = np.arange(
                        np.floor(np.min(adjusted_theta_deg) / checker_size_y)
                        * checker_size_y,
                        np.ceil(np.max(adjusted_theta_deg) / checker_size_y)
                        * checker_size_y
                        + 1,
                        checker_size_y,
                    )

                    # Draw vertical grid lines (lines of constant phi)
                    for phi_value in phi_grid_lines:
                        phi_mask = np.abs(phi_deg - phi_value) < 0.3
                        grid_img[phi_mask] = [255, 0, 0]  # Red color

                    # Draw horizontal grid lines (lines of constant theta)
                    for theta_value in theta_grid_lines:
                        # Skip the equator line as we've already drawn it in green
                        if abs(theta_value) < 0.5:
                            continue

                        theta_mask = np.abs(adjusted_theta_deg - theta_value) < 0.3
                        grid_img[theta_mask] = [255, 0, 0]  # Red color

                    # Draw bar outline using mask approach for consistent appearance
                    if direction == "left-to-right":
                        # Left and right edges of bar are lines of constant phi
                        left_edge = bar_pos - bar_width / 2
                        right_edge = bar_pos + bar_width / 2

                        # Create masks for left and right edges
                        left_mask = np.abs(phi_deg - left_edge) < 0.3
                        right_mask = np.abs(phi_deg - right_edge) < 0.3

                        # Apply masks to draw bar edges
                        bar_img[left_mask] = [0, 0, 255]  # Blue color
                        bar_img[right_mask] = [0, 0, 255]  # Blue color

                    else:  # bottom-to-top
                        # Bottom and top edges of bar are lines of constant theta
                        bottom_edge = bar_pos - bar_width / 2
                        top_edge = bar_pos + bar_width / 2

                        # Create masks for bottom and top edges
                        bottom_mask = np.abs(adjusted_theta_deg - bottom_edge) < 0.3
                        top_mask = np.abs(adjusted_theta_deg - top_edge) < 0.3

                        # Apply masks to draw bar edges
                        bar_img[bottom_mask] = [0, 0, 255]  # Blue color
                        bar_img[top_mask] = [0, 0, 255]  # Blue color

                    # Apply dilation to make the bar outline slightly thicker
                    bar_img = cv2.dilate(
                        bar_img, np.ones((2, 2), np.uint8), iterations=1
                    )

                    # Combine the original frame with grids and bar outlines
                    overlay = cv2.addWeighted(frame_rgb, 1.0, grid_img, 0.7, 0)
                    overlay = cv2.addWeighted(overlay, 1.0, bar_img, 0.7, 0)

                    # Use the overlay image for display
                    display_image = overlay
                    cmap = None  # Use RGB colors
                else:
                    # Use the original black and white frame
                    display_image = frame
                    cmap = "gray"  # Use grayscale colormap

                # Create the subplot
                ax = axs[row, col]

                # Display the frame with correct extent matching the visual field
                im = ax.imshow(display_image, extent=extent, aspect="auto", cmap=cmap)

                # Force the correct aspect ratio on the axes to match the pixel aspect ratio
                ax.set_box_aspect(1 / pixel_aspect_ratio)  # Make width > height

                # Set axis limits to exactly match the visual field
                ax.set_xlim(azimuth_min, azimuth_max)
                ax.set_ylim(altitude_min, altitude_max)

                # Set tick positions to match grid lines (25-degree intervals)
                ax.set_xticks(azimuth_lines)
                ax.set_yticks(altitude_lines)

                # Format tick labels to be cleaner (no decimal places)
                ax.set_xticklabels(
                    [f"{int(x)}" if x == int(x) else f"{x:.1f}" for x in azimuth_lines]
                )
                ax.set_yticklabels(
                    [f"{int(y)}" if y == int(y) else f"{y:.1f}" for y in altitude_lines]
                )

                # Reduce tick label size to prevent overlapping
                ax.tick_params(axis="both", labelsize=8)

                # Add title with angular position
                if row == 0:  # Horizontal sweep
                    angle = horizontal_angles[col]
                    ax.set_title(f"Azimuth: {angle:.1f}°")
                else:  # Vertical sweep
                    angle = vertical_angles[col]
                    ax.set_title(f"Altitude: {angle:.1f}°")

                # Set axis labels
                if col == 0:
                    ax.set_ylabel("Altitude (°)")
                if row == 1:
                    ax.set_xlabel("Azimuth (°)")

                # Draw the center point as reference
                ax.plot(0, 0, "+", color="k", markersize=8)

        # Add row titles
        fig.text(0.02, 0.75, "Left to Right", fontsize=14, rotation=90)
        fig.text(0.02, 0.25, "Bottom to Top", fontsize=14, rotation=90)

        # Add a simple legend using rectangular patches
        from matplotlib.patches import Patch

        if use_overlays:
            legend_elements = [
                Patch(facecolor="red", alpha=0.7, label="25° Grid"),
                Patch(facecolor="blue", alpha=0.7, label="20° Bar Width"),
                Patch(facecolor="green", alpha=0.7, label="Equator (0°)"),
            ]
        else:
            legend_elements = [
                Patch(facecolor="gray", alpha=0.7, label="25° Checkerboard Pattern"),
                Patch(facecolor="black", alpha=1.0, label="20° Bar Width"),
            ]

        axs[0, 0].legend(handles=legend_elements, loc="lower left", fontsize=8)

        # Add a title explaining the visualization
        plt.figtext(
            0.5,
            0.02,
            "Drifting bar (9°/s) with 6Hz counterphase flickering checkerboard pattern (20° bar width, 25° grid)",
            ha="center",
            fontsize=12,
        )

        # Adjust layout
        plt.tight_layout(rect=(0.03, 0, 1, 0.95))  # Adjust for row titles

        return fig

    # Create and save the black and white version (for paper)
    bw_fig = create_figure(use_overlays=False)
    bw_fig.savefig("test_output/comparison_grid_bw.png", dpi=150, bbox_inches="tight")
    plt.close(bw_fig)

    # Create and save the debug version (with grid overlays)
    debug_fig = create_figure(use_overlays=True)
    debug_fig.savefig(
        "test_output/comparison_grid_debug.png", dpi=150, bbox_inches="tight"
    )
    plt.close(debug_fig)

    print("Comparison grid created successfully!")
    print("Saved black and white version for paper (comparison_grid_bw.png)")
    print("Saved debug version with grid overlays (comparison_grid_debug.png)")


if __name__ == "__main__":
    create_composite_view()
