# compare_results.py
"""
Script to display a grid of the generated frames for comparison, labeled with angular position.
"""

import matplotlib.pyplot as plt
import os
import numpy as np


def create_composite_view():
    # Create figure with 2 rows (directions) and 5 columns (selected frames)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Drifting Bar with Spherical Correction", fontsize=16)

    # Import the DriftingBarStimulus class here to generate frames directly
    from src.stimuli.drifting_bar import DriftingBarStimulus
    from src.stimuli.spherical_correction import SphericalCorrection

    # Set up parameters for the stimulus
    params = {
        "resolution": (800, 600),
        "width": 800,
        "height": 600,
        "bar_width": 20.0,
        "grid_spacing_x": 25.0,
        "grid_spacing_y": 25.0,
        "drift_speed": 15.0,  # faster for testing
        "temporal_freq": 6.0,
    }

    # Visual field extents
    x_size = 147.0  # degrees horizontal
    y_size = 153.0  # degrees vertical
    bar_width = 20.0  # degrees

    # Define progress points to sample (0.0 to 1.0)
    progress_points = [0.0, 0.22, 0.44, 0.67, 0.89]

    # Define angular extent for the plots
    azimuth_min, azimuth_max = -x_size / 2, x_size / 2
    altitude_min, altitude_max = -y_size / 2, y_size / 2

    # Calculate aspect ratio to match frame dimensions
    frame_aspect_ratio = params["width"] / params["height"]  # 800/600 = 4/3
    data_aspect_ratio = x_size / y_size  # 147/153 ≈ 0.96

    # Generate and display frames for each direction and progress point
    directions = ["left-to-right", "bottom-to-top"]

    # Calculate start and end positions for each direction
    h_start = -x_size / 2 - bar_width  # off-screen left
    h_end = x_size / 2 + bar_width  # off-screen right
    v_start = -y_size / 2 - bar_width  # off-screen bottom
    v_end = y_size / 2 + bar_width  # off-screen top

    # Calculate the actual angular positions
    horizontal_angles = []
    vertical_angles = []

    for progress in progress_points:
        # Calculate positions (in degrees)
        h_pos = h_start + progress * (h_end - h_start)
        v_pos = v_end - progress * (v_end - v_start)  # Inverted for bottom-to-top

        horizontal_angles.append(h_pos)
        vertical_angles.append(v_pos)

    # Create spherical correction instances - one for each type of transformation
    # Make sure horizontal_meridian_offset is 0 to ensure true centering at 0,0
    sc_params_base = {
        "width": params["width"],
        "height": params["height"],
        "x_size": x_size,
        "y_size": y_size,
        "screen_distance": 10.0,
        "screen_angle": 20.0,
        "horizontal_meridian_offset": 0.0,  # Set to 0 to center exactly at 0,0
    }

    # Create separate instances for the two different transformation directions
    sc_azimuth = SphericalCorrection(sc_params_base.copy())
    sc_azimuth.parameters["correct_axis"] = "azimuth"

    sc_altitude = SphericalCorrection(sc_params_base.copy())
    sc_altitude.parameters["correct_axis"] = "altitude"

    # Create a special instance with inverted transformation for the grid
    # This will make the grid appear concave instead of convex
    sc_azimuth_inverted = SphericalCorrection(sc_params_base.copy())
    sc_azimuth_inverted.parameters["correct_axis"] = "azimuth"

    # Define inverted transformation functions without storing the original
    def inverted_transform_azimuth(self, x_deg, y_deg):
        # For azimuth correction, inverted means dividing by cos instead of multiplying
        # Convert degrees to radians
        x_rad = np.radians(x_deg)
        y_rad = np.radians(y_deg - self.horizontal_meridian_offset)

        # Calculate the inverse scaling factor
        cos_longitude = np.cos(x_rad)
        # Use division where the original uses multiplication (or vice versa)
        y_transformed = y_deg.copy()  # Create a copy to avoid modifying the original
        where_nonzero = np.abs(cos_longitude) > 1e-6
        y_transformed[where_nonzero] = (
            y_deg[where_nonzero] / cos_longitude[where_nonzero]
        )

        # Apply the offset back
        y_transformed = y_transformed + self.horizontal_meridian_offset
        return x_deg, y_transformed

    sc_azimuth_inverted._apply_spherical_transformation = (
        lambda x_deg, y_deg: inverted_transform_azimuth(
            sc_azimuth_inverted, x_deg, y_deg
        )
    )

    sc_altitude_inverted = SphericalCorrection(sc_params_base.copy())
    sc_altitude_inverted.parameters["correct_axis"] = "altitude"

    def inverted_transform_altitude(self, x_deg, y_deg):
        # For altitude correction, inverted means dividing by cos instead of multiplying
        # Convert degrees to radians
        x_rad = np.radians(x_deg)
        y_rad = np.radians(y_deg - self.horizontal_meridian_offset)

        # Calculate the inverse scaling factor
        cos_latitude = np.cos(y_rad)
        # Use division where the original uses multiplication (or vice versa)
        x_transformed = x_deg.copy()  # Create a copy to avoid modifying the original
        where_nonzero = np.abs(cos_latitude) > 1e-6
        x_transformed[where_nonzero] = (
            x_deg[where_nonzero] / cos_latitude[where_nonzero]
        )

        # Apply the offset back
        y_transformed = y_deg
        return x_transformed, y_transformed

    sc_altitude_inverted._apply_spherical_transformation = (
        lambda x_deg, y_deg: inverted_transform_altitude(
            sc_altitude_inverted, x_deg, y_deg
        )
    )

    # Create grid lines in degree space (centered around 0,0)
    # Use the same grid spacing as in the stimulus parameters (25 degrees)
    grid_spacing = params["grid_spacing_x"]  # 25.0 degrees

    # Create a sequence of grid lines that spans the entire visual field
    # Starting exactly at multiples of the grid spacing
    azimuth_min_grid = np.floor(azimuth_min / grid_spacing) * grid_spacing
    azimuth_max_grid = np.ceil(azimuth_max / grid_spacing) * grid_spacing
    altitude_min_grid = np.floor(altitude_min / grid_spacing) * grid_spacing
    altitude_max_grid = np.ceil(altitude_max / grid_spacing) * grid_spacing

    azimuth_lines = np.arange(
        azimuth_min_grid, azimuth_max_grid + grid_spacing / 2, grid_spacing
    )
    altitude_lines = np.arange(
        altitude_min_grid, altitude_max_grid + grid_spacing / 2, grid_spacing
    )

    # Filter to only include lines within the visual field
    azimuth_lines = azimuth_lines[
        (azimuth_lines >= azimuth_min) & (azimuth_lines <= azimuth_max)
    ]
    altitude_lines = altitude_lines[
        (altitude_lines >= altitude_min) & (altitude_lines <= altitude_max)
    ]

    # Generate frames for each direction and position
    for row, direction in enumerate(directions):
        # Select the appropriate spherical correction for this direction
        if direction in ["left-to-right", "right-to-left"]:
            spherical_correction = sc_azimuth
            grid_correction = sc_azimuth_inverted
        else:
            spherical_correction = sc_altitude
            grid_correction = sc_altitude_inverted

        # Update the sweep direction for the stimulus
        params["sweep_direction"] = direction

        # Create the stimulus
        stimulus = DriftingBarStimulus(params)

        # Calculate frame indices for each progress point
        frame_count = stimulus.frame_count
        frame_indices = [int(p * (frame_count - 1)) for p in progress_points]

        for col, (idx, progress) in enumerate(zip(frame_indices, progress_points)):
            # Generate the frame
            frame = stimulus.get_frame(idx)

            # Display the frame with proper angular coordinates (center at 0,0)
            extent = [
                azimuth_min,
                azimuth_max,
                altitude_max,
                altitude_min,
            ]  # Note: y-axis is flipped

            # Create the subplot with the correct aspect ratio
            ax = axs[row, col]
            im = ax.imshow(frame, extent=extent, cmap="gray", aspect=frame_aspect_ratio)

            # Set axis limits to exactly match the visual field
            ax.set_xlim(azimuth_min, azimuth_max)
            ax.set_ylim(altitude_max, altitude_min)  # Flipped for image coordinates

            # Set tick positions to match grid lines (25-degree intervals)
            ax.set_xticks(azimuth_lines)
            ax.set_yticks(altitude_lines)

            # Format tick labels to be cleaner (no decimal places)
            ax.set_xticklabels(
                [f"{int(x)}" if x == int(x) else f"{x}" for x in azimuth_lines]
            )
            ax.set_yticklabels(
                [f"{int(y)}" if y == int(y) else f"{y}" for y in altitude_lines]
            )

            # Reduce tick label size to prevent overlapping
            ax.tick_params(axis="both", labelsize=8)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Intensity")

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

            # Draw transformed grid lines that match the image transformation
            # Create straight lines in degree space and apply the inverse transformation
            # to make the grid appear concave rather than convex

            # Draw transformed vertical lines (constant azimuth)
            for azimuth in azimuth_lines:
                # Create straight lines in degree space
                y_points = np.linspace(altitude_min, altitude_max, 200)
                x_points = np.full_like(y_points, azimuth)

                # Apply the inverted spherical transformation for the grid
                x_transformed, y_transformed = (
                    grid_correction._apply_spherical_transformation(x_points, y_points)
                )

                # Plot the line
                line_style = "-"
                alpha = 0.6 if np.isclose(azimuth, 0, atol=1e-6) else 0.3
                linewidth = 1.2 if np.isclose(azimuth, 0, atol=1e-6) else 0.8
                ax.plot(
                    x_transformed,
                    y_transformed,
                    "b" + line_style,
                    alpha=alpha,
                    linewidth=linewidth,
                )

            # Draw transformed horizontal lines (constant altitude)
            for altitude in altitude_lines:
                # Create straight lines in degree space
                x_points = np.linspace(azimuth_min, azimuth_max, 200)
                y_points = np.full_like(x_points, altitude)

                # Apply the inverted spherical transformation for the grid
                x_transformed, y_transformed = (
                    grid_correction._apply_spherical_transformation(x_points, y_points)
                )

                # Plot the line
                line_style = "-"
                alpha = 0.6 if np.isclose(altitude, 0, atol=1e-6) else 0.3
                linewidth = 1.2 if np.isclose(altitude, 0, atol=1e-6) else 0.8
                ax.plot(
                    x_transformed,
                    y_transformed,
                    "b" + line_style,
                    alpha=alpha,
                    linewidth=linewidth,
                )

            # Mark the exact center (0,0) with a small crosshair
            ax.plot(0, 0, "+", color="r", markersize=8, markeredgewidth=2)

    # Add row titles
    fig.text(0.02, 0.75, "Left to Right", fontsize=14, rotation=90)
    fig.text(0.02, 0.25, "Bottom to Top", fontsize=14, rotation=90)

    # Ensure tight layout with proper spacing
    plt.tight_layout(rect=(0.03, 0, 1, 0.95))
    plt.savefig("test_output/comparison_grid.png", dpi=150)
    print("Saved composite view to test_output/comparison_grid.png")


if __name__ == "__main__":
    create_composite_view()
