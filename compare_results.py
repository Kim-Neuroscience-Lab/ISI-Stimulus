# compare_results.py
"""
Script to display a grid of the generated frames for comparison, labeled with angular position.
"""

import matplotlib.pyplot as plt
import os
import numpy as np


def create_composite_view():
    # Get screen resolution parameters
    width, height = 800, 600
    aspect_ratio = width / height  # 800/600 = 4/3

    # Create figure with 2 rows (directions) and 5 columns (selected frames)
    # Adjust figure size to maintain proper aspect ratio for each subplot
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
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

    # Get screen resolution parameters
    width, height = params["resolution"]
    aspect_ratio = width / height  # 800/600 = 4/3

    # Calculate data aspect ratio for proper rendering
    data_aspect = (azimuth_max - azimuth_min) / (altitude_max - altitude_min)

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
        else:
            spherical_correction = sc_altitude

        # Update the sweep direction for the stimulus
        params["sweep_direction"] = direction

        # Create the stimulus
        stimulus = DriftingBarStimulus(params)

        # Get checker parameters to ensure grid lines match exactly
        checker_params = stimulus.checker_parameters

        # Use checker size for grid instead of the default grid spacing
        checker_size = checker_params["checker_size"]

        # Create grid lines at exactly the checker cell boundaries
        # Start at even multiples of checker_size to align with the checkerboard pattern
        azimuth_checker_min = np.floor(azimuth_min / checker_size) * checker_size
        azimuth_checker_max = np.ceil(azimuth_max / checker_size) * checker_size
        altitude_checker_min = np.floor(altitude_min / checker_size) * checker_size
        altitude_checker_max = np.ceil(altitude_max / checker_size) * checker_size

        # Generate grid lines at exactly the checker cell boundaries
        azimuth_checker_lines = np.arange(
            azimuth_checker_min, azimuth_checker_max + checker_size / 2, checker_size
        )
        altitude_checker_lines = np.arange(
            altitude_checker_min, altitude_checker_max + checker_size / 2, checker_size
        )

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

            # Display the frame - use display aspect ratio for proper screen representation
            im = ax.imshow(frame, extent=extent, cmap="gray", aspect=aspect_ratio)

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

            # Remove colorbar - no longer needed
            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # cbar.set_label("Intensity")

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
            # Use exactly the same approach as in drifting_bar.py for transforming the coordinates

            # For each frame, get the entire grid of spherical coordinates
            # This is needed to calculate the same offsets as the checkerboard texture
            # Generate a complete set of theta and phi values for the entire frame
            y_grid_raw, x_grid_raw = np.mgrid[
                altitude_min:altitude_max:600j, azimuth_min:azimuth_max:800j
            ]

            # Convert to screen coordinates
            screen_distance = checker_params["screen_distance"]
            x_rad = np.radians(x_grid_raw)
            y_rad = np.radians(y_grid_raw)

            # Calculate screen coordinates
            y_screen = screen_distance * np.tan(x_rad)
            z_screen = screen_distance * np.tan(y_rad)
            x0 = screen_distance

            # Calculate spherical coordinates
            sqrt_term = np.sqrt(x0**2 + y_screen**2 + z_screen**2)

            # Calculate θ and φ
            z_over_sqrt = np.zeros_like(z_screen)
            valid_indices = sqrt_term > 0
            z_over_sqrt[valid_indices] = (
                z_screen[valid_indices] / sqrt_term[valid_indices]
            )
            z_over_sqrt = np.clip(z_over_sqrt, -1.0, 1.0)
            theta = np.pi / 2 - np.arccos(z_over_sqrt)

            phi = np.zeros_like(y_screen)
            nonzero_indices = np.abs(x0) > 1e-10
            phi[nonzero_indices] = np.arctan2(-y_screen[nonzero_indices], x0)

            # Convert back to degrees
            theta_deg = np.degrees(theta)
            phi_deg = np.degrees(phi)

            # Calculate the exact same offsets used in drifting_bar.py
            # These ensure alignment between grid and texture
            checker_size = checker_params["checker_size"]
            theta_offset = np.floor(np.min(theta_deg) / checker_size) * checker_size
            phi_offset = np.floor(np.min(phi_deg) / checker_size) * checker_size

            # Draw checker cell boundaries using the exact same alignment as the texture
            # Generate grid lines at exact checker boundaries where the pattern changes
            # Theta grid lines (horizontal in spherical coordinates)
            # Need to include every position where the pattern changes (each integer multiple + offset)
            theta_transitions = []
            # Ensure we cover the full range with a small buffer to catch all checkerboard boundaries
            min_theta = np.min(theta_deg) - checker_size / 2
            max_theta = np.max(theta_deg) + checker_size / 2
            for i in range(
                int(np.floor(min_theta / checker_size)),
                int(np.ceil(max_theta / checker_size)) + 1,
            ):
                theta_transitions.append(i * checker_size + theta_offset)

            # Special case: ensure we have a line under the equator by explicitly checking
            equator_line = -checker_size + theta_offset  # Line just under 0
            if equator_line not in theta_transitions:
                theta_transitions.append(equator_line)

            # Sort to ensure proper rendering order
            theta_transitions.sort()

            # Phi grid lines (vertical in spherical coordinates)
            phi_transitions = []
            # Similar buffer for phi values
            min_phi = np.min(phi_deg) - checker_size / 2
            max_phi = np.max(phi_deg) + checker_size / 2
            for i in range(
                int(np.floor(min_phi / checker_size)),
                int(np.ceil(max_phi / checker_size)) + 1,
            ):
                phi_transitions.append(i * checker_size + phi_offset)
            # Sort to ensure proper rendering order
            phi_transitions.sort()

            # Draw vertical checker boundaries (constant phi)
            for phi_value in phi_transitions:
                # Create points along this phi value
                y_points = np.linspace(altitude_min, altitude_max, 200)
                x_points = np.full_like(y_points, 0)  # Placeholder

                # Apply the same transformation as for the texture
                screen_distance = checker_params["screen_distance"]
                x_rad = np.radians(x_points)
                y_rad = np.radians(y_points)

                # Calculate screen coordinates
                y_screen = screen_distance * np.tan(x_rad)
                z_screen = screen_distance * np.tan(y_rad)
                x0 = screen_distance

                # Calculate spherical coordinates for each point
                sqrt_term = np.sqrt(x0**2 + y_screen**2 + z_screen**2)

                # We need to find points with constant phi value
                # Instead of creating points at constant azimuth and transforming,
                # we'll calculate what azimuth values would give our desired phi
                # phi = arctan2(-y, x0) => -y = x0 * tan(phi)
                y_desired = -x0 * np.tan(np.radians(phi_value))

                # Now calculate what azimuth values would give these y values
                # y = screen_distance * tan(x_rad) => x_rad = arctan(y/screen_distance)
                azimuth_for_phi = np.degrees(np.arctan2(y_desired, screen_distance))

                # Create points at these azimuth values
                x_points = np.full_like(y_points, azimuth_for_phi)

                # Apply the spherical transformation to get the exact coordinates
                x_rad = np.radians(x_points)
                y_rad = np.radians(y_points)

                # Calculate screen coordinates
                y_screen = screen_distance * np.tan(x_rad)
                z_screen = screen_distance * np.tan(y_rad)

                # Calculate spherical coordinates
                sqrt_term = np.sqrt(x0**2 + y_screen**2 + z_screen**2)

                z_over_sqrt = np.zeros_like(z_screen)
                valid_indices = sqrt_term > 0
                z_over_sqrt[valid_indices] = (
                    z_screen[valid_indices] / sqrt_term[valid_indices]
                )
                z_over_sqrt = np.clip(z_over_sqrt, -1.0, 1.0)
                theta = np.pi / 2 - np.arccos(z_over_sqrt)

                phi = np.zeros_like(y_screen)
                nonzero_indices = np.abs(x0) > 1e-10
                phi[nonzero_indices] = np.arctan2(-y_screen[nonzero_indices], x0)

                # Convert back to degrees
                theta_deg = np.degrees(theta)
                phi_deg = np.degrees(phi)

                # Plot checker cell boundaries
                ax.plot(
                    phi_deg,
                    theta_deg,
                    color="r",
                    linestyle="-",
                    alpha=0.7,  # Darker
                    linewidth=1.0,  # Thicker
                )

            # Draw horizontal checker boundaries (constant theta)
            for theta_value in theta_transitions:
                # Create points along this theta value
                x_points = np.linspace(azimuth_min, azimuth_max, 200)
                y_points = np.zeros_like(x_points)  # Will be filled with altitudes

                # We need to find points with constant theta value
                # For each x point, calculate the altitude that gives this theta

                # Convert to radians
                theta_rad = np.radians(theta_value)
                x_rad = np.radians(x_points)

                # Calculate y screen coordinates
                y_screen = screen_distance * np.tan(x_rad)

                # For a constant theta, we need to solve for z:
                # theta = π/2 - arccos(z/sqrt(x0^2 + y^2 + z^2))
                # z/sqrt(...) = sin(theta)
                # z^2 = sin^2(theta) * (x0^2 + y^2 + z^2)
                # z^2(1 - sin^2(theta)) = sin^2(theta) * (x0^2 + y^2)
                # z^2 = sin^2(theta) * (x0^2 + y^2) / (1 - sin^2(theta))
                # z^2 = sin^2(theta) * (x0^2 + y^2) / cos^2(theta)
                # z = sin(theta) * sqrt(x0^2 + y^2) / cos(theta)

                sin_theta = np.sin(theta_rad)
                cos_theta = np.cos(theta_rad)

                # Avoid division by zero
                if abs(cos_theta) < 1e-6:
                    continue

                # Calculate z coordinate
                z = sin_theta * np.sqrt(x0**2 + y_screen**2) / cos_theta

                # Convert back to degrees for altitude
                altitude = np.degrees(np.arctan2(z, x0))
                y_points = altitude

                # Plot constant theta line
                ax.plot(
                    x_points,
                    y_points,
                    color="r",
                    linestyle="-",
                    alpha=0.7,  # Darker
                    linewidth=1.0,  # Thicker
                )

            # Add a clean border around the plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

            # Mark the exact center (0,0) with a small red crosshair
            ax.plot(0, 0, "+", color="r", markersize=6, markeredgewidth=1.5)

    # Add row titles
    fig.text(0.02, 0.75, "Left to Right", fontsize=14, rotation=90)
    fig.text(0.02, 0.25, "Bottom to Top", fontsize=14, rotation=90)

    # Ensure tight layout with proper spacing
    plt.tight_layout(rect=(0.03, 0, 1, 0.95))
    plt.savefig("test_output/comparison_grid.png", dpi=150)
    print("Saved composite view to test_output/comparison_grid.png")


if __name__ == "__main__":
    create_composite_view()
