# create_video.py
"""
Create a video of the drifting bar with spherical correction.
"""

import numpy as np
import cv2
import os
from src.stimuli.drifting_bar import DriftingBarStimulus


def create_videos():
    """Create videos for both horizontal and vertical bar directions with proper spherical transformation"""
    # Create output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)

    # Set up parameters
    params = {
        "resolution": (2560, 1440),  # Higher resolution as requested
        "width": 2560,
        "height": 1440,
        "bar_width": 20.0,
        "grid_spacing_x": 25.0,
        "grid_spacing_y": 25.0,
        "drift_speed": 9.0,  # 9 degrees per second as requested
        "temporal_freq": 6.0,  # Exactly 6Hz (166 ms period) as specified
        "fps": 60.0,
        "screen_distance": 10.0,
        "screen_angle": 20.0,
        "horizontal_meridian_offset": 0.0,  # For proper alignment with grid
        "x_size": 147.0,  # degrees horizontal
        "y_size": 153.0,  # degrees vertical
        "drift_repeats": 1,  # Explicitly set to 1 to ensure no repeats
    }

    # Generate all four directions, but with correct naming for the output files
    # Fix: The actual movement is opposite to what the direction name suggests in DriftingBarStimulus
    direction_mapping = {
        "right-to-left": "left-to-right",  # Bar appears to move left-to-right when using right-to-left setting
        "left-to-right": "right-to-left",  # Bar appears to move right-to-left when using left-to-right setting
        "top-to-bottom": "bottom-to-top",  # Bar appears to move bottom-to-top when using top-to-bottom setting
        "bottom-to-top": "top-to-bottom",  # Bar appears to move top-to-bottom when using bottom-to-top setting
    }

    # Use internal directions for DriftingBarStimulus, but label videos with actual observed directions
    internal_directions = [
        "right-to-left",
        "left-to-right",
        "top-to-bottom",
        "bottom-to-top",
    ]

    for internal_direction in internal_directions:
        # Get the actual direction for file naming and display
        actual_direction = direction_mapping[internal_direction]

        # Update direction
        direction_params = params.copy()
        direction_params["sweep_direction"] = internal_direction

        # Create the stimulus with proper configuration for this direction
        stimulus = DriftingBarStimulus(direction_params)
        print(
            f"Generating {actual_direction} video with {stimulus.frame_count} frames..."
        )

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = f"test_output/{actual_direction.replace('-', '_')}_video.mp4"
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            stimulus.fps,
            (params["resolution"][0], params["resolution"][1]),
        )

        # Also create a pure black and white version with no overlays
        bw_output_path = (
            f"test_output/{actual_direction.replace('-', '_')}_video_bw.mp4"
        )
        bw_video_writer = cv2.VideoWriter(
            bw_output_path,
            fourcc,
            stimulus.fps,
            (params["resolution"][0], params["resolution"][1]),
        )

        # Generate frames for a complete sweep
        frame_count = stimulus.frame_count
        start_frame = 0
        end_frame = frame_count  # Use all frames for complete sweep
        step = 1  # Use every frame for smoother video

        # Calculate angular range
        is_horizontal = internal_direction in ["left-to-right", "right-to-left"]
        if is_horizontal:
            field_size = stimulus.x_size  # horizontal visual field size in degrees
            axis_label = "Azimuth"
        else:
            field_size = stimulus.y_size  # vertical visual field size in degrees
            axis_label = "Altitude"

        bar_width = stimulus.bar_width
        start_pos = -field_size / 2 - bar_width
        end_pos = field_size / 2 + bar_width
        travel_distance = end_pos - start_pos

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # white
        thickness = 2

        for i in range(start_frame, end_frame, step):
            # Generate frame with proper transformation
            frame = stimulus.get_frame(i)

            # Calculate progress and angular position
            progress = i / max(1, frame_count - 1)

            # Calculate position based on direction - ensure correct movement
            if internal_direction == "left-to-right":
                position = start_pos + progress * travel_distance
            elif internal_direction == "right-to-left":
                position = end_pos - progress * travel_distance
            elif internal_direction == "bottom-to-top":
                position = start_pos + progress * travel_distance
            else:  # top-to-bottom
                position = end_pos - progress * travel_distance

            # Save the pure black and white frame with no overlays
            # Convert to 3-channel but preserve grayscale appearance
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            bw_video_writer.write(frame_bw)

            # For the annotated version, convert to RGB
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_GRAY2BGR
            )  # Preserve black and white appearance

            # Add position text overlay
            position_text = f"{axis_label}: {position:.1f}°"

            # Get text size
            text_size = cv2.getTextSize(position_text, font, font_scale, thickness)[0]

            # Position text at bottom-right
            text_x = frame_rgb.shape[1] - text_size[0] - 10
            text_y = frame_rgb.shape[0] - 10

            # Add black background for text visibility
            cv2.rectangle(
                frame_rgb,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1,
            )

            # Draw text
            cv2.putText(
                frame_rgb,
                position_text,
                (text_x, text_y),
                font,
                font_scale,
                font_color,
                thickness,
            )

            # Draw center point (0°, 0°)
            center_x = frame_rgb.shape[1] // 2
            center_y = frame_rgb.shape[0] // 2
            cv2.drawMarker(
                frame_rgb,
                (center_x, center_y),
                (0, 255, 0),  # Green marker
                cv2.MARKER_CROSS,
                20,
                2,
            )

            # Draw degree grid lines (0° vertical and horizontal)
            height, width = frame_rgb.shape[:2]

            # Convert degree coordinates to pixel coordinates
            center_x = width // 2
            center_y = height // 2

            # Draw horizontal equator (0° latitude)
            cv2.line(frame_rgb, (0, center_y), (width, center_y), (128, 128, 128), 1)

            # Draw vertical meridian (0° longitude)
            cv2.line(frame_rgb, (center_x, 0), (center_x, height), (128, 128, 128), 1)

            # Add a note about the transformation
            note_text = f"Spherically-transformed bar (9°/s, 6Hz) - {actual_direction}"
            note_size = cv2.getTextSize(
                note_text, font, font_scale * 0.8, thickness - 1
            )[0]
            cv2.putText(
                frame_rgb,
                note_text,
                (10, 30),
                font,
                font_scale * 0.8,
                (200, 200, 200),
                thickness - 1,
            )

            # Write to video
            video_writer.write(frame_rgb)

        # Release video writers
        video_writer.release()
        bw_video_writer.release()
        print(f"Annotated video saved to {output_path}")
        print(f"Black and white video saved to {bw_output_path}")


if __name__ == "__main__":
    create_videos()
    print("Videos created successfully!")
