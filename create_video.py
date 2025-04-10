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
        "resolution": (800, 600),
        "width": 800,
        "height": 600,
        "bar_width": 20.0,
        "grid_spacing_x": 25.0,
        "grid_spacing_y": 25.0,
        "drift_speed": 15.0,  # faster for testing
        "temporal_freq": 6.0,
        "fps": 60.0,
        "screen_distance": 10.0,
        "screen_angle": 20.0,
        "horizontal_meridian_offset": 20.0,
        "x_size": 147.0,  # degrees horizontal
        "y_size": 153.0,  # degrees vertical
    }

    directions = ["left-to-right", "bottom-to-top"]

    for direction in directions:
        # Update direction
        direction_params = params.copy()
        direction_params["sweep_direction"] = direction

        # Create the stimulus with proper configuration for this direction
        stimulus = DriftingBarStimulus(direction_params)
        print(f"Generating {direction} video with {stimulus.frame_count} frames...")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = f"test_output/{direction.replace('-', '_')}_video.mp4"
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            stimulus.fps,
            (params["resolution"][0], params["resolution"][1]),
        )

        # Generate fewer frames for testing (every 4th frame)
        frame_count = stimulus.frame_count
        step = 4  # Use every 4th frame to speed up generation

        # Generate frames for a shorter segment (1/4 of total)
        start_frame = 0
        end_frame = min(frame_count, 200)  # limit to at most 200 frames

        # Calculate angular range
        is_horizontal = direction in ["left-to-right", "right-to-left"]
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

            if direction == "left-to-right":
                position = start_pos + progress * travel_distance
            elif direction == "right-to-left":
                position = end_pos - progress * travel_distance
            elif direction == "bottom-to-top":
                position = start_pos + progress * travel_distance
            else:  # top-to-bottom
                position = end_pos - progress * travel_distance

            # Convert to RGB for video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

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
                (0, 0, 255),  # red
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
            cv2.line(frame_rgb, (0, center_y), (width, center_y), (0, 0, 128), 1)

            # Draw vertical meridian (0° longitude)
            cv2.line(frame_rgb, (center_x, 0), (center_x, height), (0, 0, 128), 1)

            # Add a note about the transformation
            note_text = "Spherically-transformed bar"
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

        # Release video writer
        video_writer.release()
        print(f"Video saved to {output_path}")


if __name__ == "__main__":
    create_videos()
    print("Videos created successfully!")
