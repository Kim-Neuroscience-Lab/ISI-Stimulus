# src/utils/video_converter.py

"""
Utility functions for converting between video formats.
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_avi_to_mp4(input_file, output_file=None):
    """
    Convert an AVI file to MP4 format using ffmpeg.

    Args:
        input_file (str): Path to the input AVI file
        output_file (str, optional): Path to the output MP4 file. If None,
            the output file will have the same name as the input file but with .mp4 extension.

    Returns:
        str: Path to the output MP4 file if successful, None otherwise
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return None

    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix(".mp4"))

    try:
        # Using ffmpeg to convert without re-encoding (-c copy)
        # If this fails, we'll use a more robust conversion
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-c:v",
            "h264",
            "-crf",
            "23",
            "-preset",
            "fast",
            "-y",
            output_file,
        ]

        logger.info(f"Converting {input_file} to {output_file}")
        result = subprocess.run(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            logger.warning(
                f"Initial conversion failed, trying with pixel format conversion"
            )
            # Try again with pixel format conversion
            cmd = [
                "ffmpeg",
                "-i",
                input_file,
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "h264",
                "-crf",
                "23",
                "-preset",
                "fast",
                "-y",
                output_file,
            ]
            result = subprocess.run(
                cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )

        if result.returncode != 0:
            logger.error(f"Failed to convert video: {result.stderr}")
            return None

        if os.path.exists(output_file):
            logger.info(f"Successfully converted to {output_file}")
            return output_file
        else:
            logger.error(f"Output file was not created: {output_file}")
            return None

    except Exception as e:
        logger.error(f"Error converting video: {str(e)}")
        return None


def optimize_video_for_web(input_file, output_file=None, target_size_mb=None):
    """
    Optimize a video for web viewing by reducing file size while maintaining reasonable quality.

    Args:
        input_file (str): Path to the input video file
        output_file (str, optional): Path to the output video file. If None,
            the output file will have "_web" appended to the original name.
        target_size_mb (float, optional): Target file size in megabytes. If None,
            a default compression will be applied.

    Returns:
        str: Path to the optimized video file if successful, None otherwise
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return None

    if output_file is None:
        input_path = Path(input_file)
        output_file = str(
            input_path.parent / f"{input_path.stem}_web{input_path.suffix}"
        )

    try:
        if target_size_mb is not None:
            # Calculate bitrate based on target size
            # Get video duration
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                input_file,
            ]
            result = subprocess.run(
                probe_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )
            if result.returncode != 0:
                logger.error(f"Failed to get video duration: {result.stderr}")
                return None

            duration = float(result.stdout.strip())
            # Calculate target bitrate (bits/second)
            # target_size_bytes = target_size_mb * 1024 * 1024 * 8 (convert to bits)
            # bitrate = target_size_bits / duration
            target_bitrate = int((target_size_mb * 1024 * 1024 * 8) / duration)

            cmd = [
                "ffmpeg",
                "-i",
                input_file,
                "-c:v",
                "h264",
                "-b:v",
                f"{target_bitrate}",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                output_file,
            ]
        else:
            # Standard web optimization
            cmd = [
                "ffmpeg",
                "-i",
                input_file,
                "-c:v",
                "h264",
                "-crf",
                "28",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                output_file,
            ]

        logger.info(f"Optimizing {input_file} for web viewing")
        result = subprocess.run(
            cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            logger.error(f"Failed to optimize video: {result.stderr}")
            return None

        if os.path.exists(output_file):
            logger.info(f"Successfully optimized to {output_file}")
            return output_file
        else:
            logger.error(f"Output file was not created: {output_file}")
            return None

    except Exception as e:
        logger.error(f"Error optimizing video: {str(e)}")
        return None
