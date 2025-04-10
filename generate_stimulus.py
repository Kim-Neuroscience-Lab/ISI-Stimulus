# generate_stimulus.py
"""
Command-line utility for generating visual stimuli based on MMC1 specifications.

This script allows users to generate drifting bar or drifting grating stimuli
as described in the MMC1 document for neuroscience experiments.
"""

import os
import sys
import argparse
import numpy as np
import logging
import cv2
import multiprocessing
import platform
import subprocess
from typing import Dict, Any, Union, cast

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.stimuli.factory import create_mmc1_stimulus
from src.stimuli.base_stimulus import BaseStimulus
from src.stimuli.drifting_bar import DriftingBarStimulus


def setup_logging(verbose=False):
    """
    Set up logging configuration.

    Args:
        verbose: Whether to enable debug-level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def detect_gpu():
    """
    Detect and log GPU capabilities.

    Returns:
        bool: True if GPU is detected, False otherwise
    """
    logger = logging.getLogger("gpu_detection")
    gpu_detected = False

    logger.info("Checking GPU capabilities:")
    logger.info(f"Platform: {platform.system()} {platform.release()}")

    try:
        # Check OpenCV version
        logger.info(f"OpenCV version: {cv2.__version__}")

        # Check for CUDA support
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                gpu_detected = True
                logger.info(
                    f"✓ CUDA GPU acceleration available with {cuda_devices} device(s)"
                )

                # Try to get more CUDA info
                try:
                    device_id = cv2.cuda.getDevice()
                    logger.info(f"  CUDA device ID: {device_id}")
                except Exception as e:
                    logger.debug(f"Could not get CUDA device ID: {e}")
            else:
                logger.info("✗ CUDA not available (0 devices)")
        else:
            logger.info("✗ CUDA not supported in this OpenCV build")

        # Check for OpenCL support
        if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "haveOpenCL"):
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.useOpenCL():
                    gpu_detected = True
                    logger.info("✓ OpenCL GPU acceleration available")

                    # Try to get OpenCL device info
                    try:
                        # Try different methods to get OpenCL device info
                        openCL_info = "OpenCL device"
                        if hasattr(cv2.ocl, "Device"):
                            try:
                                device = cv2.ocl.Device.getDefault()
                                if hasattr(device, "name"):
                                    openCL_info = device.name()
                            except:
                                pass
                        logger.info(f"  {openCL_info}")
                    except Exception as e:
                        logger.debug(f"Could not get OpenCL device info: {e}")
                else:
                    logger.info("✗ OpenCL available but not usable")
            else:
                logger.info("✗ OpenCL not available")
        else:
            logger.info("✗ OpenCL not supported in this OpenCV build")

        # Check for Apple Silicon
        if platform.system() == "Darwin":
            try:
                is_apple_silicon = False
                # On macOS, platform.processor() may return 'i386' even on Apple Silicon
                # Try multiple methods to detect Apple Silicon
                if "Apple" in platform.processor():
                    is_apple_silicon = True
                else:
                    # Alternative detection methods for Apple Silicon
                    # Check for arm64 architecture
                    try:
                        import subprocess

                        result = subprocess.run(
                            ["uname", "-m"], capture_output=True, text=True
                        )
                        if "arm64" in result.stdout.lower():
                            is_apple_silicon = True
                    except:
                        pass

                    # Check via sysctl if above method failed
                    if not is_apple_silicon:
                        try:
                            result = subprocess.run(
                                ["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True,
                                text=True,
                            )
                            if "Apple" in result.stdout:
                                is_apple_silicon = True
                        except:
                            pass

                if is_apple_silicon:
                    gpu_detected = True
                    logger.info(f"✓ Apple Silicon detected (M-series)")
                else:
                    logger.info(f"✗ Apple Silicon not detected (Intel Mac)")
            except Exception as e:
                logger.info(f"✗ Could not determine Apple Silicon status: {e}")

    except Exception as e:
        logger.warning(f"Error during GPU detection: {e}")

    return gpu_detected


def save_stimulus_frames(
    frames: np.ndarray,
    output_dir: str,
    direction: str,
    downscale_factor: float = 1.0,
    codec: str = "mp4v",
) -> None:
    """
    Save stimulus frames as a video file, with optional downscaling.

    Args:
        frames: Numpy array of frames
        output_dir: Directory to save the video
        direction: Direction label for the filename
        downscale_factor: Factor to downscale the output (1.0 = original size, 0.5 = half size)
        codec: Video codec to use (mp4v, avc1, xvid, mjpg)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a VideoWriter object
    height, width = frames[0].shape

    # Apply downscaling if requested
    if downscale_factor != 1.0:
        # Calculate new dimensions
        new_width = int(width * downscale_factor)
        new_height = int(height * downscale_factor)

        # Ensure dimensions are even (required by some codecs)
        new_width = new_width + (new_width % 2)
        new_height = new_height + (new_height % 2)

        # Ensure minimum dimensions of 64x64 for compatibility (was 32x32)
        new_width = max(64, new_width)
        new_height = max(64, new_height)

        # Log the downscaling information
        logging.info(
            f"Downscaling output from {width}x{height} to {new_width}x{new_height}"
        )

        # Set dimensions for video writer
        output_width, output_height = new_width, new_height
    else:
        output_width, output_height = width, height

    # Use integer value for codec instead of VideoWriter_fourcc
    try:
        if codec == "mp4v":
            fourcc = int(cv2.VideoWriter.fourcc("m", "p", "4", "v"))
            ext = ".mp4"
        elif codec == "avc1":
            fourcc = int(cv2.VideoWriter.fourcc("a", "v", "c", "1"))
            ext = ".mp4"
        elif codec == "xvid":
            fourcc = int(cv2.VideoWriter.fourcc("X", "V", "I", "D"))
            ext = ".avi"  # Use AVI for XVID codec
        elif codec == "mjpg":
            fourcc = int(cv2.VideoWriter.fourcc("M", "J", "P", "G"))
            ext = ".avi"  # Use AVI for MJPG codec
        else:
            logging.warning(f"Unknown codec {codec}, falling back to mp4v")
            fourcc = int(cv2.VideoWriter.fourcc("m", "p", "4", "v"))
            ext = ".mp4"

        logging.info(f"Using codec: {codec} with file extension: {ext}")
    except Exception as e:
        logging.warning(f"Error with {codec} codec: {e}, falling back to basic codec")
        fourcc = int(cv2.VideoWriter.fourcc("X", "V", "I", "D"))
        ext = ".avi"

    video_path = os.path.join(output_dir, f"mmc1_drifting_bar_{direction}{ext}")

    # VideoWriter requires RGB, not grayscale (isColor must be True)
    out = cv2.VideoWriter(
        video_path, fourcc, 30.0, (output_width, output_height), isColor=True
    )

    if not out.isOpened():
        logging.error(
            f"Failed to open video writer for {video_path}. Trying alternative approach."
        )
        # Try a fallback approach with a different codec
        if ext == ".mp4":
            fallback_path = os.path.join(
                output_dir, f"mmc1_drifting_bar_{direction}.avi"
            )
            logging.info(f"Trying with AVI format instead: {fallback_path}")
            fallback_fourcc = int(cv2.VideoWriter.fourcc("X", "V", "I", "D"))
            out = cv2.VideoWriter(
                fallback_path,
                fallback_fourcc,
                30.0,
                (output_width, output_height),
                isColor=True,
            )
            if out.isOpened():
                video_path = fallback_path
            else:
                logging.error(
                    "Fallback approach also failed. Unable to create video file."
                )
                return

    # Write each frame to the video
    frame_count = 0
    for frame in frames:
        # Apply downscaling if requested
        if downscale_factor != 1.0:
            frame = cv2.resize(
                frame, (output_width, output_height), interpolation=cv2.INTER_AREA
            )

        # Convert to 3-channel RGB (not grayscale)
        # Use a colormap for better visibility
        frame_color = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        out.write(frame_color)
        frame_count += 1

    # Release the VideoWriter
    out.release()

    # Verify the output file
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        if file_size < 1000:  # Less than 1KB is suspicious
            logging.warning(
                f"Warning: Output file {video_path} is very small ({file_size} bytes)"
            )
        else:
            logging.info(
                f"Saved video to {video_path} ({file_size} bytes, {frame_count} frames)"
            )
    else:
        logging.error(f"Error: Output file {video_path} was not created")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visual stimuli based on MMC1 specifications"
    )

    parser.add_argument(
        "--type",
        choices=["drifting_bar", "drifting_grating"],
        default="drifting_bar",
        help="Type of stimulus to generate",
    )

    parser.add_argument(
        "--mode",
        choices=["intrinsic", "two-photon"],
        default="intrinsic",
        help="Imaging mode (affects drift speed)",
    )

    parser.add_argument(
        "--output",
        default="./videos",
        help="Output directory for stimulus videos",
    )

    parser.add_argument(
        "--resolution",
        default="1920x1080",
        help="Resolution of the stimulus in format WIDTHxHEIGHT",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate a quick preview with lower resolution and fewer frames",
    )

    # Performance optimization options
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration if available",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration even if available",
    )

    parser.add_argument(
        "--detect-gpu",
        action="store_true",
        help="Detect and display GPU capabilities, then exit",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=min(multiprocessing.cpu_count(), 8),
        help="Number of processes to use for multiprocessing (default: auto)",
    )

    parser.add_argument(
        "--precompute-all",
        action="store_true",
        help="Precompute all possible transformations (uses more memory but faster)",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per direction (default: 1, original MMC1: 10)",
    )

    parser.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Factor to downscale output video (e.g., 0.5 for half size, 1.0 for original size)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process only every Nth frame (e.g., 10 = 10x faster processing but choppier video)",
    )

    parser.add_argument(
        "--codec",
        choices=["mp4v", "avc1", "xvid", "mjpg"],
        default="mp4v",
        help="Video codec to use (try avc1 or xvid if mp4v gives problems)",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before generating new files",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Run GPU detection if requested
    if args.detect_gpu:
        gpu_available = detect_gpu()
        print(
            f"\nGPU Detection Results: {'GPU Available' if gpu_available else 'No Compatible GPU Found'}"
        )
        sys.exit(0)

    # Parse resolution
    width, height = map(int, args.resolution.split("x"))

    # Convert imaging mode to boolean for two-photon
    is_two_photon = args.mode == "two-photon"

    # Process GPU options
    use_gpu = None  # Let the system decide by default
    if args.gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False

    # Set up parameters based on arguments
    parameters = {
        "resolution": (width, height),
        "is_two_photon": is_two_photon,
        "use_gpu": use_gpu,
        "num_processes": args.processes,
        "precompute_all": args.precompute_all,
        "drift_repeats": args.repeats,
    }

    # If preview mode, reduce resolution and other parameters for quicker generation
    if args.preview:
        parameters["resolution"] = (640, 360)
        parameters["drift_repeats"] = 1  # Only one repeat per direction

    # Log optimization information
    logging.info(f"Performance settings:")
    logging.info(
        f"  GPU acceleration: {'Auto' if use_gpu is None else 'Enabled' if use_gpu else 'Disabled'}"
    )
    logging.info(f"  Multiprocessing: {args.processes} processes")
    logging.info(f"  Precomputation: {'Full' if args.precompute_all else 'Standard'}")
    logging.info(f"  Repeats per direction: {parameters['drift_repeats']}")
    if args.downscale != 1.0:
        logging.info(f"  Output downscaling: {args.downscale:.2f}x")
    logging.info(
        f"Generating {args.type} stimulus for {args.mode} imaging at {width}x{height}"
    )

    # Create stimulus with additional parameters
    try:
        # Clean output directory if requested
        if args.clean and os.path.exists(args.output):
            logging.info(f"Cleaning output directory: {args.output}")
            for filename in os.listdir(args.output):
                if filename.startswith("mmc1_drifting_bar_"):
                    filepath = os.path.join(args.output, filename)
                    try:
                        os.remove(filepath)
                        logging.debug(f"Removed: {filepath}")
                    except Exception as e:
                        logging.warning(f"Could not remove {filepath}: {e}")

        stimulus = create_mmc1_stimulus(
            stimulus_type=args.type,
            is_two_photon=is_two_photon,
            additional_params=parameters,
        )

        if args.type == "drifting_bar":
            # Cast to DriftingBarStimulus to access generate_full_sequence method
            drifting_bar_stimulus = cast(DriftingBarStimulus, stimulus)

            # Add frame skipping info to log if enabled
            if args.frame_skip > 1:
                logging.info(f"Frame skip: Processing every {args.frame_skip}th frame")

            # Use a simplified approach with frame skipping in get_all_frames
            # Store original parameters to restore later
            original_params = drifting_bar_stimulus.parameters.copy()

            # Dictionary to store video segments
            video_segments = {}

            # Generate stimulus for each direction
            directions = [
                "right-to-left",
                "left-to-right",
                "bottom-to-top",
                "top-to-bottom",
            ]

            for direction in directions:
                # Update parameters for this direction
                drifting_bar_stimulus.parameters["sweep_direction"] = direction

                # Generate the frames for this direction
                logging.info(
                    f"Generating {drifting_bar_stimulus.repeats} repeats of {direction} drifting bar"
                )

                # Generate single sequence with frame skipping
                frames = drifting_bar_stimulus.get_all_frames(args.frame_skip)

                # Optimize memory usage for repeated frames
                if drifting_bar_stimulus.repeats > 1:
                    # For large repeats, use memory-efficient approach
                    if (
                        drifting_bar_stimulus.repeats > 5 and frames.nbytes > 1e9
                    ):  # >1GB
                        logging.info(
                            f"Using memory-efficient approach for {drifting_bar_stimulus.repeats} repeats"
                        )
                        # Create a function to generate repeated frames on demand
                        repeated_frames_generator = np.tile(
                            frames, (drifting_bar_stimulus.repeats, 1, 1)
                        )
                        video_segments[direction] = repeated_frames_generator
                    else:
                        # Standard approach for smaller datasets
                        repeated_frames = np.tile(
                            frames, (drifting_bar_stimulus.repeats, 1, 1)
                        )
                        video_segments[direction] = repeated_frames
                else:
                    video_segments[direction] = frames

            # Restore original parameters
            drifting_bar_stimulus.parameters = original_params

            # Save each direction as a separate video
            for direction, frames in video_segments.items():
                save_stimulus_frames(
                    frames, args.output, direction, args.downscale, args.codec
                )
        else:
            # For drifting grating, generate all frames
            frames = stimulus.get_all_frames()

            # Save as a single video
            save_stimulus_frames(
                frames, args.output, "grating", args.downscale, args.codec
            )

        logging.info("Stimulus generation complete")

    except Exception as e:
        logging.error(f"Error generating stimulus: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        else:
            logging.error("Use --verbose for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
