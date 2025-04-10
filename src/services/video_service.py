"""Video generation service for exporting stimuli."""

import numpy as np
import os
from typing import Dict, Any, Optional
import logging


class VideoService:
    """Service for generating and exporting stimulus videos."""

    def __init__(self):
        """Initialize the video service."""
        self.logger = logging.getLogger("VideoService")

        # Configure basic logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Detect GPU capabilities
        self.gpu_info = self._detect_gpu_capabilities()

    def _detect_gpu_capabilities(self) -> Dict[str, Any]:
        """
        Detect available GPU acceleration capabilities.

        Returns:
            Dict with GPU capabilities information
        """
        gpu_info = {
            "has_gpu": False,
            "acceleration_type": None,
            "device_name": None,
            "platform": None,
            "encoder": None,
        }

        try:
            import platform

            system = platform.system()
            gpu_info["platform"] = system

            if system == "Darwin":  # macOS
                # Check for Metal support on Apple
                try:
                    # Check if running on Apple Silicon
                    is_apple_silicon = platform.processor() == "arm"

                    # On macOS, we can use VideoToolbox via ffmpeg
                    import subprocess

                    result = subprocess.run(
                        ["ffmpeg", "-hwaccels"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )

                    if "videotoolbox" in result.stdout.lower():
                        gpu_info["has_gpu"] = True
                        gpu_info["acceleration_type"] = "metal"
                        gpu_info["encoder"] = "videotoolbox"
                        gpu_info["device_name"] = "Apple VideoToolbox"
                        self.logger.info(
                            "Apple Metal GPU acceleration available via VideoToolbox"
                        )
                    else:
                        # Try to determine if Metal is available through pyobjc
                        try:
                            import ctypes

                            metal_framework = ctypes.cdll.LoadLibrary(
                                "/System/Library/Frameworks/Metal.framework/Metal"
                            )
                            gpu_info["has_gpu"] = True
                            gpu_info["acceleration_type"] = "metal"
                            gpu_info["encoder"] = "h264_videotoolbox"
                            gpu_info["device_name"] = "Apple Metal"
                            self.logger.info("Apple Metal framework detected")
                        except (ImportError, OSError):
                            self.logger.info("Apple Metal framework not detected")
                except Exception as e:
                    self.logger.warning(f"Error detecting Metal support: {e}")

            elif system == "Linux" or system == "Windows":
                # Check for CUDA support
                try:
                    import cv2

                    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                    if cuda_devices > 0:
                        gpu_info["has_gpu"] = True
                        gpu_info["acceleration_type"] = "cuda"
                        gpu_info["device_name"] = f"CUDA device ({cuda_devices} found)"

                        # Set appropriate encoder based on platform
                        if system == "Windows":
                            gpu_info["encoder"] = "h264_nvenc"
                        else:  # Linux
                            gpu_info["encoder"] = "h264_nvenc"

                        self.logger.info(
                            f"NVIDIA CUDA acceleration available with {cuda_devices} devices"
                        )
                        return gpu_info
                except (ImportError, AttributeError):
                    self.logger.info("CUDA support not available in OpenCV")

                # Check for OpenCL support (AMD, Intel)
                try:
                    import cv2

                    if cv2.ocl.haveOpenCL():
                        cv2.ocl.setUseOpenCL(True)
                        if cv2.ocl.useOpenCL():
                            gpu_info["has_gpu"] = True
                            gpu_info["acceleration_type"] = "opencl"
                            gpu_info["device_name"] = "OpenCL"

                            # Determine encoder
                            # Try to get more detailed OpenCL device info
                            try:
                                import pyopencl as cl

                                platforms = cl.get_platforms()
                                if platforms:
                                    first_platform = platforms[0]
                                    devices = first_platform.get_devices()
                                    if devices:
                                        first_device = devices[0]
                                        vendor = first_device.vendor
                                        gpu_info["device_name"] = (
                                            f"{vendor} {first_device.name}"
                                        )

                                        # Select encoder based on vendor
                                        if "AMD" in vendor:
                                            gpu_info["encoder"] = (
                                                "h264_amf"
                                                if system == "Windows"
                                                else "h264_vaapi"
                                            )
                                        elif "Intel" in vendor:
                                            gpu_info["encoder"] = (
                                                "h264_qsv"
                                                if system == "Windows"
                                                else "h264_vaapi"
                                            )
                                        else:
                                            gpu_info["encoder"] = "libx264"  # Fallback
                            except ImportError:
                                # Default encoders if pyopencl not available
                                gpu_info["encoder"] = "libx264"

                            self.logger.info("OpenCL acceleration available")
                            return gpu_info
                        else:
                            self.logger.info("OpenCL initialization failed")
                except (ImportError, AttributeError):
                    self.logger.info("OpenCL support not available in OpenCV")

        except Exception as e:
            self.logger.warning(f"Error during GPU capability detection: {e}")

        # If no acceleration is available
        if not gpu_info["has_gpu"]:
            self.logger.info("No GPU acceleration detected, using CPU processing")

        return gpu_info

    def export_stimulus_video(
        self,
        stimulus_type: str,
        parameters: Dict[str, Any],
        output_path: str,
        orientation: str = "horizontal",
        duration: float = 2.0,
        format: str = "mp4",
    ) -> str:
        """
        Generate and export a stimulus video.

        Args:
            stimulus_type: Type of stimulus (e.g., 'PG' for periodic grating)
            parameters: Parameters for the stimulus
            output_path: Directory to save the video
            orientation: 'horizontal' or 'vertical'
            duration: Duration in seconds
            format: Video format ('mp4', 'avi', etc.)

        Returns:
            Path to the generated video file
        """
        try:
            # Adjust orientation if vertical
            if orientation == "vertical":
                parameters = parameters.copy()
                parameters["orientation"] = 90.0  # Vertical orientation

            # Ensure duration is set
            if duration != parameters.get("duration", 0):
                parameters = parameters.copy()
                parameters["duration"] = duration

            # Create stimulus object based on type
            stimulus = self._create_stimulus(stimulus_type, parameters)

            # Generate frames
            self.logger.info(
                f"Generating {orientation} {stimulus_type} stimulus frames..."
            )
            frames = stimulus.get_all_frames()

            # Create a descriptive filename that includes key parameters
            pattern_str = parameters.get("pattern", "bars")
            progressive_str = (
                "_progressive" if parameters.get("progressive", False) else ""
            )
            spherical_str = (
                "_spherical" if parameters.get("spherical_correction", False) else ""
            )

            # Create filename based on stimulus type
            file_name = f"{stimulus_type}_{orientation}{progressive_str}{spherical_str}.{format}"

            file_path = os.path.join(output_path, file_name)

            self.logger.info(f"Saving video to {file_path}...")
            self._save_video(
                frames, file_path, fps=int(parameters.get("refresh_rate", 60))
            )

            self.logger.info(f"Generated video: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error generating video: {str(e)}")
            raise

    def _create_stimulus(self, stimulus_type: str, parameters: Dict[str, Any]):
        """Create the stimulus object based on type."""
        if stimulus_type == "PG":
            from src.stimuli.gratings.per_grater import PeriodicGratingStimulus

            return PeriodicGratingStimulus(parameters)
        elif stimulus_type == "CHECKERBOARD":
            from src.stimuli.checkerboard.checkerboard_stimulus import (
                CheckerboardStimulus,
            )

            # Ensure pattern is set to checkerboard
            params = parameters.copy()
            params["pattern"] = "checkerboard"
            return CheckerboardStimulus(params)
        # Add other stimulus types as needed
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")

    def _save_video(self, frames: np.ndarray, file_path: str, fps: int = 60):
        """
        Save frames as a video file.

        Args:
            frames: Numpy array of frames
            file_path: Output file path
            fps: Frames per second
        """
        try:
            # Import opencv here to avoid dependency if not used
            import cv2

            # Validate frames
            if frames is None or len(frames) == 0:
                raise ValueError("No frames to save")

            self.logger.info(f"Frame array shape: {frames.shape}")
            self.logger.info(f"Frame data type: {frames.dtype}")
            self.logger.info(f"Frame value range: [{frames.min()}, {frames.max()}]")

            # Get video dimensions
            if len(frames.shape) == 3:  # (frames, height, width)
                height, width = frames[0].shape
                is_color = False
            elif len(frames.shape) == 4:  # (frames, height, width, channels)
                height, width, _ = frames[0].shape
                is_color = True
            else:
                raise ValueError(f"Unexpected frame shape: {frames.shape}")

            self.logger.info(f"Video dimensions: {width}x{height}, color: {is_color}")

            # Initialize temp_avi_path to None
            temp_avi_path = None

            # Define video writer fourcc function to handle potential attribute errors
            if hasattr(cv2, "VideoWriter_fourcc"):

                def fourcc_func(*args):
                    return cv2.VideoWriter_fourcc(*args)

            elif hasattr(cv2, "cv") and hasattr(cv2.cv, "CV_FOURCC"):

                def fourcc_func(*args):
                    return cv2.cv.CV_FOURCC(*args)

            else:

                def fourcc_func(*args):
                    return sum(ord(c) << 8 * i for i, c in enumerate(args))

            # Check if we can use GPU acceleration
            using_gpu = False

            # If requesting MP4 output
            if file_path.lower().endswith(".mp4"):
                # Try direct GPU encoding if available
                if self.gpu_info["has_gpu"]:
                    try:
                        import subprocess
                        import os
                        import tempfile

                        # Create a temporary directory for frames
                        with tempfile.TemporaryDirectory() as temp_dir:
                            self.logger.info(
                                f"Using GPU acceleration: {self.gpu_info['acceleration_type']} with {self.gpu_info['device_name']}"
                            )

                            # Save frames as PNG files for ffmpeg
                            self.logger.info("Saving frames for GPU encoding...")
                            frame_files = []
                            for i, frame in enumerate(frames):
                                frame_path = os.path.join(
                                    temp_dir, f"frame_{i:06d}.png"
                                )
                                if not is_color:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                                cv2.imwrite(frame_path, frame)
                                frame_files.append(frame_path)

                            # Determine hwaccel and encoder options based on detected capabilities
                            encoder = self.gpu_info.get("encoder", "libx264")
                            accel_type = self.gpu_info.get("acceleration_type")

                            # Set up ffmpeg command with hwaccel
                            ffmpeg_cmd = ["ffmpeg", "-y"]

                            # Add proper hardware acceleration flags
                            if accel_type == "metal":
                                ffmpeg_cmd += ["-hwaccel", "videotoolbox"]
                            elif accel_type == "cuda":
                                ffmpeg_cmd += ["-hwaccel", "cuda"]
                            elif accel_type == "opencl" and "vaapi" in encoder:
                                ffmpeg_cmd += ["-hwaccel", "vaapi"]
                            elif accel_type == "opencl" and "qsv" in encoder:
                                ffmpeg_cmd += ["-hwaccel", "qsv"]

                            # Add input options and path
                            ffmpeg_cmd += [
                                "-framerate",
                                str(fps),
                                "-i",
                                os.path.join(temp_dir, "frame_%06d.png"),
                                "-c:v",
                                encoder,
                                "-preset",
                                "medium",
                                "-crf",
                                "23",
                                "-pix_fmt",
                                "yuv420p",  # Ensure compatibility
                                file_path,
                            ]

                            # Execute ffmpeg
                            self.logger.info(
                                f"Encoding video with GPU acceleration using {encoder}..."
                            )
                            result = subprocess.run(
                                ffmpeg_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=False,
                            )

                            if result.returncode == 0:
                                self.logger.info(
                                    f"Successfully encoded video using GPU acceleration: {file_path}"
                                )
                                using_gpu = True
                            else:
                                self.logger.warning(
                                    "GPU encoding failed, falling back to CPU encoding"
                                )
                                self.logger.debug(
                                    f"ffmpeg error: {result.stderr.decode()}"
                                )
                    except Exception as e:
                        self.logger.warning(
                            f"Error during GPU encoding: {e}, falling back to CPU"
                        )

            # If GPU encoding wasn't used, fall back to standard OpenCV encoding
            if not using_gpu:
                # Use H264 codec directly for MP4 files
                if file_path.lower().endswith(".mp4"):
                    try:
                        fourcc = fourcc_func(*"H264")
                        out = cv2.VideoWriter(
                            file_path, fourcc, fps, (width, height), isColor=True
                        )

                        if out is None or not out.isOpened():
                            self.logger.warning(
                                "H264 codec not available, falling back to XVID with conversion"
                            )
                            temp_avi_path = file_path.replace(".mp4", "_temp.avi")
                            fourcc = fourcc_func(*"XVID")
                            out = cv2.VideoWriter(
                                temp_avi_path,
                                fourcc,
                                fps,
                                (width, height),
                                isColor=True,
                            )

                            if out is None or not out.isOpened():
                                raise RuntimeError(
                                    "Failed to create video writer. Check if codec is supported."
                                )
                    except Exception as e:
                        self.logger.warning(
                            f"H264 codec failed: {str(e)}, using XVID with conversion"
                        )
                        temp_avi_path = file_path.replace(".mp4", "_temp.avi")
                        fourcc = fourcc_func(*"XVID")
                        out = cv2.VideoWriter(
                            temp_avi_path, fourcc, fps, (width, height), isColor=True
                        )

                        if out is None or not out.isOpened():
                            raise RuntimeError(
                                "Failed to create video writer. Check if codec is supported."
                            )
                else:
                    # For other formats (like AVI), use XVID
                    fourcc = fourcc_func(*"XVID")
                    out = cv2.VideoWriter(
                        file_path, fourcc, fps, (width, height), isColor=True
                    )

                if out is None or not out.isOpened():
                    raise RuntimeError(
                        "Failed to create video writer. Check if codec is supported."
                    )

                # Write frames
                for i, frame in enumerate(frames):
                    # Convert to 3-channel BGR (required by VideoWriter)
                    if not is_color:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_bgr = frame

                    # Ensure frame is in the correct format
                    if frame_bgr.dtype != np.uint8:
                        frame_bgr = (frame_bgr * 255).astype(np.uint8)

                    out.write(frame_bgr)

                    # Log progress every 10% of frames
                    if i % max(1, len(frames) // 10) == 0:
                        self.logger.debug(f"Writing frame {i}/{len(frames)}")

                # Release resources
                out.release()

                # If we used a temporary AVI file, convert it to MP4 and delete the AVI
                if file_path.lower().endswith(".mp4") and temp_avi_path:
                    try:
                        import subprocess
                        import os

                        self.logger.info(
                            "Converting temporary AVI to MP4 using ffmpeg..."
                        )
                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-i",
                            temp_avi_path,
                            "-c:v",
                            "libx264",
                            "-preset",
                            "medium",
                            "-crf",
                            "23",
                            "-pix_fmt",
                            "yuv420p",  # Ensure compatibility
                            "-y",
                            file_path,
                        ]

                        subprocess.run(
                            ffmpeg_cmd,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )

                        self.logger.info(f"Successfully converted to MP4: {file_path}")

                        # Remove the temporary AVI file
                        os.remove(temp_avi_path)
                        self.logger.info(f"Removed temporary AVI file: {temp_avi_path}")

                    except Exception as e:
                        self.logger.warning(f"Error during conversion: {e}")
                        # Still return the path to the original file

            self.logger.info(f"Video saved to {file_path}")
            return file_path

        except ImportError:
            self.logger.error(
                "OpenCV is required for video export. Install with: pip install opencv-python"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error saving video: {str(e)}")
            raise
