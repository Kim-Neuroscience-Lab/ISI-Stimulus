"""GPU acceleration mixin for stimuli."""

import logging
import numpy as np
from typing import Dict, Any, Optional


class GPUAccelerationMixin:
    """Mixin class for GPU acceleration capabilities."""

    def __init__(self):
        """Initialize GPU acceleration capabilities."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_gpu = False
        self.gpu_type = None
        self._gpu_initialized = False

    def is_gpu_available(self) -> bool:
        """
        Check if GPU acceleration is available.

        Returns:
            bool: True if GPU acceleration is available, False otherwise
        """
        # Initialize GPU if not already done
        if not self._gpu_initialized:
            self._check_gpu_capabilities()
            self._gpu_initialized = True

        return self.use_gpu

    def apply_gpu_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply GPU-accelerated processing to a frame.

        Args:
            frame: Input frame

        Returns:
            np.ndarray: Processed frame
        """
        if not self.use_gpu:
            return frame

        try:
            # Apply different processing based on GPU type
            if self.gpu_type == "cuda":
                return self._apply_cuda_processing(frame)
            elif self.gpu_type == "opencl":
                return self._apply_opencl_processing(frame)
            elif self.gpu_type == "metal":
                return self._apply_metal_processing(frame)
            else:
                return frame
        except Exception as e:
            self.logger.warning(f"GPU processing error: {e}, falling back to CPU")
            return frame

    def _apply_cuda_processing(self, frame: np.ndarray) -> np.ndarray:
        """Apply CUDA-based processing to a frame."""
        try:
            import cv2

            # Only proceed if CUDA is available
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Convert to GPU
                gpu_frame = cv2.cuda.GpuMat()
                gpu_frame.upload(frame)

                # Apply any needed processing here
                # For now, we're just demonstrating GPU transfer

                # Download result back to CPU
                result = gpu_frame.download()
                return result
        except Exception as e:
            self.logger.warning(f"CUDA processing error: {e}")

        return frame

    def _apply_opencl_processing(self, frame: np.ndarray) -> np.ndarray:
        """Apply OpenCL-based processing to a frame."""
        try:
            import cv2

            # Only proceed if OpenCL is available and enabled
            if hasattr(cv2, "ocl") and cv2.ocl.useOpenCL():
                # Create UMat from frame for OpenCL processing
                gpu_frame = cv2.UMat(np.ascontiguousarray(frame))

                # Apply any needed processing here
                # For now, we're just demonstrating GPU transfer

                # Get result back to CPU
                result = gpu_frame.get()
                return result
        except Exception as e:
            self.logger.warning(f"OpenCL processing error: {e}")

        return frame

    def _apply_metal_processing(self, frame: np.ndarray) -> np.ndarray:
        """Apply Metal-based processing for macOS."""
        # Metal processing would be implemented here if available
        # Currently, we don't have direct Metal API access through OpenCV
        # This is a placeholder for future implementation
        return frame

    def _check_gpu_capabilities(self) -> None:
        """
        Check for available GPU acceleration capabilities.

        Sets the following attributes:
        - use_gpu: bool - Whether GPU acceleration is available
        - gpu_type: Optional[str] - Type of GPU acceleration ('cuda', 'opencl', 'metal', or None)
        """
        try:
            import cv2
            import platform

            self.use_gpu = False
            self.gpu_type = None

            # Check for CUDA support
            if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                if cuda_devices > 0:
                    self.use_gpu = True
                    self.gpu_type = "cuda"
                    self.logger.info(
                        f"CUDA GPU acceleration available with {cuda_devices} device(s)"
                    )

            # Check for OpenCL support (if CUDA not available)
            if (
                not self.use_gpu
                and hasattr(cv2, "ocl")
                and hasattr(cv2.ocl, "haveOpenCL")
            ):
                if cv2.ocl.haveOpenCL():
                    cv2.ocl.setUseOpenCL(True)
                    if cv2.ocl.useOpenCL():
                        self.use_gpu = True
                        self.gpu_type = "opencl"
                        self.logger.info("OpenCL GPU acceleration available")

            # Check for Metal support on macOS (if neither CUDA nor OpenCL available)
            if not self.use_gpu and platform.system() == "Darwin":
                try:
                    import platform

                    if "Apple" in platform.processor():
                        self.use_gpu = True
                        self.gpu_type = "metal"
                        self.logger.info("Apple Metal GPU acceleration available")
                except Exception as e:
                    self.logger.debug(f"Error checking Metal availability: {e}")

            if not self.use_gpu:
                self.logger.info("No GPU acceleration available, using CPU")

        except ImportError as e:
            self.logger.warning(f"GPU acceleration library import error: {e}")
            self.use_gpu = False
            self.gpu_type = None

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get information about GPU capabilities.

        Returns:
            Dict[str, Any]: Dictionary containing GPU information
        """
        # Initialize GPU if not already done
        if not self._gpu_initialized:
            self._check_gpu_capabilities()
            self._gpu_initialized = True

        return {
            "has_gpu": self.use_gpu,
            "acceleration_type": self.gpu_type,
            "device_name": self._get_gpu_device_name(),
            "platform": self._get_gpu_platform(),
            "encoder": self._get_gpu_encoder(),
        }

    def _get_gpu_device_name(self) -> Optional[str]:
        """Get the name of the GPU device."""
        if not self.use_gpu:
            return None

        try:
            if self.gpu_type == "cuda":
                import cv2

                try:
                    device_id = cv2.cuda.getDevice()
                    device_info = cv2.cuda.DeviceInfo()
                    return f"CUDA device {device_id}: {str(device_info)}"
                except:
                    return f"CUDA device"
            elif self.gpu_type == "opencl":
                import cv2

                return "OpenCL device"
            elif self.gpu_type == "metal":
                import platform

                return f"Apple {platform.processor()}"
        except Exception as e:
            self.logger.debug(f"Error getting GPU device name: {e}")

        return None

    def _get_gpu_platform(self) -> Optional[str]:
        """Get the GPU platform name."""
        if not self.use_gpu:
            return None

        try:
            import platform

            return platform.system()
        except:
            pass

        return None

    def _get_gpu_encoder(self) -> Optional[str]:
        """Get the appropriate encoder for the GPU platform."""
        if not self.use_gpu:
            return None

        try:
            import platform

            system = platform.system()

            if self.gpu_type == "cuda":
                return "h264_nvenc"
            elif self.gpu_type == "opencl":
                if system == "Windows":
                    return "h264_amf"
                else:
                    return "h264_vaapi"
            elif self.gpu_type == "metal":
                return "h264_videotoolbox"
        except:
            pass

        return None
