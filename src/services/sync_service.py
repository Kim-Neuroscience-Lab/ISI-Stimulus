"""Synchronization service for hardware timing signals."""

import logging
from typing import Optional, Callable, List


class SyncService:
    """
    Synchronization service for hardware timing signals.

    This class handles synchronization with external hardware devices,
    such as frame grabbers and stimulus presentation hardware.
    """

    # Singleton instance
    _instance = None

    def __init__(self):
        """Initialize the sync service."""
        self.logger = logging.getLogger("SyncService")
        self.initialized = False
        self.callbacks: List[Callable] = []
        self.input_device = None

    def _initialize_hardware(self) -> bool:
        """
        Initialize the hardware interface.

        This would use a hardware-specific library like PyDAQmx or similar.

        Returns:
            bool: True if initialized successfully, False otherwise
        """
        # This is a placeholder for actual hardware initialization
        try:
            # In a real implementation, this would use a hardware access library
            # For example, with PyDAQmx:
            # import PyDAQmx as daq
            # self.input_device = daq.Task()
            # self.input_device.CreateDIChan("Dev1/port0/line0", "sync_input", daq.DAQmx_Val_ChanForAllLines)

            self.logger.info("Simulated hardware initialization successful")
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Error initializing hardware: {str(e)}")
            self.initialized = False
            return False

    def add_sync_callback(self, callback: Callable):
        """
        Add a callback function to be called on sync signal.

        Args:
            callback: The callback function
        """
        self.callbacks.append(callback)

    def remove_sync_callback(self, callback: Callable) -> bool:
        """
        Remove a callback function.

        Args:
            callback: The callback function to remove

        Returns:
            bool: True if removed, False if not found
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            return True
        return False

    def _sync_handler(self, data: Optional[dict] = None):
        """
        Handler called when a sync signal is received.

        Args:
            data: Optional data associated with the sync event
        """
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in sync callback: {str(e)}")

    def start_sync_detection(self) -> bool:
        """
        Start detecting sync signals.

        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.initialized:
            self.logger.error("Cannot start sync detection - not initialized")
            return False

        try:
            # In a real implementation, this would start the hardware task
            # For example:
            # self.input_device.StartTask()

            self.logger.info("Started sync detection")
            return True
        except Exception as e:
            self.logger.error(f"Error starting sync detection: {str(e)}")
            return False

    def stop_sync_detection(self) -> bool:
        """
        Stop detecting sync signals.

        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.initialized:
            return False

        try:
            # In a real implementation, this would stop the hardware task
            # For example:
            # self.input_device.StopTask()

            self.logger.info("Stopped sync detection")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping sync detection: {str(e)}")
            return False

    @classmethod
    def get_instance(cls) -> "SyncService":
        """
        Get the singleton instance of the SyncService.

        Returns:
            SyncService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = SyncService()
        return cls._instance

    @classmethod
    def initialize_sync_inputs(cls) -> bool:
        """
        Initialize synchronization inputs.

        Returns:
            bool: True if initialized successfully, False otherwise
        """
        instance = cls.get_instance()
        return instance._initialize_hardware()
