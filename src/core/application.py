"""Core application class that manages the application lifecycle."""

from typing import Dict, Optional


class Application:
    """
    Application Core class for managing application lifecycle.

    Implements the Singleton pattern.
    """

    # Singleton instance
    _instance = None

    # Default stimulus module
    DEFAULT_STIMULUS = "PG"

    def __init__(self):
        """Initialize the application - private constructor."""
        if Application._instance is not None:
            raise RuntimeError(
                "Application is a singleton. Use get_instance() instead."
            )

        # Configuration containers
        self._parameter_config = None
        self._machine_config = None
        self._loop_config = None
        self._stimulus_modules: Dict[str, object] = {}

    @classmethod
    def get_instance(cls) -> "Application":
        """
        Get the singleton instance of the Application.

        Returns:
            Application: The singleton instance
        """
        if cls._instance is None:
            cls._instance = Application()
        return cls._instance

    def initialize(self):
        """Initialize application components."""
        print("Initializing ISI Stimulus application...")

        # Initialize configuration
        self._initialize_configurations()

        # Initialize communication modules
        self._initialize_communication()

        # Initialize synchronization inputs
        self._initialize_sync_inputs()

        print("Initialization complete.")

    def _initialize_configurations(self):
        """Initialize configuration objects."""
        from src.config.parameter_config import ParameterConfig
        from src.config.machine_config import MachineConfig
        from src.config.loop_config import LoopConfig

        self._parameter_config = ParameterConfig()
        self._machine_config = MachineConfig()
        self._loop_config = LoopConfig()

        # Load default configurations
        self._parameter_config.load_module(self.DEFAULT_STIMULUS)

    def _initialize_communication(self):
        """Initialize communication with stimulus computer."""
        from src.services.communication_service import CommunicationService

        CommunicationService.initialize_display_communication()

    def _initialize_sync_inputs(self):
        """Initialize sync inputs for ISI acquisition timing."""
        from src.services.sync_service import SyncService

        SyncService.initialize_sync_inputs()

    def get_parameter_config(self):
        """
        Get parameter configuration.

        Returns:
            ParameterConfig: The parameter configuration object
        """
        return self._parameter_config

    def get_machine_config(self):
        """
        Get machine configuration.

        Returns:
            MachineConfig: The machine configuration object
        """
        return self._machine_config

    def get_loop_config(self):
        """
        Get loop configuration.

        Returns:
            LoopConfig: The loop configuration object
        """
        return self._loop_config

    def set_current_stimulus_module(self, module_id: str):
        """
        Change the current stimulus module.

        Args:
            module_id: The ID of the module to load
        """
        self._parameter_config.load_module(module_id)
