"""Machine configuration management for experiment setup."""

from typing import Dict, Any


class MachineConfig:
    """
    Machine configuration manager.

    This class replaces the configureMstate functionality from MATLAB,
    providing a more object-oriented and modular approach.
    """

    def __init__(self):
        """Initialize with default machine configuration."""
        # Default machine state
        self._config: Dict[str, Any] = {
            # Subject information
            "animal_id": "xx0",
            "unit": "000",
            "experiment": "000",
            # Display settings
            "hemisphere": "left",
            "screen_distance": 25,  # cm
            "monitor": "TEL",  # Monitor identifier
            "sync_size": 4,  # Size of the screen sync in cm
            # Application state
            "running": False,
            # Paths and communication
            "analyzer_root": "C:/neurodata/AnalyzerFiles_new",
            "stimulus_ip": "10.1.38.61",  # Stimulus computer IP
        }

        # Update monitor-specific values
        self._update_monitor_values()

    def _update_monitor_values(self):
        """Update monitor-specific configuration values."""
        monitor_id = self._config["monitor"]

        # This would load from a monitor configuration file in a real implementation
        if monitor_id == "TEL":
            self._config.update(
                {
                    "monitor_width": 40,  # cm
                    "monitor_height": 30,  # cm
                    "resolution_width": 1920,  # pixels
                    "resolution_height": 1080,  # pixels
                    "refresh_rate": 60,  # Hz
                }
            )

    def get_config(self, config_name: str) -> Any:
        """
        Get a configuration value.

        Args:
            config_name: The configuration parameter name

        Returns:
            The configuration value

        Raises:
            KeyError: If the configuration doesn't exist
        """
        if config_name not in self._config:
            raise KeyError(f"Configuration {config_name} not found")

        return self._config[config_name]

    def set_config(self, config_name: str, value: Any):
        """
        Set a configuration value.

        Args:
            config_name: The configuration parameter name
            value: The configuration value
        """
        self._config[config_name] = value

        # Special case for monitor - update related values
        if config_name == "monitor":
            self._update_monitor_values()

    def get_all_configs(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Dict: All configuration values
        """
        return self._config.copy()

    def set_running(self, is_running: bool):
        """
        Set the running state.

        Args:
            is_running: The running state
        """
        self._config["running"] = is_running
