"""Logging mixin for stimuli."""

import logging
from typing import Optional


class LoggingMixin:
    """Mixin class for logging functionality."""

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the logging mixin.

        Args:
            name: Name of the logger (defaults to class name)
        """
        if name is None:
            name = self.__class__.__name__

        self.logger = logging.getLogger(name)

        # Configure basic logging if not already configured
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

    def set_log_level(self, level: str) -> None:
        """
        Set the logging level.

        Args:
            level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        """
        level = level.upper()
        if hasattr(logging, level):
            self.logger.setLevel(getattr(logging, level))
        else:
            self.logger.warning(f"Invalid log level: {level}, using INFO")
            self.logger.setLevel(logging.INFO)
