"""Communication service for inter-process and device communication."""

import socket
import logging
from typing import Optional, Dict, Any, Union


class CommunicationService:
    """
    Communication service for display control and data acquisition.

    This class handles network communication with the stimulus display
    and acquisition computers.
    """

    # Singleton instances
    _display_instance = None
    _acquisition_instance = None

    def __init__(self, host: str, port: int, comm_type: str):
        """
        Initialize a communication service.

        Args:
            host: The host address to connect to
            port: The port to connect to
            comm_type: The type of communication ('display' or 'acquisition')
        """
        self.host = host
        self.port = port
        self.comm_type = comm_type
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.logger = logging.getLogger(f"CommunicationService_{comm_type}")

    def connect(self) -> bool:
        """
        Connect to the remote host.

        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.logger.info(
                f"Connected to {self.comm_type} at {self.host}:{self.port}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.comm_type}: {str(e)}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the remote host."""
        if self.socket:
            try:
                self.socket.close()
            except Exception as e:
                self.logger.error(f"Error closing socket: {str(e)}")
            finally:
                self.socket = None
                self.connected = False

    def send_command(self, command: str) -> bool:
        """
        Send a command to the remote host.

        Args:
            command: The command to send

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected")
            return False

        try:
            self.socket.sendall(command.encode("utf-8"))
            return True
        except Exception as e:
            self.logger.error(f"Error sending command: {str(e)}")
            self.connected = False
            return False

    def receive_response(self, buffer_size: int = 1024) -> Optional[str]:
        """
        Receive a response from the remote host.

        Args:
            buffer_size: The buffer size for receiving

        Returns:
            str: The response or None if error
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected")
            return None

        try:
            data = self.socket.recv(buffer_size)
            return data.decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error receiving response: {str(e)}")
            return None

    @classmethod
    def initialize_display_communication(cls) -> "CommunicationService":
        """
        Initialize communication with the display computer.

        Returns:
            CommunicationService: The display communication service
        """
        # Get the IP address from machine config
        from src.core.application import Application

        app = Application.get_instance()
        machine_config = app.get_machine_config()

        host = machine_config.get_config("stimulus_ip")
        port = 5000  # Default port for display communication

        # Create instance if it doesn't exist
        if cls._display_instance is None:
            cls._display_instance = CommunicationService(host, port, "display")

        # Try to connect
        cls._display_instance.connect()

        return cls._display_instance

    @classmethod
    def get_display_communication(cls) -> Optional["CommunicationService"]:
        """
        Get the display communication service.

        Returns:
            CommunicationService: The display communication service or None
        """
        return cls._display_instance

    @classmethod
    def initialize_acquisition_communication(cls) -> "CommunicationService":
        """
        Initialize communication with the acquisition computer.

        Returns:
            CommunicationService: The acquisition communication service
        """
        # Usually the acquisition computer connects to us, so we'd set up a server
        # This is a placeholder for the actual implementation
        host = "0.0.0.0"  # Listen on all interfaces
        port = 5001  # Default port for acquisition communication

        # Create instance if it doesn't exist
        if cls._acquisition_instance is None:
            cls._acquisition_instance = CommunicationService(host, port, "acquisition")

        return cls._acquisition_instance

    @classmethod
    def get_acquisition_communication(cls) -> Optional["CommunicationService"]:
        """
        Get the acquisition communication service.

        Returns:
            CommunicationService: The acquisition communication service or None
        """
        return cls._acquisition_instance
