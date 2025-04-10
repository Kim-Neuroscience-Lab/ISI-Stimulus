"""Main UI controller for the ISI Stimulus application."""

import tkinter as tk
from tkinter import messagebox
import logging
from typing import Dict, Any, Optional

from src.core.application import Application
from src.ui.main_window import MainWindow
from src.ui.parameter_window import ParameterWindow
from src.ui.looper_window import LooperWindow


class MainController:
    """
    Main UI controller for the ISI Stimulus application.

    This class coordinates the UI components and handles user interactions.
    """

    def __init__(self):
        """Initialize the main UI controller."""
        self.logger = logging.getLogger("MainController")

        # Application instance
        self.app = Application.get_instance()

        # UI components
        self.root: Optional[tk.Tk] = None
        self.main_window: Optional[MainWindow] = None
        self.parameter_window: Optional[ParameterWindow] = None
        self.looper_window: Optional[LooperWindow] = None

    def start(self):
        """Start the UI controller and create the main window."""
        # Create the root window
        self.root = tk.Tk()
        self.root.title("ISI Stimulus")

        # Create UI components
        self._create_ui_components()

        # Set up event handlers
        self._setup_event_handlers()

        # Start the main loop
        try:
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def _create_ui_components(self):
        """Create the UI components."""
        if not self.root:
            return

        try:
            # Create main window
            self.main_window = MainWindow(self.root, self)

            # Create parameter window
            self.parameter_window = ParameterWindow(self.root, self)

            # Create looper window
            self.looper_window = LooperWindow(self.root, self)

            # Arrange windows
            self.main_window.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.parameter_window.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.looper_window.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        except Exception as e:
            self.logger.error(f"Error creating UI components: {str(e)}")
            messagebox.showerror("Error", f"Failed to create UI: {str(e)}")

    def _setup_event_handlers(self):
        """Set up event handlers for UI components."""
        # Set up close handler
        if self.root:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Handle window close event."""
        # Confirm before closing
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            # Clean up resources
            self._cleanup()

            # Destroy the root window
            if self.root:
                self.root.destroy()

    def _cleanup(self):
        """Clean up resources before closing."""
        # Stop any running processes
        try:
            # Stop sync detection if running
            from src.services.sync_service import SyncService

            sync_service = SyncService.get_instance()
            sync_service.stop_sync_detection()

            # Disconnect communication services
            from src.services.communication_service import CommunicationService

            display_comm = CommunicationService.get_display_communication()
            if display_comm:
                display_comm.disconnect()

            acq_comm = CommunicationService.get_acquisition_communication()
            if acq_comm:
                acq_comm.disconnect()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def on_change_stimulus_module(self, module_id: str):
        """
        Handle change of stimulus module.

        Args:
            module_id: The module ID to change to
        """
        try:
            # Change the current stimulus module
            self.app.set_current_stimulus_module(module_id)

            # Update the parameter window
            if self.parameter_window:
                self.parameter_window.update_parameters()

        except Exception as e:
            self.logger.error(f"Error changing stimulus module: {str(e)}")
            messagebox.showerror("Error", f"Failed to change stimulus module: {str(e)}")

    def on_start_experiment(self):
        """Handle start experiment button click."""
        try:
            # Get config objects
            machine_config = self.app.get_machine_config()
            loop_config = self.app.get_loop_config()

            # Check if already running
            if machine_config.get_config("running"):
                messagebox.showwarning("Warning", "Experiment is already running.")
                return

            # Generate trial sequence
            loop_config.generate_trial_sequence()

            # Set running state
            machine_config.set_running(True)

            # Start sync detection
            from src.services.sync_service import SyncService

            sync_service = SyncService.get_instance()
            if not sync_service.start_sync_detection():
                messagebox.showerror("Error", "Failed to start sync detection.")
                machine_config.set_running(False)
                return

            # Update UI
            if self.main_window:
                self.main_window.update_running_state(True)

            if self.looper_window:
                self.looper_window.update_experiment_status()

        except Exception as e:
            self.logger.error(f"Error starting experiment: {str(e)}")
            messagebox.showerror("Error", f"Failed to start experiment: {str(e)}")

    def on_stop_experiment(self):
        """Handle stop experiment button click."""
        try:
            # Get config objects
            machine_config = self.app.get_machine_config()

            # Check if running
            if not machine_config.get_config("running"):
                messagebox.showwarning("Warning", "No experiment is running.")
                return

            # Set running state
            machine_config.set_running(False)

            # Stop sync detection
            from src.services.sync_service import SyncService

            sync_service = SyncService.get_instance()
            sync_service.stop_sync_detection()

            # Update UI
            if self.main_window:
                self.main_window.update_running_state(False)

            if self.looper_window:
                self.looper_window.update_experiment_status()

        except Exception as e:
            self.logger.error(f"Error stopping experiment: {str(e)}")
            messagebox.showerror("Error", f"Failed to stop experiment: {str(e)}")
