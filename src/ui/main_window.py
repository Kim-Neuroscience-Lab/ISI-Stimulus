"""Main window UI component for the ISI Stimulus application."""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

# Import only for type checking to avoid circular imports
if TYPE_CHECKING:
    from src.ui.main_controller import MainController


class MainWindow(ttk.Frame):
    """
    Main window UI component.

    This is the primary window of the application, containing
    experiment control buttons and machine state information.
    """

    def __init__(self, parent, controller: "MainController"):
        """
        Initialize the main window.

        Args:
            parent: The parent tkinter container
            controller: The UI controller
        """
        super().__init__(parent)
        self.parent = parent
        self.controller = controller
        self.logger = logging.getLogger("MainWindow")

        # Get application and config
        from src.core.application import Application

        self.app = Application.get_instance()
        self.machine_config = self.app.get_machine_config()

        # Create UI elements
        self._create_ui()

    def _create_ui(self):
        """Create the UI elements."""
        # Create a frame for machine state
        self.machine_frame = ttk.LabelFrame(self, text="Machine State")
        self.machine_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create machine state fields
        self._create_machine_fields()

        # Create a frame for experiment control
        self.control_frame = ttk.LabelFrame(self, text="Experiment Control")
        self.control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create experiment control buttons
        self._create_control_buttons()

        # Create status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Update UI with current values
        self.update_machine_fields()

    def _create_machine_fields(self):
        """Create machine state input fields."""
        # Create a grid of labels and entry fields
        fields = [
            ("Animal ID:", "animal_id"),
            ("Unit:", "unit"),
            ("Experiment:", "experiment"),
            ("Hemisphere:", "hemisphere"),
            ("Screen Distance (cm):", "screen_distance"),
            ("Monitor:", "monitor"),
        ]

        self.machine_entries = {}

        for i, (label_text, field_name) in enumerate(fields):
            # Create label
            label = ttk.Label(self.machine_frame, text=label_text)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)

            # Create entry
            entry = ttk.Entry(self.machine_frame)
            entry.grid(row=i, column=1, sticky=tk.W + tk.E, padx=5, pady=2)

            # Store reference to entry
            self.machine_entries[field_name] = entry

            # Add update handler
            entry.bind(
                "<FocusOut>", lambda event, name=field_name: self._on_field_update(name)
            )

        # Add save button
        save_button = ttk.Button(
            self.machine_frame, text="Save", command=self._on_save_machine_state
        )
        save_button.grid(row=len(fields), column=0, columnspan=2, pady=5)

    def _create_control_buttons(self):
        """Create experiment control buttons."""
        # Create start button
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Experiment",
            command=self._on_start_experiment,
        )
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create stop button
        self.stop_button = ttk.Button(
            self.control_frame,
            text="Stop Experiment",
            command=self._on_stop_experiment,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Create reset button
        self.reset_button = ttk.Button(
            self.control_frame, text="Reset", command=self._on_reset_experiment
        )
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

    def update_machine_fields(self):
        """Update machine state fields with current values."""
        # Get all configurations
        configs = self.machine_config.get_all_configs()

        # Update each entry field
        for field_name, entry in self.machine_entries.items():
            if field_name in configs:
                # Clear current value
                entry.delete(0, tk.END)

                # Set new value
                entry.insert(0, str(configs[field_name]))

    def _on_field_update(self, field_name: str):
        """
        Handle field update event.

        Args:
            field_name: The name of the field being updated
        """
        # Get the entry widget
        entry = self.machine_entries.get(field_name)
        if not entry:
            return

        # Get the new value
        new_value = entry.get()

        # Convert numeric values
        if field_name == "screen_distance":
            try:
                new_value = float(new_value)
            except ValueError:
                messagebox.showerror("Invalid Value", f"{field_name} must be a number.")
                self.update_machine_fields()  # Reset to current value
                return

        # Update configuration
        try:
            self.machine_config.set_config(field_name, new_value)
            self.logger.info(f"Updated {field_name} to {new_value}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update {field_name}: {str(e)}")
            self.update_machine_fields()  # Reset to current value

    def _on_save_machine_state(self):
        """Handle save button click."""
        # This would save the machine state to a file
        messagebox.showinfo("Save", "Machine state would be saved to a file.")

    def _on_start_experiment(self):
        """Handle start experiment button click."""
        self.controller.on_start_experiment()

    def _on_stop_experiment(self):
        """Handle stop experiment button click."""
        self.controller.on_stop_experiment()

    def _on_reset_experiment(self):
        """Handle reset experiment button click."""
        # Confirm reset
        if not messagebox.askyesno(
            "Reset", "Are you sure you want to reset the experiment?"
        ):
            return

        # Reset experiment state
        try:
            # Reset loop configuration
            loop_config = self.app.get_loop_config()
            loop_config.set_config("current_trial", 0)
            loop_config.set_config("trials_completed", 0)

            # Update status
            self.set_status("Experiment reset")

            # Update UI in other windows
            if (
                hasattr(self.controller, "looper_window")
                and self.controller.looper_window
            ):
                self.controller.looper_window.update_experiment_status()

        except Exception as e:
            self.logger.error(f"Error resetting experiment: {str(e)}")
            messagebox.showerror("Error", f"Failed to reset experiment: {str(e)}")

    def update_running_state(self, is_running: bool):
        """
        Update UI based on running state.

        Args:
            is_running: Whether the experiment is running
        """
        if is_running:
            # Disable start button
            self.start_button.config(state=tk.DISABLED)

            # Enable stop button
            self.stop_button.config(state=tk.NORMAL)

            # Set status
            self.set_status("Experiment running")
        else:
            # Enable start button
            self.start_button.config(state=tk.NORMAL)

            # Disable stop button
            self.stop_button.config(state=tk.DISABLED)

            # Set status
            self.set_status("Experiment stopped")

    def set_status(self, status: str):
        """
        Set the status bar text.

        Args:
            status: The status text
        """
        self.status_bar.config(text=status)
