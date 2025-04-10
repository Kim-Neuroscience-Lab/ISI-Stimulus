"""Loop configuration management for experiment control."""

from typing import Dict, Any, List, Optional


class LoopConfig:
    """
    Loop configuration manager for experiment control.

    This class replaces the configureLstate functionality from MATLAB,
    providing a more object-oriented and modular approach.
    """

    def __init__(self):
        """Initialize with default loop configuration."""
        # Default loop state
        self._config: Dict[str, Any] = {
            # Basic loop parameters
            "number_of_conditions": 0,
            "number_of_repetitions": 0,
            "block_size": 1,  # Number of repetitions per block
            "randomization_method": "random",  # 'random', 'sequential', 'block'
            # Timing
            "pre_stimulus_time": 0.5,  # seconds
            "stimulus_time": 1.0,  # seconds
            "post_stimulus_time": 0.5,  # seconds
            "isi_time": 1.0,  # Inter-stimulus interval in seconds
            # Trial information
            "current_trial": 0,
            "total_trials": 0,
            "trials_completed": 0,
            # Condition-repetition pairs
            "condition_repetition_pairs": [],
        }

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

        # Update dependent values if necessary
        self._update_dependent_values()

    def _update_dependent_values(self):
        """Update values that depend on other configuration parameters."""
        # Calculate total number of trials
        num_conds = self._config["number_of_conditions"]
        num_reps = self._config["number_of_repetitions"]

        if num_conds > 0 and num_reps > 0:
            self._config["total_trials"] = num_conds * num_reps
        else:
            self._config["total_trials"] = 0

    def generate_trial_sequence(self):
        """
        Generate the trial sequence based on the current configuration.

        This creates the condition-repetition pairs for all trials.
        """
        num_conds = self._config["number_of_conditions"]
        num_reps = self._config["number_of_repetitions"]
        rand_method = self._config["randomization_method"]
        block_size = self._config["block_size"]

        pairs = []

        if rand_method == "random":
            # Completely random order
            import random

            for rep in range(num_reps):
                for cond in range(num_conds):
                    pairs.append((cond, rep))
            random.shuffle(pairs)

        elif rand_method == "sequential":
            # Sequential order
            for rep in range(num_reps):
                for cond in range(num_conds):
                    pairs.append((cond, rep))

        elif rand_method == "block":
            # Block randomization - randomize within each block
            import random

            for b in range(0, num_reps, block_size):
                block_reps = min(block_size, num_reps - b)
                for rep in range(b, b + block_reps):
                    block_pairs = [(cond, rep) for cond in range(num_conds)]
                    random.shuffle(block_pairs)
                    pairs.extend(block_pairs)

        self._config["condition_repetition_pairs"] = pairs
        self._config["current_trial"] = 0
        self._config["trials_completed"] = 0

    def get_current_condition_repetition(self) -> Optional[tuple]:
        """
        Get the current condition and repetition.

        Returns:
            Tuple: (condition, repetition) or None if not available
        """
        pairs = self._config["condition_repetition_pairs"]
        current_trial = self._config["current_trial"]

        if not pairs or current_trial >= len(pairs):
            return None

        return pairs[current_trial]

    def advance_trial(self):
        """
        Advance to the next trial.

        Returns:
            bool: True if there are more trials, False if completed
        """
        self._config["current_trial"] += 1
        self._config["trials_completed"] += 1

        return self._config["current_trial"] < len(
            self._config["condition_repetition_pairs"]
        )

    def get_all_configs(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Dict: All configuration values
        """
        return self._config.copy()
