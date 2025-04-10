"""Parameter configuration management for stimulus modules."""

from typing import Dict, Any, Optional


class ParameterConfig:
    """
    Parameter configuration manager for stimulus parameters.

    This class replaces the configurePstate functionality from MATLAB,
    providing a more object-oriented and modular approach.
    """

    # Registered module factories
    _MODULE_FACTORIES = {}

    def __init__(self):
        """Initialize the parameter configuration."""
        # Current module ID
        self._current_module_id: Optional[str] = None

        # Current parameter state
        self._parameters: Dict[str, Any] = {}

        # Register standard modules
        self._register_standard_modules()

    def _register_standard_modules(self):
        """Register standard stimulus modules."""
        from src.stimuli.gratings.per_grater import register_module as reg_per_grater
        from src.stimuli.gratings.flash_grater import (
            register_module as reg_flash_grater,
        )
        from src.stimuli.rain.rain_dots import register_module as reg_rain
        from src.stimuli.noise.noise import register_module as reg_noise
        from src.stimuli.mapper.mapper import register_module as reg_mapper
        from src.stimuli.motion.coherent_motion import register_module as reg_coh_motion

        # Register each module
        reg_per_grater(self)
        reg_flash_grater(self)
        reg_rain(self)
        reg_noise(self)
        reg_mapper(self)
        reg_coh_motion(self)

    @classmethod
    def register_module_factory(cls, module_id: str, factory_function):
        """
        Register a module factory function.

        Args:
            module_id: The module identifier
            factory_function: The factory function that creates default parameters
        """
        cls._MODULE_FACTORIES[module_id] = factory_function

    def load_module(self, module_id: str):
        """
        Load a stimulus module configuration.

        Args:
            module_id: The module identifier to load

        Raises:
            ValueError: If the module ID is not registered
        """
        if module_id not in self._MODULE_FACTORIES:
            raise ValueError(f"Unknown stimulus module: {module_id}")

        # Store current module ID
        self._current_module_id = module_id

        # Call the factory function to get default parameters
        self._parameters = self._MODULE_FACTORIES[module_id]()

    def get_parameter(self, param_name: str) -> Any:
        """
        Get a parameter value.

        Args:
            param_name: The parameter name

        Returns:
            The parameter value

        Raises:
            KeyError: If the parameter doesn't exist
        """
        if param_name not in self._parameters:
            raise KeyError(f"Parameter {param_name} not found")

        return self._parameters[param_name]

    def set_parameter(self, param_name: str, value: Any):
        """
        Set a parameter value.

        Args:
            param_name: The parameter name
            value: The parameter value

        Raises:
            KeyError: If the parameter doesn't exist
        """
        if param_name not in self._parameters:
            raise KeyError(f"Parameter {param_name} not found")

        self._parameters[param_name] = value

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters.

        Returns:
            Dict: All parameters
        """
        return self._parameters.copy()

    def get_current_module_id(self) -> Optional[str]:
        """
        Get the current module ID.

        Returns:
            The current module ID, or None if no module is loaded
        """
        return self._current_module_id
