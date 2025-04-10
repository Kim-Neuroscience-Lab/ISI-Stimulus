# src/__main__.py
"""Entry point for the ISI Stimulus application."""

from src.core.application import Application


def main():
    """Main entry point for the application."""
    try:
        # Initialize core application
        app = Application.get_instance()

        # Load configuration and initialize
        app.initialize()

        # Launch UI
        from src.ui.main_controller import MainController

        controller = MainController()
        controller.start()

    except Exception as e:
        import traceback

        print(f"Error initializing application: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
