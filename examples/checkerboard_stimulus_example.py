# examples/checkerboard_stimulus_example.py
"""
Example script demonstrating how to generate checkerboard, grid and bar pattern stimuli with
various options including spherical correction and progressive appearance.
"""

import os
import sys
import argparse

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Run the example script."""
    parser = argparse.ArgumentParser(
        description="Generate checkerboard, grid and bar pattern stimuli"
    )

    # Output directory
    parser.add_argument("--output", default="./videos", help="Output directory")

    # Choose pattern: grid, bars, or checkerboard
    parser.add_argument(
        "--pattern",
        choices=["grid", "bars", "checkerboard"],
        default="checkerboard",
        help="Pattern type: grid, bars, or checkerboard",
    )

    # Spherical correction option
    parser.add_argument(
        "--spherical",
        action="store_true",
        help="Apply spherical correction for rodent vision",
    )

    # Progressive appearance option
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Use progressive appearance instead of periodic movement",
    )

    # Sweep direction for progressive mode
    parser.add_argument(
        "--sweep",
        choices=["left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"],
        default="left-to-right",
        help="Sweep direction for progressive appearance",
    )

    args = parser.parse_args()

    print(f"Generating {args.pattern} pattern stimuli...")

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Import the CLI module to use its generation capabilities
    from src.cli import main as cli_main

    # Generate a horizontal and vertical pattern
    # First, construct the command line arguments
    cmd_args = [
        "generate-video",
        "--type",
        "CHECKERBOARD",
        "--output",
        args.output,
        "--orientation",
        "both",
        "--pattern",
        args.pattern,
        "--duration",
        "2.0",
        "--resolution",
        "1920x1080",
        "--framerate",
        "60",
        "--spatial-freq",
        "0.1",
        "--temporal-freq",
        "1.0",
        "--contrast",
        "100",
        "--grid-spacing-x",
        "5.0",
        "--grid-spacing-y",
        "5.0",
        "--line-width",
        "0.5",
        "--spherical-correction",
        "1" if args.spherical else "0",
        "--progressive",
        "1" if args.progressive else "0",
        "--sweep-direction",
        args.sweep,
    ]

    # Launch the CLI with our arguments
    result = cli_main(cmd_args)

    if result == 0:
        print("\nStimulus generation completed successfully.")
        print(f"Videos have been saved to: {args.output}\n")

        # Show the command to generate the same stimulus from the command line
        cmd_str = "python generate_stimulus.py " + " ".join(cmd_args)
        print("Command to generate the same stimulus from the command line:")
        print(cmd_str)
    else:
        print("\nError generating stimuli.")

    return result


if __name__ == "__main__":
    sys.exit(main())
