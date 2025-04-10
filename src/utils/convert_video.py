# src/utils/convert_video.py

"""
Command-line utility for converting and optimizing video files.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.video_converter import convert_avi_to_mp4, optimize_video_for_web


def setup_logging(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    """Process command line arguments and convert videos."""
    parser = argparse.ArgumentParser(description="Convert and optimize video files.")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert AVI to MP4
    convert_parser = subparsers.add_parser("convert", help="Convert AVI to MP4")
    convert_parser.add_argument("input", help="Input video file")
    convert_parser.add_argument("-o", "--output", help="Output video file")

    # Optimize for web
    optimize_parser = subparsers.add_parser("optimize", help="Optimize video for web")
    optimize_parser.add_argument("input", help="Input video file")
    optimize_parser.add_argument("-o", "--output", help="Output video file")
    optimize_parser.add_argument("-s", "--size", type=float, help="Target size in MB")

    # Batch convert
    batch_parser = subparsers.add_parser(
        "batch", help="Batch convert videos in directory"
    )
    batch_parser.add_argument("directory", help="Directory with video files")
    batch_parser.add_argument(
        "-t",
        "--type",
        choices=["convert", "optimize"],
        default="convert",
        help="Operation type (default: convert)",
    )
    batch_parser.add_argument(
        "-s", "--size", type=float, help="Target size in MB for optimization"
    )
    batch_parser.add_argument(
        "-e",
        "--extension",
        default=".avi",
        help="File extension to process (default: .avi)",
    )

    # Add verbose option to all subparsers
    for subparser in [convert_parser, optimize_parser, batch_parser]:
        subparser.add_argument(
            "-v", "--verbose", action="store_true", help="Verbose output"
        )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    logger = logging.getLogger(__name__)

    if args.command == "convert":
        # Convert a single file
        result = convert_avi_to_mp4(args.input, args.output)
        if result:
            logger.info(f"Successfully converted to {result}")
        else:
            logger.error("Conversion failed")
            return 1

    elif args.command == "optimize":
        # Optimize a single file
        result = optimize_video_for_web(args.input, args.output, args.size)
        if result:
            logger.info(f"Successfully optimized to {result}")
        else:
            logger.error("Optimization failed")
            return 1

    elif args.command == "batch":
        # Process all files in a directory
        if not os.path.isdir(args.directory):
            logger.error(f"Directory not found: {args.directory}")
            return 1

        input_dir = Path(args.directory)
        files = list(input_dir.glob(f"*{args.extension}"))

        if not files:
            logger.error(
                f"No files with extension {args.extension} found in {args.directory}"
            )
            return 1

        logger.info(f"Found {len(files)} files to process")

        success_count = 0
        for file_path in files:
            try:
                if args.type == "convert":
                    result = convert_avi_to_mp4(str(file_path))
                else:  # optimize
                    result = optimize_video_for_web(
                        str(file_path), target_size_mb=args.size
                    )

                if result:
                    success_count += 1
                    logger.info(f"Processed {file_path.name} → {Path(result).name}")
                else:
                    logger.error(f"Failed to process {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")

        logger.info(f"Successfully processed {success_count} out of {len(files)} files")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
