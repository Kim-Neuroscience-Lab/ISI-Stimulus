# ISI-Stimulus

A stimulus generation package for neuroscience experiments, following protocols described in the MMC1 document.

## Overview

This package implements visual stimuli for neuroscience experiments as described in the MMC1 document, particularly focusing on:

1. Drifting bar stimuli with counter-phase checkerboard patterns
2. Drifting grating stimuli with proper spherical correction
3. Appropriate transformation corrections for wide field-of-view visual stimulation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ISI-Stimulus.git
cd ISI-Stimulus

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Generate a drifting bar stimulus as described in MMC1:

```bash
python -m src.cli mmc1 --output ./output --mode intrinsic
```

This will generate a drifting bar stimulus with:

- 20° wide bar drifted in four cardinal directions
- Counter-phase checkerboard pattern (25° squares with 166 ms period)
- 10 repeats per direction
- Appropriate spherical correction based on direction
- Drift speed of 8.5-9.5°/s for intrinsic imaging

For two-photon imaging with faster drift speeds (12-14°/s):

```bash
python -m src.cli mmc1 --output ./output --mode two-photon
```

To generate a quick preview with reduced quality:

```bash
python -m src.cli mmc1 --output ./output --preview
```

## Custom Stimulus Generation

Create a custom stimulus with specific parameters:

```bash
python -m src.cli custom --type drifting_bar --output ./output \
  --duration 10 --bar-width 20 --grid-spacing 25 \
  --transformation spherical --direction right-to-left
```

Available stimulus types:

- `drifting_bar`: Bar with counter-phase checkerboard pattern (as in MMC1)
- `drifting_grating`: Sinusoidal grating pattern with controllable spatial/temporal frequency
- `checkerboard`: Checkerboard pattern with various display options

## MMC1 Stimulus Parameters

The visual stimulation procedure follows the specifications in the MMC1 document:

### Screen Setup

- Large LCD display (68 x 121 cm)
- 60 Hz refresh rate
- Stimulus subtending 153° vertical, 147° horizontal visual field
- Screen placed at an angle of 20° relative to the animal
- Screen distance of 10 cm from eye

### Drifting Bar Stimulus

- 20° wide bar
- Drifted 10 times in each of the four cardinal directions
- Counter-phase checkerboard pattern (25° squares with 166 ms period)
- Drift speed: 8.5-9.5°/s for intrinsic imaging, 12-14°/s for two-photon imaging

### Transformation

Spherical corrections are applied to account for distortions created by displaying stimuli on a flat monitor:

```
For a drifting grating defined as S = cos(2πfsθ - tft):
- fs is spatial frequency in cycles/degree
- θ is the position
- ft is temporal frequency in Hz
- t is time
```

The transformation ensures constant spatial and temporal frequency across the entire visual field.

## Programmatic Usage

You can also use the codebase programmatically in your own scripts:

```python
from src.stimuli.factory import create_mmc1_stimulus

# Create a stimulus object
stimulus = create_mmc1_stimulus(stimulus_type="drifting_bar", is_two_photon=False)

# Generate frames for all directions
video_segments = stimulus.generate_full_sequence()

# Process the frames as needed
for direction, frames in video_segments.items():
    print(f"Generated {len(frames)} frames for {direction} direction")
    # Do something with the frames...
```

## Implementation Details

The codebase follows SOLID principles and clean code practices:

- **Single Responsibility Principle**: Each class has a single responsibility
- **Open/Closed Principle**: Extend functionality through new classes rather than modifying existing ones
- **Liskov Substitution Principle**: Child classes should be substitutable for their parent classes
- **Interface Segregation Principle**: Clients should not depend on interfaces they don't use
- **Dependency Inversion Principle**: Depend on abstractions, not concretions

Key components:

- `src/stimuli/drifting_bar.py`: Implementation of the drifting bar stimulus
- `src/stimuli/spherical_correction.py`: Spherical transformation as described in MMC1
- `src/stimuli/factory.py`: Factory for creating stimulus instances
- `src/cli.py`: Command-line interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.
