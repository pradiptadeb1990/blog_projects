# ImageAInary

An advanced stable diffusion-based style transfer tool that allows you to apply the artistic style of one image to the content of another using ControlNet and IP-Adapter technologies.

## Features

- Advanced style transfer using ControlNet for structure preservation and IP-Adapter for style fidelity
- Create grids of style variations to choose from
- Command-line interface for easy use
- Highly customizable style transfer parameters
- Support for various image formats
- Optimized for high-quality results

## Installation

ImageAInary uses Poetry for dependency management. To install:

```bash
# Clone the repository
git clone <your-repo-url>
cd ImageAInary

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

### Command Line Interface

ImageAInary provides a simple command-line interface with multiple commands:

```bash
# Get help
imageainary --help

# Display information about ImageAInary
imageainary info

# Transfer style from a style image to a content image
imageainary transfer <content_image_path> <style_image_path> --output-dir ./outputs

# Create a grid of style variations
imageainary grid <content_image_path> <style_image_path> --output-path style_grid.png
```

### Style Transfer Options

```bash
imageainary transfer \
    path/to/content.jpg \
    path/to/style.jpg \
    --output-dir ./outputs \
    --use-controlnet \
    --ip-adapter-scale 0.5 \
    --controlnet-scale 0.7 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --width 768 \
    --height 768 \
    --seed 42
```

#### Transfer Parameters:

- `content_image`: Path to the content image (required)
- `style_image`: Path to the style image (required)
- `--output-dir`: Directory to save output images (default: ./outputs)
- `--use-controlnet`: Use ControlNet for better structure preservation (flag)
- `--strength`: Strength of style transfer when not using ControlNet, 0.0 to 1.0 (default: 0.8)
- `--ip-adapter-scale`: Scale for IP-Adapter influence, 0.0 to 1.0 (default: 0.5)
- `--controlnet-scale`: Scale for ControlNet conditioning, 0.0 to 1.0 (default: 0.7)
- `--num-inference-steps`: Number of inference steps (default: 50)
- `--guidance-scale`: Guidance scale for stable diffusion (default: 7.5)
- `--width`: Output image width (default: 768)
- `--height`: Output image height (default: 768)
- `--seed`: Random seed for reproducibility (optional)

### Creating Style Variation Grids

```bash
imageainary grid \
    path/to/content.jpg \
    path/to/style.jpg \
    --output-path style_grid.png \
    --variations 3 \
    --ip-adapter-scale 0.5 \
    --num-inference-steps 30 \
    --seed 42
```

#### Grid Parameters:

- `content_image`: Path to the content image (required)
- `style_image`: Path to the style image (required)
- `--output-path`: Path to save the grid image (default: ./style_grid.png)
- `--variations`: Number of style variations to generate (default: 3)
- `--use-controlnet`: Use ControlNet for better structure preservation (flag)
- `--ip-adapter-scale`: Scale for IP-Adapter influence, 0.0 to 1.0 (default: 0.5)
- `--num-inference-steps`: Number of inference steps (default: 30)
- `--seed`: Random seed for reproducibility (optional)

## How It Works

ImageAInary uses advanced stable diffusion models with ControlNet and IP-Adapter to transfer styles between images:

1. **Content Preservation**: The content image is processed through a Canny edge detector to extract structural information
2. **Style Extraction**: The style image is processed by the IP-Adapter to extract artistic elements
3. **Prompt Enhancement**: CLIP is used to generate descriptive prompts that enhance the style transfer
4. **Generation**: ControlNet ensures structural fidelity while IP-Adapter applies the artistic style
5. **Post-processing**: The generated image is enhanced for optimal visual quality

This approach combines the best of both worlds - maintaining the structure of the content image while faithfully applying the artistic style of the style image.

## Requirements

- Python >=3.9, <3.13
- PyTorch >=2.0.0
- Diffusers >=0.25.0
- Transformers >=4.30.0
- ControlNet-Aux >=0.0.7
- Accelerate >=0.15.0

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
