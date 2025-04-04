# ImageAInary Examples

This directory contains example scripts demonstrating how to use ImageAInary for style transfer.

## Basic Style Transfer Example

The `basic_style_transfer.py` script demonstrates how to use ImageAInary programmatically to transfer style between two images.

### Usage

```bash
python basic_style_transfer.py --content path/to/content.jpg --style path/to/style.jpg --output stylized_output.png
```

### Parameters

- `--content`: Path to the content image (required)
- `--style`: Path to the style image (required)
- `--output`: Path to save the output image (default: stylized_output.png)
- `--strength`: Style transfer strength, 0.0 to 1.0 (default: 0.8)
- `--steps`: Number of inference steps (default: 50)

## Sample Images

The `sample_images` directory contains sample content and style images you can use to test the style transfer functionality.

To use these sample images, you'll need to add your own images to this directory.

## Getting Started

1. Add your content and style images to the `sample_images` directory
2. Run the example script with your images:

```bash
python basic_style_transfer.py --content sample_images/your_content.jpg --style sample_images/your_style.jpg
```

3. Check the output image to see the result of the style transfer
