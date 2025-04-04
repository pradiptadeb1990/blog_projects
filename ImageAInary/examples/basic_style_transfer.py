"""
Basic style transfer example using ImageAInary.

This example demonstrates how to use the ImageAInary library programmatically
to perform style transfer between two images using ControlNet and IP-Adapter.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from imageainary.pipeline.style_transfer import StyleTransferPipeline
from imageainary.utils.image_processing import create_image_grid


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced style transfer example")
    parser.add_argument("--content", type=str, required=True, help="Path to content image")
    parser.add_argument("--style", type=str, required=True, help="Path to style image")
    parser.add_argument("--output", type=str, default="stylized_output.png", help="Output image path")
    parser.add_argument("--use-controlnet", action="store_true", help="Use ControlNet for better structure preservation")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.5, help="Scale for IP-Adapter (0.0-1.0)")
    parser.add_argument("--controlnet-scale", type=float, default=0.7, help="Scale for ControlNet conditioning (0.0-1.0)")
    parser.add_argument("--strength", type=float, default=0.8, help="Style transfer strength (0.0-1.0) when not using ControlNet")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--variations", type=int, default=3, help="Number of style variations to generate")
    parser.add_argument("--create-grid", action="store_true", help="Create a grid of style variations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    """Run the style transfer example."""
    # Parse arguments
    args = parse_args()
    
    # Load images
    content_image_path = Path(args.content)
    style_image_path = Path(args.style)
    
    if not content_image_path.exists():
        raise FileNotFoundError(f"Content image not found: {content_image_path}")
    if not style_image_path.exists():
        raise FileNotFoundError(f"Style image not found: {style_image_path}")
    
    content_image = Image.open(content_image_path).convert("RGB")
    style_image = Image.open(style_image_path).convert("RGB")
    
    # Initialize the pipeline
    print("Initializing style transfer pipeline...")
    pipeline = StyleTransferPipeline(use_controlnet=args.use_controlnet)
    
    if args.create_grid:
        # Create a grid of style variations
        print(f"Generating {args.variations} style variations...")
        grid_image = pipeline.create_style_grid(
            content_image=content_image,
            style_image=style_image,
            variations=args.variations,
            ip_adapter_scale=args.ip_adapter_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
        
        # Save the grid image
        output_path = Path(args.output)
        grid_image.save(output_path)
        print(f"Style grid saved to: {output_path}")
        
        # Display the grid
        plt.figure(figsize=(15, 10))
        plt.imshow(grid_image)
        plt.title("Style Transfer Grid (Content, Style, Variations)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        # Perform single style transfer
        print(f"Transferring style with {'ControlNet' if args.use_controlnet else 'standard pipeline'}...")
        output_image = pipeline.transfer_style(
            content_image=content_image,
            style_image=style_image,
            strength=args.strength,
            ip_adapter_scale=args.ip_adapter_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
        
        # Save the output image
        output_path = Path(args.output)
        output_image.save(output_path)
        print(f"Output saved to: {output_path}")
        
        # Display the images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(content_image)
        axes[0].set_title("Content Image")
        axes[0].axis("off")
        
        axes[1].imshow(style_image)
        axes[1].set_title("Style Image")
        axes[1].axis("off")
        
        axes[2].imshow(output_image)
        axes[2].set_title("Stylized Output")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
