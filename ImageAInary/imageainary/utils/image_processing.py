"""
Image processing utilities for ImageAInary.
"""
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def preprocess_image(
    image: Image.Image, target_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """
    Preprocess an image for the style transfer pipeline.

    Args:
        image: The input image
        target_size: The target size for the image (width, height)

    Returns:
        The preprocessed image
    """
    # Resize the image while maintaining aspect ratio
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > 1:  # Landscape
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Portrait or square
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize the image using high-quality resampling
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the target size and paste the resized image
    new_image = Image.new("RGB", target_size, (255, 255, 255))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))

    return new_image


def postprocess_image(image: Image.Image) -> Image.Image:
    """
    Postprocess an image from the style transfer pipeline.

    Args:
        image: The input image

    Returns:
        The postprocessed image
    """
    # Convert to numpy array
    img_array = np.array(image)

    # Enhance contrast
    p_low, p_high = np.percentile(img_array, (2, 98))
    img_array = np.clip(
        (img_array - p_low) * (255.0 / (p_high - p_low)), 0, 255
    ).astype(np.uint8)

    # Convert back to PIL Image
    enhanced_image = Image.fromarray(img_array)

    # Apply additional enhancements
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(1.2)  # Slightly sharpen

    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(1.1)  # Slightly increase contrast

    enhancer = ImageEnhance.Color(enhanced_image)
    enhanced_image = enhancer.enhance(1.1)  # Slightly increase color saturation

    return enhanced_image


def create_image_grid(
    images: List[Image.Image], cols: int = 4, cell_size: Tuple[int, int] = (512, 512)
) -> Image.Image:
    """
    Create a grid of images.

    Args:
        images: List of images to include in the grid
        cols: Number of columns in the grid
        cell_size: Size of each cell in the grid (width, height)

    Returns:
        A grid image containing all input images
    """
    # Calculate grid dimensions
    n_images = len(images)
    rows = (n_images + cols - 1) // cols

    # Create a new image for the grid
    grid_width = cols * cell_size[0]
    grid_height = rows * cell_size[1]
    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))

    # Resize and paste each image into the grid
    for i, img in enumerate(images):
        # Skip if we've run out of images
        if i >= n_images:
            break

        # Resize the image to fit the cell
        img = img.resize(cell_size, Image.LANCZOS)

        # Calculate position
        row = i // cols
        col = i % cols
        x = col * cell_size[0]
        y = row * cell_size[1]

        # Paste the image into the grid
        grid.paste(img, (x, y))

    return grid


def extract_edges(
    image: Image.Image, low_threshold: int = 100, high_threshold: int = 200
) -> Image.Image:
    """
    Extract edges from an image using a simple edge detection approach.
    This is a simplified version of Canny edge detection for when controlnet_aux is not available.

    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        Edge image
    """
    # Convert to grayscale
    gray_image = ImageOps.grayscale(image)

    # Apply Gaussian blur to reduce noise
    blurred = gray_image.filter(ImageFilter.GaussianBlur(radius=2))

    # Apply edge detection filter
    edges = blurred.filter(ImageFilter.FIND_EDGES)

    # Enhance edges
    enhancer = ImageEnhance.Contrast(edges)
    edges = enhancer.enhance(2.0)

    # Convert to numpy for thresholding
    edge_array = np.array(edges)

    # Apply thresholding
    edge_array = np.where(edge_array > low_threshold, 255, 0).astype(np.uint8)

    # Convert back to PIL
    edge_image = Image.fromarray(edge_array)

    return edge_image
