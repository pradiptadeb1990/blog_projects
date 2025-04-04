"""
Style Transfer Pipeline - Core functionality for stable diffusion based style transfer.
"""
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from controlnet_aux import CannyDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
)
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

from imageainary.utils.image_processing import postprocess_image, preprocess_image


class StyleTransferPipeline:
    """
    Pipeline for style transfer using stable diffusion with ControlNet and IP-Adapter.
    """

    def __init__(
        self,
        model_id: str = "Yntec/AbsoluteReality",
        controlnet_model: str = "lllyasviel/sd-controlnet-canny",
        ip_adapter_model: str = "h94/IP-Adapter",
        device: str = None,
        use_controlnet: bool = True,
    ):
        """
        Initialize the style transfer pipeline.

        Args:
            model_id: The model ID to use for stable diffusion
            controlnet_model: The ControlNet model to use
            ip_adapter_model: The IP-Adapter model to use
            device: The device to use for inference (cuda, cpu, mps)
            use_controlnet: Whether to use ControlNet for better structure preservation
        """
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"  # Use MPS for Apple Silicon if available

        self.device = device
        self.model_id = model_id
        self.use_controlnet = use_controlnet
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # Initialize ControlNet and edge detector if using ControlNet
        if use_controlnet:
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_model, torch_dtype=self.torch_dtype
            )

            # Move ControlNet to device if not using CUDA (which will use CPU offload)
            if device != "cuda":
                self.controlnet = self.controlnet.to(device)

            # Load the stable diffusion pipeline with ControlNet
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                controlnet=self.controlnet,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            )

            # Load IP-Adapter for better style transfer
            self.pipeline.load_ip_adapter(
                ip_adapter_model, subfolder="models", weight_name="ip-adapter_sd15.bin"
            )

            # Initialize edge detector
            self.edge_detector = CannyDetector()

            # Note: CannyDetector doesn't support direct device placement
            # It will use CPU for preprocessing regardless of the device setting

            # Enable CPU offload to save VRAM for CUDA or move to device for MPS
            if device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            elif device == "mps":
                self.pipeline = self.pipeline.to(device)
        else:
            # Fallback to standard img2img pipeline if not using ControlNet
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, safety_checker=None
            )
            self.pipeline = self.pipeline.to(device)

        # Load CLIP model for style extraction (used as fallback or for prompt enhancement)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model = self.clip_model.to(device)

    def _extract_style_prompt(self, style_image: Image.Image) -> str:
        """
        Extract a text prompt describing the style of the image using CLIP.

        Args:
            style_image: The style image

        Returns:
            A text prompt describing the style
        """
        # Preprocess the image for CLIP
        inputs = self.clip_processor(images=style_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get image features
        with torch.no_grad():
            self.clip_model.get_image_features(**inputs)

        # Define style descriptors with more detailed artistic styles
        style_descriptors = [
            "impressionist painting with vibrant brushstrokes",
            "cubist artwork with geometric shapes",
            "pop art with bold colors",
            "delicate watercolor painting",
            "textured oil painting",
            "detailed digital art",
            "fine pencil sketch",
            "elegant ink drawing",
            "abstract art with dynamic composition",
            "surrealist painting with dreamlike elements",
            "minimalist art with clean lines",
            "expressionist painting with emotional intensity",
            "photorealistic rendering",
            "vintage photograph",
            "anime illustration",
            "concept art",
        ]

        # For a more advanced implementation, we would compare image_features with text embeddings
        # of style_descriptors to find the closest match
        return f"in the style of {style_descriptors[random.randint(0, len(style_descriptors)-1)]}"

    def _prepare_controlnet_image(
        self, image: Image.Image, width: int = 768, height: int = 768
    ) -> Image.Image:
        """
        Prepare an image for ControlNet by extracting edges.

        Args:
            image: The input image
            width: Target width
            height: Target height

        Returns:
            The edge-detected image for ControlNet
        """
        # Resize image to target dimensions
        image = image.resize((width, height))

        # Extract edges using Canny edge detector
        edge_image = self.edge_detector(
            image, detect_resolution=512, image_resolution=max(width, height)
        )

        return edge_image

    def transfer_style(
        self,
        content_image: Union[str, Path, Image.Image],
        style_image: Union[str, Path, Image.Image],
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.7,
        ip_adapter_scale: float = 0.5,
        num_images: int = 1,
        width: int = 768,
        height: int = 768,
        seed: Optional[int] = None,
        return_all_images: bool = False,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Transfer the style from a style image to a content image using ControlNet and IP-Adapter.

        Args:
            content_image: The content image (path or PIL Image)
            style_image: The style image (path or PIL Image)
            strength: The strength of the style transfer (0.0 to 1.0)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for stable diffusion
            controlnet_conditioning_scale: Scale for ControlNet conditioning
            ip_adapter_scale: Scale for IP-Adapter (higher means stronger style influence)
            num_images: Number of images to generate
            width: Output image width
            height: Output image height
            seed: Random seed for reproducibility
            return_all_images: If True, return all generated images as a list

        Returns:
            The stylized image or a list of stylized images if return_all_images is True
        """
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Load and preprocess images
        if not isinstance(content_image, Image.Image):
            content_image = Image.open(content_image).convert("RGB")

        if not isinstance(style_image, Image.Image):
            style_image = Image.open(style_image).convert("RGB")

        # Extract style prompt from style image to enhance the generation
        style_prompt = self._extract_style_prompt(style_image)

        # Create a prompt that describes the content and style
        prompt = f"""(photorealistic:1.2), raw, masterpiece, high quality, 
        detailed image {style_prompt}, beautiful composition, 8k resolution"""

        negative_prompt = "low quality, blurry, distorted, deformed, disfigured, bad anatomy, watermark"

        if self.use_controlnet:
            # Prepare content image for ControlNet
            controlnet_image = self._prepare_controlnet_image(
                content_image, width, height
            )

            # Set IP-Adapter scale for style transfer
            self.pipeline.set_ip_adapter_scale(ip_adapter_scale)

            # Run the ControlNet pipeline with IP-Adapter
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=controlnet_image,  # ControlNet conditioning image (edges)
                ip_adapter_image=style_image,  # Style image for IP-Adapter
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_images_per_prompt=num_images,
            ).images
        else:
            # Fallback to standard img2img pipeline
            # Preprocess images for standard pipeline
            content_image = preprocess_image(content_image, (width, height))

            # Run the stable diffusion pipeline
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=content_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
            ).images

        # Postprocess the output images
        processed_output = [postprocess_image(img) for img in output]

        # Return all images or just the first one
        if return_all_images:
            return processed_output
        else:
            return processed_output[0]

    def create_style_grid(
        self,
        content_image: Union[str, Path, Image.Image],
        style_image: Union[str, Path, Image.Image],
        variations: int = 3,
        **kwargs,
    ) -> Image.Image:
        """
        Create a grid of style transfer variations.

        Args:
            content_image: The content image
            style_image: The style image
            variations: Number of style variations to generate
            **kwargs: Additional arguments for transfer_style

        Returns:
            A grid image showing the original content, style, and variations
        """
        # Load images if they are paths
        if not isinstance(content_image, Image.Image):
            content_image = Image.open(content_image).convert("RGB")

        if not isinstance(style_image, Image.Image):
            style_image = Image.open(style_image).convert("RGB")

        # Generate style variations
        variations = self.transfer_style(
            content_image=content_image,
            style_image=style_image,
            num_images=variations,
            return_all_images=True,
            **kwargs,
        )

        # Create a grid with original content, style, and variations
        all_images = [content_image, style_image] + variations

        # Resize all images to the same size for the grid
        target_size = (512, 512)
        all_images = [img.resize(target_size) for img in all_images]

        # Calculate grid dimensions
        cols = min(4, len(all_images))
        rows = (len(all_images) + cols - 1) // cols

        # Create the grid
        grid_width = cols * target_size[0]
        grid_height = rows * target_size[1]
        grid = Image.new("RGB", (grid_width, grid_height))

        # Paste images into the grid
        for i, img in enumerate(all_images):
            row = i // cols
            col = i % cols
            grid.paste(img, (col * target_size[0], row * target_size[1]))

        return grid
