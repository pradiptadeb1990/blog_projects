"""
Style Transfer Pipeline V2 - Enhanced functionality for stable diffusion based style transfer.
Includes support for multiple ControlNet conditioning types and scheduler optimization.
"""
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.utils import load_image, make_image_grid
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

from imageainary.utils.image_processing import postprocess_image, preprocess_image


class StyleTransferPipelineV2:
    """
    Enhanced pipeline for style transfer using stable diffusion with multiple ControlNet options,
    scheduler configurations, and advanced IP-Adapter integration.
    """

    SCHEDULER_TYPES = {
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "euler_ancestral": EulerAncestralDiscreteScheduler,
        "dpm_solver": DPMSolverMultistepScheduler,
    }

    CONTROL_MODES = {
        "canny": "Edge-based structure preservation",
        "pose": "Human pose preservation",
        "depth": "3D spatial relationship preservation",
    }

    def __init__(
        self,
        model_id: str = "Yntec/AbsoluteReality",
        controlnet_models: dict = None,
        ip_adapter_model: str = "h94/IP-Adapter",
        device: str = None,
        use_controlnet: bool = True,
        scheduler_type: str = "ddim",
    ):
        """
        Initialize the enhanced style transfer pipeline.

        Args:
            model_id: The model ID to use for stable diffusion
            controlnet_models: Dictionary mapping control modes to model paths
                               If None, defaults will be used
            ip_adapter_model: The IP-Adapter model to use
            device: The device to use for inference (cuda, cpu, mps)
            use_controlnet: Whether to use ControlNet for better structure preservation
            scheduler_type: Type of scheduler to use ('ddim', 'pndm', 'euler_ancestral', 'dpm_solver')
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

        # Set up default controlnet models if not provided
        if controlnet_models is None:
            self.controlnet_models = {
                "canny": "lllyasviel/sd-controlnet-canny",
                "pose": "lllyasviel/sd-controlnet-openpose",
                "depth": "lllyasviel/sd-controlnet-depth",
            }
        else:
            self.controlnet_models = controlnet_models

        # Initialize controlnets and detectors
        self.active_controlnet = None
        self.active_control_mode = "canny"  # Default

        if use_controlnet:
            # Initialize with the default "canny" controlnet
            self._initialize_controlnet("canny")

            # Initialize all detector modules
            self.detectors = {
                "canny": CannyDetector(),
                "pose": OpenposeDetector(),
                "depth": MidasDetector(),
            }
        else:
            # Fallback to standard img2img pipeline if not using ControlNet
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, safety_checker=None
            )
            self.pipeline = self.pipeline.to(device)

        # Set up the scheduler
        self.set_scheduler(scheduler_type)

        # Load CLIP model for style extraction (used as fallback or for prompt enhancement)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip_model = self.clip_model.to(device)

        # Style vocabulary for enhanced prompt creation
        self.style_vocabulary = [
            "impressionist painting",
            "cubist artwork",
            "surrealist composition",
            "oil painting",
            "watercolor",
            "acrylic painting",
            "digital art",
            "pencil sketch",
            "ink drawing",
            "pastel drawing",
            "charcoal drawing",
            "abstract",
            "realistic",
            "photorealistic",
            "expressionist",
            "minimalist",
            "pop art",
            "art nouveau",
            "baroque",
            "renaissance",
            "ukiyo-e",
            "gothic",
            "romantic",
            "neoclassical",
            "art deco",
            "pixel art",
            "vaporwave",
            "cyberpunk",
            "steampunk",
            "fantasy",
            "3D render",
            "concept art",
            "manga",
            "anime",
            "comic book",
            "graffiti",
            "street art",
            "storybook illustration",
            "vintage poster",
            "movie still",
            "film noir",
            "technicolor",
            "black and white",
        ]

        # Prompts to avoid (for negative prompting)
        self.negative_prompts = [
            "low quality",
            "blurry",
            "distorted",
            "deformed",
            "disfigured",
            "bad anatomy",
            "watermark",
            "signature",
            "text",
            "logo",
            "oversaturated",
            "undersaturated",
            "high contrast",
            "low contrast",
        ]

    def _initialize_controlnet(self, control_mode):
        """
        Initialize or switch the active ControlNet based on the control mode.

        Args:
            control_mode: The control mode to use ('canny', 'pose', 'depth')
        """
        if control_mode not in self.controlnet_models:
            raise ValueError(
                f"Unsupported control mode: {control_mode}. Supported modes: {list(self.controlnet_models.keys())}"
            )

        # Don't reload if it's already the active one
        if (
            control_mode == self.active_control_mode
            and self.active_controlnet is not None
        ):
            return

        # Load the ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_models[control_mode], torch_dtype=self.torch_dtype
        )

        # Move ControlNet to device if not using CUDA (which will use CPU offload)
        if self.device != "cuda":
            controlnet = controlnet.to(self.device)

        # Load or update the stable diffusion pipeline with the new ControlNet
        if self.active_controlnet is None:
            # First initialization
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            )

            # Load IP-Adapter for better style transfer
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
            )

            # Enable CPU offload to save VRAM for CUDA or move to device for MPS
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
            elif self.device == "mps":
                self.pipeline = self.pipeline.to(self.device)
        else:
            # Swap out the ControlNet
            self.pipeline.controlnet = controlnet

        # Update tracking variables
        self.active_controlnet = controlnet
        self.active_control_mode = control_mode

    def set_scheduler(self, scheduler_type, **kwargs):
        """
        Configure the diffusion scheduler.

        Args:
            scheduler_type: Type of scheduler ('ddim', 'pndm', 'euler_ancestral', 'dpm_solver')
            **kwargs: Additional scheduler parameters
        """
        if scheduler_type not in self.SCHEDULER_TYPES:
            raise ValueError(
                f"Unsupported scheduler: {scheduler_type}. Supported schedulers: {list(self.SCHEDULER_TYPES.keys())}"
            )

        # Create a new scheduler from the pipeline's current config
        scheduler_class = self.SCHEDULER_TYPES[scheduler_type]
        new_scheduler = scheduler_class.from_config(
            self.pipeline.scheduler.config, **kwargs
        )

        # Update the pipeline's scheduler
        self.pipeline.scheduler = new_scheduler

        # Store the current scheduler type
        self.active_scheduler = scheduler_type

    def _extract_style_prompt(self, style_image: Image.Image) -> str:
        """
        Extract a text prompt describing the style of the image using CLIP.

        Args:
            style_image: The style image

        Returns:
            A text prompt describing the style
        """
        # Preprocess the image for CLIP
        image = self.clip_processor(images=style_image, return_tensors="pt").to(
            self.device
        )

        # Extract image features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate similarity with style vocabulary
        text_inputs = self.clip_processor(
            text=self.style_vocabulary, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarities
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get top 3 matches
        values, indices = similarities[0].topk(3)
        top_styles = [self.style_vocabulary[idx.item()] for idx in indices]

        # Create detailed prompt
        base_prompt = f"A high-quality {', '.join(top_styles)}"
        detailed_prompt = (
            f"{base_prompt}, detailed, professional, award-winning, masterpiece"
        )

        return detailed_prompt

    def _prepare_controlnet_image(
        self, image: Image.Image, width: int = 768, height: int = 768
    ) -> Image.Image:
        """
        Prepare an image for ControlNet based on the active control mode.

        Args:
            image: The input image
            width: Target width
            height: Target height

        Returns:
            The processed image for ControlNet
        """
        # Preprocess the image to the target size
        processed_image = preprocess_image(image, (width, height))

        # Apply the appropriate processor based on the control mode
        if self.active_control_mode == "canny":
            # Analyze image statistics for adaptive thresholding
            img_array = np.array(processed_image.convert("L"))
            low = np.percentile(img_array, 20)
            high = np.percentile(img_array, 80)

            # Apply Canny edge detection with adaptive thresholds
            control_image = self.detectors["canny"](
                processed_image, low_threshold=low, high_threshold=high
            )
        elif self.active_control_mode == "pose":
            # Extract human poses
            control_image = self.detectors["pose"](processed_image)
        elif self.active_control_mode == "depth":
            # Extract depth map
            control_image = self.detectors["depth"](processed_image, bg_threshold=0.1)
        else:
            raise ValueError(f"Unsupported control mode: {self.active_control_mode}")

        return control_image

    def set_control_mode(self, mode: str):
        """
        Change the ControlNet conditioning mode.

        Args:
            mode: The control mode to use ('canny', 'pose', 'depth')
        """
        if not self.use_controlnet:
            raise ValueError(
                "ControlNet is disabled. Enable use_controlnet in the constructor."
            )

        if mode not in self.controlnet_models:
            raise ValueError(
                f"Unsupported control mode: {mode}. Supported modes: {list(self.controlnet_models.keys())}"
            )

        self._initialize_controlnet(mode)

    def get_available_control_modes(self):
        """
        Get information about available control modes.

        Returns:
            Dictionary of control modes and their descriptions
        """
        return self.CONTROL_MODES

    def get_available_schedulers(self):
        """
        Get information about available scheduler types.

        Returns:
            List of available scheduler types
        """
        return list(self.SCHEDULER_TYPES.keys())

    def transfer_style(
        self,
        content_image: Union[str, Path, Image.Image],
        style_image: Union[str, Path, Image.Image],
        strength: float = 0.8,
        prompt: str = None,
        negative_prompt: str = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
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
            strength: The strength of the style transfer (when not using ControlNet)
            prompt: Optional text prompt to guide the generation
            negative_prompt: Optional negative prompt to avoid certain attributes
            guidance_scale: The guidance scale for the diffusion model
            num_inference_steps: Number of diffusion steps
            controlnet_conditioning_scale: The scale for ControlNet conditioning
            ip_adapter_scale: The scale for IP-Adapter style influence
            num_images: Number of images to generate
            width: Output image width
            height: Output image height
            seed: Random seed for reproducibility
            return_all_images: If True, return all generated images as a list

        Returns:
            The stylized image or a list of stylized images
        """
        # Load images if they are paths using diffusers utility
        if not isinstance(content_image, Image.Image):
            content_image = load_image(content_image)
            content_image = content_image.convert("RGB")

        if not isinstance(style_image, Image.Image):
            style_image = load_image(style_image)
            style_image = style_image.convert("RGB")

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(
                random.randint(0, 2147483647)
            )

        # Extract a prompt from the style image if not provided
        if prompt is None:
            prompt = self._extract_style_prompt(style_image)

        # Use default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = ", ".join(self.negative_prompts)

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
                image=controlnet_image,  # ControlNet conditioning image
                ip_adapter_image=style_image,  # Style image for IP-Adapter
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_images_per_prompt=num_images,
                generator=generator,
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
                generator=generator,
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
        seed: Optional[int] = None,
        use_different_seeds: bool = True,
        **kwargs,
    ) -> Image.Image:
        """
        Create a grid of style transfer variations.

        Args:
            content_image: The content image
            style_image: The style image
            variations: Number of style variations to generate
            seed: Base random seed for reproducibility
            use_different_seeds: If True, use different seeds for each variation
            **kwargs: Additional arguments for transfer_style

        Returns:
            A grid image showing the original content, style, and variations
        """
        # Load images if they are paths
        if not isinstance(content_image, Image.Image):
            content_image = load_image(content_image)

        if not isinstance(style_image, Image.Image):
            style_image = load_image(style_image)

        # Generate style variations
        if use_different_seeds and seed is not None:
            # Generate a list of seeds derived from the base seed
            seeds = [seed + i for i in range(variations)]

            # Generate each variation with its own seed
            variations_list = []
            for seed_value in seeds:
                variation = self.transfer_style(
                    content_image=content_image,
                    style_image=style_image,
                    num_images=1,
                    seed=seed_value,
                    **kwargs,
                )
                variations_list.append(variation)
        else:
            # Generate all variations at once with same seed
            variations_list = self.transfer_style(
                content_image=content_image,
                style_image=style_image,
                num_images=variations,
                seed=seed,
                return_all_images=True,
                **kwargs,
            )

        # Create a grid with original content, style, and variations
        all_images = [content_image, style_image] + variations_list

        # Create grid using diffusers utility
        n_images = len(all_images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols

        # Resize all images to the same size for the grid
        all_images = [img.resize((512, 512)) for img in all_images]

        # Create the grid using diffusers utility
        grid = make_image_grid(all_images, rows=rows, cols=cols)

        return grid

    def batch_process(
        self,
        content_images: List[Union[str, Path, Image.Image]],
        style_image: Union[str, Path, Image.Image],
        output_dir: Union[str, Path] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Batch process multiple content images with the same style.

        Args:
            content_images: List of content images
            style_image: The style image to apply to all content images
            output_dir: Optional directory to save processed images
            **kwargs: Additional arguments for transfer_style

        Returns:
            List of stylized images
        """
        # Create output directory if provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

        # Process each content image
        results = []
        for i, content_image in enumerate(content_images):
            # Apply style transfer
            stylized = self.transfer_style(
                content_image=content_image, style_image=style_image, **kwargs
            )

            # Save if output directory provided
            if output_dir is not None:
                # Extract content image name if it's a path
                if isinstance(content_image, (str, Path)):
                    filename = Path(content_image).stem
                else:
                    filename = f"stylized_{i}"

                # Save the stylized image
                output_path = output_dir / f"{filename}_stylized.png"
                stylized.save(output_path)

            # Add to results
            results.append(stylized)

        return results
