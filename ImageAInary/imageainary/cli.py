"""
ImageAInary CLI - Command line interface for the stable diffusion based style transfer tool.
"""
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import typer

from imageainary.pipeline.style_transfer import StyleTransferPipeline
from imageainary.pipeline.style_transfer_v2 import StyleTransferPipelineV2

app = typer.Typer(
    name="imageainary",
    help="Stable diffusion based style transfer tool",
    add_completion=False,
)
console = Console()


@app.command("transfer")
def transfer_style(
    content_image: Path = typer.Argument(
        ..., help="Path to the content image", exists=True, dir_okay=False
    ),
    style_image: Path = typer.Argument(
        ..., help="Path to the style image", exists=True, dir_okay=False
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"), help="Directory to save generated images", dir_okay=True
    ),
    use_controlnet: bool = typer.Option(
        True,
        help="Enable ControlNet for preserving the structural elements of the content image",
    ),
    strength: float = typer.Option(
        0.8,
        help="Strength of style transfer (0.0 to 1.0), used when ControlNet is disabled",
        min=0.0,
        max=1.0,
    ),
    ip_adapter_scale: float = typer.Option(
        0.5,
        help="Scale for IP-Adapter (higher means stronger style influence)",
        min=0.0,
        max=1.0,
    ),
    controlnet_scale: float = typer.Option(
        0.7, help="Scale for ControlNet conditioning", min=0.0, max=1.0
    ),
    num_inference_steps: int = typer.Option(
        50, help="Number of inference steps", min=10, max=150
    ),
    guidance_scale: float = typer.Option(
        7.5, help="Guidance scale for stable diffusion", min=1.0, max=20.0
    ),
    width: int = typer.Option(768, help="Output image width", min=256, max=1024),
    height: int = typer.Option(768, help="Output image height", min=256, max=1024),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    """
    Transfer the style from a style image to a content image using stable diffusion with ControlNet and IP-Adapter.

    Args:
        content_image: Path to the content image
        style_image: Path to the style image
        output_dir: Directory to save output images
        use_controlnet: Whether to use ControlNet for better structure preservation
        strength: Strength of style transfer (0.0 to 1.0), used when ControlNet is disabled
        ip_adapter_scale: Scale for IP-Adapter (higher means stronger style influence)
        controlnet_scale: Scale for ControlNet conditioning
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for stable diffusion
        width: Output image width
        height: Output image height
        seed: Random seed for reproducibility
    """

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold red]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting the style transfer pipeline...", total=None)

        # Initialize the pipeline
        pipeline = StyleTransferPipeline(use_controlnet=use_controlnet)

        progress.update(
            task, description=f"Processing style transfer for image {content_image}..."
        )

        # Run style transfer
        output_image = pipeline.transfer_style(
            content_image=content_image,
            style_image=style_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            ip_adapter_scale=ip_adapter_scale,
            width=width,
            height=height,
            seed=seed,
        )

        # Save the output image
        output_path = (
            output_dir / f"stylized_{content_image.stem}_with_{style_image.stem}.png"
        )
        output_image.save(output_path)

        progress.update(
            task, description=f"Style transfer complete! Saved to {output_path}"
        )

    console.print(
        f"[bold green]Style transfer complete![/] Output saved to: {output_path}"
    )


@app.command("grid")
def create_style_grid(
    content_image: Path = typer.Argument(
        ..., help="Path to the content image", exists=True, dir_okay=False
    ),
    style_image: Path = typer.Argument(
        ..., help="Path to the style image", exists=True, dir_okay=False
    ),
    output_path: Path = typer.Option(
        Path("./style_grid.png"), help="Path to save the grid image"
    ),
    variations: int = typer.Option(
        3, help="Number of style variations to generate", min=1, max=10
    ),
    use_controlnet: bool = typer.Option(
        True, help="Use ControlNet for better structure preservation"
    ),
    ip_adapter_scale: float = typer.Option(
        0.5,
        help="Scale for IP-Adapter (higher means stronger style influence)",
        min=0.0,
        max=1.0,
    ),
    num_inference_steps: int = typer.Option(
        30, help="Number of inference steps", min=10, max=100
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    """
    Create a grid of style transfer variations showing the original content, style, and multiple style variations.

    Args:
        content_image: Path to the content image
        style_image: Path to the style image
        output_path: Path to save the grid image
        variations: Number of style variations to generate
        use_controlnet: Whether to use ControlNet for better structure preservation
        ip_adapter_scale: Scale for IP-Adapter (higher means stronger style influence)
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
    """
    # create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating ", total=None)

        # Initialize the pipeline
        pipeline = StyleTransferPipeline(use_controlnet=use_controlnet)

        progress.update(
            task,
            description=f"Generating {variations} style variations for image {content_image}...",
        )

        # Create style grid
        grid_image = pipeline.create_style_grid(
            content_image=content_image,
            style_image=style_image,
            variations=variations,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        # Save the grid image
        grid_image.save(output_path)

        progress.update(
            task, description=f"Style grid completed! Saved to {output_path}"
        )

    console.print(
        f"[bold green]Style grid completed![/] Output saved to: {output_path}"
    )


@app.command("advanced-transfer")
def advanced_transfer(
    content_image: Path = typer.Argument(
        ..., help="Path to the content image", exists=True, dir_okay=False
    ),
    style_image: Path = typer.Argument(
        ..., help="Path to the style image", exists=True, dir_okay=False
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"), help="Directory to save output images", dir_okay=True
    ),
    control_mode: str = typer.Option(
        "canny", help="ControlNet mode to use (canny, pose, depth)"
    ),
    scheduler: str = typer.Option(
        "ddim", help="Scheduler to use (ddim, pndm, euler_ancestral, dpm_solver)"
    ),
    use_controlnet: bool = typer.Option(
        True, help="Use ControlNet for better structure preservation"
    ),
    strength: float = typer.Option(
        0.8,
        help="Strength of style transfer (0.0 to 1.0), used when ControlNet is disabled",
        min=0.0,
        max=1.0,
    ),
    ip_adapter_scale: float = typer.Option(
        0.5,
        help="Scale for IP-Adapter (higher means stronger style influence)",
        min=0.0,
        max=1.0,
    ),
    controlnet_scale: float = typer.Option(
        0.7, help="Scale for ControlNet conditioning", min=0.0, max=1.0
    ),
    num_inference_steps: int = typer.Option(
        50, help="Number of inference steps", min=10, max=150
    ),
    guidance_scale: float = typer.Option(
        7.5, help="Guidance scale for stable diffusion", min=1.0, max=20.0
    ),
    width: int = typer.Option(768, help="Output image width", min=256, max=1024),
    height: int = typer.Option(768, help="Output image height", min=256, max=1024),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    """
    Advanced style transfer with enhanced control options using StyleTransferPipelineV2.

    Features multiple ControlNet modes and scheduler options for better results.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Creating advanced style transfer pipeline...", total=None
        )

        # Initialize the pipeline with scheduler, default scheduler is ddim
        pipeline = StyleTransferPipelineV2(
            use_controlnet=use_controlnet, scheduler_type=scheduler
        )

        # Set control mode if using ControlNet
        if use_controlnet:
            try:
                pipeline.set_control_mode(control_mode)
                progress.update(
                    task,
                    description=f"Using {control_mode} control mode with {scheduler} scheduler...",
                )
            except ValueError as e:
                console.print(f"[bold red]Error:[/] {str(e)}")
                available_modes = list(pipeline.get_available_control_modes().keys())
                console.print(
                    f"Only use among the available control modes: {', '.join(available_modes)}"
                )
                return

        progress.update(task, description="Processing advanced style transfer...")

        # Run style transfer
        output_image = pipeline.transfer_style(
            content_image=content_image,
            style_image=style_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            ip_adapter_scale=ip_adapter_scale,
            width=width,
            height=height,
            seed=seed,
        )

        # Save the output image
        output_path = (
            output_dir
            / f"stylized_{content_image.stem}_with_{style_image.stem}_{control_mode}_{scheduler}.png"
        )
        output_image.save(output_path)

        progress.update(
            task,
            description=f"Advanced style transfer completed! Saved to {output_path}",
        )

    console.print(
        f"[bold green]Advanced style transfer completed![/] Output saved to: {output_path}"
    )


@app.command("advanced-grid")
def advanced_style_grid(
    content_image: Path = typer.Argument(
        ..., help="Path to the content image", exists=True, dir_okay=False
    ),
    style_image: Path = typer.Argument(
        ..., help="Path to the style image", exists=True, dir_okay=False
    ),
    output_path: Path = typer.Option(
        Path("./style_grid.png"), help="Path to save the grid image"
    ),
    control_mode: str = typer.Option(
        "canny", help="ControlNet mode to use (canny, pose, depth)"
    ),
    scheduler: str = typer.Option(
        "ddim", help="Scheduler to use (ddim, pndm, euler_ancestral, dpm_solver)"
    ),
    variations: int = typer.Option(
        3, help="Number of style variations to generate", min=1, max=6
    ),
    use_different_seeds: bool = typer.Option(
        True, help="Use different seeds for each variation"
    ),
    use_controlnet: bool = typer.Option(
        True, help="Use ControlNet for better structure preservation"
    ),
    ip_adapter_scale: float = typer.Option(
        0.5,
        help="Scale for IP-Adapter (higher means stronger style influence)",
        min=0.0,
        max=1.0,
    ),
    controlnet_scale: float = typer.Option(
        0.7, help="Scale for ControlNet conditioning", min=0.0, max=1.0
    ),
    num_inference_steps: int = typer.Option(
        30, help="Number of inference steps", min=10, max=100
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    """
    Create an advanced grid of style transfer variations with enhanced controls.

    Features multiple ControlNet modes, scheduler options, and seed variations.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating advanced style grid...", total=None)

        # Initialize the pipeline
        pipeline = StyleTransferPipelineV2(
            use_controlnet=use_controlnet, scheduler_type=scheduler
        )

        # Set control mode if using ControlNet
        if use_controlnet:
            try:
                pipeline.set_control_mode(control_mode)
                progress.update(
                    task,
                    description=f"Using {control_mode} control mode with {scheduler} scheduler...",
                )
            except ValueError as e:
                console.print(f"[bold red]Error:[/] {str(e)}")
                available_modes = list(pipeline.get_available_control_modes().keys())
                console.print(
                    f"Only use among the available control modes: {', '.join(available_modes)}"
                )
                return

        progress.update(
            task, description=f"Generating {variations} style variations..."
        )

        # Create style grid
        grid_image = pipeline.create_style_grid(
            content_image=content_image,
            style_image=style_image,
            variations=variations,
            use_different_seeds=use_different_seeds,
            seed=seed,
            ip_adapter_scale=ip_adapter_scale,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
        )

        # Save the grid image
        output_path_with_params = Path(
            f"{output_path.parent}/{output_path.stem}_{control_mode}_{scheduler}{output_path.suffix}"
        )
        grid_image.save(output_path_with_params)

        progress.update(
            task,
            description=f"Advanced style grid completed! Saved to {output_path_with_params}",
        )

    console.print(
        f"[bold green]Advanced style grid completed![/] Output saved to: {output_path_with_params}"
    )


@app.command("controls")
def show_control_options():
    """
    Display all the available control modes and scheduler options for advanced style transfer.
    """
    # Initialize pipeline just to get the options
    pipeline = StyleTransferPipelineV2(use_controlnet=True)

    # Show control modes
    console.print("[bold]Available Control Modes:[/]")
    control_table = Table(show_header=True)
    control_table.add_column("Mode")
    control_table.add_column("Description")

    for mode, description in pipeline.get_available_control_modes().items():
        control_table.add_row(mode, description)

    console.print(control_table)
    console.print("")

    # Show scheduler options
    console.print("[bold]Available Schedulers:[/]")
    scheduler_descriptions = {
        "ddim": "Denoising Diffusion Implicit Models - Deterministic, fewer steps needed",
        "pndm": "Pseudo Numerical Methods for Diffusion Models - Higher quality with fewer steps",
        "euler_ancestral": "Euler Ancestral Sampler - More variety in outputs",
        "dpm_solver": "DPM-Solver - Fast convergence with high quality",
    }

    scheduler_table = Table(show_header=True)
    scheduler_table.add_column("Type")
    scheduler_table.add_column("Description")

    for scheduler in pipeline.get_available_schedulers():
        scheduler_table.add_row(
            scheduler,
            scheduler_descriptions.get(scheduler, "Advanced diffusion scheduler"),
        )

    console.print(scheduler_table)


@app.command("batch-process")
def batch_process(
    content_directory: Path = typer.Argument(
        ..., help="Directory containing content images", exists=True, file_okay=False
    ),
    style_image: Path = typer.Argument(
        ..., help="Path to the style image", exists=True, dir_okay=False
    ),
    output_dir: Path = typer.Option(
        Path("./outputs/batch"), help="Directory to save output images", dir_okay=True
    ),
    control_mode: str = typer.Option(
        "canny", help="ControlNet mode to use (canny, pose, depth)"
    ),
    scheduler: str = typer.Option(
        "ddim", help="Scheduler to use (ddim, pndm, euler_ancestral, dpm_solver)"
    ),
    file_extension: str = typer.Option(
        "*.jpg", help="File extension pattern to match (e.g., *.jpg, *.png)"
    ),
    use_controlnet: bool = typer.Option(
        True, help="Use ControlNet for better structure preservation"
    ),
    ip_adapter_scale: float = typer.Option(
        0.5, help="Scale for IP-Adapter", min=0.0, max=1.0
    ),
    controlnet_scale: float = typer.Option(
        0.7, help="Scale for ControlNet conditioning", min=0.0, max=1.0
    ),
    num_inference_steps: int = typer.Option(
        30, help="Number of inference steps", min=10, max=100
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
):
    """
    Batch process multiple content images with the same style.

    Processes all matching images in the specified directory.
    """
    # Find all matching content images
    content_images = list(content_directory.glob(file_extension))

    if not content_images:
        console.print(
            f"[bold red]Error:[/] No files matching '{file_extension}' found in {content_directory}"
        )
        return

    console.print(f"Found {len(content_images)} images to process")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Initializing advanced style transfer pipeline...", total=None
        )

        # Initialize the pipeline
        pipeline = StyleTransferPipelineV2(
            use_controlnet=use_controlnet, scheduler_type=scheduler
        )

        # Set control mode if using ControlNet
        if use_controlnet:
            try:
                pipeline.set_control_mode(control_mode)
                progress.update(
                    task,
                    description=f"Using {control_mode} control mode with {scheduler} scheduler...",
                )
            except ValueError as e:
                console.print(f"[bold red]Error:[/] {str(e)}")
                available_modes = list(pipeline.get_available_control_modes().keys())
                console.print(f"Available control modes: {', '.join(available_modes)}")
                return

        progress.update(task, description=f"Processing {len(content_images)} images...")

        # Batch process images
        results = pipeline.batch_process(
            content_images=content_images,
            style_image=style_image,
            output_dir=output_dir,
            controlnet_conditioning_scale=controlnet_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        progress.update(
            task,
            description=f"Batch processing complete! {len(results)} images saved to {output_dir}",
        )

    console.print(
        f"[bold green]Batch processing complete![/] {len(results)} images saved to: {output_dir}"
    )


@app.command("info")
def show_info():
    """
    Display information about ImageAInary.
    """
    console.print("[bold]ImageAInary[/] - Stable diffusion based style transfer tool")
    console.print("Version: 0.2.0")
    console.print("Created by: Pradipta Deb")
    console.print("\nFeatures:")
    console.print("  • Advanced style transfer using ControlNet and IP-Adapter")
    console.print("  • Multiple ControlNet modes (canny, pose, depth)")
    console.print(
        "  • Various diffusion schedulers for different quality/speed tradeoffs"
    )
    console.print("  • Create style variation grids")
    console.print("  • Batch processing capabilities")
    console.print("  • Customizable style transfer parameters")
    console.print("\nCommands:")
    console.print(
        "  • [bold]imageainary transfer[/] - Basic style transfer between images"
    )
    console.print(
        "  • [bold]imageainary grid[/] - Create a basic grid of style variations"
    )
    console.print(
        "  • [bold]imageainary advanced-transfer[/] - Enhanced style transfer with multiple control modes"
    )
    console.print(
        "  • [bold]imageainary advanced-grid[/] - Create an enhanced grid with multiple control options"
    )
    console.print(
        "  • [bold]imageainary batch-process[/] - Process multiple images with the same style"
    )
    console.print(
        "  • [bold]imageainary controls[/] - Display available control modes and schedulers"
    )
    console.print("  • [bold]imageainary info[/] - Display this information")
    console.print(
        "\nUse [bold]imageainary COMMAND --help[/] to see command-specific options"
    )


if __name__ == "__main__":
    app()
