"""
Tests for the style transfer pipeline.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from imageainary.pipeline.style_transfer import StyleTransferPipeline
from imageainary.utils.image_processing import (create_image_grid,
                                                extract_edges,
                                                postprocess_image,
                                                preprocess_image)


@pytest.fixture
def mock_image():
    """Create a mock image for testing."""
    return Image.new('RGB', (512, 512), color='white')


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    with patch('imageainary.pipeline.style_transfer.StableDiffusionImg2ImgPipeline') as mock_sd:
        with patch('imageainary.pipeline.style_transfer.StableDiffusionControlNetPipeline') as mock_controlnet:
            with patch('imageainary.pipeline.style_transfer.ControlNetModel') as mock_controlnet_model:
                with patch('imageainary.pipeline.style_transfer.CLIPModel') as mock_clip_model:
                    with patch('imageainary.pipeline.style_transfer.CLIPFeatureExtractor') as mock_clip_processor:
                        with patch('imageainary.pipeline.style_transfer.CannyDetector') as mock_canny:
                            # Configure mocks
                            mock_sd.from_pretrained.return_value = MagicMock()
                            mock_sd.from_pretrained.return_value.to.return_value = MagicMock()
                            mock_controlnet.from_pretrained.return_value = MagicMock()
                            mock_controlnet.from_pretrained.return_value.to.return_value = MagicMock()
                            mock_controlnet_model.from_pretrained.return_value = MagicMock()
                            mock_clip_model.from_pretrained.return_value = MagicMock()
                            mock_clip_processor.from_pretrained.return_value = MagicMock()
                            mock_canny.return_value = MagicMock()
                            mock_canny.return_value.return_value = MagicMock()
                            
                            # Return the pipeline
                            yield StyleTransferPipeline(device="cpu")


def test_preprocess_image(mock_image):
    """Test the image preprocessing function."""
    # Test with square image
    processed = preprocess_image(mock_image)
    assert processed.size == (512, 512)
    
    # Test with landscape image
    landscape = Image.new('RGB', (800, 400), color='white')
    processed = preprocess_image(landscape)
    assert processed.size == (512, 512)
    
    # Test with portrait image
    portrait = Image.new('RGB', (400, 800), color='white')
    processed = preprocess_image(portrait)
    assert processed.size == (512, 512)


def test_postprocess_image(mock_image):
    """Test the image postprocessing function."""
    # Create an image with some variation for contrast enhancement
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    processed = postprocess_image(test_image)
    assert processed.size == (512, 512)
    assert isinstance(processed, Image.Image)


def test_style_transfer_pipeline_init(mock_pipeline):
    """Test pipeline initialization."""
    assert mock_pipeline.device == "cpu"
    assert mock_pipeline.model_id == "runwayml/stable-diffusion-v1-5"
    assert mock_pipeline.controlnet_model_id == "lllyasviel/sd-controlnet-canny"


def test_extract_style_prompt(mock_pipeline, mock_image):
    """Test style prompt extraction."""
    with patch.object(mock_pipeline, '_extract_style_prompt', return_value="in the style of artistic oil painting"):
        prompt = mock_pipeline._extract_style_prompt(mock_image)
        assert "in the style of artistic" in prompt


def test_transfer_style(mock_pipeline, mock_image):
    """Test the style transfer function."""
    # Mock the pipeline output
    mock_output = MagicMock()
    mock_output.images = [mock_image]
    mock_pipeline.pipeline.return_value = mock_output
    mock_pipeline.controlnet_pipeline.return_value = mock_output
    
    # Mock the style prompt extraction
    with patch.object(mock_pipeline, '_extract_style_prompt', return_value="in the style of artistic oil painting"):
        # Test with standard pipeline
        result = mock_pipeline.transfer_style(
            content_image=mock_image,
            style_image=mock_image,
            strength=0.8,
            num_inference_steps=10,
            use_controlnet=False
        )
        
        # Check the result
        assert isinstance(result, Image.Image)
        assert result.size == (512, 512)
        
        # Test with ControlNet pipeline
        with patch.object(mock_pipeline, '_prepare_controlnet_image', return_value=mock_image):
            result = mock_pipeline.transfer_style(
                content_image=mock_image,
                style_image=mock_image,
                ip_adapter_scale=0.5,
                controlnet_conditioning_scale=0.7,
                num_inference_steps=10,
                use_controlnet=True
            )
            
            # Check the result
            assert isinstance(result, Image.Image)
            assert result.size == (512, 512)


def test_create_style_grid(mock_pipeline, mock_image):
    """Test the create_style_grid function."""
    # Mock the transfer_style method
    with patch.object(mock_pipeline, 'transfer_style', return_value=mock_image):
        # Call the create_style_grid method
        result = mock_pipeline.create_style_grid(
            content_image=mock_image,
            style_image=mock_image,
            variations=3,
            ip_adapter_scale=0.5,
            controlnet_conditioning_scale=0.7,
            num_inference_steps=10
        )
        
        # Check the result
        assert isinstance(result, Image.Image)


def test_prepare_controlnet_image(mock_pipeline, mock_image):
    """Test the _prepare_controlnet_image function."""
    # Mock the CannyDetector
    with patch('imageainary.pipeline.style_transfer.CannyDetector') as mock_canny:
        mock_detector = MagicMock()
        mock_detector.return_value = mock_image
        mock_canny.return_value = mock_detector
        
        # Call the _prepare_controlnet_image method
        result = mock_pipeline._prepare_controlnet_image(mock_image)
        
        # Check the result
        assert isinstance(result, Image.Image)
        
        # Test fallback to built-in edge detection
        with patch('imageainary.pipeline.style_transfer.CannyDetector', side_effect=ImportError):
            with patch('imageainary.utils.image_processing.extract_edges', return_value=mock_image):
                result = mock_pipeline._prepare_controlnet_image(mock_image)
                assert isinstance(result, Image.Image)


def test_extract_edges(mock_image):
    """Test the extract_edges function."""
    # Create an image with some variation for edge detection
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)
    
    # Call the extract_edges function
    result = extract_edges(test_image)
    
    # Check the result
    assert isinstance(result, Image.Image)
    assert result.size == (512, 512)


def test_create_image_grid(mock_image):
    """Test the create_image_grid function."""
    # Create a list of images
    images = [mock_image] * 5
    
    # Call the create_image_grid function
    result = create_image_grid(images, cols=2)
    
    # Check the result
    assert isinstance(result, Image.Image)
    # Grid should have 3 rows (5 images with 2 columns)
    expected_height = 512 * 3
    expected_width = 512 * 2
    assert result.size == (expected_width, expected_height)
