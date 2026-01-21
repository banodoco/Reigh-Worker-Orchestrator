import os
import sys
import asyncio
import json
import logging
import tempfile
import base64
import random
from typing import Any, Dict, Optional
from io import BytesIO

import httpx
import fal_client
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env at import time so module-level reads work
load_dotenv()

# Configure structured logging before importing internal modules
from .logging_config import setup_logging, set_current_cycle, get_db_logging_stats
from .database import DatabaseClient

# Initial setup without database client
setup_logging()

# Import utility modules
from .storage_utils import process_external_url_result, upload_to_supabase_storage, upload_to_supabase_storage_only
from .task_utils import count_tasks, claim_next_task, mark_complete, mark_failed
from .wavespeed_utils import call_wavespeed_api
from .video_utils import download_video_to_temp, remove_last_frame_from_video, join_videos, extract_first_frame_bytes

CONCURRENCY = int(os.getenv("API_WORKER_CONCURRENCY", "20"))
RUN_TYPE = "api"  # Hardcoded for API workers - they process API tasks
PARENT_POLL_SEC = int(os.getenv("API_PARENT_POLL_SEC", "10"))


logger = logging.getLogger(__name__)


async def call_fal_api(
    endpoint: str,
    arguments: Dict[str, Any],
    client: httpx.AsyncClient,
) -> Dict[str, Any]:
    """
    Generic helper for fal.ai API calls.
    
    Args:
        endpoint: The fal.ai endpoint (e.g., "fal-ai/qwen-image")
        arguments: Dictionary of arguments to pass to the API
        client: httpx client (unused but kept for consistency with wavespeed pattern)
    
    Returns:
        Transformed result dictionary with 'output_url' key
    """
    logger.info(f"Calling fal.ai endpoint: {endpoint}")
    logger.info(f"Arguments: {json.dumps(arguments, default=str)[:500]}...")
    
    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                logger.info(f"fal.ai: {log['message']}")
    
    # Run the blocking fal_client.subscribe in a thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: fal_client.subscribe(
            endpoint,
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update,
        )
    )
    
    logger.info(f"fal.ai result: {result}")
    
    # Transform fal.ai result format to expected format
    # fal.ai typically returns: {'image': {'url': '...', 'width': ..., 'height': ...}, ...}
    # Or for multiple images: {'images': [{'url': '...', ...}, ...], ...}
    # We need: {'output_url': '...'}
    
    if isinstance(result, dict):
        # Handle single image response
        if 'image' in result:
            image_data = result['image']
            if isinstance(image_data, dict) and 'url' in image_data:
                return {
                    'output_url': image_data['url'],
                    'width': image_data.get('width'),
                    'height': image_data.get('height'),
                    'seed': result.get('seed'),
                    'content_type': image_data.get('content_type')
                }
        
        # Handle multiple images response (take first)
        if 'images' in result and result['images']:
            image_data = result['images'][0]
            if isinstance(image_data, dict) and 'url' in image_data:
                return {
                    'output_url': image_data['url'],
                    'width': image_data.get('width'),
                    'height': image_data.get('height'),
                    'seed': result.get('seed'),
                    'content_type': image_data.get('content_type'),
                    'all_images': [img.get('url') for img in result['images'] if img.get('url')]
                }
        
        # Handle direct URL response
        if 'url' in result:
            return {
                'output_url': result['url'],
                'width': result.get('width'),
                'height': result.get('height'),
                'seed': result.get('seed'),
            }
    
    # Fallback if format is unexpected
    logger.warning(f"Unexpected fal.ai result format, returning as-is: {result}")
    return result


def build_fal_lora_list(params: Dict[str, Any]) -> list:
    """
    Build a list of LoRA configurations from task params.
    
    Supports multiple input formats:
    - params["loras"]: list of {"path": ..., "scale": ...} or {"url": ..., "strength": ...}
    - params["additional_loras"]: dict of {path: scale, ...}
    
    Returns:
        List of LoRA configs in fal.ai format: [{"path": ..., "scale": ...}, ...]
    """
    loras = []
    
    # Handle list format
    loras_list = params.get("loras", [])
    for lora in loras_list:
        if isinstance(lora, dict):
            lora_path = lora.get("path") or lora.get("url", "")
            lora_scale = lora.get("scale", lora.get("strength", 1.0))
            if lora_path:
                loras.append({"path": lora_path, "scale": float(lora_scale)})
    
    # Handle dict format (additional_loras)
    additional_loras = params.get("additional_loras", {})
    if isinstance(additional_loras, dict):
        for lora_path, scale in additional_loras.items():
            loras.append({"path": lora_path, "scale": float(scale)})
    
    return loras


def normalize_resolution(resolution: str, min_dimension: int = 512, max_dimension: int = 1200) -> Optional[str]:
    """
    Normalize a resolution string to fit within min/max dimension constraints.
    
    Args:
        resolution: Resolution string like "400x225" or "1920*1080"
        min_dimension: Minimum size for the shortest side (default 512)
        max_dimension: Maximum size for the longest side (default 1200)
    
    Returns:
        Normalized resolution string like "512*288" or None if parsing failed
    """
    if not resolution:
        return None
    
    parts = resolution.replace("*", "x").split("x")
    if len(parts) != 2:
        logger.warning(f"Invalid resolution format '{resolution}', expected WIDTHxHEIGHT")
        return None
    
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        logger.warning(f"Could not parse resolution '{resolution}' as integers")
        return None
    
    original_width, original_height = width, height
    
    # First, scale UP if below minimum (use shortest side)
    shortest_side = min(width, height)
    if shortest_side < min_dimension:
        ratio = min_dimension / shortest_side
        width = int(width * ratio)
        height = int(height * ratio)
        logger.info(f"Scaled up resolution from {original_width}x{original_height} to {width}x{height} (min {min_dimension}px)")
    
    # Then, scale DOWN if above maximum (use longest side)
    longest_side = max(width, height)
    if longest_side > max_dimension:
        ratio = max_dimension / longest_side
        width = int(width * ratio)
        height = int(height * ratio)
        logger.info(f"Capped resolution to {width}x{height} (max {max_dimension}px)")
    
    return f"{width}*{height}"


async def get_image_dimensions(client: httpx.AsyncClient, image_url: str) -> tuple[int, int] | None:
    """
    Fetch image dimensions from a URL by downloading just enough to read the header.
    
    Args:
        client: httpx client for downloading
        image_url: URL of the image
    
    Returns:
        Tuple of (width, height) or None if failed
    """
    try:
        # Stream just enough bytes to get image header (usually < 64KB is enough)
        async with client.stream("GET", image_url, timeout=30.0) as response:
            response.raise_for_status()
            chunks = []
            bytes_read = 0
            max_bytes = 64 * 1024  # 64KB should be enough for any image header
            
            async for chunk in response.aiter_bytes(chunk_size=8192):
                chunks.append(chunk)
                bytes_read += len(chunk)
                if bytes_read >= max_bytes:
                    break
            
            data = b"".join(chunks)
            img = Image.open(BytesIO(data))
            width, height = img.size
            logger.info(f"Got image dimensions from URL: {width}x{height}")
            return width, height
            
    except Exception as e:
        logger.warning(f"Failed to get image dimensions from {image_url}: {e}")
        return None


def scale_dimensions_to_megapixels(
    width: int, 
    height: int, 
    target_megapixels: float = 1.0,
    round_to: int = 8
) -> dict[str, int]:
    """
    Scale dimensions to approximately target megapixels while preserving aspect ratio.
    
    Args:
        width: Original width
        height: Original height
        target_megapixels: Target total pixels in millions (default 1.0 = 1024x1024)
        round_to: Round dimensions to nearest multiple of this (default 8 for model compatibility)
    
    Returns:
        Dict with {"width": scaled_width, "height": scaled_height}
    """
    import math
    
    current_pixels = width * height
    target_pixels = target_megapixels * 1_000_000
    
    # Calculate scale factor
    scale = math.sqrt(target_pixels / current_pixels)
    
    # Apply scale and round to nearest multiple
    new_width = round(width * scale / round_to) * round_to
    new_height = round(height * scale / round_to) * round_to
    
    # Ensure minimum dimensions
    new_width = max(new_width, round_to)
    new_height = max(new_height, round_to)
    
    actual_pixels = new_width * new_height
    logger.info(f"Scaled {width}x{height} ({current_pixels/1e6:.2f}MP) -> {new_width}x{new_height} ({actual_pixels/1e6:.2f}MP)")
    
    return {"width": new_width, "height": new_height}


async def create_masked_composite_image(
    client: httpx.AsyncClient,
    task_id: str,
    image_url: str,
    mask_url: str,
    filename_prefix: str = "composite"
) -> str:
    """
    Download image and mask, create green-overlay composite, upload to Supabase.
    Returns the public URL of the uploaded composite image.
    
    Args:
        client: httpx client for downloading
        task_id: task ID for upload naming
        image_url: URL of the original image
        mask_url: URL of the mask (white = areas to edit)
        filename_prefix: prefix for uploaded filename
    
    Returns:
        Public URL of the uploaded composite image
    """
    try:
        # Download the original image
        logger.info("Downloading original image...")
        image_response = await client.get(image_url)
        image_response.raise_for_status()
        image = Image.open(BytesIO(image_response.content)).convert("RGB")
        logger.info(f"Image downloaded: {image.size[0]}x{image.size[1]}")
        
        # Download the mask
        logger.info("Downloading mask...")
        mask_response = await client.get(mask_url)
        mask_response.raise_for_status()
        mask = Image.open(BytesIO(mask_response.content)).convert("L")  # Convert to grayscale
        logger.info(f"Mask downloaded: {mask.size[0]}x{mask.size[1]}")
        
        # Resize image to a reasonable size if it's too large
        # Keep max 1200px on the widest side while maintaining aspect ratio
        max_dimension = 1200
        if image.size[0] > max_dimension or image.size[1] > max_dimension:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_dimension / image.size[0], max_dimension / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            logger.info(f"Resizing image from {image.size[0]}x{image.size[1]} to {new_size[0]}x{new_size[1]}")
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Resize mask to match image dimensions if needed
        if mask.size != image.size:
            logger.info(f"Resizing mask from {mask.size[0]}x{mask.size[1]} to {image.size[0]}x{image.size[1]}")
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            # Apply binary threshold to eliminate gray values from interpolation
            # This ensures crisp black/white boundaries and prevents graininess
            mask = mask.point(lambda x: 255 if x > 127 else 0)
            logger.info("Applied binary threshold to mask after resizing")
        
        # Create a pure green overlay where the mask is white
        # Create a green image of the same size
        green_overlay = Image.new("RGB", image.size, (0, 255, 0))
        
        # Composite: where mask is white (255), use green; where black (0), use original image
        # Image.composite uses the mask as an alpha channel
        composite = Image.composite(green_overlay, image, mask)
        
        logger.info("Created composite image with green mask overlay")
        
        # Upload composite image to Supabase storage
        # Use JPEG format with quality setting to reduce file size
        composite_bytes = BytesIO()
        composite.save(composite_bytes, format='JPEG', quality=95, optimize=True)
        composite_bytes.seek(0)
        
        file_size_mb = len(composite_bytes.getvalue()) / (1024 * 1024)
        logger.info(f"Composite image size: {file_size_mb:.2f}MB")
        
        # Upload composite WITHOUT marking task complete (we need to call Wavespeed API first)
        composite_url = await upload_to_supabase_storage_only(
            client,
            task_id,
            composite_bytes.getvalue(),
            filename=f"{filename_prefix}_{task_id}.jpg"
        )
        logger.info(f"Uploaded composite image to: {composite_url}")
        
        return composite_url
        
    except Exception as e:
        logger.error(f"Failed to process images for masked composite: {e}")
        raise Exception(f"Image processing failed: {str(e)}")


async def process_api_task(task: Dict[str, Any], client: httpx.AsyncClient) -> Dict[str, Any]:
    """Process Wavespeed AI tasks"""
    params = task.get("params") or {}
    task_type = task.get("task_type", "unknown")
    task_id = task.get("task_id") or task.get("id")
    
    # Parse params if it's a JSON string
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse params JSON: {e}")
            raise Exception(f"Invalid JSON in params field: {e}")
    
    if task_type == "qwen_image_edit" or params.get("api_type") == "wavespeed":
        # Determine which model/endpoint to use based on qwen_edit_model parameter
        qwen_edit_model = params.get("qwen_edit_model", "qwen-edit")  # default to current behavior

        # Check if we have LoRAs to determine if we need LoRA endpoint variant
        loras = params.get("loras", [])
        has_loras = bool(loras)

        # Map model names to endpoints
        # 2511 and 2509 (edit-plus) use "images" array format
        # Default edit-lora uses "image" string format
        if qwen_edit_model == "qwen-edit-2511":
            endpoint_path = "wavespeed-ai/qwen-image/edit-2511-lora" if has_loras else "wavespeed-ai/qwen-image/edit-2511"
            use_images_array = True
        elif qwen_edit_model == "qwen-edit-2509":
            endpoint_path = "wavespeed-ai/qwen-image/edit-plus-lora" if has_loras else "wavespeed-ai/qwen-image/edit-plus"
            use_images_array = True
        else:  # "qwen-edit" or default
            endpoint_path = "wavespeed-ai/qwen-image/edit-lora"
            use_images_array = False

        logger.info(f"Using qwen_edit_model: {qwen_edit_model} -> endpoint: {endpoint_path}")

        # Build parameters based on API format
        image_url = params.get("image", "")

        if use_images_array:
            # 2511/2509 APIs use "images" array format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "images": [image_url] if image_url else [],
                "output_format": params.get("output_format", "jpeg"),
                "prompt": params.get("prompt", ""),
                "seed": params.get("seed", -1),
            }

            # Add size parameter if resolution is provided
            resolution = params.get("resolution", "")
            if resolution:
                normalized_size = normalize_resolution(resolution)
                if normalized_size:
                    wavespeed_params["size"] = normalized_size
                    logger.info(f"Using resolution/size: {normalized_size}")

            # Add LoRAs if using lora endpoint variant
            if has_loras:
                wavespeed_params["loras"] = []
                for lora in loras:
                    if isinstance(lora, dict):
                        lora_url = lora.get("url") or lora.get("path", "")
                        lora_strength = lora.get("strength", lora.get("scale", 1.0))
                        if lora_url:
                            wavespeed_params["loras"].append({
                                "path": lora_url,
                                "scale": float(lora_strength)
                            })
                            logger.info(f"Added LoRA: {lora_url} with strength {lora_strength}")
                logger.info(f"Processing with {len(wavespeed_params['loras'])} LoRAs")
            else:
                logger.info("Processing without LoRAs")
        else:
            # Default edit-lora API uses "image" string format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "image": image_url,  # API expects "image" as string (not array)
                "output_format": params.get("output_format", "jpeg"),
                "prompt": params.get("prompt", ""),
                "seed": params.get("seed", -1),
                "loras": []
            }

            # Add size parameter if resolution is provided
            resolution = params.get("resolution", "")
            if resolution:
                normalized_size = normalize_resolution(resolution)
                if normalized_size:
                    wavespeed_params["size"] = normalized_size
                    logger.info(f"Using resolution/size: {normalized_size}")

            # Map loras - support both {"url": ..., "strength": ...} and {"path": ..., "scale": ...} formats
            for lora in loras:
                if isinstance(lora, dict):
                    # Support both "url"/"strength" and "path"/"scale" formats
                    lora_url = lora.get("url") or lora.get("path", "")
                    lora_strength = lora.get("strength", lora.get("scale", 1.0))

                    if lora_url:
                        wavespeed_params["loras"].append({
                            "path": lora_url,
                            "scale": float(lora_strength)
                        })
                        logger.info(f"Added LoRA: {lora_url} with strength {lora_strength}")

            if loras:
                logger.info(f"Processing with {len(wavespeed_params['loras'])} LoRAs")
            else:
                logger.info("Processing without LoRAs")

        result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)
        
        # Process external URL with automatic screenshot extraction for videos
        result = await process_external_url_result(client, task_id, result)
        
        logger.info(f"Processed {task_type} task via Wavespeed API")
        return result
        
    elif task_type == "qwen_image_style":
        # Wavespeed AI Qwen image style transfer with LoRA
        logger.info(f"Processing {task_type} task via Wavespeed API")
        
        # Determine which model/endpoint to use based on qwen_edit_model parameter
        # Style transfer always uses LoRA endpoints since it needs style/subject/scene LoRAs
        qwen_edit_model = params.get("qwen_edit_model", "qwen-edit")  # default to current behavior
        
        # Map model names to endpoints
        # 2511 and 2509 (edit-plus) use "images" array format
        # Default edit-lora uses "image" string format
        if qwen_edit_model == "qwen-edit-2511":
            endpoint_path = "wavespeed-ai/qwen-image/edit-2511-lora"
            use_images_array = True
        elif qwen_edit_model == "qwen-edit-2509":
            endpoint_path = "wavespeed-ai/qwen-image/edit-plus-lora"
            use_images_array = True
        else:  # "qwen-edit" or default
            endpoint_path = "wavespeed-ai/qwen-image/edit-lora"
            use_images_array = False
        
        logger.info(f"Using qwen_edit_model: {qwen_edit_model} -> endpoint: {endpoint_path}")
        
        # Build the prompt with style and subject modifications
        original_prompt = params.get("prompt", "")
        modified_prompt = original_prompt
        
        # Get style, subject, and scene parameters
        style_strength = params.get("style_reference_strength", 0.0)
        subject_strength = params.get("subject_strength", 0.0)
        scene_strength = params.get("scene_reference_strength", 0.0)
        subject_description = params.get("subject_description", "")
        in_this_scene = params.get("in_this_scene", False)
        
        # Build prompt modifications
        prompt_parts = []
        has_style_prefix = False
        
        # Add style prefix if style_strength > 0
        if style_strength > 0.0:
            prompt_parts.append("In the style of this image,")
            has_style_prefix = True
        
        # Add subject prefix if subject_strength > 0
        if subject_strength > 0.0 and subject_description:
            # Use lowercase 'make' if style prefix is already present
            make_word = "make" if has_style_prefix else "Make"
            if in_this_scene:
                prompt_parts.append(f"{make_word} an image of this {subject_description} in this scene:")
            else:
                prompt_parts.append(f"{make_word} an image of this {subject_description}:")
        
        # Combine prompt parts with original prompt
        if prompt_parts:
            modified_prompt = " ".join(prompt_parts) + " " + original_prompt
            logger.info(f"Modified prompt from '{original_prompt}' to '{modified_prompt}'")
        
        # Determine which reference image to use (they should be the same)
        reference_image = params.get("style_reference_image") or params.get("subject_reference_image", "")
        
        logger.info(f"Using reference image: {reference_image}")
        
        # Build LoRA list for style transfer
        style_loras = []
        
        # Add LoRA configuration for style transfer
        # Use a default style transfer LoRA if style_reference_strength is provided
        if style_strength > 0.0:
            # Default style transfer LoRA path - can be overridden via params
            default_lora_path = "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/style_transfer_qwen_edit_2_000011250.safetensors"
            lora_path = params.get("style_lora_path", default_lora_path)
            
            style_loras.append({
                "path": lora_path,
                "scale": float(style_strength)
            })
            logger.info(f"Added style transfer LoRA: {lora_path} with strength {style_strength}")
        
        # Add subject LoRA if subject_strength > 0
        if subject_strength > 0.0:
            # Add subject LoRA
            subject_lora_path = "https://huggingface.co/peteromallet/mystery_models/resolve/main/in_subject_qwen_edit_2_000006750.safetensors"
            style_loras.append({
                "path": subject_lora_path,
                "scale": float(subject_strength)
            })
            logger.info(f"Added subject LoRA: {subject_lora_path} with strength {subject_strength}")
        
        # Add scene LoRA if scene_strength > 0
        if scene_strength > 0.0:
            # Add scene LoRA for "in the same scene" transformations
            scene_lora_path = "https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/in_scene_different_perspective_000019000.safetensors"
            style_loras.append({
                "path": scene_lora_path,
                "scale": float(scene_strength)
            })
            logger.info(f"Added scene LoRA: {scene_lora_path} with strength {scene_strength}")
        
        # Add any additional LoRAs from params
        additional_loras = params.get("loras", [])
        if additional_loras:
            for lora in additional_loras:
                if isinstance(lora, dict) and "path" in lora and "scale" in lora:
                    style_loras.append({
                        "path": lora["path"],
                        "scale": float(lora["scale"])
                    })
            logger.info(f"Added {len(additional_loras)} additional LoRAs")
        
        # Build parameters based on API format
        if use_images_array:
            # 2511/2509 APIs use "images" array format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "images": [reference_image] if reference_image else [],
                "output_format": params.get("output_format", "jpeg"),
                "prompt": modified_prompt,
                "seed": params.get("seed", -1),
                "loras": style_loras
            }
        else:
            # Default edit-lora API uses "image" string format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "output_format": params.get("output_format", "jpeg"),
                "prompt": modified_prompt,
                "seed": params.get("seed", -1),
                "image": reference_image,
                "model_id": params.get("model_id", "wavespeed-ai/qwen-image/edit-lora"),
                "loras": style_loras
            }
        
        # Add size parameter if resolution is provided
        resolution = params.get("resolution", "")
        if resolution:
            normalized_size = normalize_resolution(resolution)
            if normalized_size:
                wavespeed_params["size"] = normalized_size
                logger.info(f"Using resolution/size: {normalized_size}")
        
        result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)
        
        # Process external URL with automatic screenshot extraction for videos
        result = await process_external_url_result(client, task_id, result)
        
        logger.info(f"Processed {task_type} task via Wavespeed API")
        return result
        
    elif task_type == "wan_2_2_t2i":
        # Wavespeed AI WAN 2.2 Text-to-Image with LoRA
        endpoint_path = "wavespeed-ai/wan-2.2/text-to-image-lora"
        logger.info(f"Calling Wavespeed API endpoint: {endpoint_path}")
        
        # Extract orchestrator details or use top-level params
        orchestrator_details = params.get("orchestrator_details", {})
        effective_params = {**params, **orchestrator_details}
        
        # Map parameters to Wavespeed API format
        wavespeed_params = {
            "enable_base64_output": False,
            "enable_sync_mode": False,
            "output_format": "jpeg",
            "prompt": effective_params.get("prompt", ""),
            "seed": effective_params.get("seed", -1),
            "size": effective_params.get("resolution", "256*256").replace("x", "*"),
            "high_noise_loras": [],
            "low_noise_loras": [],
            "loras": []
        }
        
        # Extract and format LoRAs from additional_loras
        additional_loras = effective_params.get("additional_loras", {})
        if additional_loras:
            for lora_path, scale in additional_loras.items():
                wavespeed_params["loras"].append({
                    "path": lora_path,
                    "scale": float(scale)
                })
            logger.info(f"Added {len(additional_loras)} LoRAs to request")
        
        result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)
        
        # Process external URL with automatic screenshot extraction for videos
        result = await process_external_url_result(client, task_id, result)
        
        logger.info(f"Processed {task_type} task via Wavespeed API")
        return result
        
    elif task_type == "animate_character":
        # Wavespeed AI WAN 2.2 Character Animation
        endpoint_path = "wavespeed-ai/wan-2.2/animate"
        logger.info(f"Processing {task_type} task via Wavespeed API endpoint: {endpoint_path}")
        
        # Extract orchestrator details or use top-level params
        orchestrator_details = params.get("orchestrator_details", {})
        effective_params = {**params, **orchestrator_details}
        
        # Map parameters to Wavespeed API format for character animation
        wavespeed_params = {
            "image": effective_params.get("character_image_url", ""),
            "mode": effective_params.get("mode", "animate"),
            "prompt": effective_params.get("prompt", ""),
            "resolution": effective_params.get("resolution", "480p"),
            "seed": effective_params.get("seed", -1),
            "video": effective_params.get("motion_video_url", "")
        }
        
        logger.info(f"Character animation params: image={wavespeed_params['image'][:50]}..., "
                   f"video={wavespeed_params['video'][:50]}..., "
                   f"mode={wavespeed_params['mode']}, "
                   f"resolution={wavespeed_params['resolution']}, "
                   f"seed={wavespeed_params['seed']}")
        
        result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)
        
        # Process external URL with automatic screenshot extraction for videos
        result = await process_external_url_result(client, task_id, result)
        
        logger.info(f"Processed {task_type} task via Wavespeed API")
        return result
        
    elif task_type == "wan_2_2_i2v":
        # Wavespeed AI WAN 2.2 Image-to-Video
        logger.info(f"Processing {task_type} task via Wavespeed API")
        
        # Extract orchestrator details or use top-level params
        orchestrator_details = params.get("orchestrator_details", {})
        effective_params = {**params, **orchestrator_details}
        
        # Check if we have LoRAs to determine which endpoint to use
        additional_loras = effective_params.get("additional_loras", {})
        has_loras = bool(additional_loras)
        
        # Get input images and prompts
        input_images = effective_params.get("input_image_paths_resolved", [])
        base_prompts = effective_params.get("base_prompts_expanded", [])
        negative_prompts = effective_params.get("negative_prompts_expanded", [])
        
        # Fall back to base_prompt (singular) if base_prompts_expanded is empty or not provided
        if not base_prompts or (len(base_prompts) == 1 and not base_prompts[0]):
            base_prompt_singular = effective_params.get("base_prompt", "")
            if base_prompt_singular:
                base_prompts = [base_prompt_singular]
                logger.info(f"Using base_prompt fallback: '{base_prompt_singular}'")
        
        # If we have more than 2 images, generate pairwise transitions and join
        if len(input_images) > 2:
            logger.info(f"Processing {len(input_images)} images as pairwise transitions")
            video_segments: list[str] = []
            num_segments = max(0, len(input_images) - 1)

            try:
                # Build per-segment videos for each consecutive image pair
                for i in range(num_segments):
                    image_url = input_images[i]
                    next_image_url = input_images[i + 1]
                    logger.info(f"Processing segment {i+1}/{num_segments}: {image_url} -> {next_image_url}")

                    # Per-segment prompts
                    prompt = base_prompts[i] if i < len(base_prompts) else (base_prompts[0] if base_prompts else "")
                    negative_prompt = negative_prompts[i] if i < len(negative_prompts) else (negative_prompts[0] if negative_prompts else "")

                    if has_loras:
                        endpoint_path = "wavespeed-ai/wan-2.2/i2v-480p-lora"
                        wavespeed_params = {
                            "duration": effective_params.get("duration", 5),
                            "high_noise_loras": [],
                            "image": image_url,
                            "last_image": next_image_url,
                            "loras": [],
                            "low_noise_loras": [],
                            "negative_prompt": negative_prompt,
                            "prompt": prompt,
                            "seed": effective_params.get("seed_base", -1) + i
                        }
                        for lora_path, scale in additional_loras.items():
                            wavespeed_params["loras"].append({"path": lora_path, "scale": float(scale)})
                    else:
                        endpoint_path = "wavespeed-ai/wan-2.2/i2v-480p"
                        wavespeed_params = {
                            "seed": effective_params.get("seed_base", -1) + i,
                            "image": image_url,
                            "last_image": next_image_url,
                            "prompt": prompt,
                            "duration": effective_params.get("duration", 5),
                            "negative_prompt": negative_prompt,
                            "model_id": "wavespeed-ai/wan-2.2/i2v-480p"
                        }

                    # Call Wavespeed for this segment
                    segment_result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)

                    # Extract video URL directly from Wavespeed result
                    video_url = None
                    if isinstance(segment_result, dict):
                        if 'output_url' in segment_result:
                            video_url = segment_result['output_url']
                        elif 'url' in segment_result:
                            video_url = segment_result['url']
                        elif 'video_url' in segment_result:
                            video_url = segment_result['video_url']
                        elif 'outputs' in segment_result and segment_result['outputs']:
                            video_url = segment_result['outputs'][0]

                    if not video_url:
                        raise Exception(f"No video URL found in segment {i+1} result")

                    # Download to temp file
                    temp_video_path = await download_video_to_temp(client, video_url)
                    if not temp_video_path:
                        raise Exception(f"Failed to download video for segment {i+1}")

                    video_segments.append(temp_video_path)
                    logger.info(f"Segment {i+1}/{num_segments} ready: {temp_video_path}")

                # Remove last frame from all but the final segment and join
                if len(video_segments) > 1:
                    processed_segments: list[str] = []
                    for i, video_path in enumerate(video_segments[:-1]):
                        processed_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                        processed_file.close()
                        if remove_last_frame_from_video(video_path, processed_file.name):
                            processed_segments.append(processed_file.name)
                            logger.info(f"Removed last frame from segment {i+1}")
                        else:
                            processed_segments.append(video_path)
                            logger.warning(f"Could not remove last frame from segment {i+1}; using original")
                    processed_segments.append(video_segments[-1])

                    final_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                    final_video.close()
                    if not join_videos(processed_segments, final_video.name):
                        raise Exception("Failed to join video segments")
                    logger.info(f"Joined {len(processed_segments)} segments -> {final_video.name}")

                    # Upload final video to Supabase with first frame screenshot
                    with open(final_video.name, 'rb') as f:
                        file_bytes = f.read()
                    screenshot_bytes = extract_first_frame_bytes(file_bytes)
                    first_frame_b64 = base64.b64encode(screenshot_bytes).decode('utf-8') if screenshot_bytes else None
                    public_url = await upload_to_supabase_storage(
                        client,
                        task_id,
                        file_bytes,
                        filename=f"joined_{task_id}.mp4",
                        first_frame_data=first_frame_b64
                    )

                    # Cleanup temp files
                    for temp_path in video_segments + processed_segments:
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
                    try:
                        os.unlink(final_video.name)
                    except Exception:
                        pass

                    result = {
                        "output_location": public_url,
                        "output_url": public_url,
                        "segments_processed": num_segments,
                        "message": f"Successfully processed {num_segments} transitions into joined video",
                        "_task_completed_by_upload": True
                    }
                else:
                    # Degenerate case; return the only segment after uploading
                    only_path = video_segments[0]
                    with open(only_path, 'rb') as f:
                        file_bytes = f.read()
                    screenshot_bytes = extract_first_frame_bytes(file_bytes)
                    first_frame_b64 = base64.b64encode(screenshot_bytes).decode('utf-8') if screenshot_bytes else None
                    public_url = await upload_to_supabase_storage(
                        client,
                        task_id,
                        file_bytes,
                        filename=f"segment_{task_id}_0.mp4",
                        first_frame_data=first_frame_b64
                    )
                    try:
                        os.unlink(only_path)
                    except Exception:
                        pass
                    result = {
                        "output_location": public_url,
                        "output_url": public_url,
                        "segments_processed": 1,
                        "message": "Single transition processed",
                        "_task_completed_by_upload": True
                    }

            except Exception as e:
                # Clean up any temporary files on error
                for temp_path in list(locals().get('video_segments', [])):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
                raise e
        
        else:
            # Original logic for 1-2 images
            if has_loras:
                # Use LoRA endpoint: wan-2.2/i2v-480p-lora
                endpoint_path = "wavespeed-ai/wan-2.2/i2v-480p-lora"
                logger.info(f"Using LoRA endpoint: {endpoint_path}")
                
                # Map parameters to Wavespeed API format for LoRA endpoint
                wavespeed_params = {
                    "duration": effective_params.get("duration", 5),
                    "high_noise_loras": [],
                    "image": input_images[0] if input_images else "",
                    "last_image": "",
                    "loras": [],
                    "low_noise_loras": [],
                    "negative_prompt": negative_prompts[0] if negative_prompts else "",
                    "prompt": base_prompts[0] if base_prompts else "",
                    "seed": effective_params.get("seed_base", -1)
                }
                
                # Add last_image if we have multiple input images
                if len(input_images) > 1:
                    wavespeed_params["last_image"] = input_images[1]
                    logger.info(f"Using first image: {input_images[0]}")
                    logger.info(f"Using last image: {input_images[1]}")
                else:
                    logger.info(f"Using single image: {input_images[0] if input_images else 'None'}")
                
                # Add LoRAs from additional_loras
                for lora_path, scale in additional_loras.items():
                    wavespeed_params["loras"].append({
                        "path": lora_path,
                        "scale": float(scale)
                    })
                logger.info(f"Added {len(additional_loras)} LoRAs to i2v request")
                
            else:
                # Use non-LoRA endpoint: wan-2.2/i2v-480p
                endpoint_path = "wavespeed-ai/wan-2.2/i2v-480p"
                logger.info(f"Using non-LoRA endpoint: {endpoint_path}")
                
                # Map parameters to Wavespeed API format for non-LoRA endpoint
                wavespeed_params = {
                    "seed": effective_params.get("seed_base", -1),
                    "image": input_images[0] if input_images else "",
                    "prompt": base_prompts[0] if base_prompts else "",
                    "duration": effective_params.get("duration", 5),
                    "negative_prompt": negative_prompts[0] if negative_prompts else "",
                    "model_id": "wavespeed-ai/wan-2.2/i2v-480p"
                }
                
                # Set last_image to second input image if available
                if len(input_images) > 1:
                    wavespeed_params["last_image"] = input_images[1]
                    logger.info(f"Using first image: {input_images[0]}")
                    logger.info(f"Using last image (second input): {input_images[1]}")
                else:
                    wavespeed_params["last_image"] = ""
                    logger.info(f"Using single image: {input_images[0] if input_images else 'None'}")
            
            result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)
            
            # Process external URL with automatic screenshot extraction for videos
            result = await process_external_url_result(client, task_id, result)
        
        logger.info(f"Processed {task_type} task via Wavespeed API")
        return result
        
    elif task_type == "image_inpaint":
        # Wavespeed AI Qwen Image Inpainting with LoRA
        logger.info(f"Processing {task_type} task via Wavespeed API")
        
        # Extract parameters
        image_url = params.get("image_url") or params.get("image", "")
        mask_url = params.get("mask_url", "")
        prompt = params.get("prompt", "")
        
        if not image_url:
            raise Exception("image_url parameter is required for image_inpaint task")
        if not mask_url:
            raise Exception("mask_url parameter is required for image_inpaint task")
        
        logger.info(f"Inpainting image: {image_url}")
        logger.info(f"Using mask: {mask_url}")
        logger.info(f"Prompt: {prompt}")
        
        # Create masked composite image
        composite_url = await create_masked_composite_image(
            client, task_id, image_url, mask_url, filename_prefix="inpaint_composite"
        )
        
        # Generate random seed if not provided
        seed = params.get("seed")
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"Generated random seed: {seed}")
        
        # Determine which model/endpoint to use based on qwen_edit_model parameter
        # Inpaint always uses LoRA endpoints since it needs the inpainting LoRA
        qwen_edit_model = params.get("qwen_edit_model", "qwen-edit")  # default to current behavior
        
        # Map model names to endpoints
        # 2511 and 2509 (edit-plus) use "images" array format
        # Default edit-lora uses "image" string format
        if qwen_edit_model == "qwen-edit-2511":
            endpoint_path = "wavespeed-ai/qwen-image/edit-2511-lora"
            use_images_array = True
        elif qwen_edit_model == "qwen-edit-2509":
            endpoint_path = "wavespeed-ai/qwen-image/edit-plus-lora"
            use_images_array = True
        else:  # "qwen-edit" or default
            endpoint_path = "wavespeed-ai/qwen-image/edit-lora"
            use_images_array = False
        
        logger.info(f"Using qwen_edit_model: {qwen_edit_model} -> endpoint: {endpoint_path}")
        
        # Build the inpainting LoRA list
        inpaint_loras = [
            {
                "path": "https://huggingface.co/ostris/qwen_image_edit_inpainting/resolve/main/qwen_image_edit_inpainting.safetensors",
                "scale": 1.0
            }
        ]
        
        # Add any additional LoRAs from params
        additional_loras = params.get("loras", [])
        if additional_loras:
            for lora in additional_loras:
                if isinstance(lora, dict):
                    # Support both "url"/"strength" and "path"/"scale" formats
                    lora_url = lora.get("url") or lora.get("path", "")
                    lora_strength = lora.get("strength", lora.get("scale", 1.0))
                    
                    if lora_url:
                        inpaint_loras.append({
                            "path": lora_url,
                            "scale": float(lora_strength)
                        })
                        logger.info(f"Added additional inpaint LoRA: {lora_url} with strength {lora_strength}")
            logger.info(f"Added {len(additional_loras)} additional LoRAs to inpaint request")
        
        # Build parameters based on API format
        if use_images_array:
            # 2511/2509 APIs use "images" array format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "images": [composite_url],
                "output_format": params.get("output_format", "jpeg"),
                "prompt": prompt,
                "seed": seed,
                "loras": inpaint_loras
            }
        else:
            # Default edit-lora API uses "image" string format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "output_format": params.get("output_format", "jpeg"),
                "prompt": prompt,
                "seed": seed,
                "image": composite_url,  # Use the composite image with green mask
                "loras": inpaint_loras
            }
        
        # Add size parameter if resolution is provided
        resolution = params.get("resolution", "")
        if resolution:
            normalized_size = normalize_resolution(resolution)
            if normalized_size:
                wavespeed_params["size"] = normalized_size
                logger.info(f"Using resolution/size: {normalized_size}")
        
        logger.info(f"Calling inpainting API with composite image and LoRA")
        
        result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)
        
        # Process external URL with automatic screenshot extraction for videos
        result = await process_external_url_result(client, task_id, result)
        
        logger.info(f"Processed {task_type} task via Wavespeed API")
        return result
        
    elif task_type == "annotated_image_edit":
        # Wavespeed AI Qwen Image Editing with Scene Annotation LoRAs
        logger.info(f"Processing {task_type} task via Wavespeed API")
        
        # Extract parameters
        image_url = params.get("image_url") or params.get("image", "")
        mask_url = params.get("mask_url", "")
        prompt = params.get("prompt", "")
        
        if not image_url:
            raise Exception("image_url parameter is required for annotated_image_edit task")
        if not mask_url:
            raise Exception("mask_url parameter is required for annotated_image_edit task")
        
        logger.info(f"Annotated image editing: {image_url}")
        logger.info(f"Using mask: {mask_url}")
        logger.info(f"Prompt: {prompt}")
        
        # Create masked composite image
        composite_url = await create_masked_composite_image(
            client, task_id, image_url, mask_url, filename_prefix="annotated_edit_composite"
        )
        
        # Generate random seed if not provided
        seed = params.get("seed")
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"Generated random seed: {seed}")
        
        # Determine which model/endpoint to use based on qwen_edit_model parameter
        # Annotated edit always uses LoRA endpoints since it needs the annotation LoRA
        qwen_edit_model = params.get("qwen_edit_model", "qwen-edit")  # default to current behavior
        
        # Map model names to endpoints
        # 2511 and 2509 (edit-plus) use "images" array format
        # Default edit-lora uses "image" string format
        if qwen_edit_model == "qwen-edit-2511":
            endpoint_path = "wavespeed-ai/qwen-image/edit-2511-lora"
            use_images_array = True
        elif qwen_edit_model == "qwen-edit-2509":
            endpoint_path = "wavespeed-ai/qwen-image/edit-plus-lora"
            use_images_array = True
        else:  # "qwen-edit" or default
            endpoint_path = "wavespeed-ai/qwen-image/edit-lora"
            use_images_array = False
        
        logger.info(f"Using qwen_edit_model: {qwen_edit_model} -> endpoint: {endpoint_path}")
        
        # Build the annotation LoRA list
        annotation_loras = [
            {
                "path": "https://huggingface.co/peteromallet/random_junk/resolve/main/in_scene_pure_squares_flipped_450_lr_000006700.safetensors",
                "scale": 1.0
            }
        ]
        
        # Add any additional LoRAs from params
        additional_loras = params.get("loras", [])
        if additional_loras:
            for lora in additional_loras:
                if isinstance(lora, dict):
                    # Support both "url"/"strength" and "path"/"scale" formats
                    lora_url = lora.get("url") or lora.get("path", "")
                    lora_strength = lora.get("strength", lora.get("scale", 1.0))
                    
                    if lora_url:
                        annotation_loras.append({
                            "path": lora_url,
                            "scale": float(lora_strength)
                        })
                        logger.info(f"Added additional annotated edit LoRA: {lora_url} with strength {lora_strength}")
            logger.info(f"Added {len(additional_loras)} additional LoRAs to annotated edit request")
        
        # Build parameters based on API format
        if use_images_array:
            # 2511/2509 APIs use "images" array format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "images": [composite_url],
                "output_format": params.get("output_format", "jpeg"),
                "prompt": prompt,
                "seed": seed,
                "loras": annotation_loras
            }
        else:
            # Default edit-lora API uses "image" string format
            wavespeed_params = {
                "enable_base64_output": params.get("enable_base64_output", False),
                "enable_sync_mode": params.get("enable_sync_mode", False),
                "output_format": params.get("output_format", "jpeg"),
                "prompt": prompt,
                "seed": seed,
                "image": composite_url,  # Use the composite image with green mask
                "loras": annotation_loras
            }
        
        # Add size parameter if resolution is provided
        resolution = params.get("resolution", "")
        if resolution:
            normalized_size = normalize_resolution(resolution)
            if normalized_size:
                wavespeed_params["size"] = normalized_size
                logger.info(f"Using resolution/size: {normalized_size}")
        
        logger.info(f"Calling annotated edit API with composite image and scene annotation LoRAs")
        
        result = await call_wavespeed_api(endpoint_path, wavespeed_params, client)
        
        # Process external URL with automatic screenshot extraction for videos
        result = await process_external_url_result(client, task_id, result)
        
        logger.info(f"Processed {task_type} task via Wavespeed API")
        return result
        
    elif task_type == "image-upscale":
        # fal.ai Image Upscale
        logger.info(f"Processing {task_type} task via fal.ai API")
        
        # Extract parameters - support both "image_url" and "image" parameter names
        image_url = params.get("image_url") or params.get("image", "")
        # Support both "upscale_factor" and "scale_factor" parameter names
        upscale_factor = params.get("upscale_factor") or params.get("scale_factor", 2)
        
        if not image_url:
            logger.error(f"Missing image parameter. Available params: {list(params.keys())}")
            raise Exception("image_url or image parameter is required for image-upscale task")
        
        logger.info(f"Upscaling image: {image_url} with factor: {upscale_factor}")
        
        # Call fal.ai API (run in thread pool since it's a blocking call)
        try:
            def on_queue_update(update):
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        logger.info(f"fal.ai: {log['message']}")
            
            # Run the blocking fal_client.subscribe in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: fal_client.subscribe(
                    "fal-ai/seedvr/upscale/image",
                    arguments={
                        "image_url": image_url,
                        "upscale_factor": upscale_factor
                    },
                    with_logs=True,
                    on_queue_update=on_queue_update,
                )
            )
            
            logger.info(f"fal.ai upscale result: {result}")
            
            # Transform fal.ai result format to expected format
            # fal.ai returns: {'image': {'url': '...', 'width': ..., 'height': ...}, 'seed': ...}
            # We need: {'output_url': '...'}
            if isinstance(result, dict) and 'image' in result:
                image_data = result['image']
                if isinstance(image_data, dict) and 'url' in image_data:
                    transformed_result = {
                        'output_url': image_data['url'],
                        'width': image_data.get('width'),
                        'height': image_data.get('height'),
                        'seed': result.get('seed'),
                        'content_type': image_data.get('content_type')
                    }
                    logger.info(f"Transformed fal.ai result to standard format: {transformed_result['output_url']}")
                    
                    # Process external URL with automatic download and upload to Supabase
                    transformed_result = await process_external_url_result(client, task_id, transformed_result)
                    
                    logger.info(f"Processed {task_type} task via fal.ai API")
                    return transformed_result
            
            # Fallback if format is unexpected
            logger.warning(f"Unexpected fal.ai result format, returning as-is: {result}")
            return result
            
        except Exception as e:
            logger.error(f"fal.ai API call failed: {e}")
            raise Exception(f"fal.ai upscale failed: {str(e)}")
    
    elif task_type in ("qwen_image", "qwen_image_2512", "z_image_turbo"):
        # fal.ai image generation models with optional LoRA support
        # Maps task_type to fal.ai endpoint base path
        FALAI_IMAGE_MODELS = {
            "qwen_image": "fal-ai/qwen-image",
            "qwen_image_2512": "fal-ai/qwen-image-2512",
            "z_image_turbo": "fal-ai/z-image/turbo",
        }
        
        endpoint = FALAI_IMAGE_MODELS[task_type]
        logger.info(f"Processing {task_type} task via fal.ai API")
        logger.info(f"Using endpoint: {endpoint}")
        
        # Build LoRA list from params (same format as qwen_image_style)
        # Supports: loras=[{"path": ..., "scale": ...}] or additional_loras={path: scale}
        loras = build_fal_lora_list(params)
        if loras:
            logger.info(f"Including {len(loras)} LoRAs in request")
        
        # Build arguments - following similar naming to qwen_image_style
        fal_args = {
            "prompt": params.get("prompt", ""),
            "seed": params.get("seed", -1),
        }
        
        # Add size/resolution if provided
        # fal.ai accepts either preset strings or {"width": x, "height": y}
        resolution = params.get("resolution", "")
        if resolution:
            # Parse resolution string like "512x512" or "1920*1080"
            parts = resolution.replace("*", "x").split("x")
            if len(parts) == 2:
                try:
                    width = int(parts[0])
                    height = int(parts[1])
                    fal_args["image_size"] = {"width": width, "height": height}
                    logger.info(f"Using resolution: {width}x{height}")
                except ValueError:
                    logger.warning(f"Could not parse resolution '{resolution}', using default")
        
        # Add negative prompt if provided
        negative_prompt = params.get("negative_prompt", "")
        if negative_prompt:
            fal_args["negative_prompt"] = negative_prompt
        
        # Add LoRAs to args if provided
        if loras:
            fal_args["loras"] = loras
        
        try:
            result = await call_fal_api(endpoint, fal_args, client)
            result = await process_external_url_result(client, task_id, result)
            logger.info(f"Processed {task_type} task via fal.ai API")
            return result
        except Exception as e:
            logger.error(f"fal.ai API call failed: {e}")
            raise Exception(f"fal.ai {task_type} failed: {str(e)}")
    
    elif task_type == "z_image_turbo_i2i":
        # fal.ai Z-Image Turbo image-to-image
        # Automatically scales to ~1 megapixel while preserving aspect ratio
        # Uses /lora endpoint when LoRAs are provided
        logger.info(f"Processing {task_type} task via fal.ai API")
        
        # Build LoRA list from params
        loras = build_fal_lora_list(params)
        
        # Select endpoint based on whether LoRAs are provided
        if loras:
            endpoint = "fal-ai/z-image/turbo/image-to-image/lora"
            logger.info(f"Using LoRA endpoint with {len(loras)} LoRAs")
        else:
            endpoint = "fal-ai/z-image/turbo/image-to-image"
            logger.info(f"Using standard endpoint (no LoRAs)")
        
        # Get required image_url
        image_url = params.get("image_url") or params.get("image")
        if not image_url:
            raise Exception("image_url or image parameter is required for z_image_turbo_i2i task")
        
        # Determine output image_size
        # If explicit image_size is provided (not "auto"), use it directly
        # Otherwise, fetch input dimensions and scale to target megapixels
        explicit_image_size = params.get("image_size")
        if explicit_image_size and explicit_image_size != "auto":
            # User provided explicit size - use as-is
            image_size = explicit_image_size
            logger.info(f"Using explicit image_size: {image_size}")
        else:
            # Auto-scale to target megapixels (default ~1MP = 1024x1024 equivalent)
            target_mp = params.get("target_megapixels", 1.0)
            dimensions = await get_image_dimensions(client, image_url)
            
            if dimensions:
                width, height = dimensions
                image_size = scale_dimensions_to_megapixels(width, height, target_megapixels=target_mp)
                logger.info(f"Auto-scaled to {image_size['width']}x{image_size['height']} (~{target_mp}MP)")
            else:
                # Fallback to "auto" if we couldn't get dimensions
                image_size = "auto"
                logger.warning(f"Could not get input dimensions, falling back to image_size='auto'")
        
        # Build arguments
        fal_args = {
            "prompt": params.get("prompt", ""),
            "image_url": image_url,
            "strength": params.get("strength", 0.6),
            "image_size": image_size,
            "num_inference_steps": params.get("num_inference_steps", 8),
            "num_images": params.get("num_images", 1),
            "enable_safety_checker": params.get("enable_safety_checker", True),
            "output_format": params.get("output_format", "png"),
        }
        
        # Add acceleration (default "high" for non-LoRA, "none" for LoRA)
        if loras:
            fal_args["acceleration"] = params.get("acceleration", "none")
            fal_args["loras"] = loras
        else:
            fal_args["acceleration"] = params.get("acceleration", "high")
        
        # Add optional seed if provided
        seed = params.get("seed")
        if seed is not None:
            fal_args["seed"] = seed
        
        # Add negative prompt if provided
        negative_prompt = params.get("negative_prompt", "")
        if negative_prompt:
            fal_args["negative_prompt"] = negative_prompt
        
        logger.info(f"Z-Image Turbo i2i: image={image_url}, strength={fal_args['strength']}, steps={fal_args['num_inference_steps']}, size={image_size}")
        
        try:
            result = await call_fal_api(endpoint, fal_args, client)
            result = await process_external_url_result(client, task_id, result)
            logger.info(f"Processed {task_type} task via fal.ai API")
            return result
        except Exception as e:
            logger.error(f"fal.ai API call failed: {e}")
            raise Exception(f"fal.ai {task_type} failed: {str(e)}")
        
    else:
        # Unsupported task type
        raise Exception(f"Unsupported task type: {task_type}. Supported types: 'qwen_image_edit', 'qwen_image_style', 'wan_2_2_t2i', 'wan_2_2_i2v', 'animate_character', 'image-upscale', 'image_inpaint', 'annotated_image_edit', 'qwen_image', 'qwen_image_2512', 'z_image_turbo', 'z_image_turbo_i2i'.")


async def worker_loop(index: int, worker_id: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> None:
    # Deprecated continuous claim-per-worker loop; kept for reference
    while True:
        await asyncio.sleep(1)


def validate_api_environment():
    """Validate all required environment variables for API orchestrator."""
    logger.info("🔍 [STARTUP] Validating API orchestrator environment...")
    
    # Required environment variables
    required_vars = {
        'SUPABASE_URL': 'Supabase database URL for task management',
        'SUPABASE_SERVICE_ROLE_KEY': 'Supabase service role key for database access',
        'WAVESPEED_API_KEY': 'Wavespeed API key for processing tasks',
        'FAL_KEY': 'fal.ai API key for image upscaling tasks',
    }
    
    # Optional but important environment variables
    important_vars = {
        'API_WORKER_ID': 'Unique identifier for this API worker instance',
        'CONCURRENCY': 'Number of concurrent tasks to process',
        'RUN_TYPE': 'Type of tasks to process (cloud/local)',
        'PARENT_POLL_SEC': 'Polling interval for task checking',
        'LOG_LEVEL': 'Logging level (DEBUG/INFO/WARNING/ERROR)',
    }
    
    # Check required variables
    missing_required = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_required.append(f"  ❌ {var}: {description}")
            logger.error(f"[STARTUP] Missing required environment variable: {var}")
        else:
            # Show partial value for security
            if 'KEY' in var or 'SECRET' in var:
                display_value = f"{value[:10]}..." if len(value) > 10 else "***"
            else:
                display_value = value
            logger.info(f"[STARTUP]   ✅ {var}: {display_value}")
    
    # Check important variables
    for var, description in important_vars.items():
        value = os.getenv(var)
        if not value:
            logger.warning(f"[STARTUP]   ⚠️  {var}: {description} (using default)")
        else:
            logger.info(f"[STARTUP]   ✅ {var}: {value}")
    
    # Report results
    if missing_required:
        logger.error("[STARTUP] ❌ CRITICAL: Missing required environment variables:")
        for msg in missing_required:
            logger.error(f"[STARTUP] {msg}")
        logger.error("[STARTUP] 🛑 Cannot start API orchestrator without required configuration!")
        return False
    
    logger.info("[STARTUP] ✅ Environment validation completed successfully")
    return True

async def main_async() -> None:
    logger.info(f"[STARTUP] API Orchestrator starting...")
    
    # Validate environment before proceeding
    if not validate_api_environment():
        logger.error("[STARTUP] 🛑 Exiting due to missing required configuration!")
        sys.exit(1)
    
    # Initialize database client for centralized logging
    try:
        db_client = DatabaseClient()
        
        # Re-initialize logging with database client if DB logging is enabled
        from .logging_config import setup_logging as reinit_logging
        reinit_logging(db_client=db_client, source_type="orchestrator_api")
        
        logger.info("[STARTUP] ✅ Centralized database logging initialized")
    except Exception as e:
        logger.warning(f"[STARTUP] ⚠️  Could not initialize database logging: {e}")
        logger.warning("[STARTUP] Continuing with console logging only...")
    
    # Log additional startup configuration
    logger.info(f"[STARTUP] CONCURRENCY: {CONCURRENCY}")
    logger.info(f"[STARTUP] RUN_TYPE: {RUN_TYPE}")
    logger.info(f"[STARTUP] PARENT_POLL_SEC: {PARENT_POLL_SEC}")

    worker_id = os.getenv("API_WORKER_ID", "api-worker-main")
    limits = httpx.Limits(max_connections=max(64, CONCURRENCY * 4), max_keepalive_connections=max(32, CONCURRENCY * 2))
    active_tasks: set[asyncio.Task] = set()

    async def spawn_task(task_payload: Dict[str, Any], client: httpx.AsyncClient):
        task_id = task_payload.get("task_id") or task_payload.get("id")
        
        # Set task context for logging
        from .logging_config import set_current_task
        set_current_task(str(task_id))
        
        try:
            logger.info(f"Starting task {task_id}")
            result = await process_api_task(task_payload, client)
            
            # Check if task completion was already handled by the upload process
            if result.get("_task_completed_by_upload"):
                logger.info(f"Task {task_id} completion already handled by upload process, skipping additional mark_complete call")
            else:
                # Only call mark_complete if the upload process didn't handle it
                success = await mark_complete(client, task_id, result)
                if not success:
                    # If we can't mark the task complete, treat it as a failure to prevent stuck tasks
                    logger.error(f"Task {task_id} processed successfully but failed to save to database")
                    await mark_failed(client, task_id, "Task processed successfully but failed to save completion status to database")
                else:
                    logger.info(f"Task {task_id} completed successfully")
        except Exception as exc:
            logger.error(f"Task {task_id} failed with exception: {exc}")
            logger.error(f"Exception type: {type(exc).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            failed_success = await mark_failed(client, task_id, str(exc))
            if not failed_success:
                logger.error(f"DOUBLE FAILURE: Task {task_id} failed AND could not mark as failed in database!")
        finally:
            # Clear task context
            set_current_task(None)

    async with httpx.AsyncClient(limits=limits, timeout=20.0) as client:
        logger.info(f"[WORKER_LOOP] Starting worker loop for {worker_id} with RUN_TYPE: {RUN_TYPE}, CONCURRENCY: {CONCURRENCY}, POLL_SEC: {PARENT_POLL_SEC}")
        
        loop_count = 0
        while True:
            loop_count += 1
            
            # prune finished subtasks
            done = {t for t in active_tasks if t.done()}
            if done:
                logger.debug(f"[WORKER_LOOP] Pruned {len(done)} completed tasks")
                active_tasks -= done

            capacity = max(0, CONCURRENCY - len(active_tasks))
            
            # Log every 10 loops or when there's activity
            if loop_count % 10 == 1 or capacity > 0 or len(active_tasks) > 0:
                logger.info(f"[WORKER_LOOP] Loop #{loop_count}: Active tasks: {len(active_tasks)}, Capacity: {capacity}")
            
            # Log database logging stats periodically (every 100 loops)
            if loop_count % 100 == 0:
                db_stats = get_db_logging_stats()
                if db_stats:
                    logger.info(f"[WORKER_LOOP] 📊 Database logging stats: {db_stats}")
            
            if capacity > 0:
                logger.debug(f"[WORKER_LOOP] Checking for available tasks...")
                count_info = await count_tasks(client, RUN_TYPE)
                available_tasks = int(count_info.get("queued_plus_active") or 0)
                to_claim = min(capacity, available_tasks)
                
                logger.debug(f"[WORKER_LOOP] Task count result: {count_info}")
                logger.info(f"[WORKER_LOOP] Available tasks: {available_tasks}, Will attempt to claim: {to_claim}")
                
                if to_claim > 0:
                    logger.info(f"[WORKER_LOOP] Claiming {to_claim} tasks (capacity: {capacity}, available: {available_tasks})")
                    
                    claimed_count = 0
                    for i in range(to_claim):
                        logger.debug(f"[WORKER_LOOP] Attempting to claim task {i+1}/{to_claim}")
                        claimed = await claim_next_task(client, worker_id, RUN_TYPE)
                        if not claimed:
                            logger.warning(f"[WORKER_LOOP] Failed to claim task {i+1}/{to_claim} - no tasks available despite count showing {available_tasks}")
                            break
                        claimed_count += 1
                        logger.info(f"[WORKER_LOOP] Successfully claimed task {i+1}/{to_claim}: {claimed.get('task_id')}")
                        t = asyncio.create_task(spawn_task(claimed, client))
                        active_tasks.add(t)
                        
                    if claimed_count > 0:
                        logger.info(f"[WORKER_LOOP] Spawned {claimed_count} tasks")
                    elif available_tasks > 0:
                        logger.error(f"[WORKER_LOOP] CRITICAL: Could not claim any tasks despite {available_tasks} being available - check task filters and dependencies")
                elif available_tasks == 0:
                    logger.debug(f"[WORKER_LOOP] No tasks available to claim")
                else:
                    logger.debug(f"[WORKER_LOOP] No capacity to claim tasks")

            await asyncio.sleep(PARENT_POLL_SEC)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


