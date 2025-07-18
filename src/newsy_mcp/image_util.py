#!/usr/bin/env python3

"""
Image Generation and Management System

Features:
1. Image Generation
   - Integration with OpenAI's GPT Image model (gpt-image-1)
   - Support for multiple image sizes (square, landscape, portrait)
   - Handling of base64-encoded image data
   - Concurrent image generation using threading
   - Configurable image count per request
   - Model-specific quality parameter handling

2. File Management
   - Automatic directory creation and management
   - Consistent PNG file format enforcement
   - Configurable output directory support
   - Thread-safe file operations

3. Error Handling
   - Graceful error handling for API and file operations
   - Clear error messages for debugging
   - Thread-safe error reporting

4. Debug Support
   - Debug mode for detailed operation tracking
   - Thread operation monitoring
   - File operation logging
"""

import openai
import threading
import os
import base64
import uuid
from enum import Enum
from typing import List
from utils import dprint


MODEL = "gpt-image-1"
openai.api_key = os.getenv("RODION_OPENAI_API_KEY")


class ImageSize(Enum):
    """Supported image size configurations"""

    AUTO = "auto"
    SQUARE = "1024x1024"
    LANDSCAPE = "1536x1024"
    PORTRAIT = "1024x1536"


class ImageQuality(Enum):
    """Supported quality settings per model"""

    AUTO = "auto"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImageUtil:
    def __init__(self) -> None:
        """Initialize ImageUtil with thread lock and default directory"""
        self.lock = threading.Lock()
        self.images_dir = ".images"

        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            dprint(f"Created images directory: {self.images_dir}")

    def send_prompt(
        self,
        prompt: str,
        count: int = 1,
        size: ImageSize = ImageSize.PORTRAIT,
        quality: ImageQuality = ImageQuality.AUTO,
        output_dir: str | None = None,
    ) -> List[str]:
        """
        Generate images from a text prompt using GPT Image model.

        Args:
            prompt: Text description of the desired image
            count: Number of images to generate
            size: Target image dimensions
            quality: Image quality setting
            output_dir: Optional custom save location

        Returns:
            List of paths to generated image files
        """
        save_dir = output_dir if output_dir else self.images_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            dprint(f"Created output directory: {save_dir}")

        image_paths: List[str] = []

        def send_prompt_helper() -> None:
            try:
                dprint(f"Generating image with quality: {quality.value}")
                response = openai.images.generate(
                    model=MODEL,
                    prompt=prompt,
                    size=size.value,
                    quality=quality.value,
                    n=1,
                )

                # Try to get the image data
                image_data = response.data[0]

                # Check if we have base64 data (gpt-image-1)
                if hasattr(image_data, "b64_json") and image_data.b64_json:
                    # Save base64 data directly
                    image_path = self.save_base64_image(image_data.b64_json, save_dir)
                    with self.lock:
                        image_paths.append(image_path)
                    dprint(f"Generated and saved base64 image: {image_path}")
                # Check if we have a URL (DALL-E 3)
                elif hasattr(image_data, "url") and image_data.url:
                    image_path = self.save_image_from_url(image_data.url, save_dir)
                    with self.lock:
                        image_paths.append(image_path)
                    dprint(f"Generated and saved image from URL: {image_path}")
                else:
                    # Log the available attributes for debugging
                    available_attrs = dir(image_data)
                    dprint(
                        f"Available attributes in response.data[0]: {available_attrs}"
                    )
                    raise Exception(f"No image data found in response: {image_data}")

            except Exception as e:
                print(f"Error generating image: {str(e)}")

        threads = []
        for i in range(count):
            dprint(f"Starting image generation thread {i + 1}/{count}")
            thread = threading.Thread(target=send_prompt_helper)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        return image_paths

    def save_base64_image(self, b64_data: str, output_dir: str) -> str:
        """
        Save a base64-encoded image to a file.

        Args:
            b64_data: Base64-encoded image data
            output_dir: Directory to save the image

        Returns:
            Path to saved image file
        """
        try:
            # Create a unique filename
            filename = os.path.join(output_dir, f"image_{uuid.uuid4()}.png")

            # Decode and save the image
            with open(filename, "wb") as file:
                file.write(base64.b64decode(b64_data))

            dprint(f"Saved base64 image to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving base64 image: {str(e)}")
            raise

    def save_image_from_url(self, url: str, output_dir: str) -> str:
        """
        Download and save an image from a URL.

        Args:
            url: Source URL of the image
            output_dir: Directory to save the image

        Returns:
            Path to saved image file
        """
        try:
            import requests

            response = requests.get(url)
            filename = os.path.join(output_dir, os.path.basename(url))

            if not filename.lower().endswith(".png"):
                filename += ".png"

            with open(filename, "wb") as file:
                file.write(response.content)
            dprint(f"Saved image from URL to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving image from URL: {str(e)}")
            raise
