import base64
import os
import tempfile
import time
from pathlib import Path
from typing import List

from diffusionkit.mlx import DiffusionPipeline, FluxPipeline

from .schema import ImageGenerationRequest, ImageObject, ResponseFormat


class MLXImageGenerator:
    def __init__(self, model_version: str = "argmaxinc/mlx-FLUX.1-schnell"):
        self.model_version = model_version
        self.height = {
            "argmaxinc/mlx-stable-diffusion-3-medium": 512,
            "argmaxinc/mlx-stable-diffusion-3.5-large": 1024,
            "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": 1024,
            "argmaxinc/mlx-FLUX.1-schnell": 512,
            "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 512,
            "argmaxinc/mlx-FLUX.1-dev": 512,
        }[model_version]

        self.width = {
            "argmaxinc/mlx-stable-diffusion-3-medium": 512,
            "argmaxinc/mlx-stable-diffusion-3.5-large": 1024,
            "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": 1024,
            "argmaxinc/mlx-FLUX.1-schnell": 512,
            "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 512,
            "argmaxinc/mlx-FLUX.1-dev": 512,
        }[model_version]

        self.shift = {
            "argmaxinc/mlx-stable-diffusion-3-medium": 3.0,
            "argmaxinc/mlx-stable-diffusion-3.5-large": 3.0,
            "argmaxinc/mlx-stable-diffusion-3.5-large-4bit-quantized": 3.0,
            "argmaxinc/mlx-FLUX.1-schnell": 1.0,
            "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized": 1.0,
            "argmaxinc/mlx-FLUX.1-dev": 1.0,
        }[model_version]

        self.pipeline_class = (
            FluxPipeline if "FLUX" in model_version else DiffusionPipeline
        )

    def generate(
        self,
        request: ImageGenerationRequest,
        output_path: str,
        low_memory_mode: bool = True,
        **extra_params,
    ):
        """Generate image using MLX DiffusionKit with request parameters"""
        # Extract size parameters from request
        width, height = self._parse_size(request.size)

        # Get extra parameters from request
        request_extra_params = request.get_extra_params()

        # Merge all extra parameters, with passed extra_params taking precedence
        all_extra_params = {**request_extra_params, **extra_params}

        # Initialize model with appropriate settings
        sd = self.pipeline_class(
            w16=True,
            shift=all_extra_params.pop("shift", self.shift),
            use_t5=all_extra_params.pop(
                "use_t5", False
            ),  # Can be overridden by extra_params
            model_version=self.model_version,
            low_memory_mode=low_memory_mode,
            a16=True,
        )

        # Default parameters for generate_image
        gen_params = {
            "text": request.prompt,  # prompt parameter is renamed to text
            "num_steps": all_extra_params.pop("num_steps", 50),
            "cfg_weight": all_extra_params.pop(
                "cfg_weight", 0.0 if "FLUX" in self.model_version else 5.0
            ),
            "negative_text": all_extra_params.pop("negative_prompt", ""),
            "latent_size": all_extra_params.pop(
                "latent_size", (height // 8, width // 8)
            ),
            "seed": all_extra_params.pop("seed", None),
            "verbose": all_extra_params.pop("verbose", False),
            "image_path": all_extra_params.pop("image_path", None),
            "denoise": all_extra_params.pop("denoise", 1.0),
        }

        # Add any remaining extra parameters
        gen_params.update(all_extra_params)

        # Generate image
        image, _ = sd.generate_image(**gen_params)

        # Save generated image
        image.save(output_path)
        return image

    def _parse_size(self, size_str: str) -> tuple[int, int]:
        """Parse size string into width and height"""
        try:
            width, height = map(int, size_str.split("x"))
            return width, height
        except (ValueError, AttributeError):
            # Default to model's default size if parsing fails
            return self.width, self.height


class ImagesService:
    def __init__(self):
        # Use system temporary directory
        self.output_dir = Path(tempfile.gettempdir()) / "mlx_omni_server" / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, uid: str) -> str:
        """Generate unique output path for image"""
        return str(self.output_dir / f"{uid}.png")

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _cleanup_image(self, image_path: str):
        """Clean up temporary image file"""
        try:
            os.unlink(image_path)
        except Exception as e:
            print(f"Error cleaning up image {image_path}: {str(e)}")

    def generate_images(
        self,
        request: ImageGenerationRequest,
    ) -> List[ImageObject]:
        """Generate images based on the request"""
        generated_images = []
        generator = MLXImageGenerator(model_version=request.model)

        for i in range(request.n):
            # Generate unique identifier for this image
            uid = f"{int(time.time())}_{i}"
            output_path = self._get_output_path(uid)

            try:
                # Generate the image
                generator.generate(
                    request=request, output_path=output_path, low_memory_mode=True
                )

                # Create response object based on format
                image_object = ImageObject(revised_prompt=request.prompt)

                # Response All Format
                image_object.b64_json = self._image_to_base64(output_path)
                image_object.url = f"file://{output_path}"

                generated_images.append(image_object)

            except Exception as e:
                raise Exception(f"Error generating image: {str(e)}")
            finally:
                # Clean up temporary file if using base64 format
                if request.response_format == ResponseFormat.B64_JSON:
                    self._cleanup_image(output_path)

        return generated_images
