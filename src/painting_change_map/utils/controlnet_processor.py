from typing import Any, List, Union

import torch
from controlnet_aux import CannyDetector, LineartAnimeDetector, LineartDetector
from PIL import Image
from transformers import pipeline


class ControlnetProcessor:
    """
    Loads and runs one of several image processors depending on `mode`.

    Parameters
    ----------
    mode : str
        One of: "canny", "depth-anything-v2", "lineart", "lineart-anime", "grayscale".
    offload_after : bool
        If True, move the model back to CPU (and clear the CUDA cache) after
        each :meth:`__call__` invocation to free VRAM between runs.
    device: Device to load the model if applicable.
    """

    def __init__(self, mode: str, offload_after: bool = True):
        self.mode = mode.lower()
        self.offload_after = offload_after

        self.processor: Any = None
        self._init_processor()

    def _init_processor(self):
        """Initializes the processor based on the selected mode."""
        if self.mode == "canny":
            self.processor = CannyDetector()

        elif self.mode == "depth-anything-v2":
            # We initialize on CPU first, device management is handled dynamically
            self.processor = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")

        elif self.mode == "lineart":
            self.processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

        elif self.mode == "lineart-anime":
            self.processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        elif self.mode == "grayscale":
            pass
        else:
            raise ValueError(
                f"Unknown mode: '{self.mode}'. Supported modes: "
                "'canny', 'depth-anything-v2', 'lineart', 'lineart-anime', 'grayscale'."
            )

    def _move_to_device(self, target_device: torch.device):
        """Safely moves the underlying models to the specified device."""
        if self.mode == "canny" or self.mode == "grayscale":
            # Canny relies on OpenCV and doesn't utilize PyTorch device management
            pass
        elif self.mode == "depth-anything-v2":
            self.processor.model.to(device=target_device)  # , dtype=torch.float32)
            self.processor.device = target_device
        else:
            # For controlnet_aux neural network detectors
            self.processor = self.processor.to(device=target_device)  # , dtype=torch.float32)

    @torch.no_grad()
    def __call__(
        self, images: List[Image.Image], device: Union[torch.device, str] = torch.device("cpu")
    ) -> List[Image.Image]:
        """Processes a list of PIL images."""
        if not images:
            return []

        if isinstance(device, str):
            device = torch.device(device)

        # 1. Load model to GPU if available
        if device.type != "cpu":
            self._move_to_device(device)

        # 2. Process the images
        processed_images = []

        # Pytorch lightning was forcing the model to BFloat16, this disables it
        autocast_device = device.type if device.type in ["cuda", "cpu"] else "cuda"

        with torch.autocast(device_type=autocast_device, enabled=False):
            # The HuggingFace pipeline processes lists efficiently natively
            if self.mode == "depth-anything-v2":
                results = self.processor(images)
                processed_images = [res["depth"].convert("RGB") for res in results]
            elif self.mode == "grayscale":
                processed_images = [image.convert("L").convert("RGB") for image in images]
            else:
                # Process frames one by one for controlnet_aux detectors
                for frame in images:
                    og_size = frame.size
                    if self.mode == "canny":
                        processed = (
                            self.processor(frame, low_threshold=10, high_threshold=60)
                            .resize(size=og_size)
                            .convert("RGB")
                        )
                    else:
                        # Lineart and Lineart Anime
                        processed = self.processor(frame).resize(size=og_size).convert("RGB")
                    processed_images.append(processed)

        # 3. Offload model back to CPU and clear VRAM cache if requested
        if self.offload_after and device.type != "cpu":
            self._move_to_device(torch.device("cpu"))
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        return processed_images
