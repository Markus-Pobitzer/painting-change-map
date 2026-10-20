"""Utiltiy functions for images."""

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps


def pil_resize(
    image: Image.Image,
    target_size: Tuple[int, int],
    pad_input: bool = False,
    padding_color: Union[str, int, Tuple[int, ...]] = "white",
) -> Image.Image:
    """Resizing it to the target size.

    Args:
        image: Input image to be processed.
        target_size: Target size (width, height).
        pad_input: If set resizes the image while keeping the aspect ratio and pads the unfilled part.
        padding_color: The color for the padded pixels.

    Returns:
        The resized image
    """
    if pad_input:
        # Resize image, keep aspect ratio
        image = ImageOps.contain(image, size=target_size)
        # Pad while keeping image in center
        image = ImageOps.pad(image, size=target_size, color=padding_color)
    else:
        image = image.resize(target_size)
    return image


def undo_pil_resize(
    image: Image.Image,
    target_size: Tuple[int, int],
) -> Image.Image:
    """Undo the resizing and padding of the input image to the a new image with size target_size.

    Args:
        image: Input image to be processed.
        target_size: Target size (width, height).

    Returns:
        The resized image
    """
    tmp_img = Image.new(mode="RGB", size=target_size)
    # Get the resized image size
    tmp_img = ImageOps.contain(tmp_img, size=image.size)

    # Undo padding by center cropping
    width, height = image.size
    tmp_width, tmp_height = tmp_img.size

    left = int(round((width - tmp_width) / 2.0))
    top = int(round((height - tmp_height) / 2.0))
    right = left + tmp_width
    bottom = top + tmp_height
    cropped = image.crop((left, top, right, bottom))

    # Undo resizing
    ret = cropped.resize(target_size)
    return ret


def resize_and_center_crop_pil(
    image: Image.Image, target_width: int, target_height: int, resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio, then center crop to target dimensions.

    Args:
        image (Image.Image): Input PIL image.
        target_width (int): Desired width after cropping.
        target_height (int): Desired height after cropping.
        resample (int): Resampling filter to use for resizing. Defaults to LANCZOS.

    Returns:
        Image.Image: The resized and center-cropped image.
    """
    original_width, original_height = image.size

    scale = max(target_width / original_width, target_height / original_height)
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = image.resize((new_width, new_height), resample=resample)

    # Calculate coordinates for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def pil_transform(image: Image.Image, resolution: int):
    """Resize and center crop.

    Resizes smaller side to resolution while keeping aspect ratio.
    Center crop to (resolution, resolution).

    Args:
        image: PIL Image.
        resolution: The final resolution.

    Returns:
        The resized and center cropped PIL Image.
    """
    width, height = image.size
    if width < height:
        new_width = resolution
        new_height = int(resolution * height / width)
    else:
        new_height = resolution
        new_width = int(resolution * width / height)

    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    # Step 2: Center crop to (resolution, resolution)
    left = (new_width - resolution) // 2
    top = (new_height - resolution) // 2
    right = left + resolution
    bottom = top + resolution

    image = image.crop((left, top, right, bottom))
    return image


def load_images_and_filenames_from_folder(img_folder: str) -> Tuple[List[Image.Image], List[str]]:
    images: List[Image.Image] = []
    filenames = []
    for filename in os.listdir(img_folder):
        file_path = os.path.join(img_folder, filename)
        if os.path.isfile(file_path):
            try:
                img = Image.open(file_path)
                images.append(img)
                filenames.append(filename)
            except IOError:
                pass
    return images, filenames


def tensor_to_pil_list(video_tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert a [C, T, H, W] or [T, H, W, C] uint8 tensor (values [0, 255])
    into a list of PIL images.

    Accepts either layout; [C, T, H, W] is transposed automatically.
    """
    if video_tensor.shape[0] in (1, 3, 4) and video_tensor.ndim == 4:
        # Assume [C, T, H, W] -> [T, H, W, C]
        video_tensor = video_tensor.permute(1, 2, 3, 0)
    frames_numpy = video_tensor.detach().cpu().float().numpy().astype(np.uint8)
    return [Image.fromarray(frame, mode="RGB") for frame in frames_numpy]
