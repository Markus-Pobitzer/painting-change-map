from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ChangeMapMode = Literal["interpolate", "pack_24bit", "history", "interpolate_lineart", "local_interpolate_lineart"]


def prepare_change_video_tensor(
    change_video: Union[List[torch.Tensor], List[Image.Image], torch.Tensor],
) -> torch.Tensor:
    """
    Convert a list of raw video frame tensors, a list of PIL images, or a single
    batched video tensor into a channel-first tensor normalized to [0, 1].

    If a list of tensors is provided:
        Each input tensor is expected to have shape ``[1, H, W, C]`` (a single frame
        with a leading batch dimension). The function removes that leading dimension
        and stacks all frames into a batch.

    If a list of PIL images is provided:
        Each image is converted to a uint8 NumPy array of shape ``[H, W, C]`` and
        then stacked into a batch. Grayscale (mode ``"L"``) images produce a single
        channel (C=1). All images must share the same spatial dimensions and mode.

    If a single tensor is provided:
        It is expected to have shape ``[T, H, W, C]`` where T is the number of frames.

    Finally, the function normalizes pixel values from ``[0, 255]`` to ``[0, 1]``
    and reorders the axes to the PyTorch-standard ``[B, C, H, W]`` layout.

    Args:
        change_video: One of:
            - A list of ``N`` tensors, each of shape ``[1, H, W, C]``.
            - A list of ``N`` PIL ``Image`` objects, all with the same size and mode.
            - A single tensor of shape ``[N, H, W, C]``.
            All tensor/array values should lie in ``[0, 255]``.

    Returns:
        A single tensor of shape ``[N, C, H, W]`` and dtype ``torch.bfloat16``
        with values in ``[0.0, 1.0]``.

    Raises:
        ValueError: If a list is empty, a single tensor lacks 4 dimensions, or PIL
            images have inconsistent sizes or modes.
        RuntimeError: If tensors in the list have inconsistent spatial dimensions.
        TypeError: If the input or list elements are of an unsupported type.
    """
    if isinstance(change_video, torch.Tensor):
        if change_video.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [T, H, W, C], but got {change_video.ndim} dimensions.")
        frames = change_video  # Already [T, H, W, C]

    elif isinstance(change_video, list):
        if len(change_video) == 0:
            raise ValueError("change_video must not be empty; received an empty list.")

        first = change_video[0]

        if isinstance(first, Image.Image):
            ref_size = first.size  # (W, H)
            ref_mode = first.mode
            for i, img in enumerate(change_video):
                if not isinstance(img, Image.Image):
                    raise TypeError(f"Mixed list: expected PIL Image at index {i}, got {type(img)}.")
                if img.size != ref_size:
                    raise ValueError(
                        f"PIL Image at index {i} has size {img.size}; expected {ref_size} (same as index 0)."
                    )
                if img.mode != ref_mode:
                    raise ValueError(
                        f"PIL Image at index {i} has mode '{img.mode}'; expected '{ref_mode}' (same as index 0)."
                    )

            arrays = [np.array(img) for img in change_video]  # list of [H, W, C] or [H, W]
            # Ensure channel dim exists for grayscale images
            arrays = [a[..., np.newaxis] if a.ndim == 2 else a for a in arrays]
            frames = torch.from_numpy(np.stack(arrays))  # [T, H, W, C]

        elif isinstance(first, torch.Tensor):
            tensor_frames: List[torch.Tensor] = []
            for i, frame in enumerate(change_video):
                if not isinstance(frame, torch.Tensor):
                    raise TypeError(f"Mixed list: expected torch.Tensor at index {i}, got {type(frame)}.")
                if frame.ndim != 4:
                    raise ValueError(f"Tensor at index {i} has {frame.ndim} dimensions; expected 4 ([1, H, W, C]).")
                if frame.shape[0] != 1:
                    raise ValueError(
                        f"Tensor at index {i} has a leading dimension of {frame.shape[0]}; "
                        "expected 1 (a single squeezable batch dimension)."
                    )
                tensor_frames.append(frame.squeeze(0))
            frames = torch.stack(tensor_frames)  # [T, H, W, C]

        else:
            raise TypeError(f"List elements must be torch.Tensor or PIL.Image.Image, got {type(first)}.")

    else:
        raise TypeError(f"Expected a list of tensors/PIL images or a single tensor, got {type(change_video)}.")

    # Shared processing path for all input types
    frames = frames.to(torch.bfloat16).div_(255.0)  # normalize in-place to [0, 1]
    return frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]


def compute_change_map(
    diffs: torch.Tensor,
    threshold: float = 0.1,
    reduction: str = "l2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a binary spatial change map from a diff tensor.

    Args:
        diffs:                 BFloat16 tensor of shape (T, C, W, H)
        threshold:             Scalar threshold after normalization
        reduction:             How to collapse C → scalar per position:
                                 'l2'   - L2 norm across channels (default)
                                 'mean' - mean absolute value across channels
                                 'max'  - max absolute value across channels

    Returns:
        Boolean tensor of shape (T, W, H): True where change > threshold.
    """
    x = diffs.to(torch.bfloat16)  # (T, C, W, H)

    # Reduce across channel dimension → (T, W, H)
    if reduction == "l2":
        magnitude = x.norm(p=2, dim=1)  # L2 norm across C
    elif reduction == "mean":
        magnitude = x.abs().mean(dim=1)  # mean |diff| across C
    elif reduction == "max":
        magnitude = x.abs().amax(dim=1)  # max |diff| across C
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

    return magnitude > threshold, magnitude  # (T, W, H) bool, (T, W, H)


def filter_global_motion_frames(
    change_map: torch.Tensor,
    threshold_pct: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero out frames where changed pixels exceed a percentage of total pixels.

    Args:
        change_map:     Bool tensor (T, H, W)
        threshold_pct:  Frames with more than this % of changed pixels are zeroed (default: 4%)

    Returns:
        filtered_map:   (T, H, W) — change_map with global-motion frames zeroed out
        is_global:      (T,)      — True for frames that were zeroed out
    """
    _T, H, W = change_map.shape
    total_pixels = H * W

    pixel_change = change_map.sum(dim=(1, 2))  # (T,)
    pct_changed = pixel_change.float() / total_pixels * 100  # (T,)

    is_global = pct_changed > threshold_pct  # (T,)
    filtered_map = change_map & ~is_global.view(-1, 1, 1)  # (T, H, W)

    return filtered_map, is_global


def downsample_change_map(
    change_map: torch.Tensor,
    factor: int = 8,
    method: Literal["max", "avg"] = "max",
    avg_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Spatially downsample a boolean change map by an integer factor.

    Args:
        change_map:     Bool tensor (T, H, W)
        factor:         Spatial reduction factor (default: 8)
        method:         'max' — any change in cell → True  (conservative, keeps more)
                        'avg' — fraction of changed pixels must exceed avg_threshold
        avg_threshold:  Threshold for 'avg' method (default: 0.5)

    Returns:
        Bool tensor (T, H//factor, W//factor)
    """
    T, H, W = change_map.shape

    # Trim to factor-divisible size
    H_t, W_t = (H // factor) * factor, (W // factor) * factor
    x = change_map[:, :H_t, :W_t].float().unsqueeze(1)  # (T, 1, H_t, W_t)

    if method == "max":
        pooled = F.max_pool2d(x, kernel_size=factor, stride=factor)  # (T, 1, H//f, W//f)
        result = pooled.squeeze(1) > 0.0

    elif method == "avg":
        pooled = F.avg_pool2d(x, kernel_size=factor, stride=factor)  # (T, 1, H//f, W//f)
        result = pooled.squeeze(1) > avg_threshold

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'max' or 'avg'.")

    return result


def compute_last_change_timestep(
    change_map: torch.Tensor,
    reverse_map: bool = False,
    normalize: bool = True,
    norm_number: int = -1,
    no_change_fill: float = 0.0,
) -> torch.Tensor:
    """
    For each position (w, h) and each time step t, compute the index of the
    last time step <= t at which a change was detected.

    Args:
        change_map:     Bool tensor of shape (T, W, H)
        reverse_map:    If True, look forward in time (>= t) for the closest change.
                        If False, look backward in time (<= t) for the most recent change.
        normalize:      If True, normalize valid indices to [0, 1] by dividing by (T-1)
        norm_number:    If normalize is set:
                           If norm_number > 0, normalize by norm_number.
                           Else, normalize by first dimension of change_map - 1.
        no_change_fill: Value assigned to positions that have never changed up to time t.
                        0.0 a neutral "no history" value.

    Returns:
        Float tensor of shape (T, W, H):
            - At position (t, w, h): the (normalized) index of the most recent
              change at (w, h) up to and including step t.
            - no_change_fill where no change has been seen yet.
    """
    T, W, H = change_map.shape
    device = change_map.device

    # 1. Base allocation: expand creates a zero-memory view, clone allocates exactly once (~1.6GB)
    t_idx = torch.arange(T, device=device, dtype=torch.float32).view(T, 1, 1).expand(T, W, H)
    last_idx = t_idx.clone()

    # 2. Use in-place masking instead of torch.where + torch.full
    not_changed = ~change_map
    if reverse_map:
        last_idx.masked_fill_(not_changed, float("inf"))
        del not_changed  # Free the 400MB mask immediately

        last_idx = last_idx.flip(0)

        # Explicitly catch the indices tensor and delete it immediately to save 3.2GB VRAM
        last_idx, unused_indices = last_idx.cummin(dim=0)
        del unused_indices

        last_idx = last_idx.flip(0)
    else:
        last_idx.masked_fill_(not_changed, float("-inf"))
        del not_changed

        # Catch and delete indices immediately
        last_idx, unused_indices = last_idx.cummax(dim=0)
        del unused_indices

    # 3. In-place normalization BEFORE sentinel fill
    # (Since inf / norm_number is still inf, this is mathematically safe and avoids extra masking)
    if normalize and T > 1:
        if norm_number < 1:
            norm_number = T - 1
        last_idx.div_(norm_number)  # In-place division

    # 4. Fill sentinels in-place
    never_changed = last_idx.isinf()
    last_idx.masked_fill_(never_changed, no_change_fill)

    return last_idx


@torch.inference_mode()
def prepare_painting_changes_tensor(
    change_video: torch.Tensor,
    device: Any,
    sampled_frame_indices: List[int] = [],
    reverse_map: bool = False,
    threshold: float = 0.1,
    skip_global_movements: bool = True,
    threshold_pct: float = 3.2,
    downsample: bool = False,
    downsample_factor: int = 16,
    downsample_method: Literal["max", "avg"] = "max",
    normalize_to_abs_finish: bool = True,
    total_num_frame_changes: int = -1,
) -> torch.Tensor:
    """
    Compute the last change map of a video: for every selected frame and every
    pixel, the normalized timestep at which the pixel last changed.

    The method computes per-pixel change maps from frame differences,
    optionally filters out frames with global motion, optionally downsamples
    spatially, and then derives the last change timestep per pixel
    (see ``compute_last_change_timestep``).

    Args:
        change_video:
            Video tensor of shape ``[N, C, H, W]`` with pixel values in
            ``[0.0, 1.0]`` (see ``prepare_change_video_tensor``), where ``N`` is
            the number of frames.
        device:
            Device to compute the change map on.
        sampled_frame_indices:
            Indices of the frames to retain from the full change map.
            If ascending, forward frame order is assumed; if descending,
            reverse frame order is assumed and indices are mirrored
            accordingly. Empty (default) keeps all frames.
        reverse_map:
            If True, look forward in time (>= t) for the closest change.
            If False, look backward in time (<= t) for the most recent change.
        threshold:
            Per-pixel change threshold, see ``compute_change_map``.
        skip_global_movements:
            If True, discard frames with global motion, see ``filter_global_motion_frames``.
        threshold_pct:
            Percentage of changed pixels above which a frame counts as global motion.
        downsample:
            If True, spatially downsample the change map, see ``downsample_change_map``.
        downsample_factor:
            Spatial reduction factor if ``downsample`` is set.
        downsample_method:
            Pooling method if ``downsample`` is set.
        normalize_to_abs_finish:
            If True and ``total_num_frame_changes > 1``, normalize the timesteps
            by ``total_num_frame_changes - 1`` instead of ``N - 1``.
        total_num_frame_changes:
            Total number of frames of the **full** (un-sampled) painting process,
            used to normalize to an absolute timescale. ``-1`` (default)
            normalizes by ``N - 1``.

    Returns:
        Float tensor of shape ``[T, H, W]`` (``[T, H // downsample_factor,
        W // downsample_factor]`` if ``downsample``), where ``T`` is the number of
        sampled frames (or ``N``). Positions without a change are 0.
    """
    change_video = change_video.to(device)
    # Frame difference
    diffs = change_video[1:] - change_video[:-1]
    del change_video
    # Change map
    change_map, _magnitude = compute_change_map(diffs, threshold=threshold)
    H, W = change_map.shape[-2:]
    del diffs
    del _magnitude
    # Pad first/last entry since it got lost by the difference computation
    empty_mask = torch.zeros(1, H, W, dtype=torch.bool, device=change_map.device)
    if reverse_map:
        change_map = torch.cat([change_map, empty_mask], dim=0)  # [T, H, W]
    else:
        change_map = torch.cat([empty_mask, change_map], dim=0)  # [T, H, W]
    del empty_mask

    if skip_global_movements:
        change_map, _is_global = filter_global_motion_frames(change_map, threshold_pct=threshold_pct)
        del _is_global

    if downsample:
        change_map = downsample_change_map(change_map, factor=downsample_factor, method=downsample_method)

    if normalize_to_abs_finish and total_num_frame_changes > 1:
        norm_number = total_num_frame_changes - 1
    else:
        norm_number = -1
    # Compute when the last change happened for each pixel and each frame
    last_change_map = compute_last_change_timestep(change_map, reverse_map=reverse_map, norm_number=norm_number)

    if len(sampled_frame_indices) > 0:
        # Select only the ones we need
        last_change_indices = _sampled_to_change_indices(sampled_frame_indices)
        # Select only the frame changes we want
        last_change_map = torch.index_select(
            last_change_map, 0, torch.LongTensor(last_change_indices).to(last_change_map.device)
        )
    return last_change_map


def _sampled_to_change_indices(sampled_frame_indices: List[int]) -> List[int]:
    """
    Map sampled video frame indices to indices into the change map, which starts at the first
    sampled frame. Ascending indices assume forward frame order, descending ones reverse order.
    """
    if sampled_frame_indices[0] < sampled_frame_indices[1]:
        return [frame_idx - sampled_frame_indices[0] for frame_idx in sampled_frame_indices]
    # Reverse frame order
    return [sampled_frame_indices[0] - frame_idx for frame_idx in sampled_frame_indices]


def compute_sampled_frame_timesteps(
    num_change_frames: int,
    sampled_frame_indices: List[int] = [],
    normalize_to_abs_finish: bool = True,
    total_num_frame_changes: int = -1,
) -> torch.Tensor:
    """
    Timestep of each frame returned by ``prepare_painting_changes_tensor``, on the same
    (normalized) scale as the values of the last change map.

    Args:
        num_change_frames:       Number of frames of the change video (``change_video.shape[0]``).
        sampled_frame_indices:   Same as passed to ``prepare_painting_changes_tensor``.
        normalize_to_abs_finish: Same as passed to ``prepare_painting_changes_tensor``.
        total_num_frame_changes: Same as passed to ``prepare_painting_changes_tensor``.

    Returns:
        Float tensor (T,) with the timestep of each frame of the last change map.
    """
    if len(sampled_frame_indices) > 0:
        change_indices = _sampled_to_change_indices(sampled_frame_indices)
    else:
        change_indices = list(range(num_change_frames))
    timesteps = torch.tensor(change_indices, dtype=torch.float32)

    # Mirrors the normalization in prepare_painting_changes_tensor / compute_last_change_timestep
    if num_change_frames > 1:
        if normalize_to_abs_finish and total_num_frame_changes > 1:
            norm_number = total_num_frame_changes - 1
        else:
            norm_number = num_change_frames - 1
        timesteps.div_(norm_number)
    return timesteps


def last_change_to_image_tensor(
    last_change: torch.Tensor,
    mode: ChangeMapMode = "interpolate",
    color_start: tuple[int, int, int] = (0, 0, 0),
    color_end: tuple[int, int, int] = (255, 255, 255),
    no_change_color: tuple[int, int, int] = (0, 0, 0),
    reverse_map: bool = False,
    frame_timesteps: Optional[torch.Tensor] = None,
    local_min_value: float = 0.2,
) -> torch.Tensor:
    """
    Convert a last-change-timestep tensor to a RGB tensor.

    Modes:
        - "interpolate": Standard lerp between color_start and color_end.
        - "pack_24bit":  Maps the [0,1] value losslessly across all 16.7M RGB colors.
        - "history":     Computes the last 3 changes over time and maps them to R, G, B.
        - "interpolate_lineart": Combines interpolate with lineart processor.
        - "local_interpolate_lineart": Like interpolate_lineart, but the lerp only spans the changes
                         between the neighbouring frame and the current frame (needs frame_timesteps).

    Args:
        last_change:     Float tensor (T, W, H), values in [0, 1].
        mode:            The visualization mode to use.
        color_start:     RGB for normalized value = 0  (most recent)
        color_end:       RGB for normalized value = 1  (oldest / longest ago)
        no_change_color: RGB for positions that have never changed
        reverse_map:
            If True, look forward in time (>= t) for the closest change.
            If False, look backward in time (<= t) for the most recent change.
        frame_timesteps:
            Float tensor (T,) with the timestep of each frame on the scale of last_change,
            see ``compute_sampled_frame_timesteps``. Only used (and required) by
            "local_interpolate_lineart".
        local_min_value:
            "local_interpolate_lineart" only: the interval is mapped to [local_min_value, 1]
            instead of [0, 1], so changes at the dark end stay distinguishable from no change.

    Returns:
        Tensor corresponding to a list of RGB images.
    """
    assert last_change.ndim == 3, f"last_change must have shape (T, W, H), got {last_change.shape}"
    T, W, H = last_change.shape

    # Move to CPU and float32 once at the start
    x = last_change.float().cpu()

    start = torch.tensor(color_start, dtype=torch.float32)
    end = torch.tensor(color_end, dtype=torch.float32)
    fill = torch.tensor(no_change_color, dtype=torch.float32)

    if mode == "interpolate" or mode == "interpolate_lineart":
        # Mode 0: Standard interpolation (255 distinct steps)
        t_norm = x.clamp(0, 1).unsqueeze(-1)  # (T, W, H, 1)
        colors = (1 - t_norm) * start + t_norm * end
        colors = torch.where(x.unsqueeze(-1) > 0, colors, fill)

    elif mode == "local_interpolate_lineart":
        # Same as interpolate, but zoomed into the timestep interval between the neighbouring frame
        # and the current frame. Changes outside of it are treated as no change.
        if frame_timesteps is None:
            raise ValueError(f"mode={mode!r} requires frame_timesteps, see compute_sampled_frame_timesteps.")
        ts = torch.as_tensor(frame_timesteps, dtype=torch.float32).flatten().cpu()
        if ts.numel() != T:
            raise ValueError(f"frame_timesteps has {ts.numel()} entries, but last_change has {T} frames.")

        if reverse_map:
            # Looks forward: changes in [current, next frame), the last frame runs to the end (1.0)
            neighbour = torch.cat([ts[1:], ts.new_ones(1)])
        else:
            # Looks backward: changes in (previous frame, current], the first frame starts at 0.0
            neighbour = torch.cat([ts.new_zeros(1), ts[:-1]])

        # Keep the orientation of the global values (low -> color_start, high -> color_end)
        lo = torch.minimum(ts, neighbour).view(T, 1, 1)
        hi = torch.maximum(ts, neighbour).view(T, 1, 1)
        span = hi - lo
        valid_span = span > 0
        t_local = (x - lo) / torch.where(valid_span, span, torch.ones_like(span))  # (T, W, H)

        # Tolerance for timesteps computed on another device (e.g. CUDA vs. CPU division)
        eps = 1e-6
        if reverse_map:
            # The upper bound belongs to the next frame, except for the end of the video
            is_end = hi >= 1.0 - eps
            in_bounds = (x >= lo - eps) & ((x < hi - eps) | is_end)
        else:
            # The lower bound belongs to the previous frame
            in_bounds = (x > lo + eps) & (x <= hi + eps)
        in_interval = (x > 0) & valid_span & in_bounds

        t_local = local_min_value + (1 - local_min_value) * t_local.clamp(0, 1)
        t_norm = t_local.unsqueeze(-1)  # (T, W, H, 1)
        colors = (1 - t_norm) * start + t_norm * end
        colors = torch.where(in_interval.unsqueeze(-1), colors, fill)

    elif mode == "pack_24bit":
        # Mode 1: Use all 3 channels for interpolation (16,777,215 steps)
        v = (x.clamp(0, 1) * 16777215).long()

        r = (v >> 16) & 255
        g = (v >> 8) & 255
        b = v & 255

        colors = torch.stack((r, g, b), dim=-1).float()
        colors = torch.where(x.unsqueeze(-1) > 0, colors, fill)

    elif mode == "history":
        # Mode 2:
        # if reverse_map == True:
        #      R=last change, G=next change, B=next next change
        # else:
        #      R=last change, G=previous change, B=older change
        # Compute the history tensor (T, W, H, 3) by tracking changes along T
        history = torch.zeros((T, W, H, 3), dtype=torch.float32)
        state = torch.zeros((W, H, 3), dtype=torch.float32)

        if not reverse_map:
            # Look backward in time (<= t)
            # R=last change, G=previous change, B=older change
            for t in range(T):
                curr = x[t]
                if t == 0:
                    state[..., 0] = curr
                else:
                    prev = x[t - 1]
                    changed = curr != prev

                    # Shift historical values: 2nd becomes 3rd, 1st becomes 2nd
                    state[..., 2] = torch.where(changed, state[..., 1], state[..., 2])
                    state[..., 1] = torch.where(changed, prev, state[..., 1])
                    state[..., 0] = curr
                history[t] = state
        else:
            # Look forward in time (>= t)
            # R=last change, G=next change, B=next next change
            for t in range(T - 1, -1, -1):
                curr = x[t]
                if t == T - 1:
                    state[..., 0] = curr
                else:
                    ahead = x[t + 1]
                    changed = curr != ahead

                    # Shift forward-looking values: next becomes next-next
                    state[..., 2] = torch.where(changed, state[..., 1], state[..., 2])
                    state[..., 1] = torch.where(changed, ahead, state[..., 1])
                    state[..., 0] = curr
                history[t] = state

        # Interpolate using the broadcasted start/end colors over the 3 history channels
        t_norm = history.clamp(0, 1)  # (T, W, H, 3)
        colors = (1 - t_norm) * start + t_norm * end

        # Apply sentinel fill if the tracked history value is 0
        colors = torch.where(history.sum(dim=-1, keepdim=True) > 0, colors, fill)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return colors.clamp(0, 255)


def last_change_to_pil(
    last_change: torch.Tensor,
    mode: ChangeMapMode = "interpolate",
    reverse_map: bool = False,
    frame_timesteps: Optional[torch.Tensor] = None,
) -> list[Image.Image]:
    """
    Convert a last-change-timestep tensor to a list of RGB PIL images.

    Modes:
        - "interpolate": Standard lerp between color_start and color_end.
        - "pack_24bit":  Maps the [0,1] value losslessly across all 16.7M RGB colors.
        - "history":     Computes the last 3 changes over time and maps them to R, G, B.
        - "interpolate_lineart": Combines interpolate with lineart processor.
        - "local_interpolate_lineart": Like interpolate_lineart, but the lerp only spans the changes
                         between the neighbouring frame and the current frame (needs frame_timesteps).

    Args:
        last_change:     Float tensor (T, W, H), values in [0, 1].
        mode:            The visualization mode to use.
        reverse_map:
            If True, look forward in time (>= t) for the closest change.
            If False, look backward in time (<= t) for the most recent change.
        frame_timesteps:
            Float tensor (T,) with the timestep of each frame, see ``compute_sampled_frame_timesteps``.
            Required for "local_interpolate_lineart".

    Returns:
        List of T RGB PIL images.
    """
    color_start: tuple[int, int, int] = (0, 0, 0)
    color_end: tuple[int, int, int] = (255, 255, 255)
    no_change_color: tuple[int, int, int] = get_no_change_color(mode=mode)
    T, _, _ = last_change.shape

    colors = last_change_to_image_tensor(
        last_change,
        mode=mode,
        color_start=color_start,
        color_end=color_end,
        no_change_color=no_change_color,
        reverse_map=reverse_map,
        frame_timesteps=frame_timesteps,
    )

    # Convert to uint8 PIL images
    colors_u8 = colors.clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(colors_u8[t], mode="RGB") for t in range(T)]


def get_no_change_color(
    mode: ChangeMapMode = "interpolate",
) -> tuple[int, int, int]:
    """
    Returns the default no change color for the given mode.

    Modes:
        - "interpolate": Standard lerp between color_start and color_end.
        - "pack_24bit":  Maps the [0,1] value losslessly across all 16.7M RGB colors.
        - "history":     Computes the last 3 changes over time and maps them to R, G, B.
        - "interpolate_lineart": Combines interpolate with lineart processor.
        - "local_interpolate_lineart": Like interpolate_lineart, but the lerp only spans the changes
                         between the neighbouring frame and the current frame (needs frame_timesteps).

    Args:
        mode:            The visualization mode to use.

    Returns:
        Tuple of RGB color.
    """
    no_change_color: tuple[int, int, int] = (0, 0, 0)

    if mode == "history":
        # Gives more contrast
        no_change_color = (255, 255, 255)

    return no_change_color


def merge_control_videos(control_video_0: List[Image.Image], control_video_1: List[Image.Image]) -> List[Image.Image]:
    """
    Merge two control videos into a single RGB video using channel packing:
    ``control_video_0`` goes into the red channel, ``control_video_1`` into green
    and blue (cyan). All frames are converted to grayscale first.

    For the lineart modes, pass the change map as ``control_video_0`` and the
    lineart as ``control_video_1``: red change map, cyan lineart.
    """
    if len(control_video_0) != len(control_video_1):
        raise ValueError("The two video sequences must have the same number of frames.")

    merged_video = []
    for frame_0, frame_1 in zip(control_video_0, control_video_1):
        # Image.merge needs single-channel ('L') images
        r_channel = frame_0.convert("L")
        gb_channel = frame_1.convert("L")
        # R = control_video_0, G = B = control_video_1
        merged_video.append(Image.merge("RGB", (r_channel, gb_channel, gb_channel)))

    return merged_video


def video_to_change_map_images(
    video: List[Image.Image],
    sampled_frame_indices: List[int],
    mode: ChangeMapMode = "interpolate_lineart",
    device: Any = "cpu",
    reverse_map: bool = True,
    threshold: float = 0.13,
    threshold_pct: float = 4.0,
    lineart_processor: Any = None,
) -> List[Image.Image]:
    """
    Compute the change map images of a painting video for the sampled frames.

    With ``reverse_map`` (default) the video is processed like during training: from the latest
    sampled frame back to the blank canvas.

    Args:
        video:                 All frames of the painting video, from blank canvas to finished painting.
        sampled_frame_indices: Ascending indices into ``video`` of the frames to return change maps for.
        mode:                  The visualization mode, see ``last_change_to_image_tensor``.
        device:                Device to compute the change map (and lineart) on.
        reverse_map:           See ``prepare_painting_changes_tensor``.
        threshold:             Per-pixel change threshold, see ``compute_change_map``.
        threshold_pct:         Global motion threshold in %, see ``filter_global_motion_frames``.
        lineart_processor:     Optional ``ControlnetProcessor(mode="lineart")`` to reuse for the
                               lineart modes, otherwise one is created.

    Returns:
        List of RGB PIL images, one per entry of ``sampled_frame_indices``.
    """
    if len(sampled_frame_indices) < 2 or sorted(set(sampled_frame_indices)) != list(sampled_frame_indices):
        raise ValueError(f"Expected at least 2 unique, ascending indices, got {sampled_frame_indices}.")

    if reverse_map:
        # Latest sampled frame -> blank canvas
        change_frames = video[sampled_frame_indices[-1] :: -1]
        frame_indices = sampled_frame_indices[::-1]
        total_num_frame_changes = sampled_frame_indices[-1] + 1
    else:
        # First sampled frame -> finished painting
        change_frames = video[sampled_frame_indices[0] :]
        frame_indices = list(sampled_frame_indices)
        total_num_frame_changes = len(video) - sampled_frame_indices[0]

    change_video = prepare_change_video_tensor(change_frames)
    last_change = prepare_painting_changes_tensor(
        change_video,
        device=device,
        sampled_frame_indices=frame_indices,
        reverse_map=reverse_map,
        threshold=threshold,
        threshold_pct=threshold_pct,
        total_num_frame_changes=total_num_frame_changes,
    )
    frame_timesteps = compute_sampled_frame_timesteps(
        change_video.shape[0], frame_indices, total_num_frame_changes=total_num_frame_changes
    )
    del change_video
    images = last_change_to_pil(last_change, mode=mode, reverse_map=reverse_map, frame_timesteps=frame_timesteps)

    if mode in ("interpolate_lineart", "local_interpolate_lineart"):
        if lineart_processor is None:
            # Imported here, so the change map functions do not require loading controlnet_aux
            from painting_change_map.utils.controlnet_processor import ControlnetProcessor

            lineart_processor = ControlnetProcessor(mode="lineart")
        lineart = lineart_processor([video[i] for i in frame_indices], device=device)
        # R = change map, G = B = lineart
        images = merge_control_videos(images, lineart)

    # Back to the order of sampled_frame_indices
    return images[::-1] if reverse_map else images
