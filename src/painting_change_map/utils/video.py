from pathlib import Path
from typing import List, Union

import av
from PIL import Image


def load_video(path: Union[str, Path]) -> List[Image.Image]:
    """
    Decode all frames of a video file.

    Args:
        path: Path to the video (any container/codec supported by FFmpeg, e.g. H.264 MP4).

    Returns:
        List of RGB PIL images, one per frame.
    """
    with av.open(str(path)) as container:
        return [frame.to_image() for frame in container.decode(video=0)]
