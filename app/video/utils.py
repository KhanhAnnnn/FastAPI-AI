from io import BytesIO

import av
from PIL import Image


def frames_to_mp4(frames: list[Image.Image], fps: int = 7) -> bytes:
    if not frames:
        raise ValueError("At least one video frame is required.")
    width, height = frames[0].size
    if width <= 0 or height <= 0 or any(frame.size != (width, height) for frame in frames):
        raise ValueError("Video frames must have consistent, non-zero dimensions.")
    buffer = BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image.convert("RGB"))):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    buffer.seek(0)
    return buffer.read()
