from io import BytesIO

import av
import numpy as np
import pytest
from PIL import Image

from app.audio.utils import waveform_to_wav
from app.image.utils import image_to_png
from app.threed.utils import mesh_to_obj
from app.video.utils import frames_to_mp4


def test_image_to_png_returns_decodable_png() -> None:
    source = Image.new("RGB", (32, 24), "red")
    result = image_to_png(source)
    with Image.open(BytesIO(result)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.size == (32, 24)


def test_waveform_to_wav_returns_decodable_audio() -> None:
    waveform = np.sin(np.linspace(0, 2 * np.pi, 800, dtype=np.float32))
    result = waveform_to_wav(waveform, 8_000)
    audio, sample_rate = __import__("soundfile").read(BytesIO(result))
    assert sample_rate == 8_000
    assert audio.size == waveform.size


def test_video_encoder_writes_complete_mp4() -> None:
    frames = [Image.new("RGB", (64, 48), color) for color in ("red", "blue")]
    result = frames_to_mp4(frames, fps=2)
    with av.open(BytesIO(result)) as container:
        stream = container.streams.video[0]
        decoded_frames = list(container.decode(stream))
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height) == (64, 48)
        assert len(decoded_frames) == 2
        assert container.duration is not None and container.duration > 0


def test_mesh_to_obj_serializes_vertices_and_faces() -> None:
    class Mesh:
        verts = __import__("torch").tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        faces = __import__("torch").tensor([[0, 1, 2]])
        vertex_channels = {}

    data, vertices, faces = mesh_to_obj(Mesh())
    text = data.decode("ascii")
    assert vertices == 3
    assert faces == 1
    assert text.count("\nv ") == 3
    assert "\nf 1 2 3\n" in text


def test_video_encoder_rejects_empty_frames() -> None:
    with pytest.raises(ValueError, match="At least one"):
        frames_to_mp4([])
