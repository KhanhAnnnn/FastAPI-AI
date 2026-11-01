from io import BytesIO

from PIL import Image


def image_to_png(image: Image.Image) -> bytes:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image.")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
