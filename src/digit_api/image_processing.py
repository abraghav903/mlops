from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


def load_image(content: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(content))
        image.load()
    except UnidentifiedImageError as exc:
        raise ImageValidationError("Uploaded file is not a valid image") from exc

    if image.width < 2 or image.height < 2:
        raise ImageValidationError("Image is too small to classify")
    return image


def preprocess_digit_image(image: Image.Image) -> np.ndarray:
    """Convert an arbitrary handwritten digit image to the notebook CNN input shape."""
    grayscale = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)

    if pixels.mean() > 127:
        pixels = 255 - pixels

    return pixels / 255.0
