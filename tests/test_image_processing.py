from PIL import Image

from digit_api.image_processing import preprocess_digit_image


def test_preprocess_digit_image_returns_28_by_28_pixels():
    image = Image.new("L", (28, 28), color=255)

    features = preprocess_digit_image(image)

    assert features.shape == (28, 28)
    assert features.min() >= 0
    assert features.max() <= 1
