import os
from io import BytesIO

from PIL import Image

from core.util.image_analysis import ImageAnalysisUtil


class TestCheckImageSize:
    def test_image_too_large(self):
        with open(
            os.path.join("test", "resources", "image", "test_image_large.jpg"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))
            assert not ImageAnalysisUtil.check_image_size(image)

    def test_image_too_small(self):
        with open(
            os.path.join("test", "resources", "image", "test_image_small.jpg"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))
            assert not ImageAnalysisUtil.check_image_size(image)

    def test_image_valid(self):
        with open(
            os.path.join("test", "resources", "image", "test_image_horizontal.jpg"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))
            assert ImageAnalysisUtil.check_image_size(image)
