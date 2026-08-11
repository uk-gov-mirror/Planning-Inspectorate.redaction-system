import os
from io import BytesIO

from PIL import Image

from core.util.image_analysis import AzureVisionUtil


class TestDetectFaces:
    def test_identifies_faces(self):
        """
        - Given I have an image with two people in it (Darth Plagueis the wise scene from Revenge of the Sith)
        - When I call AzureVisionUtil.detect_faces
        - The two faces should be identified
        """
        with open(
            os.path.join("test", "resources", "image", "image_with_faces.jpeg"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))
            response = AzureVisionUtil().detect_faces(image, confidence_threshold=0.5)
            # Azure Vision seems to be deterministic from testing

        expected_response = ((0, 2, 410, 430), (359, 7, 766, 431))
        assert expected_response == response

    def test_uses_cached_response(self):
        with open(
            os.path.join("test", "resources", "image", "image_with_faces.jpeg"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))
            response = AzureVisionUtil().detect_faces(image, confidence_threshold=0.5)
            # Azure Vision seems to be deterministic from testing
            new_response = AzureVisionUtil().detect_faces(
                image, confidence_threshold=0.5
            )

        expected_response = ((0, 2, 410, 430), (359, 7, 766, 431))

        assert expected_response == new_response
        assert response == new_response


class TestDetectText:
    def __init__(self):
        from .text_boxes import EXPECTED_TEXT_RESPONSE

        self.EXPECTED_TEXT_RESPONSE = EXPECTED_TEXT_RESPONSE

    def test_returns_text_boxes(self):
        """
        - Given I have an image containing a lot of text
        - When I call AzureVisionUtil.detect_text
        - The text content of the image should be extracted, with each line represented by a bounding box
        """
        with open(
            os.path.join("test", "resources", "image", "image_with_text.jpg"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))
            response = AzureVisionUtil().detect_text(image)

        assert self.EXPECTED_TEXT_RESPONSE == response

    def test_uses_cached_response(self):
        with open(
            os.path.join("test", "resources", "image", "image_with_text.jpg"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))
            response = AzureVisionUtil().detect_text(image)
            new_response = AzureVisionUtil().detect_text(image)

        assert self.EXPECTED_TEXT_RESPONSE == response
        assert response == new_response
