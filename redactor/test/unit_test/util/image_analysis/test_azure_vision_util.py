from unittest.mock import Mock, patch

import pytest
from PIL import Image

from core.util.image_analysis import AzureVisionUtil
from test.util.util import compare_unashable_lists


class ImageAnalysisError(Exception):
    pass


@pytest.fixture(autouse=True)
def _clear_caches():
    AzureVisionUtil._IMAGE_FACE_CACHE.clear()
    AzureVisionUtil._IMAGE_TEXT_CACHE.clear()
    yield
    AzureVisionUtil._IMAGE_FACE_CACHE.clear()
    AzureVisionUtil._IMAGE_TEXT_CACHE.clear()


def _mock_vision_result_people(people):
    mock_result = Mock()
    mock_result.people.list = people
    return mock_result


def _mock_vision_result_text():
    mock_result = Mock()

    class MockWord:
        def __init__(self, content, bounding_box):
            self.text = content
            self.bounding_polygon = bounding_box

    class MockLine:
        def __init__(self, words):
            self.words = words

    class MockBlock:
        def __init__(self, lines):
            self.lines = lines

    mock_result.read.blocks = [
        MockBlock(
            lines=[
                MockLine(
                    words=[
                        MockWord(
                            "Hello",
                            [Mock(x=10, y=20), Mock(x=40, y=20), Mock(x=30, y=40)],
                        )
                    ],
                ),
                MockLine(
                    words=[
                        MockWord(
                            "World",
                            [Mock(x=50, y=60), Mock(x=80, y=60), Mock(x=70, y=80)],
                        )
                    ],
                ),
            ]
        ),
    ]
    return mock_result


class TestDetectFaces:
    def test_returns_boxes_above_threshold(self):
        image = Mock()
        people_list = [
            Mock(bounding_box=Mock(x=10, y=20, width=30, height=40), confidence=0.9),
            Mock(bounding_box=Mock(x=50, y=60, width=10, height=10), confidence=0.4),
        ]

        with (
            patch.object(AzureVisionUtil, "check_image_size", return_value=True),
            patch.object(
                AzureVisionUtil,
                "_azure_vision_analysis",
                return_value=_mock_vision_result_people(people_list),
            ),
        ):
            result = AzureVisionUtil.detect_faces(image, confidence_threshold=0.5)

        assert result == ((10, 20, 40, 60),)

    def test_caches_result(self):
        image = Mock()
        people_list = [
            Mock(bounding_box=Mock(x=10, y=20, width=30, height=40), confidence=0.9),
        ]

        with (
            patch.object(AzureVisionUtil, "check_image_size", return_value=True),
            patch.object(
                AzureVisionUtil,
                "_azure_vision_analysis",
                return_value=_mock_vision_result_people(people_list),
            ),
        ):
            AzureVisionUtil.detect_faces(image, confidence_threshold=0.5)

        assert len(AzureVisionUtil._IMAGE_FACE_CACHE) == 1
        assert AzureVisionUtil._IMAGE_FACE_CACHE[0]["image"] == image

    def test_uses_cached_result(self):
        image = Mock()
        AzureVisionUtil._IMAGE_FACE_CACHE = [
            {"image": image, "faces": ({"box": (10, 20, 40, 60), "confidence": 0.9},)}
        ]

        with patch.object(AzureVisionUtil, "_azure_vision_analysis") as mock_analysis:
            result = AzureVisionUtil.detect_faces(image, confidence_threshold=0.5)

        mock_analysis.assert_not_called()
        assert result == ((10, 20, 40, 60),)

    def test_skips_image_too_small(self):
        image = Image.new("RGB", (49, 49))
        result = AzureVisionUtil.detect_faces(image, confidence_threshold=0.5)
        assert result == ()

    def test_skips_image_too_large(self):
        image = Image.new("RGB", (16001, 100))
        result = AzureVisionUtil.detect_faces(image, confidence_threshold=0.5)
        assert result == ()


class TestDetectFacesInImages:
    def test_returns_results_for_all_images(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(5)]
        expected_results = [(img, (i,)) for i, img in enumerate(images)]

        with patch.object(
            AzureVisionUtil,
            "detect_faces",
            side_effect=lambda img, **kw: (images.index(img),),
        ):
            actual_results = AzureVisionUtil.detect_faces_in_images(images, 0.1)

        compare_unashable_lists(expected_results, actual_results)

    def test_redacts_full_image_on_exception(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(3)]

        def mock_detect(image, **kwargs):
            if image == images[1]:
                raise ImageAnalysisError("Some exception")
            return ((0, 0, 10, 10),)

        with patch.object(AzureVisionUtil, "detect_faces", side_effect=mock_detect):
            actual_results = AzureVisionUtil.detect_faces_in_images(images, 0.1)

        failed_result = next(r for r in actual_results if r[0] == images[1])
        assert failed_result == (images[1], (0, 0, images[1].width, images[1].height))


class TestDetectText:
    def test_returns_words_with_bounding_boxes(self):
        image = Mock()

        with (
            patch.object(AzureVisionUtil, "check_image_size", return_value=True),
            patch.object(
                AzureVisionUtil,
                "_azure_vision_analysis",
                return_value=_mock_vision_result_text(),
            ),
        ):
            result = AzureVisionUtil.detect_text(image)

        assert result == (("Hello", (10, 20, 30, 40)), ("World", (50, 60, 70, 80)))

    def test_caches_result(self):
        image = Mock()

        with (
            patch.object(AzureVisionUtil, "check_image_size", return_value=True),
            patch.object(
                AzureVisionUtil,
                "_azure_vision_analysis",
                return_value=_mock_vision_result_text(),
            ),
        ):
            AzureVisionUtil.detect_text(image)

        assert len(AzureVisionUtil._IMAGE_TEXT_CACHE) == 1
        assert AzureVisionUtil._IMAGE_TEXT_CACHE[0]["image"] == image

    def test_uses_cached_result(self):
        image = Mock()
        cached_text = (("Hello", (10, 20, 30, 40)),)
        AzureVisionUtil._IMAGE_TEXT_CACHE = [{"image": image, "text": cached_text}]

        with patch.object(AzureVisionUtil, "_azure_vision_analysis") as mock_analysis:
            result = AzureVisionUtil.detect_text(image)

        mock_analysis.assert_not_called()
        assert result == cached_text


class TestDetectTextInImages:
    def test_returns_results_for_all_images(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(5)]
        expected_results = [(img, (str(i),)) for i, img in enumerate(images)]

        with patch.object(
            AzureVisionUtil,
            "detect_text",
            side_effect=lambda img: (str(images.index(img)),),
        ):
            actual_results = AzureVisionUtil.detect_text_in_images(images)

        compare_unashable_lists(expected_results, actual_results)

    def test_redacts_full_image_on_exception(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(3)]

        def mock_detect(image):
            if image == images[1]:
                raise ImageAnalysisError("Some exception")
            return (("word", (0, 0, 10, 10)),)

        with patch.object(AzureVisionUtil, "detect_text", side_effect=mock_detect):
            actual_results = AzureVisionUtil.detect_text_in_images(images)

        failed_result = next(r for r in actual_results if r[0] == images[1])
        assert failed_result == (
            images[1],
            ("TEXT DETECTION FAILED", (0, 0, images[1].width, images[1].height)),
        )
