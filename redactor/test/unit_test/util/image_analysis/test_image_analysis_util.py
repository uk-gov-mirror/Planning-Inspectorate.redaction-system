from typing import ClassVar
from unittest.mock import Mock

import pytest
from PIL import Image

from core.util.image_analysis import ImageAnalysisUtil
from test.util.util import compare_unashable_lists


class DetectionError(Exception):
    pass


class ConcreteAnalysisUtil(ImageAnalysisUtil):
    """Concrete subclass for testing the abstract base."""

    _IMAGE_TEST_CACHE: ClassVar[list] = []


@pytest.fixture(autouse=True)
def _clear_caches():
    ConcreteAnalysisUtil._IMAGE_TEST_CACHE.clear()
    yield
    ConcreteAnalysisUtil._IMAGE_TEST_CACHE.clear()


class TestCheckImageSize:
    def test_valid_image(self):
        assert ImageAnalysisUtil.check_image_size(Image.new("RGB", (100, 100))) is True

    def test_exact_minimum(self):
        assert ImageAnalysisUtil.check_image_size(Image.new("RGB", (50, 50))) is True

    def test_exact_maximum(self):
        assert (
            ImageAnalysisUtil.check_image_size(Image.new("RGB", (16000, 16000))) is True
        )

    def test_too_small_width(self):
        assert ImageAnalysisUtil.check_image_size(Image.new("RGB", (49, 100))) is False

    def test_too_small_height(self):
        assert ImageAnalysisUtil.check_image_size(Image.new("RGB", (100, 49))) is False

    def test_too_large_width(self):
        assert (
            ImageAnalysisUtil.check_image_size(Image.new("RGB", (16001, 100))) is False
        )

    def test_too_large_height(self):
        assert (
            ImageAnalysisUtil.check_image_size(Image.new("RGB", (100, 16001))) is False
        )

    def test_non_rgb_image_converted(self):
        image = Image.new("RGBA", (100, 100))
        assert ImageAnalysisUtil.check_image_size(image) is True


class TestCreateFailedRedactionOutput:
    def test_non_text_covers_full_image(self):
        image = Image.new("RGB", (200, 300))
        result = ImageAnalysisUtil._create_failed_redaction_output(image)
        assert result == (image, (0, 0, 200, 300))

    def test_text_includes_label(self):
        image = Image.new("RGB", (200, 300))
        result = ImageAnalysisUtil._create_failed_redaction_output(image, is_text=True)
        assert result == (image, ("TEXT DETECTION FAILED", (0, 0, 200, 300)))


class TestClearCache:
    def test_clears_image_caches(self):
        ConcreteAnalysisUtil._IMAGE_TEST_CACHE.append({"image": Mock()})
        assert len(ConcreteAnalysisUtil._IMAGE_TEST_CACHE) == 1

        ConcreteAnalysisUtil.clear_cache()
        assert len(ConcreteAnalysisUtil._IMAGE_TEST_CACHE) == 0


class TestImageDetection:
    def test_returns_results_for_all_images(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(5)]
        expected = [(img, (i,)) for i, img in enumerate(images)]

        results = ImageAnalysisUtil._image_detection(
            images,
            "test",
            lambda img: (images.index(img),),
        )

        compare_unashable_lists(expected, results)

    def test_passes_kwargs_to_detection_function(self):
        images = [Image.new("RGB", (51, 51))]
        captured = {}

        def detector(img, threshold=0.5):
            captured["threshold"] = threshold
            return ()

        ImageAnalysisUtil._image_detection(images, "test", detector, threshold=0.9)

        assert captured["threshold"] == 0.9

    def test_fallback_redaction_on_exception(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(3)]

        def failing_detector(image):
            if image == images[1]:
                raise DetectionError("boom")
            return ((0, 0, 10, 10),)

        results = ImageAnalysisUtil._image_detection(images, "object", failing_detector)

        failed = next(r for r in results if r[0] == images[1])
        assert failed == (images[1], (0, 0, images[1].width, images[1].height))

    def test_text_fallback_includes_label(self):
        images = [Image.new("RGB", (51, 51))]

        def failing_detector(image):
            raise DetectionError("boom")

        results = ImageAnalysisUtil._image_detection(images, "text", failing_detector)

        assert results[0] == (
            images[0],
            ("TEXT DETECTION FAILED", (0, 0, images[0].width, images[0].height)),
        )
