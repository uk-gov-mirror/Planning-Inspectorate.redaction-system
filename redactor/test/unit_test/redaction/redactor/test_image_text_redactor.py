import dataclasses
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from core.redaction.config import ImageRedactionConfig
from core.redaction.redactor import ImageTextRedactor
from core.redaction.result import ImageRedactionResult
from test.unit_test.redaction.redactor.util import TestImageTextRedactorBase


def test_get_name():
    assert ImageTextRedactor.get_name() == "ImageTextRedaction"


def test_get_redaction_config_class():
    assert ImageTextRedactor.get_redaction_config_class() == ImageRedactionConfig


class DetectNumberPlates:
    @pytest.mark.parametrize(
        "test_input",
        [
            "AB12 CDE",  # Current format
            "AB12\nCDE",  # Current format on two lines
            "A12 BCD",  # Prefix format
            "ABC 1 D",  # Suffix format with 1 digit
            "ABC 12 D",  # Suffix format with 2 digits
            "ABC 123 D",  # Suffix format with 3 digits
            "1234 A",  # Dateless format with long number prefix
            "1234 AB",  # Dateless format with long number prefix
            "1 ABC",  # Dateless format with short number prefix
            "12 AB",  # Dateless format with short number prefix
            "123 A",  # Dateless format with short number prefix
            "AB 1234",  # Dateless format with long number suffix
            "AB 123",  # Dateless format with long number suffix
            "AB 12",  # Dateless format with short number suffix
            "ABC 123",  # Dateless format with short number suffix
            "101 D 234",  # Diplomatic format
        ],
    )
    def test_valid_number_plate(self, test_input):
        """
        - Given I have some text containing UK number plates
        - When I call ImageTextRedactor.detect_number_plates
        - Then the correct number plates should be returned
        """
        assert test_input in ImageTextRedactor.detect_number_plates(test_input)

    @pytest.mark.parametrize(
        "test_input",
        [
            "something AB12 CDE",  # Current format with preceding text
            "AB12 CDE something",  # Current format with following text
        ],
    )
    def test_ignores_surrounding_text(self, test_input):
        result = ImageTextRedactor.detect_number_plates(test_input)
        assert test_input not in result
        assert "AB12 CDE" in result


def test_examine_redaction_boxes():
    """
    - Given I have some text rectangle map and a redaction string
    - When I call ImageTextRedactor.examine_redaction_boxes
    - Then the correct bounding boxes are returned
    """
    text_rect_map = [
        ("no", (10, 10, 100, 20)),
        ("yes", (10, 40, 200, 20)),
        ("yep", (10, 60, 200, 10)),
        ("negative", (10, 70, 150, 20)),
    ]
    redaction_string = "yes yep"
    expected_boxes = [(10, 40, 200, 20), (10, 60, 200, 10)]
    with patch.object(ImageTextRedactor, "__init__", return_value=None):
        actual_boxes = ImageTextRedactor().examine_redaction_boxes(
            text_rect_map, redaction_string
        )
        assert expected_boxes == actual_boxes


class TestImageTextRedactor(TestImageTextRedactorBase):
    @staticmethod
    def create_config(**kwargs) -> ImageRedactionConfig:
        return ImageRedactionConfig(
            name="config name",
            redactor_type="ImageTextRedaction",
            **kwargs,
        )


class TestRedact(TestImageTextRedactor):
    @staticmethod
    def _create_mock_number_plate_result(text_rect_map_for_image, redaction_strings):
        """Build the dict that _get_number_plate_redactions would return for one image."""
        text_rects_to_redact = [
            (bbox, text)
            for text, bbox in text_rect_map_for_image
            if text in redaction_strings
        ]
        return {
            "text_rects_to_redact": text_rects_to_redact,
            "number_plate_detection_time": 0.01,
            "bbox_time": 0.01,
        }

    @dataclasses.dataclass
    class RedactResult:
        result: ImageRedactionResult
        analyse_images: Mock
        get_number_plate_redactions: Mock

    @classmethod
    def patch_redactor_and_redact(
        cls,
        images,
        text_rect_map,
        redaction_strings=None,
    ):
        if redaction_strings is None:
            redaction_strings = []

        # Build per-image side_effect for _get_number_plate_redactions
        number_plate_side_effect = [
            cls._create_mock_number_plate_result(trm, redaction_strings)
            for trm in text_rect_map
        ]

        with (
            patch.object(ImageTextRedactor, "__init__", return_value=None),
            patch.object(
                ImageTextRedactor,
                "_analyse_images",
                return_value=(
                    list(zip(images, text_rect_map)),
                    0.5 * len(images),
                ),
            ) as mock_analyse_images,
            patch.object(
                ImageTextRedactor,
                "_get_number_plate_redactions",
                side_effect=number_plate_side_effect,
            ) as mock_get_number_plate_redactions,
        ):
            inst = ImageTextRedactor()
            inst.config = cls.create_config(images=images)
            result = inst.redact()

        return cls.RedactResult(
            result=result,
            analyse_images=mock_analyse_images,
            get_number_plate_redactions=mock_get_number_plate_redactions,
        )

    def test_returns_bbox_surrounding_number_plate(self):
        images = [Image.new("RGB", (500, 1000))]
        text_rect_map = [(("AB12", (5, 5, 20, 10)), ("CDE", (25, 5, 35, 10)))]
        redaction_strings = ["AB12 CDE"]

        r = self.patch_redactor_and_redact(
            images, text_rect_map, redaction_strings=redaction_strings
        )

        expected_results = self._create_expected_results(
            images, text_rect_map, redaction_strings=redaction_strings
        )
        self._compare_results(r.result, expected_results)

    def test_no_images_skips_analysis(self):
        """
        - Given I have a config with an empty images list
        - When I call ImageTextRedactor.redact
        - Then it should return an empty ImageRedactionResult without calling AzureVisionUtil
        """
        r = self.patch_redactor_and_redact(images=[], text_rect_map=[])

        r.analyse_images.assert_not_called()

        assert r.result.redaction_results == ()
        assert r.result.run_metrics["total_images_to_analyse"] == 0

    def test_no_text_in_image_skips_number_plate_analysis(self):
        """
        - Given I have images but OCR returns no text for them
        - When I call ImageTextRedactor.redact
        - Then the number plate analysis should not be called and the image should be skipped
        """
        images = [Image.new("RGB", (1000, 1000))]
        text_rect_map = [(("", (10, 10, 50, 50)),)]

        r = self.patch_redactor_and_redact(images=images, text_rect_map=text_rect_map)

        r.get_number_plate_redactions.assert_not_called()
        assert r.result.redaction_results == ()

    def test_no_number_plate_detected(self):
        """
        - Given I have some redaction config (containing two images)
        - When I call ImageRedactor.redact
        - If the underlying analysis tool returns no bounding boxes, then the redaction results should be empty
        """
        images = [Image.new("RGB", (1000, 1000)), Image.new("RGB", (200, 100))]
        text_rect_map = [
            (("text", (10, 10, 50, 50)),),
            (("other text", (30, 30, 50, 50)),),
        ]

        r = self.patch_redactor_and_redact(images, text_rect_map, redaction_strings=[])
        assert r.get_number_plate_redactions.call_count == 2
        assert r.result.redaction_results == ()

    def test_image_analysis_failure_returns_full_image(self):
        """
        - Given I have an image where OCR text detection fails
        - When I call redact
        - Then the full image bounding box should be returned for the failing image
        """
        images = [Image.new("RGB", (500, 1000))]
        text_rect_map = [
            (
                (
                    "Text Detection Failed",
                    (0, 0, images[0].width, images[0].height),
                ),
            )
        ]
        redaction_strings = ["Text Detection Failed"]
        r = self.patch_redactor_and_redact(
            images, text_rect_map, redaction_strings=redaction_strings
        )

        r.get_number_plate_redactions.assert_not_called()

        actual_results = r.result
        assert len(actual_results.redaction_results) == 1

        expected_results = self._create_expected_results(
            images, text_rect_map, redaction_strings=redaction_strings
        )
        self._compare_results(actual_results, expected_results)
