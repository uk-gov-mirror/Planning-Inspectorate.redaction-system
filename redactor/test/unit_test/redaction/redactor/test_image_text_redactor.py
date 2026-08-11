import dataclasses
from unittest import mock

import pytest
from PIL import Image

from core.redaction.config import ImageRedactionConfig
from core.redaction.redactor import ImageTextRedactor
from core.redaction.result import ImageRedactionResult
from core.util.image_analysis import AzureVisionUtil
from test.util.util import compare_unashable_lists


def test__image_text_redactor__get_name():
    assert ImageTextRedactor.get_name() == "ImageTextRedaction"


def test__image_text_redactor__get_redaction_config_class():
    assert ImageTextRedactor.get_redaction_config_class() == ImageRedactionConfig


def test__image_text_redactor__detect_number_plates():
    """
    - Given I have some text containing UK number plates
    - When I call ImageTextRedactor.detect_number_plates
    - Then the correct number plates should be returned
    """
    valid_number_plates = [
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
    ]
    for variant in valid_number_plates:
        assert variant in ImageTextRedactor.detect_number_plates(variant)

    invalid_number_plates = [
        "something AB12 CDE",  # Current format with preceding text
        "AB12 CDE something",  # Current format with following text
    ]
    for variant in invalid_number_plates:
        result = ImageTextRedactor.detect_number_plates(variant)
        assert variant not in result
        assert "AB12 CDE" in result


def test__image_text_redactor__examine_redaction_boxes():
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
    with mock.patch.object(ImageTextRedactor, "__init__", return_value=None):
        actual_boxes = ImageTextRedactor().examine_redaction_boxes(
            text_rect_map, redaction_string
        )
        assert expected_boxes == actual_boxes


def test__image_text_redactor__redact():
    """
    - Given I have two images which we imagine contains 5 different Star Trek species names,
      the names (Klingon, Vulcan, Romulan) are marked as sensitive
    - When I call redact
    - Then only the bounding boxes for the sensitive names should be returned,
    alongside metadata for the corresponding image
    """
    config = ImageRedactionConfig(
        name="config name",
        redactor_type="ImageTextRedaction",
        images=[
            Image.new("RGB", (500, 1000)),
        ],
    )
    text_rect_map = (("AB12", (5, 5, 20, 10)), ("CDE", (25, 5, 35, 10)))
    expected_results = ImageRedactionResult(
        rule_name="config name",
        run_metrics="",
        redaction_results=(
            ImageRedactionResult.Result(
                image_dimensions=(500, 1000),
                source_image=config.images[0],
                redaction_boxes=tuple(x[1] for x in text_rect_map),
                names=tuple("AB12 CDE" for _ in text_rect_map),
            ),
        ),
    )
    with (
        mock.patch.object(ImageTextRedactor, "__init__", return_value=None),
        mock.patch.object(AzureVisionUtil, "__init__", return_value=None),
        mock.patch.object(AzureVisionUtil, "detect_text", return_value=text_rect_map),
        mock.patch.object(
            ImageTextRedactor,
            "detect_number_plates",
            return_value=("AB12 CDE",),
        ),
    ):
        inst = ImageTextRedactor()
        inst.config = config
        actual_results = inst.redact()
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")
        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")
        assert cleaned_expected_results == cleaned_actual_results


def test__image_text_redactor__redact__no_images_skips_analysis():
    """
    - Given I have a config with an empty images list
    - When I call ImageTextRedactor.redact
    - Then it should return an empty ImageRedactionResult without calling AzureVisionUtil
    """
    config = ImageRedactionConfig(
        name="config name",
        redactor_type="ImageTextRedaction",
        images=[],
    )
    with (
        mock.patch.object(ImageTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageTextRedactor,
            "_analyse_images",
            return_value=([], 0.0),
        ),
    ):
        inst = ImageTextRedactor()
        inst.config = config
        actual_results = inst.redact()

    assert actual_results.rule_name == "config name"
    assert actual_results.redaction_results == ()
    assert actual_results.run_metrics["total_images_to_analyse"] == 0
    for metric in [
        "total_image_ocr_time",
        "total_image_text_analysis_time",
    ]:
        assert actual_results.run_metrics[metric] == pytest.approx(0.0, abs=1e-2)


def test__image_text_redactor__redact__no_text_in_image_skips_number_plate_analysis():
    """
    - Given I have images but OCR returns no text for them
    - When I call ImageTextRedactor.redact
    - Then the number plate analysis should not be called and the image should be skipped
    """
    config = ImageRedactionConfig(
        name="config name",
        redactor_type="ImageTextRedaction",
        images=[
            Image.new("RGB", (1000, 1000)),
        ],
    )
    # OCR returns empty text for the image
    detect_text_in_images_return_value = [
        (config.images[0], (("", (10, 10, 50, 50)),)),
    ]
    with (
        mock.patch.object(
            ImageTextRedactor,
            "_analyse_images",
            return_value=(detect_text_in_images_return_value, 0.0),
        ),
        mock.patch.object(
            ImageTextRedactor,
            "_get_number_plate_redactions",
        ) as mock_get_number_plate_redactions,
        mock.patch.object(ImageTextRedactor, "__init__", return_value=None),
    ):
        inst = ImageTextRedactor()
        inst.config = config
        actual_results = inst.redact()

    # LLM analysis should not have been called since text_content is empty
    mock_get_number_plate_redactions.assert_not_called()
    # No results since the image was skipped
    assert actual_results.redaction_results == ()


def test__image_redactor__no_number_plate_detected():
    """
    - Given I have some redaction config (containing two images)
    - When I call ImageRedactor.redact
    - If the underlying analysis tool returns no bounding boxes, then the redaction results should be empty
    """
    images = [Image.new("RGB", (1000, 1000)), Image.new("RGB", (200, 100))]
    config = ImageRedactionConfig(
        name="some image redaction config",
        redactor_type="ImageRedaction",
        images=images,
    )
    detect_text_in_images_return_value = [
        (images[0], (("text", (10, 10, 50, 50)),)),
        (images[1], (("other text", (30, 30, 50, 50)),)),
    ]
    number_plate_redactions_result = ((), 0.0, 0.0)
    expected_results = ImageRedactionResult(
        rule_name="some image redaction config",
        run_metrics={},
        redaction_results=(),
    )
    with (
        mock.patch.object(AzureVisionUtil, "__init__", return_value=None),
        mock.patch.object(ImageTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageTextRedactor,
            "_analyse_images",
            return_value=(detect_text_in_images_return_value, 0.0),
        ),
        mock.patch.object(
            ImageTextRedactor,
            "_get_number_plate_redactions",
            return_value=number_plate_redactions_result,
        ) as mock_get_number_plate_redactions,
    ):
        inst = ImageTextRedactor()
        inst.config = config
        actual_results = inst.redact()
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")
        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")
        assert cleaned_expected_results == cleaned_actual_results
        assert mock_get_number_plate_redactions.call_count == 2


def test__image_text_redactor__redact__with_analysis_failure():
    """
    - Given I have two images which we imagine contains some text
    - When I call redact and one of the images raises an exception during analysis
    - Then only the bounding boxes for the sensitive names should be returned,
    alongside metadata for the corresponding image
    """
    config = ImageRedactionConfig(
        name="config name",
        redactor_type="ImageTextRedaction",
        images=[
            Image.new("RGB", (500, 1000)),
        ],
    )
    expected_results = ImageRedactionResult(
        rule_name="config name",
        run_metrics="",
        redaction_results=(
            ImageRedactionResult.Result(
                image_dimensions=(500, 1000),
                source_image=config.images[0],
                redaction_boxes=(
                    (0, 0, config.images[0].width, config.images[0].height),
                ),
                names=("Text Detection Failed",),
            ),
        ),
    )
    mock_analyse_image_return_value = (
        [
            (
                config.images[0],
                (
                    (
                        "TEXT DETECTION FAILED",
                        (0, 0, config.images[0].width, config.images[0].height),
                    ),
                ),
            ),
        ],
        0.0,
    )
    with (
        mock.patch.object(ImageTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageTextRedactor,
            "_analyse_images",
            return_value=mock_analyse_image_return_value,
        ),
        mock.patch.object(
            ImageTextRedactor,
            "_get_number_plate_redactions",
        ) as mock_get_number_plate_redactions,
    ):
        inst = ImageTextRedactor()
        inst.config = config
        actual_results = inst.redact()
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")
        expected_redaction_boxes = cleaned_expected_results.pop("redaction_results")
        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")
        actual_redaction_boxes = cleaned_actual_results.pop("redaction_results")

        assert cleaned_expected_results == cleaned_actual_results
        mock_get_number_plate_redactions.assert_not_called()
        compare_unashable_lists(expected_redaction_boxes, actual_redaction_boxes)
