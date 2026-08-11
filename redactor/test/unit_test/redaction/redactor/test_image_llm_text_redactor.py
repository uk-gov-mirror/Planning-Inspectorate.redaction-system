import dataclasses
from unittest import mock

import pytest
from PIL import Image

from core.redaction.config import ImageLLMTextRedactionConfig
from core.redaction.redactor import ImageLLMTextRedactor
from core.redaction.result import ImageRedactionResult, LLMTextRedactionResult
from core.util.image_analysis import AzureVisionUtil
from core.util.llm_util import LLMUtil
from test.util.util import compare_unashable_lists


def test__image_redactor__get_name():
    assert ImageLLMTextRedactor.get_name() == "ImageLLMTextRedaction"


def test__image_redactor__get_redaction_config_class():
    assert (
        ImageLLMTextRedactor.get_redaction_config_class() == ImageLLMTextRedactionConfig
    )


def test__image_llm_text_redactor___analyse_image_text():
    """
    - Given I have image text rect map data containing text from multiple images
    - When I call _analyse_image_text
    - Then it should batch all unique text chunks into a single LLM call
      and distribute redaction strings back to the correct images
    """
    config = ImageLLMTextRedactionConfig(
        name="config name",
        redactor_type="ImageLLMTextRedaction",
        model="gpt-4.1",
        images=[
            Image.new("RGB", (1000, 1000)),
            Image.new("RGB", (200, 100)),
        ],
        system_prompt="some system prompt",
        redaction_terms=["rule A"],
    )
    image_text_rect_map = [
        (
            config.images[0],
            (
                ("Klingon", (10, 10, 50, 50)),
                ("Romulan", (100, 100, 50, 50)),
            ),
        ),
        (
            config.images[1],
            (("Vulcan", (4, 8, 12, 16)),),
        ),
    ]
    mock_llm_result = LLMTextRedactionResult(
        rule_name="config name",
        run_metrics={},
        redaction_strings=("Klingon", "Romulan", "Vulcan"),
        metadata=LLMTextRedactionResult.LLMResultMetadata(
            input_token_count=80, output_token_count=20, total_token_count=100
        ),
    )
    with (
        mock.patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
        mock.patch.object(LLMUtil, "__init__", return_value=None),
        mock.patch.object(
            LLMUtil, "analyse_text", return_value=mock_llm_result
        ) as mock_analyse_text,
    ):
        inst = ImageLLMTextRedactor()
        inst.config = config
        result = inst._analyse_image_text(image_text_rect_map)

    # LLM should be called once with the combined unique chunks
    mock_analyse_text.assert_called_once()

    # Image 0 contains "Klingon Romulan" so should get both strings
    assert "Klingon" in result[0]["redaction_strings"]
    assert "Romulan" in result[0]["redaction_strings"]
    # Image 1 contains "Vulcan" so should get that string
    assert "Vulcan" in result[1]["redaction_strings"]


def test__image_llm_text_redactor___analyse_image_text__all_empty_text():
    """
    - Given all images have empty text content
    - When I call _analyse_image_text
    - Then it should return None without calling LLMUtil
    """
    config = ImageLLMTextRedactionConfig(
        name="config name",
        redactor_type="ImageLLMTextRedaction",
        model="gpt-4.1",
        images=[Image.new("RGB", (100, 100))],
        system_prompt="some system prompt",
        redaction_terms=["rule A"],
    )
    image_text_rect_map = [
        (config.images[0], (("", (10, 10, 50, 50)),)),
    ]
    with (
        mock.patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
        mock.patch.object(LLMUtil, "__init__", return_value=None) as mock_llm_init,
        mock.patch.object(LLMUtil, "analyse_text") as mock_analyse_text,
    ):
        inst = ImageLLMTextRedactor()
        inst.config = config
        result = inst._analyse_image_text(image_text_rect_map)

    assert result is None
    mock_llm_init.assert_not_called()
    mock_analyse_text.assert_not_called()


def test__image_llm_text_redactor__redact():
    """
    - Given I have three images containing Star Trek species names,
      the names (Klingon, Vulcan, Romulan) are marked as sensitive
    - When I call redact
    - Then only the bounding boxes for the sensitive names should be returned,
      alongside metadata for the corresponding image
    """
    config = ImageLLMTextRedactionConfig(
        name="config name",
        redactor_type="ImageLLMTextRedaction",
        model="gpt-4.1",
        images=[
            Image.new("RGB", (1000, 1000)),
            Image.new("RGB", (200, 100)),
            Image.new("RGB", (1000, 1000)),
        ],
        system_prompt="some system prompt",
        redaction_terms=[
            "rule A",
            "rule B",
            "rule C",
        ],
    )
    detect_text_in_images_return_value = (
        (
            config.images[0],
            (
                ("Klingon", (10, 10, 50, 50)),
                ("Romulan", (100, 100, 50, 50)),
                ("Jem'Hadar", (1, 2, 3, 4)),
            ),
        ),
        (
            config.images[1],
            (("Cardassian", (30, 30, 50, 50)), ("Vulcan", (4, 8, 12, 16))),
        ),
        (
            config.images[2],
            (
                ("Klingon", (10, 10, 50, 50)),
                ("Klingon", (100, 100, 50, 50)),
            ),
        ),
    )
    # Mock _analyse_image_text to return results that assign redaction strings to images
    mock_analyse_result = [
        {
            "image": config.images[0],
            "text_rect_map": (
                ("Klingon", (10, 10, 50, 50)),
                ("Romulan", (100, 100, 50, 50)),
                ("Jem'Hadar", (1, 2, 3, 4)),
            ),
            "text_content": "Klingon Romulan Jem'Hadar",
            "text_chunks": ["Klingon Romulan Jem'Hadar"],
            "redaction_strings": ["Klingon", "Romulan"],
        },
        {
            "image": config.images[1],
            "text_rect_map": (
                ("Cardassian", (30, 30, 50, 50)),
                ("Vulcan", (4, 8, 12, 16)),
            ),
            "text_content": "Cardassian Vulcan",
            "text_chunks": ["Cardassian Vulcan"],
            "redaction_strings": ["Vulcan"],
        },
        {
            "image": config.images[2],
            "text_rect_map": (
                ("Klingon", (10, 10, 50, 50)),
                ("Klingon", (100, 100, 50, 50)),
            ),
            "text_content": "Klingon Klingon",
            "text_chunks": ["Klingon Klingon"],
            "redaction_strings": ["Klingon"],
        },
    ]
    expected_results = ImageRedactionResult(
        rule_name="config name",
        run_metrics={},
        redaction_results=(
            ImageRedactionResult.Result(
                image_dimensions=(1000, 1000),
                source_image=config.images[0],
                redaction_boxes=(
                    (10, 10, 50, 50),
                    (100, 100, 50, 50),
                ),
                names=("Klingon", "Romulan"),
            ),
            ImageRedactionResult.Result(
                image_dimensions=(200, 100),
                source_image=config.images[1],
                redaction_boxes=((4, 8, 12, 16),),
                names=("Vulcan",),
            ),
            ImageRedactionResult.Result(
                image_dimensions=(1000, 1000),
                source_image=config.images[2],
                redaction_boxes=(
                    (10, 10, 50, 50),
                    (100, 100, 50, 50),
                ),
                names=("Klingon", "Klingon"),
            ),
        ),
    )
    with (
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_images",
            return_value=(detect_text_in_images_return_value, 0.5),
        ),
        mock.patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_image_text",
            return_value=mock_analyse_result,
        ),
    ):
        inst = ImageLLMTextRedactor()
        inst.config = config
        actual_results = inst.redact()
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")
        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")
        assert cleaned_expected_results == cleaned_actual_results


def test__image_llm_text_redactor__redact__no_images_skips_analysis():
    """
    - Given I have a config with an empty images list
    - When I call ImageLLMTextRedactor.redact
    - Then it should return an empty ImageRedactionResult without calling AzureVisionUtil
    """
    config = ImageLLMTextRedactionConfig(
        name="config name",
        redactor_type="ImageLLMTextRedaction",
        model="gpt-4.1",
        images=[],
        system_prompt="some system prompt",
        redaction_terms=["rule A"],
    )
    with (
        mock.patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_images",
            return_value=([], 0.0),
        ),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_image_text",
        ) as mock_analyse_image_text,
    ):
        inst = ImageLLMTextRedactor()
        inst.config = config
        actual_results = inst.redact()

    mock_analyse_image_text.assert_not_called()
    assert actual_results.rule_name == "config name"
    assert actual_results.redaction_results == ()
    assert actual_results.run_metrics["total_images_to_analyse"] == 0
    for metric in [
        "total_image_ocr_time",
        "total_image_text_analysis_time",
    ]:
        assert actual_results.run_metrics[metric] == pytest.approx(0.0, abs=1e-2)


def test__image_llm_text_redactor__redact__no_text_in_images_skips_llm():
    """
    - Given I have images but OCR returns no text for any of them
    - When I call ImageLLMTextRedactor.redact
    - Then _analyse_image_text should return None and no redaction results are produced
    """
    config = ImageLLMTextRedactionConfig(
        name="config name",
        redactor_type="ImageLLMTextRedaction",
        model="gpt-4.1",
        images=[
            Image.new("RGB", (1000, 1000)),
        ],
        system_prompt="some system prompt",
        redaction_terms=["rule A"],
    )
    detect_text_in_images_return_value = (
        (config.images[0], (("", (10, 10, 50, 50)),)),
    )
    with (
        mock.patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_images",
            return_value=(detect_text_in_images_return_value, 0.1),
        ),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_image_text",
            return_value=None,
        ) as mock_analyse_image_text,
    ):
        inst = ImageLLMTextRedactor()
        inst.config = config
        actual_results = inst.redact()

    mock_analyse_image_text.assert_called_once()
    assert actual_results.redaction_results == ()


def test__image_llm_text_redactor__no_redaction_strings():
    """
    - Given I have some redaction config (containing two images)
    - When I call ImageRedactor.redact
    - If the underlying analysis tool returns no bounding boxes, then the redaction results should be empty
    """
    images = [Image.new("RGB", (1000, 1000)), Image.new("RGB", (200, 100))]
    config = ImageLLMTextRedactionConfig(
        name="config name",
        redactor_type="ImageLLMTextRedaction",
        model="gpt-4.1",
        images=images,
        system_prompt="some system prompt",
        redaction_terms=["rule A"],
    )
    detect_text_in_images_return_value = [
        (images[0], (("text", (10, 10, 50, 50)),)),
        (images[1], (("other text", (30, 30, 50, 50)),)),
    ]
    analyse_image_text_result = [
        {
            "image": images[0],
            "text_rect_map": (10, 10, 50, 50),
            "text_content": "text",
            "text_chunks": ["text"],
            "redaction_strings": [],
        },
        {
            "image": images[1],
            "text_rect_map": (30, 30, 50, 50),
            "text_content": "other text",
            "text_chunks": ["other text"],
            "redaction_strings": [],
        },
    ]
    expected_results = ImageRedactionResult(
        rule_name="config name",
        run_metrics={},
        redaction_results=(),
    )
    with (
        mock.patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_images",
            return_value=(detect_text_in_images_return_value, 0.0),
        ),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_image_text",
            return_value=analyse_image_text_result,
        ) as mock_analyse_image_text,
    ):
        inst = ImageLLMTextRedactor()
        inst.config = config
        actual_results = inst.redact()
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")
        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")
        assert cleaned_expected_results == cleaned_actual_results
        mock_analyse_image_text.assert_called_once()


def test__image_llm_text_redactor__redact__with_image_analysis_failure():
    """
    - Given I have two images which we imagine contains some text
    - When I call redact and one of the image analysis raises an exception when performing OCR
    - Then the full image bounding box should be returned for the failing image
    """
    config = ImageLLMTextRedactionConfig(
        name="config name",
        redactor_type="ImageLLMTextRedaction",
        model="gpt-4.1",
        images=[
            Image.new("RGB", (1000, 1000)),
            Image.new("RGB", (200, 100)),
            Image.new("RGB", (500, 500)),
        ],
        system_prompt="some system prompt",
        redaction_terms=[
            "rule A",
            "rule B",
            "rule C",
        ],
    )

    def mock_analyse_images(inst):
        images = inst.config.images
        return (
            (
                (
                    images[0],
                    (
                        (
                            "TEXT DETECTION FAILED",
                            (0, 0, images[0].width, images[0].height),
                        ),
                    ),
                ),
                (
                    images[1],
                    (("Cardassian", (30, 30, 50, 50)), ("Vulcan", (4, 8, 12, 16))),
                ),
                (
                    images[2],
                    (("Klingon", (10, 10, 50, 50)), ("Klingon", (100, 100, 50, 50))),
                ),
            ),
            0.5,
        )

    expected_results = ImageRedactionResult(
        rule_name="config name",
        run_metrics={},
        redaction_results=(
            ImageRedactionResult.Result(
                image_dimensions=(1000, 1000),
                source_image=config.images[0],
                redaction_boxes=((0, 0, 1000, 1000),),
                names=("Text Detection Failed",),
            ),
            ImageRedactionResult.Result(
                image_dimensions=(200, 100),
                source_image=config.images[1],
                redaction_boxes=((4, 8, 12, 16),),
                names=("Vulcan",),
            ),
            ImageRedactionResult.Result(
                image_dimensions=(500, 500),
                source_image=config.images[2],
                redaction_boxes=(
                    (10, 10, 50, 50),
                    (100, 100, 50, 50),
                ),
                names=("Klingon", "Klingon"),
            ),
        ),
    )
    mock_text_analysis_result = (
        {
            "image": config.images[0],
            "text_rect_map": (("TEXT DETECTION FAILED", (0, 0, 1000, 1000)),),
            "text_content": "TEXT DETECTION FAILED",
            "text_chunks": ["TEXT DETECTION FAILED"],
            "redaction_strings": [],
        },
        {
            "image": config.images[1],
            "text_rect_map": (
                ("Cardassian", (30, 30, 50, 50)),
                ("Vulcan", (4, 8, 12, 16)),
            ),
            "text_content": "Cardassian Vulcan",
            "text_chunks": ["Cardassian Vulcan"],
            "redaction_strings": ["Vulcan"],
        },
        {
            "image": config.images[2],
            "text_rect_map": (
                ("Klingon", (10, 10, 50, 50)),
                ("Klingon", (100, 100, 50, 50)),
            ),
            "text_content": "Klingon Klingon",
            "text_chunks": ["Klingon Klingon"],
            "redaction_strings": ["Klingon"],
        },
    )
    mock_redaction_boxes_result = [
        [(4, 8, 12, 16)],
        [(10, 10, 50, 50), (100, 100, 50, 50)],
    ]

    with (
        mock.patch.object(AzureVisionUtil, "__init__", return_value=None),
        mock.patch.object(ImageLLMTextRedactor, "_analyse_images", mock_analyse_images),
        mock.patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
        mock.patch.object(
            ImageLLMTextRedactor,
            "_analyse_image_text",
            return_value=mock_text_analysis_result,
        ),
        mock.patch.object(
            ImageLLMTextRedactor,
            "examine_redaction_boxes",
            side_effect=mock_redaction_boxes_result,
        ),
    ):
        inst = ImageLLMTextRedactor()
        inst.config = config
        actual_results = inst.redact()
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")
        expected_redaction_boxes = cleaned_expected_results.pop("redaction_results")
        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")
        actual_redaction_boxes = cleaned_actual_results.pop("redaction_results")
        assert cleaned_expected_results == cleaned_actual_results
        compare_unashable_lists(expected_redaction_boxes, actual_redaction_boxes)
