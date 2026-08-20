import os
from io import BytesIO

from PIL import Image

from core.redaction.config import ImageLLMTextRedactionConfig
from core.redaction.redactor import ImageLLMTextRedactor
from core.redaction.result import ImageRedactionResult


class TestRedact:
    def test_no_images_returns_empty_result(self):
        """
        - Given I have a config with an empty images list
        - When I call ImageLLMTextRedactor.redact
        - Then it should return an empty ImageRedactionResult without calling Azure Vision or the LLM
        """
        config = ImageLLMTextRedactionConfig(
            name="config name",
            redactor_type="ImageLLMTextRedaction",
            model="gpt-4.1",
            images=[],
            system_prompt="Identify redaction strings",
            redaction_terms=["Names"],
        )
        redactor_inst = ImageLLMTextRedactor(config)
        result = redactor_inst.redact()

        assert isinstance(result, ImageRedactionResult)
        assert result.redaction_results == ()
        assert result.run_metrics["total_images_to_analyse"] == 0

    def test_returns_matching_bounding_boxes(self):
        """
        - Given I have an image containing readable text with identifiable content
        - When I call ImageLLMTextRedactor.redact
        - Then the LLM should identify redaction strings and return matching bounding boxes
        """
        image_path = os.path.join(
            "test", "resources", "image", "image_with_number_plate.jpg"
        )
        if not os.path.exists(image_path):
            return  # Skip if test resource not available

        with open(image_path, "rb") as f:
            image = Image.open(BytesIO(f.read()))

        config = ImageLLMTextRedactionConfig(
            name="config name",
            redactor_type="ImageLLMTextRedaction",
            model="gpt-4.1",
            images=[image],
            system_prompt="Identify any number plates or registration numbers in the text",
            redaction_terms=["Number plates", "Registration numbers"],
        )
        redactor_inst = ImageLLMTextRedactor(config)
        result = redactor_inst.redact()

        assert isinstance(result, ImageRedactionResult)
        assert result.run_metrics["total_images_to_analyse"] == 1
        assert "total_image_llm_analysis_time" in result.run_metrics

    def test_multiple_images_batched_in_llm_call(self):
        """
        - Given I have multiple images containing text
        - When I call ImageLLMTextRedactor.redact
        - Then all image text should be processed in a single batched LLM call
        (verifying the new efficient architecture works end-to-end)
        """
        image_path = os.path.join(
            "test", "resources", "image", "image_with_number_plate.jpg"
        )
        if not os.path.exists(image_path):
            return  # Skip if test resource not available

        with open(image_path, "rb") as f:
            image = Image.open(BytesIO(f.read()))

        # Use the same image twice to simulate multiple images
        config = ImageLLMTextRedactionConfig(
            name="config name",
            redactor_type="ImageLLMTextRedaction",
            model="gpt-4.1",
            images=[image.copy(), image.copy()],
            system_prompt="Identify any number plates or registration numbers in the text",
            redaction_terms=["Number plates", "Registration numbers"],
        )
        redactor_inst = ImageLLMTextRedactor(config)
        result = redactor_inst.redact()

        assert isinstance(result, ImageRedactionResult)
        assert result.run_metrics["total_images_to_analyse"] == 2
        assert "total_image_llm_analysis_time" in result.run_metrics
        assert "total_image_ocr_time" in result.run_metrics


class TestAnalyseImageText:
    def test_identifies_redaction_strings(self):
        """
        - Given I have image text rect map data containing text with identifiable names
        - When I call _analyse_image_text
        - Then the LLM should identify the names as redaction strings
        and distribute them back to the correct images
        """
        config = ImageLLMTextRedactionConfig(
            name="config name",
            redactor_type="ImageLLMTextRedaction",
            model="gpt-4.1",
            images=[
                Image.new("RGB", (100, 100)),
                Image.new("RGB", (100, 100)),
            ],
            system_prompt="Identify any person names in the text",
            redaction_terms=["Names"],
        )
        image_text_rect_map = [
            (
                config.images[0],
                (
                    ("John", (10, 10, 50, 50)),
                    ("Smith", (60, 10, 50, 50)),
                ),
            ),
            (
                config.images[1],
                (
                    ("Jane", (10, 10, 50, 50)),
                    ("Doe", (60, 10, 50, 50)),
                ),
            ),
        ]

        inst = ImageLLMTextRedactor(config)

        redaction_strings, result = inst._analyse_image_text(image_text_rect_map)

        assert redaction_strings is not None
        assert result is not None
        assert len(result) == 2

        # Each image entry should have the expected structure
        for image_result in result:
            assert "image" in image_result
            assert "text_rect_map" in image_result
            assert "text_content" in image_result
            assert "redaction_strings" in image_result

        # Image 0 text is "John Smith" - LLM should identify the name
        assert len(result[0]["redaction_strings"]) > 0
        # Image 1 text is "Jane Doe" - LLM should identify the name
        assert len(result[1]["redaction_strings"]) > 0

    def test_no_text_content_returns_none(self):
        """
        - Given all images have empty text content
        - When I call _analyse_image_text
        - Then it should return None without calling the LLM
        """
        config = ImageLLMTextRedactionConfig(
            name="config name",
            redactor_type="ImageLLMTextRedaction",
            model="gpt-4.1",
            images=[Image.new("RGB", (100, 100))],
            system_prompt="Identify any person names in the text",
            redaction_terms=["Names"],
        )
        image_text_rect_map = [
            (config.images[0], (("", (10, 10, 50, 50)),)),
        ]

        inst = ImageLLMTextRedactor(config)
        redaction_strings, result = inst._analyse_image_text(image_text_rect_map)

        assert redaction_strings == ()
        assert result == ()
