import dataclasses
from unittest.mock import Mock, patch

from PIL import Image

from core.redaction.config import ImageLLMTextRedactionConfig
from core.redaction.redactor import ImageLLMTextRedactor
from core.redaction.result import ImageRedactionResult, LLMTextRedactionResult
from core.util.llm_util import LLMUtil
from test.unit_test.redaction.redactor.util import TestImageTextRedactorBase


def test_get_name():
    assert ImageLLMTextRedactor.get_name() == "ImageLLMTextRedaction"


def test_get_redaction_config_class():
    assert (
        ImageLLMTextRedactor.get_redaction_config_class() == ImageLLMTextRedactionConfig
    )


class TestImageLLMTextRedactor(TestImageTextRedactorBase):
    @staticmethod
    def create_config(**kwargs) -> ImageLLMTextRedactionConfig:
        return ImageLLMTextRedactionConfig(
            name="config name",
            redactor_type="ImageLLMTextRedaction",
            model="gpt-4.1",
            system_prompt="some system prompt",
            redaction_terms=["rule A"],
            **kwargs,
        )

    @staticmethod
    def _create_mock_analyse_image_text_result(
        images, text_rect_map, redaction_strings=None
    ):
        if redaction_strings is None:
            redaction_strings = ()
        return (
            redaction_strings,
            [
                {
                    "image": image,
                    "text_rect_map": map,
                    "text_content": " ".join(text for text, _ in map),
                    "text_chunks": [" ".join(text for text, _ in map)],
                    "redaction_strings": [
                        text for text, _ in map if text in redaction_strings
                    ],
                }
                for image, map in zip(images, text_rect_map)
            ],
        )


class TestCreateRedactionResult(TestImageLLMTextRedactor):
    def test_returns_image_redaction_result(self):
        image = Image.new("RGB", (1000, 1000))
        text_rect_map = [
            (
                ("Klingon", (10, 10, 50, 50)),
                ("Romulan", (100, 100, 50, 50)),
            ),
        ]
        redaction_strings = ["Klingon"]
        _, image_results = self._create_mock_analyse_image_text_result(
            [image], text_rect_map, redaction_strings=redaction_strings
        )
        actual_result, _ = ImageLLMTextRedactor._create_redaction_result(
            image_results[0]
        )

        assert actual_result.image_dimensions == (1000, 1000)
        assert actual_result.redaction_boxes == ((10, 10, 50, 50),)
        assert actual_result.names == ("Klingon",)


class TestAnalyseImageText(TestImageLLMTextRedactor):
    def test_text_and_redistributes_redaction_strings_to_images(self):
        """
        - Given I have image text rect map data containing text from multiple images
        - When I call _analyse_image_text
        - Then it should batch all unique text chunks into a single LLM call
        and distribute redaction strings back to the correct images
        """
        images = [
            Image.new("RGB", (1000, 1000)),
            Image.new("RGB", (200, 100)),
        ]
        config = self.create_config(images=images)
        image_text_rect_map = [
            (
                images[0],
                (
                    ("Klingon", (10, 10, 50, 50)),
                    ("Romulan", (100, 100, 50, 50)),
                ),
            ),
            (
                images[1],
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
            patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
            patch.object(LLMUtil, "__init__", return_value=None),
            patch.object(
                LLMUtil, "analyse_text", return_value=mock_llm_result
            ) as mock_analyse_text,
        ):
            inst = ImageLLMTextRedactor()
            inst.config = config
            _, image_text_content = inst._analyse_image_text(image_text_rect_map)

        # LLM should be called once with the combined unique chunks
        mock_analyse_text.assert_called_once()

        # Image 0 contains "Klingon Romulan" so should get both strings
        assert "Klingon" in image_text_content[0]["redaction_strings"]
        assert "Romulan" in image_text_content[0]["redaction_strings"]
        # Image 1 contains "Vulcan" so should get that string
        assert "Vulcan" in image_text_content[1]["redaction_strings"]

    def test_no_llm_analysis_with_empty_text_content(self):
        """
        - Given all images have empty text content
        - When I call _analyse_image_text
        - Then it should return None without calling LLMUtil
        """
        images = [Image.new("RGB", (100, 100))]
        config = self.create_config(images=images)
        image_text_rect_map = [
            (images[0], (("", (10, 10, 50, 50)),)),
        ]
        with (
            patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
            patch.object(LLMUtil, "__init__", return_value=None) as mock_llm_init,
            patch.object(LLMUtil, "analyse_text") as mock_analyse_text,
        ):
            inst = ImageLLMTextRedactor()
            inst.config = config
            redaction_strings, image_text_content = inst._analyse_image_text(
                image_text_rect_map
            )

        assert redaction_strings == ()
        assert image_text_content == ()
        mock_llm_init.assert_not_called()
        mock_analyse_text.assert_not_called()


class TestRedact(TestImageLLMTextRedactor):
    @dataclasses.dataclass
    class RedactResult:
        result: ImageRedactionResult
        analyse_images: Mock
        analyse_image_text: Mock

    @classmethod
    def patch_redactor_and_redact(
        cls,
        images,
        text_rect_map,
        redaction_strings=None,
        rendered_images=None,
    ):
        if redaction_strings is None:
            redaction_strings = []

        all_images = images + (
            [rm.image for rm in rendered_images] if rendered_images else []
        )

        with (
            patch.object(ImageLLMTextRedactor, "__init__", return_value=None),
            patch.object(
                ImageLLMTextRedactor,
                "_analyse_images",
                return_value=(
                    list(zip(images, text_rect_map)),
                    0.5 * len(images),
                ),
            ) as mock_analyse_images,
            patch.object(
                ImageLLMTextRedactor,
                "_analyse_image_text",
                return_value=cls._create_mock_analyse_image_text_result(
                    all_images,
                    text_rect_map,
                    redaction_strings=tuple(redaction_strings),
                ),
            ) as mock_analyse_image_text,
        ):
            inst = ImageLLMTextRedactor()
            inst.config = cls.create_config(
                images=images, rendered_images=rendered_images
            )
            result = inst.redact()

        return cls.RedactResult(
            result=result,
            analyse_images=mock_analyse_images,
            analyse_image_text=mock_analyse_image_text,
        )

    def test_returns_bbox_metadata_for_redaction_strings(self):
        """
        - Given I have three images containing Star Trek species names,
        the names (Klingon, Vulcan, Romulan) are marked as sensitive
        - When I call redact
        - Then only the bounding boxes for the sensitive names should be returned,
        alongside metadata for the corresponding image
        """
        images = [
            Image.new("RGB", (1000, 1000)),
            Image.new("RGB", (200, 100)),
            Image.new("RGB", (1000, 1000)),
        ]
        text_rect_map = [
            (
                ("Klingon", (10, 10, 50, 50)),
                ("Romulan", (100, 100, 50, 50)),
                ("Jem'Hadar", (1, 2, 3, 4)),
            ),
            (("Cardassian", (30, 30, 50, 50)), ("Vulcan", (4, 8, 12, 16))),
            (
                ("Klingon", (10, 10, 50, 50)),
                ("Klingon", (100, 100, 50, 50)),
            ),
        ]
        # Mock _analyse_image_text to return results that assign redaction strings to images
        redaction_strings = ["Klingon", "Romulan", "Vulcan"]

        actual_results = self.patch_redactor_and_redact(
            images, text_rect_map, redaction_strings
        )

        expected_results = self._create_expected_results(
            images, text_rect_map, redaction_strings=redaction_strings
        )

        self._compare_results(actual_results.result, expected_results)

    def test_no_images_skips_analysis(self):
        """
        - Given I have a config with an empty images list
        - When I call ImageLLMTextRedactor.redact
        - Then it should return an empty ImageRedactionResult without calling AzureVisionUtil
        """
        r = self.patch_redactor_and_redact(images=[], text_rect_map=[])

        r.analyse_images.assert_not_called()
        actual_results = r.result

        assert actual_results.run_metrics["total_images_to_analyse"] == 0
        self._compare_results(
            actual_results, self._create_expected_results([], [], redaction_strings=[])
        )

    def test_no_text_in_images_skips_llm(self):
        """
        - Given I have images but OCR returns no text for any of them
        - When I call ImageLLMTextRedactor.redact
        - Then _analyse_image_text should return None and no redaction results are produced
        """
        images = [Image.new("RGB", (1000, 1000))]
        text_rect_map = [(("", (10, 10, 50, 50)),)]
        r = self.patch_redactor_and_redact(images, text_rect_map)

        r.analyse_images.assert_called_once()
        r.analyse_image_text.assert_called_once()

        assert r.result.redaction_results == ()

    def test_no_redaction_strings_creates_empty_result(self):
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
        r = self.patch_redactor_and_redact(images, text_rect_map)

        r.analyse_image_text.assert_called_once()

        assert r.result.redaction_results == ()

    def test_image_analysis_failure_returns_full_image(self):
        """
        - Given I have two images which we imagine contains some text
        - When I call redact and one of the image analysis raises an exception when performing OCR
        - Then the full image bounding box should be returned for the failing image
        """
        images = [
            Image.new("RGB", (1000, 1000)),
            Image.new("RGB", (200, 100)),
            Image.new("RGB", (500, 500)),
        ]
        text_rect_map = [
            (
                ("Text Detection Failed", (0, 0, images[0].width, images[0].height)),
            ),  # mock image analysis failure
            (("Cardassian", (30, 30, 50, 50)), ("Vulcan", (4, 8, 12, 16))),
            (
                ("Klingon", (10, 10, 50, 50)),
                ("Klingon", (100, 100, 50, 50)),
            ),
        ]

        redaction_strings = ["Vulcan", "Klingon"]
        r = self.patch_redactor_and_redact(images, text_rect_map, redaction_strings)
        actual_results = r.result

        expected_results = self._create_expected_results(
            images, text_rect_map, redaction_strings=redaction_strings
        )
        self._compare_results(actual_results, expected_results)
