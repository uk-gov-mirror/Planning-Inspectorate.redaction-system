import os
from io import BytesIO

from PIL import Image

from core.redaction.config import ImageRedactionConfig
from core.redaction.redactor import ImageRedactor
from core.redaction.result import ImageRedactionResult


class TestRedact:
    def test_no_images_returns_empty_result(self):
        """
        - Given I have a config with an empty images list
        - When I call ImageRedactor.redact
        - Then it should return an empty ImageRedactionResult without calling Azure Vision or the LLM
        """
        config = ImageRedactionConfig(
            name="config name",
            redactor_type="ImageRedaction",
            images=[],
        )
        redactor_inst = ImageRedactor(config)
        result = redactor_inst.redact()

        assert isinstance(result, ImageRedactionResult)
        assert result.redaction_results == ()
        assert result.run_metrics["total_images_to_analyse"] == 0

    def test_returns_matching_bounding_boxes(self):
        with open(
            os.path.join("test", "resources", "image", "image_with_signature.png"),
            "rb",
        ) as f:
            image = Image.open(BytesIO(f.read()))

        config = ImageRedactionConfig(
            name="config name",
            redactor_type="ImageRedaction",
            images=[image],
        )
        redactor_inst = ImageRedactor(config)
        result = redactor_inst.redact()

        assert isinstance(result, ImageRedactionResult)
        assert result.run_metrics["total_images_to_analyse"] == 1
        assert len(result.redaction_results) == 1

        redaction_results = result.redaction_results[0]
        assert redaction_results.image_dimensions == (image.width, image.height)
        assert redaction_results.source_image == image
        assert redaction_results.redaction_boxes == ((688, 620, 872, 697),)
        assert redaction_results.names == ("Signature Detected",)
