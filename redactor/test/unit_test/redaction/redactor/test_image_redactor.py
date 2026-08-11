import dataclasses
from unittest import mock

from PIL import Image

from core.redaction.config import ImageRedactionConfig
from core.redaction.redactor import ImageRedactor
from core.redaction.result import ImageRedactionResult
from core.util.image_analysis import AzureVisionUtil
from test.util.util import compare_unashable_lists


class TestGetName:
    def test_returns_name(self):
        assert ImageRedactor.get_name() == "ImageRedaction"


class TestGetRedactionConfigClass:
    def test_returns_redaction_config_class(self):
        assert ImageRedactor.get_redaction_config_class() == ImageRedactionConfig


class ImageAnalysisError(Exception):
    pass


class TestRedactBase:
    RULE_NAME = "some image redaction config"

    def setup_image_redactor(self, images):
        config = ImageRedactionConfig(
            name=self.RULE_NAME,
            redactor_type="ImageRedaction",
            images=images,
            confidence_thresholds=ImageRedactionConfig.ConfidenceThresholdConfig(),
        )
        with (
            mock.patch.object(ImageRedactor, "__init__", return_value=None),
        ):
            inst = ImageRedactor()
            inst.config = config

            return inst


class TestRedactFaces(TestRedactBase):
    def patch_azure_vision_util(
        self, detect_faces_return_value, detect_faces_side_effects=None
    ):
        mock_avu = mock.Mock(spec=AzureVisionUtil)
        mock_avu.detect_faces_in_images.return_value = detect_faces_return_value
        mock_avu.detect_faces_in_images.side_effect = detect_faces_side_effects

        return mock.patch(
            "core.redaction.redactor.AzureVisionUtil",
            return_value=mock_avu,
        )

    def test_returns_redaction_results(self):
        """
        - Given I have some redaction config (containing two images)
        - When I call ImageRedactor.redact
        - If the underlying analysis tool returns three bounding boxes, then these should be returned alongside metedata about the analysed image
        """
        images = [Image.new("RGB", (1000, 1000)), Image.new("RGB", (200, 100))]
        detect_faces_result = [
            (images[0], ((10, 10, 50, 50), (100, 100, 50, 50))),
            (images[1], ((30, 30, 50, 50),)),
        ]
        inst = self.setup_image_redactor(images)

        expected_results = ImageRedactionResult(
            rule_name=self.RULE_NAME,
            run_metrics={},
            redaction_results=tuple(
                ImageRedactionResult.Result(
                    source_image=image,
                    image_dimensions=(image.width, image.height),
                    redaction_boxes=faces_detected,
                    names=tuple("Face Detected" for _ in faces_detected),
                )
                for i, (image, faces_detected) in enumerate(detect_faces_result)
            ),
        )
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")

        with self.patch_azure_vision_util(detect_faces_result):
            actual_results = inst.redact()

        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")

        assert cleaned_expected_results == cleaned_actual_results

    def test_no_images_skips_analysis(self):
        """
        - Given I have a config with an empty images list
        - When I call ImageRedactor.redact
        - Then it should return an empty ImageRedactionResult without calling AzureVisionUtil
        """
        inst = self.setup_image_redactor(images=[])

        with self.patch_azure_vision_util(
            detect_faces_return_value=[]
        ) as mock_azure_vision_util:
            actual_results = inst.redact()

        mock_azure_vision_util.assert_not_called()

        assert actual_results.rule_name == self.RULE_NAME
        assert actual_results.redaction_results == ()
        assert actual_results.run_metrics == {}

    def test_no_faces_detected(self):
        """
        - Given I have some redaction config (containing two images)
        - When I call ImageRedactor.redact
        - If the underlying analysis tool returns no bounding boxes, then the redaction results should be empty
        """
        images = [Image.new("RGB", (1000, 1000)), Image.new("RGB", (200, 100))]
        inst = self.setup_image_redactor(images=images)

        expected_results = ImageRedactionResult(
            rule_name=self.RULE_NAME,
            run_metrics={},
            redaction_results=(),
        )
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")

        detect_faces_result = [(images[0], ()), (images[1], ())]
        with self.patch_azure_vision_util(detect_faces_result):
            actual_results = inst.redact()

        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")

        assert cleaned_expected_results == cleaned_actual_results

    def test_with_analysis_failure(self):
        """
        - Given I have some redaction config (containing two images)
        - When I call ImageRedactor.redact
        - If the underlying analysis fails for one of the images, then the whole failed image should be redacted
        """
        images = [Image.new("RGB", (1000, 1000)), Image.new("RGB", (200, 100))]
        inst = self.setup_image_redactor(images=images)

        def detect_faces_side_effects(images, confidence):
            return [
                (
                    images[0],
                    ((0, 0, images[0].width, images[0].height),),
                ),  # Full image redaction for exception
                (images[1], ((30, 30, 50, 50),)),  # Normal detection for second image
            ]

        expected_results = ImageRedactionResult(
            rule_name=self.RULE_NAME,
            run_metrics={},
            redaction_results=(
                ImageRedactionResult.Result(
                    source_image=images[0],
                    image_dimensions=(images[0].width, images[0].height),
                    # Should contain a single redaction box set to the image's bounds
                    redaction_boxes=((0, 0, images[0].width, images[0].height),),
                    names=("Face Detection Failed",),
                ),
                ImageRedactionResult.Result(
                    source_image=images[1],
                    image_dimensions=(images[1].width, images[1].height),
                    redaction_boxes=((30, 30, 50, 50),),
                    names=("Face Detected",),
                ),
            ),
        )
        cleaned_expected_results = dataclasses.asdict(expected_results)
        cleaned_expected_results.pop("run_metrics")
        expected_redaction_boxes = cleaned_expected_results.pop("redaction_results")

        with self.patch_azure_vision_util(
            detect_faces_return_value=None,
            detect_faces_side_effects=detect_faces_side_effects,
        ):
            actual_results = inst.redact()

        cleaned_actual_results = dataclasses.asdict(actual_results)
        cleaned_actual_results.pop("run_metrics")
        actual_redaction_boxes = cleaned_actual_results.pop("redaction_results")

        assert cleaned_expected_results == cleaned_actual_results
        compare_unashable_lists(expected_redaction_boxes, actual_redaction_boxes)
