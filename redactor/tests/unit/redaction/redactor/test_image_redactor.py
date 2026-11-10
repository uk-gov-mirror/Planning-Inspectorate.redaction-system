import contextlib
import threading
from unittest import mock

from PIL import Image

from src.analysis.images import AzureImageAnalyser, SignatureDetector
from src.redaction.config import ImageRedactionConfig
from src.redaction.redactor import ImageRedactor
from src.types import ImageRedactionResult
from tests.utils import compare_unhashable_lists


class TestGetName:
    def test_returns_name(self):
        assert ImageRedactor.get_name() == "ImageRedaction"


class TestGetRedactionConfigClass:
    def test_returns_redaction_config_class(self):
        assert ImageRedactor.get_redaction_config_class() == ImageRedactionConfig


class ImageAnalysisError(Exception):
    pass


class TestImageRedactorBase:
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

    @staticmethod
    @contextlib.contextmanager
    def mock_detectors(face_results, signature_results):
        with (
            mock.patch.object(
                AzureImageAnalyser,
                "detect_faces_in_images",
                return_value=face_results,
            ),
            mock.patch.object(
                SignatureDetector,
                "detect_signatures_in_images",
                return_value=signature_results,
            ),
        ):
            yield


class TestCreateRedactionResults(TestImageRedactorBase):
    def test_combines_image_analysis_results(self):
        images = [Image.new("RGB", (100, 100)), Image.new("RGB", (200, 100))]
        face_results = (
            (
                images[0],
                (("Face Detected", (10, 10, 50, 50)),),
            ),
            (
                images[1],
                (("Face Detected", (60, 20, 50, 80)),),
            ),
        )
        sig_results = (
            (images[0], ()),
            (images[1], (("Signature Detected", (30, 30, 50, 50)),)),
        )
        inst = self.setup_image_redactor(images)

        actual_results = inst._create_redaction_results([face_results, sig_results])

        expected_results = (
            ImageRedactionResult.Result(
                source_image=images[0],
                image_dimensions=(100, 100),
                redaction_boxes=((10, 10, 50, 50),),
                names=("Face Detected",),
            ),
            ImageRedactionResult.Result(
                source_image=images[1],
                image_dimensions=(200, 100),
                redaction_boxes=(
                    (60, 20, 50, 80),
                    (30, 30, 50, 50),
                ),
                names=("Face Detected", "Signature Detected"),
            ),
        )
        compare_unhashable_lists(expected_results, actual_results)

    def test_no_result_with_no_detections(self):
        images = [Image.new("RGB", (100, 100))]
        face_results = ((images[0], ()),)
        sig_results = ((images[0], (("Signature Detected", (30, 30, 50, 50)),)),)
        inst = self.setup_image_redactor(images)

        actual_results = inst._create_redaction_results([face_results, sig_results])

        expected_results = (
            ImageRedactionResult.Result(
                source_image=images[0],
                image_dimensions=(100, 100),
                redaction_boxes=((30, 30, 50, 50),),
                names=("Signature Detected",),
            ),
        )
        compare_unhashable_lists(expected_results, actual_results)


class TestRedact(TestImageRedactorBase):
    def test_no_images_skips_analysis(self):
        inst = self.setup_image_redactor(images=[])

        with (
            mock.patch.object(
                AzureImageAnalyser, "detect_faces_in_images"
            ) as mock_faces,
            mock.patch.object(
                SignatureDetector, "detect_signatures_in_images"
            ) as mock_sigs,
        ):
            actual = inst.redact()

        mock_faces.assert_not_called()
        mock_sigs.assert_not_called()
        assert actual.rule_name == self.RULE_NAME
        assert actual.redaction_results == ()

    def test_returns_face_and_signature_results(self):
        images = [Image.new("RGB", (1000, 1000)), Image.new("RGB", (200, 100))]
        face_results = [
            (images[0], (("Face Detected", (10, 10, 50, 50)),)),
            (images[1], ()),
        ]
        sig_results = [
            (images[0], ()),
            (images[1], (("Signature Detected", (30, 30, 50, 50)),)),
        ]
        inst = self.setup_image_redactor(images)

        with self.mock_detectors(face_results, sig_results):
            actual = inst.redact()

        assert len(actual.redaction_results) == 2
        expected_results = [
            ImageRedactionResult.Result(
                source_image=images[0],
                image_dimensions=(1000, 1000),
                redaction_boxes=((10, 10, 50, 50),),
                names=("Face Detected",),
            ),
            ImageRedactionResult.Result(
                source_image=images[1],
                image_dimensions=(200, 100),
                redaction_boxes=((30, 30, 50, 50),),
                names=("Signature Detected",),
            ),
        ]
        compare_unhashable_lists(expected_results, actual.redaction_results)

    def test_no_detections_returns_empty(self):
        images = [Image.new("RGB", (100, 100))]
        face_results = [(images[0], ())]
        sig_results = [(images[0], ())]
        inst = self.setup_image_redactor(images)

        with self.mock_detectors(face_results, sig_results):
            actual = inst.redact()

        assert actual.redaction_results == ()

    def test_face_failure_redacts_full_image(self):
        images = [Image.new("RGB", (200, 100))]
        full_box = (0, 0, 200, 100)
        face_results = [(images[0], (("Face Detection Failed", full_box),))]
        sig_results = [(images[0], ())]
        inst = self.setup_image_redactor(images)

        with self.mock_detectors(face_results, sig_results):
            actual = inst.redact()

        assert len(actual.redaction_results) == 1
        result = actual.redaction_results[0]
        assert list(result.redaction_boxes) == [full_box]
        assert "Face Detection Failed" in result.names

    def test_signature_failure_redacts_full_image(self):
        images = [Image.new("RGB", (200, 100))]
        full_box = (0, 0, 200, 100)
        face_results = [(images[0], ())]
        sig_results = [(images[0], (("Signature Detection Failed", full_box),))]
        inst = self.setup_image_redactor(images)

        with self.mock_detectors(face_results, sig_results):
            actual = inst.redact()

        assert len(actual.redaction_results) == 1
        result = actual.redaction_results[0]
        assert list(result.redaction_boxes) == [full_box]
        assert "Signature Detection Failed" in result.names

    def test_aggregates_faces_and_signatures_for_same_image(self):
        images = [Image.new("RGB", (500, 500))]
        face_box = (10, 10, 50, 50)
        sig_box = (200, 200, 300, 300)
        face_results = [(images[0], (("Face Detected", face_box),))]
        sig_results = [(images[0], (("Signature Detected", sig_box),))]
        inst = self.setup_image_redactor(images)

        with self.mock_detectors(face_results, sig_results):
            actual = inst.redact()

        assert len(actual.redaction_results) == 1
        result = actual.redaction_results[0]
        assert face_box in result.redaction_boxes
        assert sig_box in result.redaction_boxes
        assert "Face Detected" in result.names
        assert "Signature Detected" in result.names

    def test_run_metrics_contains_timing_keys(self):
        images = [Image.new("RGB", (100, 100))]
        face_results = [(images[0], ())]
        sig_results = [(images[0], ())]
        inst = self.setup_image_redactor(images)

        with self.mock_detectors(face_results, sig_results):
            actual = inst.redact()

        assert "total_images_to_analyse" in actual.run_metrics
        assert "total_face_analysis_time" in actual.run_metrics
        assert "total_signature_analysis_time" in actual.run_metrics
        assert "total_image_analysis_time" in actual.run_metrics
        assert actual.run_metrics["total_images_to_analyse"] == 1

    def test_face_and_signature_detection_run_concurrently(self):
        images = [Image.new("RGB", (100, 100))]
        # Each detector waits for the other, so both only finish if they overlap
        barrier = threading.Barrier(2, timeout=5)

        def detect(*args):
            barrier.wait()
            return [(images[0], ())]

        inst = self.setup_image_redactor(images)

        with (
            mock.patch.object(
                AzureImageAnalyser, "detect_faces_in_images", side_effect=detect
            ),
            mock.patch.object(
                SignatureDetector, "detect_signatures_in_images", side_effect=detect
            ),
        ):
            actual = inst.redact()

        assert actual.redaction_results == ()
