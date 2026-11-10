from typing import ClassVar
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.analysis.images import (
    AzureImageAnalyser,
    ImageAnalyser,
    SignatureDetector,
)
from tests.utils import compare_unhashable_lists


class ImageAnalysisError(Exception):
    pass


class ConcreteAnalysisUtil(ImageAnalyser):
    """Concrete subclass for testing the abstract base."""

    _IMAGE_TEST_CACHE: ClassVar[dict] = {}


@pytest.fixture(autouse=True)
def _clear_caches():
    ConcreteAnalysisUtil._IMAGE_TEST_CACHE.clear()
    yield
    ConcreteAnalysisUtil._IMAGE_TEST_CACHE.clear()


class TestCheckImageSize:
    def test_valid_image(self):
        assert ImageAnalyser.check_image_size(Image.new("RGB", (100, 100))) is True

    def test_exact_minimum(self):
        assert ImageAnalyser.check_image_size(Image.new("RGB", (50, 50))) is True

    def test_exact_maximum(self):
        assert ImageAnalyser.check_image_size(Image.new("RGB", (16000, 16000))) is True

    def test_too_small_width(self):
        assert ImageAnalyser.check_image_size(Image.new("RGB", (49, 100))) is False

    def test_too_small_height(self):
        assert ImageAnalyser.check_image_size(Image.new("RGB", (100, 49))) is False

    def test_too_large_width(self):
        assert ImageAnalyser.check_image_size(Image.new("RGB", (16001, 100))) is False

    def test_too_large_height(self):
        assert ImageAnalyser.check_image_size(Image.new("RGB", (100, 16001))) is False

    def test_non_rgb_image_converted(self):
        image = Image.new("RGBA", (100, 100))
        assert ImageAnalyser.check_image_size(image) is True


class TestClearCache:
    def test_clears_image_caches(self):
        ConcreteAnalysisUtil._IMAGE_TEST_CACHE["hash"] = ("value",)
        assert len(ConcreteAnalysisUtil._IMAGE_TEST_CACHE) == 1

        ConcreteAnalysisUtil.clear_cache()
        assert len(ConcreteAnalysisUtil._IMAGE_TEST_CACHE) == 0


class TestImageDetection:
    def test_returns_results_for_all_images(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(5)]
        expected = [(img, (("Object Detected", (i,)),)) for i, img in enumerate(images)]

        results = ImageAnalyser._image_detection(
            images,
            "object",
            lambda img: ((images.index(img),),),
        )

        compare_unhashable_lists(expected, results)

    def test_passes_kwargs_to_detection_function(self):
        images = [Image.new("RGB", (51, 51))]
        captured = {}

        def detector(img, threshold=0.5):
            captured["threshold"] = threshold
            return ()

        ImageAnalyser._image_detection(images, "test", detector, threshold=0.9)

        assert captured["threshold"] == 0.9

    def test_fallback_redaction_on_exception(self):
        images = [Image.new("RGB", (51 + i, 51), i) for i in range(3)]

        def failing_detector(image):
            if image == images[1]:
                raise ImageAnalysisError("boom")
            else:
                return (("text", (0, 0, 10, 10)),)

        expected_results = (
            (images[0], (("text", (0, 0, 10, 10)),)),
            (
                images[1],
                (
                    (
                        "Text Detection Failed",
                        (0, 0, images[1].width, images[1].height),
                    ),
                ),
            ),
            (images[2], (("text", (0, 0, 10, 10)),)),
        )

        results = ImageAnalyser._image_detection(images, "text", failing_detector)

        compare_unhashable_lists(expected_results, results)

    @pytest.mark.parametrize(
        ("analyser", "env_var"),
        [
            (AzureImageAnalyser, "VISION_MAX_WORKERS"),
            (SignatureDetector, "SIGNATURE_MAX_WORKERS"),
        ],
    )
    def test_worker_cap_uses_service_env_var(self, analyser, env_var):
        with patch("src.analysis.images.get_max_workers", return_value=2) as mock_get:
            analyser._image_detection([], "test", lambda image: ())

        mock_get.assert_called_once_with(env_var=env_var)


class TestAzureImageAnalyserBase:
    # Avoid checking for the environment variable in tests
    @pytest.fixture(autouse=True)
    @staticmethod
    def _mock_get_endpoint(request):
        if "noendpointfixt" in request.keywords:
            yield
            return
        with patch.object(
            AzureImageAnalyser, "_get_client", return_value="http://mock-endpoint"
        ):
            yield

    @pytest.fixture(autouse=True)
    @staticmethod
    def _clear_caches():
        AzureImageAnalyser._IMAGE_FACE_CACHE.clear()
        AzureImageAnalyser._IMAGE_TEXT_CACHE.clear()
        yield
        AzureImageAnalyser._IMAGE_FACE_CACHE.clear()
        AzureImageAnalyser._IMAGE_TEXT_CACHE.clear()


class TestGetClient(TestAzureImageAnalyserBase):
    @pytest.fixture(autouse=True)
    def _reset_endpoint(self):
        AzureImageAnalyser._VISION_CLIENT = None
        yield
        AzureImageAnalyser._VISION_CLIENT = None

    @pytest.mark.noendpointfixt
    def test_returns_endpoint_from_env(self, monkeypatch):
        monkeypatch.setenv("AZURE_VISION_ENDPOINT", "http://test-endpoint")
        with patch("src.analysis.images.ImageAnalysisClient") as mock_client:
            client = AzureImageAnalyser._get_client()
        assert client is not None
        mock_client.assert_called_once()
        assert mock_client.call_args_list[0].args[0] == "http://test-endpoint"

    @pytest.mark.noendpointfixt
    def test_raises_error_if_env_not_set(self, monkeypatch):
        monkeypatch.delenv("AZURE_VISION_ENDPOINT", raising=False)
        with pytest.raises(AzureImageAnalyser.EndpointNotSetError):
            AzureImageAnalyser._get_client()


class TestAzureVisionAnalysisRateLimit(TestAzureImageAnalyserBase):
    @pytest.fixture(autouse=True)
    def _reset_rate_limiter(self):
        AzureImageAnalyser._RATE_LIMITER = None
        yield
        AzureImageAnalyser._RATE_LIMITER = None

    def test_waits_for_shared_rate_limiter_before_each_request(self, monkeypatch):
        monkeypatch.setenv("VISION_REQUESTS_PER_MINUTE", "15")
        image = Image.new("RGB", (100, 100))
        client = Mock()

        with (
            patch.object(AzureImageAnalyser, "_get_client", return_value=client),
            patch("src.analysis.images.RateLimiter") as mock_rate_limiter,
        ):
            for _ in range(2):
                AzureImageAnalyser._azure_vision_analysis(image, [])

        mock_rate_limiter.assert_called_once_with(15)
        assert mock_rate_limiter.return_value.wait.call_count == 2

    def test_no_rate_limiter_when_unset(self, monkeypatch):
        monkeypatch.delenv("VISION_REQUESTS_PER_MINUTE", raising=False)
        image = Image.new("RGB", (100, 100))

        with (
            patch.object(AzureImageAnalyser, "_get_client", return_value=Mock()),
            patch("src.analysis.images.RateLimiter") as mock_rate_limiter,
        ):
            AzureImageAnalyser._azure_vision_analysis(image, [])

        mock_rate_limiter.assert_not_called()


class TestDetectFaces(TestAzureImageAnalyserBase):
    @staticmethod
    def _mock_vision_result_people(people):
        mock_result = Mock()
        mock_result.people.list = people
        return mock_result

    def test_returns_boxes_above_threshold(self):
        image = Mock()
        people_list = [
            Mock(bounding_box=Mock(x=10, y=20, width=30, height=40), confidence=0.9),
            Mock(bounding_box=Mock(x=50, y=60, width=10, height=10), confidence=0.4),
        ]

        with (
            patch.object(AzureImageAnalyser, "check_image_size", return_value=True),
            patch.object(
                AzureImageAnalyser,
                "_azure_vision_analysis",
                return_value=self._mock_vision_result_people(people_list),
            ),
        ):
            result = AzureImageAnalyser.detect_faces(image, confidence_threshold=0.5)

        assert result == ((10, 20, 40, 60),)

    def test_caches_result(self):
        image = Image.new("RGB", (100, 100))
        people_list = [
            Mock(bounding_box=Mock(x=10, y=20, width=30, height=40), confidence=0.9),
        ]

        with (
            patch.object(AzureImageAnalyser, "check_image_size", return_value=True),
            patch.object(
                AzureImageAnalyser,
                "_azure_vision_analysis",
                return_value=self._mock_vision_result_people(people_list),
            ),
        ):
            AzureImageAnalyser.detect_faces(image, confidence_threshold=0.5)

        assert len(AzureImageAnalyser._IMAGE_FACE_CACHE) == 1
        assert (
            AzureImageAnalyser.get_image_hash(image)
            in AzureImageAnalyser._IMAGE_FACE_CACHE
        )

    def test_uses_cached_result(self):
        image = Image.new("RGB", (100, 100))
        AzureImageAnalyser._IMAGE_FACE_CACHE = {
            AzureImageAnalyser.get_image_hash(image): (
                {"box": (10, 20, 40, 60), "confidence": 0.9},
            )
        }

        with patch.object(
            AzureImageAnalyser, "_azure_vision_analysis"
        ) as mock_analysis:
            result = AzureImageAnalyser.detect_faces(image, confidence_threshold=0.5)

        mock_analysis.assert_not_called()
        assert result == ((10, 20, 40, 60),)

    def test_skips_image_too_small(self):
        image = Image.new("RGB", (49, 49))
        result = AzureImageAnalyser.detect_faces(image, confidence_threshold=0.5)
        assert result == ()

    def test_skips_image_too_large(self):
        image = Image.new("RGB", (16001, 100))
        result = AzureImageAnalyser.detect_faces(image, confidence_threshold=0.5)
        assert result == ()


class TestDetectFacesInImages(TestAzureImageAnalyserBase):
    def test_returns_results_for_all_images(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(5)]
        expected_results = [
            (img, (("Face Detected", (i,)),)) for i, img in enumerate(images)
        ]

        with patch.object(
            AzureImageAnalyser,
            "detect_faces",
            side_effect=lambda img, **kw: ((images.index(img),),),
        ):
            actual_results = AzureImageAnalyser.detect_faces_in_images(images, 0.1)

        compare_unhashable_lists(expected_results, actual_results)

    def test_redacts_full_image_on_exception(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(3)]

        def mock_detect(image, **kwargs):
            if image == images[1]:
                raise ImageAnalysisError("Some exception")
            return ((0, 0, 10, 10),)

        with patch.object(AzureImageAnalyser, "detect_faces", side_effect=mock_detect):
            actual_results = AzureImageAnalyser.detect_faces_in_images(images, 0.1)

        failed_result = next(r for r in actual_results if r[0] == images[1])
        assert failed_result == (
            images[1],
            (("Face Detection Failed", (0, 0, images[1].width, images[1].height)),),
        )


class TestDetectText(TestAzureImageAnalyserBase):
    @staticmethod
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

    def test_returns_words_with_bounding_boxes(self):
        image = Mock()

        with (
            patch.object(AzureImageAnalyser, "check_image_size", return_value=True),
            patch.object(
                AzureImageAnalyser,
                "_azure_vision_analysis",
                return_value=self._mock_vision_result_text(),
            ),
        ):
            result = AzureImageAnalyser.detect_text(image)

        assert result == (("Hello", (10, 20, 30, 40)), ("World", (50, 60, 70, 80)))

    def test_caches_result(self):
        image = Mock()

        with (
            patch.object(AzureImageAnalyser, "check_image_size", return_value=True),
            patch.object(
                AzureImageAnalyser,
                "_azure_vision_analysis",
                return_value=self._mock_vision_result_text(),
            ),
        ):
            AzureImageAnalyser.detect_text(image)

        assert len(AzureImageAnalyser._IMAGE_TEXT_CACHE) == 1
        assert (
            AzureImageAnalyser.get_image_hash(image)
            in AzureImageAnalyser._IMAGE_TEXT_CACHE
        )

    def test_uses_cached_result(self):
        image = Image.new("RGB", (100, 100))
        cached_text = (("Hello", (10, 20, 30, 40)),)
        AzureImageAnalyser._IMAGE_TEXT_CACHE = {
            AzureImageAnalyser.get_image_hash(image): cached_text
        }

        with patch.object(
            AzureImageAnalyser, "_azure_vision_analysis"
        ) as mock_analysis:
            result = AzureImageAnalyser.detect_text(image)

        mock_analysis.assert_not_called()
        assert result == cached_text


class TestDetectTextInImages(TestAzureImageAnalyserBase):
    def test_returns_results_for_all_images(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(5)]
        expected_results = [
            (img, ((str(i), (0, 0, i, i)),)) for i, img in enumerate(images)
        ]

        with patch.object(
            AzureImageAnalyser,
            "detect_text",
            side_effect=lambda img: (
                (
                    str(images.index(img)),
                    (0, 0, images.index(img), images.index(img)),
                ),
            ),
        ):
            actual_results = AzureImageAnalyser.detect_text_in_images(images)

        compare_unhashable_lists(expected_results, actual_results)

    def test_redacts_full_image_on_exception(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(3)]

        def mock_detect(image):
            if image == images[1]:
                raise ImageAnalysisError("Some exception")
            return (("word", (0, 0, 10, 10)),)

        with patch.object(AzureImageAnalyser, "detect_text", side_effect=mock_detect):
            actual_results = AzureImageAnalyser.detect_text_in_images(images)

        failed_result = next(r for r in actual_results if r[0] == images[1])
        assert failed_result == (
            images[1],
            (("Text Detection Failed", (0, 0, images[1].width, images[1].height)),),
        )


class TestSignatureDetectorBase:
    # Avoid checking for the environment variable in tests
    @pytest.fixture(autouse=True)
    @staticmethod
    def _mock_get_endpoint(request):
        if "noendpointfixt" in request.keywords:
            yield
            return
        with patch.object(
            SignatureDetector, "_get_endpoint", return_value="http://mock-endpoint"
        ):
            yield

    @pytest.fixture(autouse=True)
    @staticmethod
    def _clear_caches():
        SignatureDetector._IMAGE_CACHE.clear()
        yield
        SignatureDetector._IMAGE_CACHE.clear()


class TestGetBoundingBox(TestSignatureDetectorBase):
    def test_returns_bounding_box(self):
        detection = {
            "score": 0.9,
            "label": "signature",
            "box": {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 60},
        }
        bounding_box = SignatureDetector._get_bounding_box(detection)
        assert bounding_box == (10, 20, 50, 60)


class TestGetEndpoint(TestSignatureDetectorBase):
    @pytest.fixture(autouse=True)
    def _reset_endpoint(self):
        SignatureDetector._ENDPOINT = None
        yield
        SignatureDetector._ENDPOINT = None

    @pytest.mark.noendpointfixt
    def test_returns_endpoint_from_env(self, monkeypatch):
        monkeypatch.setenv("SIGNATURE_DETECTOR_ENDPOINT", "http://test-endpoint")
        endpoint = SignatureDetector._get_endpoint()
        assert endpoint == "http://test-endpoint"

    @pytest.mark.noendpointfixt
    def test_raises_error_if_env_not_set(self, monkeypatch):
        monkeypatch.delenv("SIGNATURE_DETECTOR_ENDPOINT", raising=False)
        with pytest.raises(SignatureDetector.EndpointNotSetError):
            SignatureDetector._get_endpoint()


class TestDetectSignatures(TestSignatureDetectorBase):
    @staticmethod
    def _mock_response(detections, status_code=200):
        mock = Mock()
        mock.status_code = status_code
        mock.json.return_value = {"detections": detections}
        mock.raise_for_status = Mock()
        return mock

    def test_returns_boxes_above_threshold(self):
        image = Image.new("RGB", (100, 100))
        detections = [
            {
                "score": 0.9,
                "label": "signature",
                "box": {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 60},
            },
            {
                "score": 0.3,
                "label": "signature",
                "box": {"x_min": 70, "y_min": 80, "x_max": 90, "y_max": 95},
            },
        ]

        with patch(
            "src.analysis.images.post",
            return_value=self._mock_response(detections),
        ):
            result = SignatureDetector.detect_signatures(
                image, confidence_threshold=0.5
            )

        assert result == ((10, 20, 50, 60),)

    def test_returns_empty_for_no_detections(self):
        image = Image.new("RGB", (100, 100))

        with patch("src.analysis.images.post", return_value=self._mock_response([])):
            result = SignatureDetector.detect_signatures(
                image, confidence_threshold=0.5
            )

        assert result == ()

    def test_skips_image_too_small(self):
        image = Image.new("RGB", (49, 49))
        result = SignatureDetector.detect_signatures(image, confidence_threshold=0.5)
        assert result == ()

    def test_skips_image_too_large(self):
        image = Image.new("RGB", (16001, 100))
        result = SignatureDetector.detect_signatures(image, confidence_threshold=0.5)
        assert result == ()

    def test_caches_result(self):
        image = Image.new("RGB", (100, 100))
        detections = [
            {
                "score": 0.9,
                "label": "signature",
                "box": {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 60},
            },
        ]

        with patch(
            "src.analysis.images.post",
            return_value=self._mock_response(detections),
        ):
            SignatureDetector.detect_signatures(image, confidence_threshold=0.5)

        assert len(SignatureDetector._IMAGE_CACHE) == 1
        assert (
            SignatureDetector._IMAGE_CACHE[SignatureDetector.get_image_hash(image)]
            == detections
        )

    def test_uses_cached_result(self):
        image = Image.new("RGB", (100, 100))
        cached_detections = [
            {
                "score": 0.9,
                "box": {"x_min": 10, "y_min": 20, "x_max": 50, "y_max": 60},
                "confidence": 0.9,
            },
        ]
        SignatureDetector._IMAGE_CACHE = {
            SignatureDetector.get_image_hash(image): cached_detections
        }

        with patch("src.analysis.images.post") as mock_post:
            result = SignatureDetector.detect_signatures(
                image, confidence_threshold=0.5
            )

        mock_post.assert_not_called()
        assert result == ((10, 20, 50, 60),)

    def test_sends_correct_payload(self):
        image = Image.new("RGB", (100, 100))

        with patch(
            "src.analysis.images.post", return_value=self._mock_response([])
        ) as mock_post:
            SignatureDetector.detect_signatures(image, confidence_threshold=0.7)

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"]["threshold"] == 0.0
        assert "image" in call_kwargs.kwargs["json"]
        assert call_kwargs.kwargs["timeout"] == 120


class TestDetectSignaturesInImages:
    def test_returns_results_for_all_images(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(3)]

        def mock_detect(img, **kwargs):
            idx = images.index(img)
            return ((idx, idx, idx, idx),)

        with patch.object(
            SignatureDetector, "detect_signatures", side_effect=mock_detect
        ):
            actual = SignatureDetector.detect_signatures_in_images(images, 0.5)

        expected = [
            (img, (("Signature Detected", (i, i, i, i)),))
            for i, img in enumerate(images)
        ]
        compare_unhashable_lists(expected, actual)

    def test_redacts_full_image_on_exception(self):
        images = [Image.new("RGB", (51, 51), i) for i in range(3)]

        class DetectionError(Exception):
            pass

        def mock_detect(image, **kwargs):
            if image == images[1]:
                raise DetectionError("endpoint down")
            return ()

        with patch.object(
            SignatureDetector, "detect_signatures", side_effect=mock_detect
        ):
            actual = SignatureDetector.detect_signatures_in_images(images, 0.5)

        failed = next(r for r in actual if r[0] == images[1])
        assert failed == (
            images[1],
            (
                (
                    "Signature Detection Failed",
                    (0, 0, images[1].width, images[1].height),
                ),
            ),
        )
