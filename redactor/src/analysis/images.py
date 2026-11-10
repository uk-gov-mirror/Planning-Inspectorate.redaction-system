import hashlib
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from io import BytesIO
from threading import Lock
from typing import ClassVar

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import ImageAnalysisResult, VisualFeatures
from azure.core.exceptions import HttpResponseError
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)
from dotenv import load_dotenv
from PIL import Image
from requests import post
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity.retry import retry_if_exception

from ..monitoring import LoggingUtil, log_to_appins
from .utils import RateLimiter, get_max_workers

load_dotenv(verbose=True)


@log_to_appins
def handle_last_retry_error(retry_state):
    LoggingUtil().log_info(
        f"All retry attempts failed: {retry_state.outcome.exception()}\n"
        "Returning None for this image."
    )


class ImageAnalyser:
    CACHE_LOCK = Lock()
    MAX_WORKERS_ENV_VAR: ClassVar[str | None] = None
    _IN_FLIGHT_REQUESTS: ClassVar[dict[tuple[int, str], Future]] = {}

    class EndpointNotSetError(EnvironmentError):
        pass

    @classmethod
    def _fetch_once(cls, cache: dict, image_hash: str, fetch: Callable):
        """
        Return the cached result for the image, calling fetch() in only one thread
        when several threads miss the cache for the same image at the same time.

        :param dict cache: The cache of results, keyed by image hash
        :param str image_hash: The content hash of the image
        :param Callable fetch: Gets the result, or None if it should not be cached
        """
        key = (id(cache), image_hash)
        with cls.CACHE_LOCK:
            if image_hash in cache:
                LoggingUtil().log_info("Using cached image analysis result.")
                return cache[image_hash]
            future = cls._IN_FLIGHT_REQUESTS.get(key)

            # Check if this thread is the one responsible for fetching the result
            is_owner = future is None
            if is_owner:
                future = cls._IN_FLIGHT_REQUESTS[key] = Future()

        # Wait for the result of the owner thread
        if not is_owner:
            LoggingUtil().log_info("Waiting for an in-progress analysis of this image.")
            return future.result()

        try:
            # Get the analysis result for this image
            result = fetch()
        except BaseException as e:
            with cls.CACHE_LOCK:
                # Remove the in-flight request entry for this key since fetching failed
                del cls._IN_FLIGHT_REQUESTS[key]

            # Notify other threads of the failure
            future.set_exception(e)
            raise

        with cls.CACHE_LOCK:
            if result is not None:
                cache[image_hash] = result
            # Remove the in-flight request entry for this key since fetching succeeded
            del cls._IN_FLIGHT_REQUESTS[key]

        # Notify other threads that the result is available
        future.set_result(result)

        return result

    @classmethod
    def clear_cache(cls):
        """Clear cached image analysis results from previous invocations."""
        with cls.CACHE_LOCK:
            for attr in dir(cls):
                if attr.startswith("_IMAGE_") and isinstance(
                    getattr(cls, attr), list | dict
                ):
                    getattr(cls, attr).clear()

    @staticmethod
    def get_image_hash(image: Image.Image) -> str:
        """Content hash of an image, memoised on the image object. Images must not be mutated after hashing."""
        image_hash = getattr(image, "_content_hash", None)
        if image_hash is None:
            digest = hashlib.sha256(f"{image.mode}{image.size}".encode())
            digest.update(image.tobytes())
            image_hash = digest.hexdigest()
            image._content_hash = image_hash
        return image_hash

    @staticmethod
    def _save_image_to_bytes(image: Image.Image) -> bytes:
        save_image = image.convert("RGB") if image.mode != "RGB" else image
        byte_stream = BytesIO()
        try:
            save_image.save(byte_stream, format="jpeg")
        except OSError:
            save_image.save(byte_stream, format="png")
        return byte_stream.getvalue()

    @classmethod
    def check_image_size(cls, image: Image.Image) -> bool:
        """
        Check if the image size is within the limits of Azure Computer Vision API:
        - The image size must be less than 20MB
        - The image dimensions must be at least 50x50 and at most 16000x16000 pixels

        :param Image.Image image: The image to check
        :returns: True if the image size is within the limits, False otherwise
        """
        image_bytes = cls._save_image_to_bytes(image)

        if len(image_bytes) > 20 * 1024 * 1024:
            LoggingUtil().log_info(
                f"Image size is {len(image_bytes)} bytes, which is larger than 20MB. "
            )
            return False

        if image.width < 50 or image.height < 50:
            LoggingUtil().log_info(
                f"Image dimensions are {image.width}x{image.height}, which is smaller "
                "than 50x50 pixels."
            )
            return False

        if image.width > 16000 or image.height > 16000:
            LoggingUtil().log_info(
                f"Image dimensions are {image.width}x{image.height}, which is larger "
                "than 16000x16000 pixels."
            )
            return False

        return True

    @classmethod
    def _image_detection(
        cls,
        images: list[Image.Image],
        object_to_detect: str,
        detection_function: callable,
        *args,
        **kwargs,
    ) -> tuple[tuple[Image.Image, tuple[str, tuple]], ...]:
        """
        Generic function to detect objects in images using a given detection function.

        :param list[Image.Image] images: The images to detect objects in
        :param detection_function: The function to use for detection
        :returns: A list of tuples containing the image and the detected objects
        """
        responses: list[tuple[Image.Image, tuple[str, tuple]]] = []
        max_workers = get_max_workers(env_var=cls.MAX_WORKERS_ENV_VAR)
        LoggingUtil().log_info(
            f"Detecting {object_to_detect}s in {len(images)} images using up to "
            f"{max_workers} workers..."
        )
        with ThreadPoolExecutor(max_workers) as tpe:
            finished_futures = 0
            futures_map = {
                tpe.submit(detection_function, image, *args, **kwargs): image
                for image in images
            }

            for future in as_completed(futures_map):
                image = futures_map[future]
                try:
                    detections = future.result()
                    if object_to_detect != "text":
                        wrapped = tuple(
                            (f"{object_to_detect.capitalize()} Detected", box)
                            for box in detections
                        )
                    else:
                        wrapped = tuple((word, box) for word, box in detections)
                    responses.append((image, wrapped))
                    finished_futures += 1
                    LoggingUtil().log_info(
                        f"Finished {object_to_detect} detection for {finished_futures}/"
                        f"{len(images)} images: {len(detections)} objects detected."
                    )
                except Exception as e:  # noqa: BLE001
                    LoggingUtil().log_exception_with_message(
                        f"Image {object_to_detect} detection failed with the following exception: ",
                        e,
                    )
                    # If object detection fails for any reason, redact the full image
                    responses.append(
                        (
                            image,
                            (
                                (
                                    f"{object_to_detect.capitalize()} Detection Failed",
                                    (0, 0, image.width, image.height),
                                ),
                            ),
                        )
                    )

        LoggingUtil().log_info(
            f"Finished detecting {object_to_detect}s in {len(images)} images."
        )
        return tuple(responses)


class AzureImageAnalyser(ImageAnalyser):
    MAX_WORKERS_ENV_VAR: ClassVar[str | None] = "VISION_MAX_WORKERS"
    _IMAGE_TEXT_CACHE: ClassVar[dict[str, tuple]] = {}
    _IMAGE_FACE_CACHE: ClassVar[dict[str, tuple]] = {}
    _VISION_CLIENT: ClassVar[ImageAnalysisClient | None] = None
    _RATE_LIMITER: ClassVar[RateLimiter | None] = None
    _RATE_LIMITER_LOCK: ClassVar[Lock] = Lock()

    @classmethod
    def _wait_for_rate_limit(cls) -> None:
        """Pace requests to VISION_REQUESTS_PER_MINUTE; unset or 0 means no limit."""
        requests_per_minute = int(os.environ.get("VISION_REQUESTS_PER_MINUTE") or 0)
        if requests_per_minute < 1:
            return
        with cls._RATE_LIMITER_LOCK:
            if cls._RATE_LIMITER is None:
                cls._RATE_LIMITER = RateLimiter(requests_per_minute)
            rate_limiter = cls._RATE_LIMITER
        rate_limiter.wait()

    @classmethod
    def _get_client(cls) -> ImageAnalysisClient:
        if cls._VISION_CLIENT is None:
            endpoint = os.environ.get("AZURE_VISION_ENDPOINT", None)
            if endpoint is None:
                raise cls.EndpointNotSetError(
                    "AZURE_VISION_ENDPOINT environment variable is not set."
                )
            cls._VISION_CLIENT = ImageAnalysisClient(
                endpoint,
                credential=ChainedTokenCredential(
                    ManagedIdentityCredential(), AzureCliCredential()
                ),
            )
        return cls._VISION_CLIENT

    @classmethod
    @retry(
        retry=retry_if_exception(
            lambda exception: (
                isinstance(exception, HttpResponseError)
                and exception.status_code in [429]
            )
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=lambda retry_state: LoggingUtil().log_info(
            "Retrying image face detection..."
        ),
        retry_error_callback=handle_last_retry_error,
    )
    def detect_faces(
        cls, image: Image.Image, confidence_threshold: float = 0.5
    ) -> tuple[tuple[float, float, float, float], ...]:
        """
        Detect faces in the given image

        :param Image.Image image: The image to analyse
        :param float confidence_threshold: Confidence threshold between 0 and 1
        :returns: Bounding boxes of faces as a 4-tuple of the form (top left corner x, top left corner y, bottom right corner x, bottom right corner y), for boxes
                  with confidence above the threshold
        """
        image_hash = cls.get_image_hash(image)

        def fetch():
            result = cls._azure_vision_analysis(image, [VisualFeatures.PEOPLE])
            if result is None:
                return None
            return tuple(
                {
                    "box": (
                        person.bounding_box.x,
                        person.bounding_box.y,
                        person.bounding_box.x + person.bounding_box.width,
                        person.bounding_box.y + person.bounding_box.height,
                    ),
                    "confidence": person.confidence,
                }
                for person in result.people.list
            )

        faces_detected = cls._fetch_once(cls._IMAGE_FACE_CACHE, image_hash, fetch)
        if faces_detected is None:
            return ()

        return tuple(
            person["box"]
            for person in faces_detected
            if person["confidence"] >= confidence_threshold
        )

    @classmethod
    def detect_faces_in_images(
        cls, images: list[Image.Image], confidence_threshold: float = 0.5
    ) -> tuple[tuple[Image.Image, tuple[str, tuple]]]:
        return cls._image_detection(
            images,
            "face",
            cls.detect_faces,
            confidence_threshold=confidence_threshold,
        )

    @classmethod
    @log_to_appins
    def detect_text_in_images(
        cls, images: list[Image.Image]
    ) -> tuple[tuple[Image.Image, tuple[tuple[str, tuple]]]]:
        return cls._image_detection(
            images,
            "text",
            cls.detect_text,
        )

    @classmethod
    def _azure_vision_analysis(
        cls, image: Image.Image, visual_features: list[VisualFeatures]
    ) -> ImageAnalysisResult | None:
        valid_image = cls.check_image_size(image)
        if not valid_image:
            LoggingUtil().log_info("Skipping image analysis due to size constraints.")
            return None

        byte_stream = BytesIO()
        save_image = image.convert("RGB") if image.mode != "RGB" else image
        save_image.save(byte_stream, format="jpeg")
        image_bytes = byte_stream.getvalue()

        LoggingUtil().log_info(
            f"Analysing image using Azure Computer Vision API for features: {visual_features}..."
        )
        try:
            cls._wait_for_rate_limit()
            result = cls._get_client().analyze(
                image_bytes,
                visual_features,
            )
            return result
        except HttpResponseError as e:
            LoggingUtil().log_exception_with_message(
                "HTTP response error analysing image", e
            )
            raise

    @classmethod
    @log_to_appins
    @retry(
        retry=retry_if_exception(
            lambda exception: (
                isinstance(exception, HttpResponseError)
                and exception.status_code in [429]
            )
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(10),
        before_sleep=lambda retry_state: LoggingUtil().log_info(
            "Retrying image text detection..."
        ),
        retry_error_callback=handle_last_retry_error,
    )
    def detect_text(
        cls, image: Image.Image
    ) -> tuple[tuple[str, tuple[float, float, float, float]]]:
        """
        Return all text content of the given image, as a 2D tuple of <word, bounding box>

        :param Image.Image image: The image to analyse
        :return Tuple[Tuple[str, Tuple[float, float, float, float]], ...]: The text content
        detected in the image, as a 2D tuple of <word, bounding box>.
        """
        image_hash = cls.get_image_hash(image)

        def fetch():
            result = cls._azure_vision_analysis(image, [VisualFeatures.READ])
            if result is None:
                return None
            return tuple(
                (
                    word.text,
                    (
                        word.bounding_polygon[0].x,
                        word.bounding_polygon[0].y,
                        word.bounding_polygon[2].x,
                        word.bounding_polygon[2].y,
                    ),
                )
                for block in result.read.blocks
                for line in block.lines
                for word in line.words
            )

        text_detected = cls._fetch_once(cls._IMAGE_TEXT_CACHE, image_hash, fetch)
        return () if text_detected is None else text_detected


class SignatureDetector(ImageAnalyser):
    MAX_WORKERS_ENV_VAR: ClassVar[str | None] = "SIGNATURE_MAX_WORKERS"
    _IMAGE_CACHE: ClassVar[dict[str, list]] = {}
    _ENDPOINT: ClassVar[str | None] = None

    @classmethod
    def _get_endpoint(cls) -> str:
        if cls._ENDPOINT is None:
            cls._ENDPOINT = os.environ.get("SIGNATURE_DETECTOR_ENDPOINT")
            if cls._ENDPOINT is None:
                raise cls.EndpointNotSetError(
                    "SIGNATURE_DETECTOR_ENDPOINT environment variable is not set."
                )
        return cls._ENDPOINT

    @classmethod
    def detect_signatures_in_images(
        cls, images: list[Image.Image], confidence_threshold: float = 0.5
    ) -> list[tuple[Image.Image, tuple[str, tuple]]]:
        return cls._image_detection(
            images,
            "signature",
            cls.detect_signatures,
            confidence_threshold=confidence_threshold,
        )

    @staticmethod
    def _get_bounding_box(detection: dict) -> tuple[float, float, float, float]:
        """
        Convert a detection dictionary to a bounding box tuple.

        :param dict detection: The detection dictionary containing the bounding box
        :return Tuple[float, float, float, float]: The bounding box as a 4-tuple of the form
        (top left corner x, top left corner y, bottom right corner x, bottom right corner y)
        """
        box = detection.get("box", {})
        return tuple(
            round(box.get(val)) for val in ["x_min", "y_min", "x_max", "y_max"]
        )

    @classmethod
    @retry(
        retry=retry_if_exception(
            lambda exception: (
                isinstance(exception, HttpResponseError)
                and exception.status_code in [429, 503]
            )
        ),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: LoggingUtil().log_info(
            "Retrying signature detection..."
        ),
        retry_error_callback=handle_last_retry_error,
    )
    def detect_signatures(
        cls, image: Image.Image, confidence_threshold: float = 0.5
    ) -> tuple[tuple[float, float, float, float], ...]:
        valid_image = AzureImageAnalyser.check_image_size(image)
        if not valid_image:
            LoggingUtil().log_info(
                "Skipping signature detection for image due to size constraints."
            )
            return ()

        image_hash = cls.get_image_hash(image)

        def fetch():
            image_bytes = cls._save_image_to_bytes(image)
            LoggingUtil().log_info(
                "Analysing image for signatures using signature detector app..."
            )
            response = post(
                f"https://{cls._get_endpoint()}/score",
                json={"image": image_bytes.hex(), "threshold": 0.0},
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("detections", [])

        detections = cls._fetch_once(cls._IMAGE_CACHE, image_hash, fetch)

        return tuple(
            cls._get_bounding_box(d)
            for d in detections
            if d["score"] >= confidence_threshold
        )
