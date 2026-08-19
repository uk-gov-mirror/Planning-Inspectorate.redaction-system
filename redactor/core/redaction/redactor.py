import json
import re
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, ClassVar

from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from core.redaction.config import (
    ImageLLMTextRedactionConfig,
    ImageRedactionConfig,
    LLMTextRedactionConfig,
    RedactionConfig,
    TextRedactionConfig,
)
from core.redaction.exceptions import (
    DuplicateRedactorNameException,
    IncorrectRedactionConfigClassException,
    RedactorNameNotFoundException,
)
from core.redaction.result import (
    ImageRedactionResult,
    LLMTextRedactionResult,
    RedactionResult,
)
from core.util.image_analysis import AzureVisionUtil, SignatureDetector
from core.util.llm_util import LLMUtil
from core.util.logging_util import LoggingUtil, log_to_appins
from core.util.metric_util import TimerUtil
from core.util.text_util import get_normalised_words


class Redactor(ABC):
    """
    Class that handles the redaction of items, according to a given config
    """

    def __init__(self, config: RedactionConfig):
        """
        :param RedactionConfig config: The configuration for the redaction
        """
        self._validate_redaction_config(config)
        self.config = config

    def __str__(self):
        return f"{self.__class__.__name__}({self.config.name})"

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        :return str: A unique name for the Redactor implementation class
        """

    @classmethod
    @abstractmethod
    def get_redaction_config_class(cls) -> type[RedactionConfig]:
        """
        :return: The RedactionConfig class that this Redactor expects
        """

    @classmethod
    def _validate_redaction_config(cls, config: RedactionConfig) -> bool:
        """
        Check that the given config is of the expected type

        :raises IncorrectRedactionConfigClassException: If the given config does
        not match the type returned by `get_redaction_config_class`
        """
        expected_class = cls.get_redaction_config_class()
        if type(config) is not expected_class:
            raise IncorrectRedactionConfigClassException(
                f"The config class provided to {cls.__qualname__}.redact is "
                f"incorrect. Expected {expected_class.__qualname__}, but was "
                f"{type(config)}"
            )

    @abstractmethod
    def redact(self) -> RedactionResult:
        """
        Perform a redaction based on the given config


        :param RedactionConfig config: The configuration for the redaction
        :returns RedactionResult: A RedactionResult that holds the result of the
        redaction
        """


class TextRedactor(Redactor):
    """
    Abstract class that represents the redaction of text
    """

    @classmethod
    def get_redaction_config_class(cls):
        return TextRedactionConfig


class LLMTextRedactor(TextRedactor):
    """
    Class that performs text redaction using an LLM

    Loosely based on https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs?view=foundry-classic&tabs=python-secure%2Cdotnet-entra-id&pivots=programming-language-python
    """

    TEXT_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=6000, chunk_overlap=250, separators=["\n\n", "\n", " ", ""]
    )

    @classmethod
    def get_name(cls) -> str:
        return "LLMTextRedaction"

    @classmethod
    def get_redaction_config_class(cls):
        return LLMTextRedactionConfig

    @log_to_appins
    def _analyse_text(self, text_to_analyse: str) -> LLMTextRedactionResult:
        # Initialisation
        # TODO Add LLM parameters to the config class
        self.config: LLMTextRedactionConfig

        if not text_to_analyse:
            LoggingUtil().log_info("No text to analyse, skipping LLM analysis")
            return LLMTextRedactionResult(
                rule_name=self.config.name,
                run_metrics={},
                redaction_strings=(),
                metadata={},
            )

        # Create system prompt from loaded config
        system_prompt = self.config.create_system_prompt()

        # The user's prompt will just be the raw text
        text_chunks = self.TEXT_SPLITTER.split_text(text_to_analyse)
        LoggingUtil().log_info(
            f"The text has been broken down into {len(text_chunks)} chunks"
        )

        # Initialise LLM interface
        llm_util = LLMUtil(self.config)

        # Identify redaction strings
        llm_redaction_result = llm_util.analyse_text(
            system_prompt,
            text_chunks,
        )
        return LLMTextRedactionResult(
            rule_name=self.config.name,
            run_metrics=llm_redaction_result.run_metrics,
            redaction_strings=llm_redaction_result.redaction_strings,
            metadata=llm_redaction_result.metadata,
        )

    def redact(self) -> LLMTextRedactionResult:
        self.config: LLMTextRedactionConfig
        return self._analyse_text(self.config.text)


class ImageRedactor(Redactor):  # pragma: no cover
    """
    Class that performs image redaction

    """

    @classmethod
    def get_name(cls) -> str:
        return "ImageRedaction"

    @classmethod
    def get_redaction_config_class(cls):
        return ImageRedactionConfig

    def redact(self) -> ImageRedactionResult:
        self.config: ImageRedactionConfig
        thresholds = self.config.confidence_thresholds

        self.total_images_to_analyse = len(self.config.images)
        run_metrics = {"total_images_to_analyse": self.total_images_to_analyse}
        if self.total_images_to_analyse == 0:
            LoggingUtil().log_info("No images to analyse, skipping image analysis")
            return ImageRedactionResult(
                rule_name=self.config.name,
                run_metrics=run_metrics,
                redaction_results=(),
            )

        detection_results = []
        for detection_function, detection_type in [
            (AzureVisionUtil.detect_faces_in_images, "face"),
            (SignatureDetector.detect_signatures_in_images, "signature"),
        ]:
            with TimerUtil() as timer:
                results: list[tuple[Image.Image, tuple[str, tuple]]] = (
                    detection_function(
                        self.config.images,
                        getattr(thresholds, f"{detection_type}_detection"),
                    )
                )
                detection_results.append(results)
            run_metrics[f"total_{detection_type}_analysis_time"] = timer.elapsed_time

        run_metrics["total_image_analysis_time"] = sum(
            run_metrics[metric] for metric in run_metrics if "analysis_time" in metric
        )

        redaction_results = self._create_redaction_results(detection_results)

        return ImageRedactionResult(
            rule_name=self.config.name,
            run_metrics=run_metrics,
            redaction_results=redaction_results,
        )

    def _create_redaction_results(
        self,
        detection_results: list[tuple[tuple[Image.Image, tuple[tuple[str, tuple]]]]],
    ) -> tuple[ImageRedactionResult.Result, ...]:
        results: list[ImageRedactionResult.Result] = []

        for i, image in enumerate(self.config.images):
            # Aggregate all detected objects across all object detection types
            bounding_boxes: list[tuple[int, int, int, int]] = []
            object_names: list[str] = []

            for detection_result in detection_results:
                image_to_redact, objects_detected = detection_result[i]

                # Should match since result is returned for all images
                if image_to_redact != image:
                    raise ValueError(
                        f"Image mismatch in detection results: expected {image}, got {image_to_redact}"
                    )

                if len(objects_detected) == 0:
                    continue

                print(objects_detected)
                for name, box in objects_detected:
                    bounding_boxes.append(box)
                    object_names.append(name)

            if not bounding_boxes:
                continue

            results.append(
                ImageRedactionResult.Result(
                    image_dimensions=(
                        image_to_redact.width,
                        image_to_redact.height,
                    ),
                    source_image=image_to_redact,
                    redaction_boxes=tuple(bounding_boxes),
                    names=tuple(object_names),
                )
            )
        return tuple(results)


class ImageTextRedactor(ImageRedactor, TextRedactor):
    """Redactors that redact text content in an image"""

    # Translations to account for common OCR misreads of 0s and 1s
    OCR_TRANSLATIONS: ClassVar[list[dict[int, int]]] = [
        str.maketrans("01", "OI"),
        str.maketrans("OI", "01"),
    ]

    @classmethod
    def get_name(cls) -> str:
        return "ImageTextRedaction"

    @classmethod
    def detect_number_plates(cls, text_to_analyse: str) -> tuple[str]:
        """
        Detect number plates in the given text

        :param str text_to_analyse: The text to analyse for number plates
        :return TextRedactionResult: The redaction result containing the detected
        number plates
        """

        # Regex pattern from https://gist.github.com/danielrbradley/7567269
        uk_number_plate_pattern = (
            r"([A-Z]{2}[0-9]{2}\s[A-Z]{3})"  # Current format: AB12 CDE
            r"|([A-Z][0-9]{1,3}\s[A-Z]{3})"  # Prefix format: A12 BCD
            r"|([A-Z]{3}\s[0-9]{1,3}\s[A-Z])"  # Suffix format: ABC 1 D
            r"|([0-9]{3}\s[DX]{1}\s[0-9]{3})"  # Diplomatic format: 101D234
            r"|([A-Z]{1,2}\s[0-9]{1,4})"  # Dateless format with long number suffix: AB 1234
            r"|([0-9]{1,3}\s[A-Z]{1,3})"  # Dateless format with short number prefix: 123 A
            r"|([0-9]{1,4}\s[A-Z]{1,2})"  # Dateless format with long number prefix: 1234 AB
            r"|([A-Z]{1,3}\s[0-9]{1,4})"  # Northern Ireland format: AIZ 1234
            r"|([A-Z]{1,3}\s[0-9]{1,3})"  # Dateless format with short number suffix: ABC 123
        )
        # Replace any 0s with Os and any 1s with Is to account for common OCR misreads
        matches = []
        for translation in cls.OCR_TRANSLATIONS:
            matches.extend(
                re.findall(
                    uk_number_plate_pattern,
                    text_to_analyse.translate(translation),
                    re.MULTILINE,
                )
            )

        return tuple(
            set(
                chain.from_iterable(
                    [item for item in match if item] for match in matches
                )
            )
        )

    @classmethod
    def examine_redaction_boxes(
        cls,
        text_rect_map: list[tuple[str, tuple[int, int, int, int]]],
        redaction_string: str,
    ) -> list[tuple[int, int, int, int]]:
        """
        Examine the text rectangles and return the bounding boxes that correspond
        to the given redaction string. If it's a multi-term redaction string, then
        the bounding boxes will only be returned if the full sequence is found in
        the correct order.

        :param str text_rect_map: A list of tuples of the form (text_at_box, bounding_box)
        :param str redaction_string: The string to redact
        :return List[Tuple[int, int, int, int]]: A list of bounding boxes that correspond
        to the redaction string
        """
        text_rects_to_redact = []
        words_to_redact = get_normalised_words(redaction_string)

        if len(words_to_redact) == 1:
            for text_at_box, bounding_box in text_rect_map:
                normalised_words = get_normalised_words(text_at_box)
                if normalised_words:
                    normalised_text = normalised_words[0]
                    if words_to_redact[0] == normalised_text:
                        text_rects_to_redact.append(bounding_box)
        else:
            # Multiple words to redact; need to match sequence
            for i, (text_at_box, bounding_box) in enumerate(text_rect_map):
                words_to_redact_copy = words_to_redact.copy()
                first_word = words_to_redact_copy.pop(0)

                # Proceed only if the first word matches
                normalised_words = get_normalised_words(text_at_box)
                if normalised_words and first_word == normalised_words[0]:
                    boxes = [bounding_box]
                    i_copy = i
                    # Check subsequent words
                    while i_copy + 1 < len(text_rect_map) and words_to_redact_copy:
                        word = words_to_redact_copy.pop(0)
                        next_text, next_bounding_box = text_rect_map[i_copy + 1]
                        text_normalised_words = get_normalised_words(next_text)
                        if text_normalised_words:
                            text_normalised = text_normalised_words[0]
                            if word == text_normalised:
                                boxes.append(next_bounding_box)
                                if not words_to_redact_copy:
                                    # All words matched
                                    text_rects_to_redact.extend(boxes)
                                i_copy += 1
                            else:
                                continue

        return text_rects_to_redact

    def _analyse_images(
        self,
    ) -> tuple[list[tuple[Image.Image, tuple[tuple[str, tuple]]]], float]:
        if len(self.config.images) == 0:
            LoggingUtil().log_info("No images to analyse, skipping image text analysis")
            return [], 0.0
        with TimerUtil() as timer:
            vision_util = AzureVisionUtil()
            image_text_rect_map = vision_util.detect_text_in_images(self.config.images)
        return image_text_rect_map, timer.elapsed_time

    def _get_number_plate_redactions(
        self, text_content, text_rect_map
    ) -> tuple[list[tuple], float, float]:
        # Detect number plates using regex
        with TimerUtil() as timer:
            redaction_strings = self.detect_number_plates(text_content)
        number_plate_detection_time = timer.elapsed_time

        # Identify text rectangles to redact based on redaction strings
        text_rects_to_redact = []
        with TimerUtil() as timer:
            for redaction_string in redaction_strings:
                for translation in self.OCR_TRANSLATIONS:
                    translated_redaction = redaction_string.translate(translation)
                    rects_found = self.examine_redaction_boxes(
                        text_rect_map,
                        translated_redaction,
                    )
                    if rects_found:
                        text_rects_to_redact.extend(
                            tuple((rect, translated_redaction) for rect in rects_found)
                        )
        bbox_time = timer.elapsed_time

        return (
            text_rects_to_redact,
            number_plate_detection_time,
            bbox_time,
        )

    @log_to_appins
    def redact(self) -> ImageRedactionResult:
        # Initialisation
        self.config: ImageRedactionConfig
        results = []
        total_images_to_analyse = len(self.config.images)

        with TimerUtil() as timer:
            image_text_rect_map, total_ocr_time = self._analyse_images()

            if not image_text_rect_map:
                timer.__exit__(None, None, None)
                LoggingUtil().log_info(
                    "No text detected in any images, skipping LLM analysis"
                )
                return ImageRedactionResult(
                    rule_name=self.config.name,
                    run_metrics={
                        "total_images_to_analyse": total_images_to_analyse,
                        "total_image_text_analysis_time": timer.elapsed_time,
                        "total_image_ocr_time": total_ocr_time,
                    },
                    redaction_results=(),
                )

            total_number_plate_detection_time = 0.0
            total_bounding_box_time = 0.0
            for image_to_redact, text_rect_map in image_text_rect_map:
                # If image analysis failed, the full image will be returned
                full_image_box = (0, 0, image_to_redact.width, image_to_redact.height)
                if len(text_rect_map) == 1 and text_rect_map[0] == (
                    "Text Detection Failed",
                    full_image_box,
                ):
                    LoggingUtil().log_info(
                        "Text detection failed for image, redacting full image"
                    )
                    results.append(
                        ImageRedactionResult.Result(
                            image_dimensions=(
                                image_to_redact.width,
                                image_to_redact.height,
                            ),
                            source_image=image_to_redact,
                            redaction_boxes=(full_image_box,),
                            names=("Text Detection Failed",),
                        )
                    )
                    continue

                try:
                    text_content = " ".join([x[0] for x in text_rect_map]).strip()
                    if not text_content:
                        LoggingUtil().log_info(
                            "No text detected in image, skipping LLM analysis"
                        )
                        continue
                    LoggingUtil().log_info(
                        f"The following text was extracted from the image: '{text_content}'"
                    )

                    text_rects_to_redact, number_plate_detection_time, bbox_time = (
                        self._get_number_plate_redactions(text_content, text_rect_map)
                    )
                    total_number_plate_detection_time += number_plate_detection_time
                    total_bounding_box_time += bbox_time

                    redaction_result = (
                        ImageRedactionResult.Result.from_image_analysis_results(
                            text_rects_to_redact, image_to_redact
                        )
                    )
                    if redaction_result:
                        results.append(redaction_result)

                except Exception as e:  # noqa: BLE001
                    LoggingUtil().log_exception_with_message(
                        "Error analysing image for text redaction:", e
                    )

        return ImageRedactionResult(
            rule_name=self.config.name,
            run_metrics={
                "total_images_to_analyse": total_images_to_analyse,
                "total_image_text_analysis_time": timer.elapsed_time,
                "total_image_ocr_time": total_ocr_time,
                "total_image_number_plate_detection_time": total_number_plate_detection_time,
                "total_image_text_bounding_box_matching_time": total_bounding_box_time,
            },
            redaction_results=tuple(results),
        )


class ImageLLMTextRedactor(ImageTextRedactor, LLMTextRedactor):
    """
    Class that performs text redaction within images

    """

    @classmethod
    def get_name(cls) -> str:
        return "ImageLLMTextRedaction"

    @classmethod
    def get_redaction_config_class(cls):
        return ImageLLMTextRedactionConfig

    def _analyse_image_text(
        self, image_text_rect_map: tuple[tuple[str, tuple[int, int, int, int]]]
    ) -> tuple[dict[str, Any]]:
        self.config: LLMTextRedactionConfig

        text_content = tuple(
            " ".join([x[0] for x in text_rect_map])
            for _, text_rect_map in image_text_rect_map
        )
        if all(not text for text in text_content):
            LoggingUtil().log_info("No text to analyse, skipping LLM analysis")
            return None
        image_text_content = tuple(
            {
                "image": image_to_redact,
                "text_rect_map": text_rect_map,
                "text_content": text_content[i],
                "text_chunks": self.TEXT_SPLITTER.split_text(text_content[i]),
                "redaction_strings": [],
            }
            for i, (image_to_redact, text_rect_map) in enumerate(image_text_rect_map)
        )

        # Flatten the text chunks from all images into a single list of unique chunks
        text_chunks = list(
            {
                chunk
                for image in image_text_content
                for chunk in image["text_chunks"]
                if image["text_content"]
                != "Text Detection Failed"  # Exclude images where text detection failed
            }
        )

        # Create system prompt from loaded config
        system_prompt = self.config.create_system_prompt()

        # Initialise LLM interface
        llm_util = LLMUtil(self.config)

        # Identify redaction strings
        llm_redaction_result = llm_util.analyse_text(system_prompt, text_chunks)

        redaction_strings = llm_redaction_result.redaction_strings
        for image in image_text_content:
            if not image["text_content"]:
                continue

            for redaction_string in redaction_strings:
                if redaction_string in image["text_content"]:
                    image["redaction_strings"].append(redaction_string)

        return image_text_content

    @classmethod
    def _create_redaction_result(
        cls,
        image_result: dict[str, Any],
    ) -> ImageRedactionResult.Result | None:
        image_to_redact = image_result["image"]
        text_rect_map = image_result["text_rect_map"]
        text_content = image_result["text_content"]

        # If the image couldn't be analysed, mark the whole image for redaction
        if len(text_rect_map) == 1 and text_content == "Text Detection Failed":
            LoggingUtil().log_info(
                "Text detection failed for image, redacting full image"
            )
            return (
                ImageRedactionResult.Result.from_image_analysis_results(
                    [(text_rect_map[0][1], text_rect_map[0][0])], image_to_redact
                ),
                0.0,  # no time spent on bounding boxes
            )

        # Identify text rectangles to redact based on redaction strings
        with TimerUtil() as timer:
            text_rects_to_redact = []
            for redaction_string in image_result["redaction_strings"]:
                rects_found = cls.examine_redaction_boxes(
                    text_rect_map,
                    redaction_string,
                )

                if len(rects_found) > 0:
                    text_rects_to_redact.extend(
                        tuple((rect, redaction_string) for rect in rects_found)
                    )

        redaction_result = ImageRedactionResult.Result.from_image_analysis_results(
            text_rects_to_redact, image_to_redact
        )
        return redaction_result, timer.elapsed_time

    @log_to_appins
    def redact(self) -> ImageRedactionResult:
        # Initialisation
        self.config: ImageLLMTextRedactionConfig
        results = []
        run_metrics = {}
        run_metrics["total_images_to_analyse"] = len(self.config.images)

        with TimerUtil() as timer:
            image_text_rect_map, total_ocr_time = self._analyse_images()
            run_metrics["total_image_ocr_time"] = total_ocr_time

            if not image_text_rect_map:
                timer.__exit__(None, None, None)
                run_metrics["total_image_text_analysis_time"] = timer.elapsed_time
                LoggingUtil().log_info(
                    "No text detected in any images, skipping LLM analysis"
                )
                return ImageRedactionResult(
                    rule_name=self.config.name,
                    run_metrics=run_metrics,
                    redaction_results=(),
                )

            with TimerUtil() as llm_timer:
                image_text_redaction_results = self._analyse_image_text(
                    image_text_rect_map
                )
            run_metrics["total_image_llm_analysis_time"] = llm_timer.elapsed_time

            if not image_text_redaction_results:
                timer.__exit__(None, None, None)
                run_metrics["total_image_text_analysis_time"] = timer.elapsed_time
                return ImageRedactionResult(
                    rule_name=self.config.name,
                    run_metrics=run_metrics,
                    redaction_results=(),
                )

            total_bounding_box_time = 0.0
            for image_result in image_text_redaction_results:
                redaction_result, bbox_time = self._create_redaction_result(
                    image_result
                )
                total_bounding_box_time += bbox_time
                if redaction_result:
                    results.append(redaction_result)

        run_metrics["total_image_text_bounding_box_matching_time"] = (
            total_bounding_box_time
        )
        run_metrics["total_image_text_analysis_time"] = timer.elapsed_time

        return ImageRedactionResult(
            rule_name=self.config.name,
            run_metrics=run_metrics,
            redaction_results=tuple(results),
        )


class RedactorFactory:
    """
    Class for generating Redactor classes by name
    """

    REDACTOR_TYPES: ClassVar[list[type[Redactor]]] = [
        LLMTextRedactor,
        ImageRedactor,
        ImageTextRedactor,
        ImageLLMTextRedactor,
    ]
    """The Redactor classes that are known to the factory"""

    @classmethod
    def _validate_redactor_types(cls):
        """
        Validate the REDACTOR_TYPES and return a map of type_name: Redactor
        """
        name_map: dict[str, list[type[Redactor]]] = {}
        for redactor_type in cls.REDACTOR_TYPES:
            type_name = redactor_type.get_name()
            if type_name in name_map:
                name_map[type_name].append(redactor_type)
            else:
                name_map[type_name] = [redactor_type]
        invalid_types = {k: v for k, v in name_map.items() if len(v) > 1}
        if invalid_types:
            raise DuplicateRedactorNameException(
                "The following Redactor implementation classes had duplicate names: "
                + json.dumps(invalid_types, indent=4, default=str)
            )
        return {k: v[0] for k, v in name_map.items()}

    @classmethod
    def get(cls, redactor_type: str) -> type[Redactor]:
        """
        Return the Redactor that is identified by the provided type name

        :param str redactor_type: The Redactor type name (which aligns with the
        get_name method of the Redactor)
        :return Type[Redactor]: The redactor instance identified by the provided
        redactor_type
        :raises RedactorNameNotFoundException if the given redactor_type is not
        found
        """
        if not isinstance(redactor_type, str):
            raise TypeError(
                f"RedactorFactory.get expected a str, but got a {type(redactor_type)}"
            )
        name_map = cls._validate_redactor_types()
        if redactor_type not in name_map:
            raise RedactorNameNotFoundException(
                f"No redactor could be found for redactor type '{redactor_type}'"
            )
        return name_map[redactor_type]
