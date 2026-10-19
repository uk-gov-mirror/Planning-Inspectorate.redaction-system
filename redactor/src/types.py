from dataclasses import dataclass, field
from typing import Annotated, Any

from PIL.Image import Image
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

# This is baed on https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types


class _ImageAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """
        We return a pydantic_core.CoreSchema that behaves in the following ways:

        * ints will be parsed as `Image` instances with the int as the x attribute
        * `Image` instances will be parsed as `Image` instances without any changes
        * Nothing else will pass validation
        * Serialization will always return just an int
        """

        def validate_from_int(value: int) -> Image:
            result = Image()
            result.x = value
            return result

        from_int_schema = core_schema.chain_schema(
            [
                core_schema.int_schema(),
                core_schema.no_info_plain_validator_function(validate_from_int),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_int_schema,
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(Image),
                    from_int_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.x
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for `int`
        return handler(core_schema.int_schema())


PydanticImage = Annotated[Image, _ImageAnnotation]


@dataclass(frozen=True)
class RedactionResult:
    rule_name: str
    """The name of the redaction rule that generated the result"""
    run_metrics: dict[str, int | float | str]
    """Any analytical metrics for the result"""


@dataclass(frozen=True)
class ImageRedactionResult(RedactionResult):
    @dataclass(frozen=True)
    class Result:
        image_dimensions: tuple[int, int]
        """The dimensions of the image"""
        source_image: PydanticImage
        """The source image"""
        redaction_boxes: tuple[tuple[int, int, int, int]] = field(
            default_factory=lambda: ()
        )
        """The list redaction boxes to draw on the image, in the image's local space. This is of the form (top left corner x, top left corner y, width, height)"""
        names: tuple[str] = field(default_factory=lambda: ())
        """The list of names associated with the redaction boxes"""

    @classmethod
    def create_result(
        cls,
        text_rects_to_redact: list[tuple[tuple[int, int, int, int], str]],
        image_to_redact: PydanticImage,
    ) -> "ImageRedactionResult.Result":
        """
        Create an ImageRedactionResult.Result from the given text rects to redact and the source image.

        :param list[tuple[tuple[int, int, int, int], str]] text_rects_to_redact: A list of tuples containing the bounding box and the associated name to redact
        :param PydanticImage image_to_redact: The source image

        :return ImageRedactionResult.Result: The resulting ImageRedactionResult.Result object
        """
        text_rects_to_redact = list(dict.fromkeys(text_rects_to_redact))
        if not text_rects_to_redact:
            return None

        redaction_boxes = tuple(rect for rect, _ in text_rects_to_redact)
        names = tuple(name for _, name in text_rects_to_redact)
        return cls.Result(
            image_dimensions=image_to_redact.size,
            source_image=image_to_redact,
            redaction_boxes=redaction_boxes,
            names=names,
        )

    redaction_results: tuple[Result]
    """A list of ImageRedactionResult.Result objects"""


@dataclass(frozen=True)
class TextRedactionResult(RedactionResult):
    redaction_strings: tuple[str] = field(default_factory=lambda: ())
    """The list of strings to redact"""


@dataclass(frozen=True)
class ImageTextRedactionResult(TextRedactionResult, ImageRedactionResult):
    pass


@dataclass(frozen=True)
class LLMTextRedactionResult(TextRedactionResult):
    @dataclass(frozen=True)
    class LLMResultMetadata:
        request_count: int = field(default=0)
        input_token_count: int = field(default=0)
        completion_token_count: int = field(default=0)
        total_token_count: int = field(default=0)
        total_cost: float = field(default=0.0)
        cached_input_token_count: int = field(default=0)
        """Input tokens served from the prompt cache (a subset of input_token_count)"""
        cache_write_token_count: int = field(default=0)
        """Tokens used for writing to the prompt cache (a subset of input_token_count)"""
        reasoning_token_count: int = field(default=0)
        """Reasoning tokens used by the model (a subset of completion_token_count)"""
        failed_chunk_count: int = field(default=0)
        """Chunks that returned no result after all retries, e.g. due to truncation"""

    metadata: LLMResultMetadata = field(default=None)
    """Any metadata provided by the LLM"""


@dataclass(frozen=True)
class ImageLLMTextRedactionResult(LLMTextRedactionResult, ImageRedactionResult):
    pass


class LLMRedactionResultFormat(BaseModel):
    redaction_strings: list[str]
