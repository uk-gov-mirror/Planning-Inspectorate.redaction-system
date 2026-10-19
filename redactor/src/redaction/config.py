from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, model_validator

from ..analysis.text import LLMTextAnalyserConfig
from ..types import PydanticImage


class InvalidRedactionConfigException(Exception):  # pragma: no cover
    pass


class RedactionConfig(BaseModel):
    name: str
    label: str | None = None
    redactor_type: str
    """The redactor the config should be fed into"""


class TextRedactionConfig(RedactionConfig):
    text: str | None = None
    """The source text to redact"""


class LLMTextRedactionConfigBase(RedactionConfig, LLMTextAnalyserConfig):
    system_prompt: str
    """The system prompt for the LLM"""
    redaction_terms: list[str]
    """A list of redaction rule strings to apply"""
    constraints: list[str] = None
    """A list of constraint strings to apply"""
    output_format: str = (
        "<OutputFormat> You respond in JSON format. You return the "
        "successfully extracted terms from the text in JSON list named "
        '"terms". List them as they appear in the text. '
        "</OutputFormat>"
    )
    chunk_size: int = Field(1500, gt=0)
    """Maximum number of tokens in each chunk of text sent to the LLM"""
    chunk_overlap: int = Field(60, ge=0)
    """Number of tokens shared between consecutive chunks"""

    @model_validator(mode="after")
    def _check_chunk_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create a token-based text splitter using this config's chunk settings"""
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.token_encoding_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )


class LLMTextRedactionConfig(TextRedactionConfig, LLMTextRedactionConfigBase):
    def create_system_prompt(self) -> str:
        system_prompt_list: list[str] = []
        # Add the system role and redaction_terms to redact
        system_prompt_list.append(xml_format(self.system_prompt, "SystemRole"))
        system_prompt_list.append(
            xml_format(self.redaction_terms, "Terms", as_list=True)
        )

        # Add the output format instructions
        system_prompt_list.append(self.output_format)

        # Add any constraints to the System prompt
        if self.constraints:
            system_prompt_list.append(
                xml_format(self.constraints, "Constraints", as_list=True)
            )

        # Add the defined redaction rules to the System prompt
        prompt_template_string = "\n\n".join(system_prompt_list)

        system_prompt_template = PromptTemplate(
            input_variables=["chunk"],
            template=prompt_template_string,
        )
        return system_prompt_template.format()


def xml_format(input: str | list, format_string: str, as_list: bool = False) -> str:
    """Wrap the input string in XML tags of the given format string"""
    if isinstance(input, list):
        if as_list:
            joined_input = "\n".join(
                ["- " + x if not x.startswith("-") else x for x in input]
            )
        else:
            joined_input = "\n".join(
                [x + "." if not x.endswith(".") else x for x in input]
            )
        return f"<{format_string}>\n{joined_input}\n</{format_string}>"
    return f"<{format_string}>\n{input}\n</{format_string}>"


class ImageRedactionConfig(RedactionConfig):
    images: list[PydanticImage] | None = None
    """The images to redact"""

    class ConfidenceThresholdConfig(BaseModel):
        """Confidence thresholds for object detect models. Values should be between 0
        and 1.0"""

        face_detection: float | None = Field(0.5, ge=0.0, le=1.0)
        """Confidence threshold for face detection model"""
        signature_detection: float | None = Field(0.5, ge=0.0, le=1.0)
        """Confidence threshold for signature detection model"""

    confidence_thresholds: ConfidenceThresholdConfig = ConfidenceThresholdConfig()
    """Confidence thresholds for object detection models"""


class ImageLLMTextRedactionConfig(LLMTextRedactionConfig):
    images: list[PydanticImage] | None = None
    """The images to redact"""

    rendered_images: list | None = None
    """list[PDFImageMetadata] for rendered page images with pre-populated text_rect_map"""
