import re

from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, field_validator, model_validator

from ..analysis.text import LLMTextAnalyserConfig
from ..types import PydanticImage

RESERVED_SECTION_TAGS = (
    "Role",
    "Task",
    "Terms",
    "MustRedact",
    "DoNotRedact",
    "Examples",
)
"""XML tags of the sections built from the standard config fields"""
EXTRA_SECTION_TAG = re.compile(r"[A-Z][A-Za-z0-9]*")
"""CamelCase names allowed for extra_sections tags"""


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
    role: str
    """A short statement of the assistant's role and the domain context for the
    redaction task"""
    task: str
    """The specific instruction describing what counts as personal data here and how
    to apply it consistently"""
    redaction_terms: list[str]
    """A list of redaction rule strings to apply"""
    must_redact: list[str] | None = None
    """A list of additional rules clarifying what must be redacted"""
    do_not_redact: list[str] | None = None
    """A list of exceptions/exclusions describing what must never be redacted"""
    examples: list[str] | None = None
    """A list of worked examples illustrating correct redaction decisions for
    ambiguous cases"""
    extra_sections: dict[str, list[str]] | None = None
    """Further prompt sections, keyed by CamelCase XML tag and rendered after the
    examples as a bullet list. Unlike the sections above, they carry no clause IDs"""
    chunk_size: int = Field(1500, gt=0)
    """Maximum number of tokens in each chunk of text sent to the LLM"""
    chunk_overlap: int = Field(60, ge=0)
    """Number of tokens shared between consecutive chunks"""

    @field_validator("extra_sections")
    @classmethod
    def _check_extra_sections(cls, sections: dict[str, list[str]] | None):
        for tag, items in (sections or {}).items():
            if not EXTRA_SECTION_TAG.fullmatch(tag):
                raise ValueError(f"Extra section tag '{tag}' must be CamelCase")
            if tag in RESERVED_SECTION_TAGS:
                raise ValueError(f"Extra section tag '{tag}' is already a section")
            if not items:
                raise ValueError(f"Extra section '{tag}' needs at least one item")
        return sections

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
        # Add the role, task and redaction_terms to redact
        system_prompt_list.append(xml_format(self.role, "Role"))
        system_prompt_list.append(xml_format(self.task, "Task"))
        system_prompt_list.append(
            xml_format(self.redaction_terms, "Terms", as_list=True)
        )

        # Add any rules clarifying what must be redacted
        if self.must_redact:
            system_prompt_list.append(
                xml_format(self.must_redact, "MustRedact", as_list=True)
            )

        # Add any exceptions/exclusions describing what must never be redacted
        if self.do_not_redact:
            system_prompt_list.append(
                xml_format(self.do_not_redact, "DoNotRedact", as_list=True)
            )

        # Add any worked examples
        if self.examples:
            system_prompt_list.append(
                xml_format(self.examples, "Examples", as_list=True)
            )

        # Add any further sections, in the order given
        for tag, items in (self.extra_sections or {}).items():
            system_prompt_list.append(xml_format(items, tag, as_list=True))

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
