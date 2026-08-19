from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from core.util.types import PydanticImage


class LLMUtilConfig(BaseModel):
    model: str
    """The LLM model to use"""
    max_tokens: int | None = 1000
    """Maximum number of tokens per completion"""
    temperature: float | None = 0.5
    """LLM sampling temperature"""
    request_rate_limit: int | None = None
    """Maximum number of requests per minute. Defaults to 20% of model max RPM."""
    token_rate_limit: int | None = None
    """Number of tokens allowed per minute. Defaults to 20% of model max TPM."""
    max_concurrent_requests: int | None = None
    """Number of concurrent requests to allow. Assigns the number of threads."""
    token_encoding_name: str | None = None
    """The token encoding name to use for estimating token counts"""
    n: int | None = 1
    """Number of completions to generate per prompt"""
    budget: float | None = None
    """The budget in GBP for LLM usage"""
    token_timeout: float | None = 60.0
    """The timeout in seconds for acquiring tokens from the token semaphore"""
    request_timeout: float | None = 60.0
    """The timeout in seconds for acquiring request semaphore"""

    def __init__(self, **data):
        super().__init__(**data)
        if self.token_encoding_name is None:
            if self.model.startswith("gpt-5") or self.model.startswith("gpt-4o"):
                self.token_encoding_name = "o200k_base"  # nosec: B105
            elif self.model.startswith("gpt-4") or self.model.startswith("gpt-3.5"):
                self.token_encoding_name = "cl100k_base"  # nosec: B105
            else:
                self.token_encoding_name = "p50k_base"  # nosec: B105


class RedactionConfig(BaseModel):
    name: str
    label: str | None = None
    redactor_type: str
    """The redactor the config should be fed into"""


class TextRedactionConfig(RedactionConfig):
    text: str | None = None
    """The source text to redact"""


class LLMTextRedactionConfigBase(RedactionConfig, LLMUtilConfig):
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
