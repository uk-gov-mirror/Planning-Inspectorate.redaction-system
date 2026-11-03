import re
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pymupdf
import pytest
from PIL import Image

from src.analysis.images import AzureImageAnalyser
from src.monitoring import LoggingUtil
from src.redaction.file_processor import NonEnglishContentException, PDFProcessor
from src.redaction.utils.pdf import (
    PDFImageMetadata,
    PDFLineMetadata,
    PDFPageMetadata,
    PDFUtil,
)
from src.redaction.utils.text import get_normalised_words, is_english_text
from src.types import (
    ImageLLMTextRedactionResult,
    ImageRedactionResult,
    LLMTextRedactionResult,
    TextRedactionResult,
)
from tests.utils import (
    REDACTED_JPG,
    SOURCE_PDF,
    TEXT_IMAGE_PROPOSED_PDF,
    TEXT_IMAGE_REDACTED_PDF,
    TRANSLATED_IMAGE_PDF,
    open_image,
    open_pdf,
)


def _make_pdf_with_text(text: str) -> BytesIO:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    b = BytesIO()
    doc.save(b)
    b.seek(0)
    return b


def _make_pdf_with_pages(*texts: str) -> BytesIO:
    doc = pymupdf.open()
    for text in texts:
        doc.new_page().insert_text((72, 72), text)
    b = BytesIO()
    doc.save(b)
    b.seek(0)
    return b


def _occurrence(chunk_index, span_offsets, **kwargs):
    return {
        "chunk_index": chunk_index,
        "chunk_id": f"id{chunk_index}",
        "span_offsets": span_offsets,
        **kwargs,
    }


class TestGetName:
    def test_returns_str(self):
        """
        - When get_name is called
        - The return value must be a string
        """
        assert isinstance(PDFProcessor.get_name(), str)


class TestExtractPDFAnnotations:
    @pytest.fixture(autouse=True)
    def _setup(self):
        mock_document = pymupdf.open()
        mock_page = mock_document.new_page()

        mock_types = [
            pymupdf.PDF_ANNOT_HIGHLIGHT,
            pymupdf.PDF_ANNOT_HIGHLIGHT,
            pymupdf.PDF_ANNOT_REDACT,
        ]
        vertices = (
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(2, 2), (2, 3), (3, 2), (3, 3)],
            [(4, 4), (4, 5), (5, 4), (5, 5)],
        )
        types = ((8, "Highlight"), (8, "Highlight"), (12, "Redact"))

        self.mock_annotations = [MagicMock(spec=pymupdf.Annot) for _ in range(3)]
        for i, mock_annotation in enumerate(self.mock_annotations):
            mock_annotation.info = {
                "content": f"Annotation {i}",
            }
            mock_annotation.type = types[i]
            mock_annotation.vertices = vertices[i]
            mock_annotation._yielded = False

        with (
            patch("pymupdf.open", return_value=mock_document),
            patch(
                "pymupdf.Page.annot_xrefs",
                return_value=zip(self.mock_annotations, mock_types),
            ),
            patch(
                "pymupdf.Page.load_annot",
                side_effect=self.mock_annotations,
            ),
            patch("pymupdf.Page.get_text", side_effect=["hello", "world"]),
            patch("pymupdf.mupdf.pdf_annot_page", return_value=mock_page),
        ):
            yield

    def test_returns_all_annotations(self):
        expected_annotations = (
            {
                "page_number": 0,
                "annotations": [
                    {
                        "annot": self.mock_annotations[0],
                        "content": "Annotation 0",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(0, 0, 1, 1),
                        "text": "hello",
                    },
                    {
                        "annot": self.mock_annotations[1],
                        "content": "Annotation 1",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(2, 2, 3, 3),
                        "text": "world",
                    },
                    {
                        "annot": self.mock_annotations[2],
                        "content": "Annotation 2",
                        "type": "Redact",
                        "rect": pymupdf.Rect(4, 4, 5, 5),
                    },
                ],
            },
        )
        actual_annotations = PDFProcessor()._extract_pdf_annotations(
            BytesIO(), return_annot=True
        )

        assert expected_annotations == actual_annotations

    def test_returns_only_highlight_annotations(self):
        expected_annotations = (
            {
                "page_number": 0,
                "annotations": [
                    {
                        "content": "Annotation 0",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(0, 0, 1, 1),
                        "text": "hello",
                    },
                    {
                        "content": "Annotation 1",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(2, 2, 3, 3),
                        "text": "world",
                    },
                ],
            },
        )
        actual_annotations = PDFProcessor()._extract_pdf_annotations(
            BytesIO(), annotation_class=[pymupdf.PDF_ANNOT_HIGHLIGHT]
        )

        assert expected_annotations == actual_annotations


class TestGetProposedRedactions:
    STROKE_COLOUR = (0.2157, 0.898, 1.0)

    def test_returns_all_highlights(self):
        creation_date = "D:20260103123456+01'00'"
        creation_timestamp = datetime(
            year=2026, month=1, day=3, hour=12, minute=34, second=56, tzinfo=UTC
        )
        annotations = (
            {
                "page_number": "0",
                "annotations": [
                    {
                        "title": "Text Redaction",
                        "content": "Redact this",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(0, 0, 1, 1),
                        "text": "Redact this",
                        "creationDate": creation_date,
                        "modDate": "",
                        "stroke": self.STROKE_COLOUR,
                    },
                    {
                        "title": "Text Redaction",
                        "content": "Redact this too",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(2, 2, 3, 3),
                        "text": "Redact this",
                        "creationDate": creation_date,
                        "modDate": "",
                        "stroke": self.STROKE_COLOUR,
                    },
                    {
                        "title": "Text Redaction",
                        "content": "Redact this too",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(0, 2, 1, 3),
                        "text": "too.",
                        "creationDate": creation_date,
                        "modDate": "",
                        "stroke": self.STROKE_COLOUR,
                    },
                ],
            },
        )
        document_bytes = BytesIO()
        expected_dict = [
            {
                "pageNumber": 0,
                "annotations": [
                    {
                        "title": "Text Redaction",
                        "annotationType": "Highlight",
                        "proposedRedaction": "Redact this",
                        "annotatedText": "Redact this",
                        "rect": (0.0, 0.0, 1.0, 1.0),
                        "creationDate": creation_timestamp,
                        "isRedactionCandidate": True,
                        "modDate": None,
                    },
                    {
                        "title": "Text Redaction",
                        "annotationType": "Highlight",
                        "proposedRedaction": "Redact this too",
                        "annotatedText": "Redact this",
                        "rect": (2.0, 2.0, 3.0, 3.0),
                        "creationDate": creation_timestamp,
                        "isRedactionCandidate": True,
                        "modDate": None,
                    },
                    {
                        "title": "Text Redaction",
                        "annotationType": "Highlight",
                        "proposedRedaction": "Redact this too",
                        "annotatedText": "too.",
                        "rect": (0.0, 2.0, 1.0, 3.0),
                        "creationDate": creation_timestamp,
                        "isRedactionCandidate": True,
                        "modDate": None,
                    },
                ],
            }
        ]
        with patch.object(
            PDFProcessor, "_extract_pdf_annotations", return_value=annotations
        ):
            actual_dict = PDFProcessor().get_proposed_redactions(document_bytes)
        assert expected_dict == actual_dict

    def test_malformed_pdf_date(self):
        """
        - Given I have annotation metadata with a malformed PDF date
        - When I call get_proposed_redactions
        - Then the malformed date should be treated as missing metadata
        """
        creation_date = "D:20260103123456"
        creation_timestamp = datetime(
            year=2026, month=1, day=3, hour=12, minute=34, second=56, tzinfo=UTC
        )
        annotations = (
            {
                "page_number": "0",
                "annotations": [
                    {
                        "title": "Text Redaction",
                        "content": "Redact this",
                        "type": "Highlight",
                        "rect": pymupdf.Rect(0, 0, 1, 1),
                        "text": "Redact this",
                        "creationDate": creation_date,
                        "modDate": "D:-001-1-1-1-1-1",
                        "stroke": self.STROKE_COLOUR,
                    },
                ],
            },
        )
        document_bytes = BytesIO()
        expected_dict = [
            {
                "pageNumber": 0,
                "annotations": [
                    {
                        "title": "Text Redaction",
                        "annotationType": "Highlight",
                        "proposedRedaction": "Redact this",
                        "annotatedText": "Redact this",
                        "rect": (0.0, 0.0, 1.0, 1.0),
                        "creationDate": creation_timestamp,
                        "isRedactionCandidate": True,
                        "modDate": None,
                    },
                ],
            }
        ]

        with patch.object(
            PDFProcessor, "_extract_pdf_annotations", return_value=annotations
        ):
            actual_dict = PDFProcessor().get_proposed_redactions(document_bytes)

        assert expected_dict == actual_dict


class TestGetRedactorLabel:
    def _create_mock_redaction_config(self, name: str, label: str | None = None):
        config = Mock()
        config.redactor_type = "llm_text"
        config.name = name
        config.label = label
        config.text = None
        config.images = None
        return config

    def test_returns_label(self):
        """
        Given I have a PDFProcessor with a text_redaction_summary attribute
        When I call _get_redactor_label with a redactor name
        Then I should receive the label for that redactor from the text_redaction_summary
        """
        pdf_processor = PDFProcessor()
        pdf_processor.redaction_rules = [
            self._create_mock_redaction_config(
                "TextRedactor1", label="Labelled Redactor"
            ),
            self._create_mock_redaction_config("TextRedactor2"),
        ]
        pdf_processor._text_redaction_summary = {
            "TextRedactor1": {"redaction_strings": ["Commander Data"]},
            "TextRedactor2": {"redaction_strings": ["Phillipa"]},
        }

        label = pdf_processor._get_redactor_label("Commander Data")
        assert label == "Labelled Redactor"

        label = pdf_processor._get_redactor_label("Phillipa")
        assert label == "TextRedactor2"


class TestExamineApplyRedactionsBase:
    @staticmethod
    def create_mock_page_metadata(
        page_number,
        text_content: str | None = None,
        lines=None,
        y0=None,
        y1=None,
        x0=None,
        x1=None,
    ):
        line_metadata = []

        if lines:
            for i, line in enumerate(lines):
                normalised_words = get_normalised_words(line)
                line_metadata.append(
                    PDFLineMetadata(
                        line_number=i,
                        words=np.array(normalised_words, dtype=str),
                        y0=y0[i],
                        y1=y1[i],
                        x0=tuple(x0[i]),
                        x1=tuple(x1[i]),
                    )
                )
        return PDFPageMetadata(
            page_number=page_number,
            lines=line_metadata,
            raw_text=text_content if text_content else "",
            rendered_image=PDFImageMetadata(
                source_image_resolution=(100, 100),
                file_format="jpg",
                page_number=page_number,
                image=Image.new("RGB", (100, 100)),
                image_transform_in_pdf=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            ),
        )


class TestExamineProvisionalRedactionsOnPage(TestExamineApplyRedactionsBase):
    def test_returns_match_on_page(self):
        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content="Hello World",
            lines=["Hello", "World"],
            y0=[0, 20],
            y1=[10, 30],
            x0=[[0], [0]],
            x1=[[10], [10]],
        )
        term = "Hello"
        rect = pymupdf.Rect(0, 0, 10, 10)

        expected_result = [(0, rect, term)]
        with patch.object(
            PDFUtil,
            "examine_provisional_text_redaction",
            return_value=expected_result,
        ):
            pdf_processor = PDFProcessor()
            pdf_processor.file_bytes = BytesIO()  # Dummy value for file_bytes
            pdf_processor.terms_found = {term: 0}

            result = pdf_processor._examine_provisional_redactions_on_page(
                [term], page_metadata
            )

        assert result == expected_result

    def test_term_is_only_audit_logged(self):
        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content="Hello World",
            lines=["Hello", "World"],
            y0=[0, 20],
            y1=[10, 30],
            x0=[[0], [0]],
            x1=[[10], [10]],
        )
        term = "Hello"

        with patch.object(
            PDFUtil, "examine_provisional_text_redaction", return_value=[]
        ):
            pdf_processor = PDFProcessor()
            pdf_processor.terms_found = {term: 0}
            pdf_processor._examine_provisional_redactions_on_page([term], page_metadata)

        audit_messages = [c.args[0] for c in LoggingUtil.log_audit.call_args_list]
        info_messages = [c.args[0] for c in LoggingUtil.log_info.call_args_list]
        assert any(term in message for message in audit_messages)
        assert not any(term in message for message in info_messages)

    def test_returns_line_broken_match(self):
        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content="Hello\nWorld",
            lines=["Hello", "World"],
            y0=[0, 20],
            y1=[10, 30],
            x0=[[0], [0]],
            x1=[[10], [10]],
        )
        term = "Hello World"
        rect = pymupdf.Rect(0, 0, 10, 10)
        next_rect = pymupdf.Rect(0, 20, 10, 30)

        expected_result = [(0, rect, term), (0, next_rect, term)]
        side_effects = [
            [(0, rect, term), (0, next_rect, term)],
            [],
        ]
        with (
            patch.object(
                PDFUtil,
                "examine_provisional_text_redaction",
                side_effect=side_effects,
            ),
        ):
            pdf_processor = PDFProcessor()
            pdf_processor.terms_found = {term: 0}
            result = pdf_processor._examine_provisional_redactions_on_page(
                [term],
                page_metadata,
            )

        assert result == expected_result


class TestApplyProvisionalTextRedactions(TestExamineApplyRedactionsBase):
    def test_skips_no_text_on_page(self):
        file_bytes = _make_pdf_with_text("")
        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content="",
            lines=[],
            y0=[],
            y1=[],
            x0=[],
            x1=[],
        )
        term = "Hello World"

        with (
            patch.object(
                PDFUtil,
                "extract_page_metadata",
                return_value=page_metadata,
            ),
            patch.object(
                PDFUtil,
                "get_next_page_metadata",
                return_value=None,
            ),
            patch.object(
                PDFProcessor,
                "_examine_provisional_redactions_on_page",
            ) as mock_examine_provisional_redactions_on_page,
            patch.object(
                PDFUtil,
                "add_provisional_redaction",
            ) as mock_add_provisional_redaction,
        ):
            pdf_processor = PDFProcessor()
            pdf_processor.terms_found = {term: 0}
            pdf_processor._apply_provisional_text_redactions(
                file_bytes,
                [term],
            )

        mock_examine_provisional_redactions_on_page.assert_not_called()
        mock_add_provisional_redaction.assert_not_called()


class TestApplyRule:
    @staticmethod
    def _image_llm_text_result():
        image = Image.new("RGB", (100, 100))
        return ImageLLMTextRedactionResult(
            rule_name="ImageRedactor1",
            run_metrics={"total_images_to_analyse": 1},
            redaction_results=(
                ImageRedactionResult.Result(
                    image_dimensions=image.size,
                    source_image=image,
                    redaction_boxes=((0, 0, 100, 100),),
                    names=("John Smith",),
                ),
            ),
            redaction_strings=("John Smith",),
        )

    def test_result_to_loggable_dict_excludes_source_images(self):
        result = self._image_llm_text_result()

        with patch("copy.deepcopy", side_effect=AssertionError("deepcopy called")):
            loggable = PDFProcessor._result_to_loggable_dict(result)

        assert loggable["redaction_strings"] == ("John Smith",)
        assert loggable["redaction_results"] == [
            {
                "image_dimensions": (100, 100),
                "source_image": None,
                "redaction_boxes": ((0, 0, 100, 100),),
                "names": ("John Smith",),
            }
        ]

    def test_result_to_loggable_dict_text_result(self):
        result = TextRedactionResult(
            rule_name="TextRedactor1",
            run_metrics={},
            redaction_strings=("John Smith",),
        )

        assert PDFProcessor._result_to_loggable_dict(result) == {
            "rule_name": "TextRedactor1",
            "run_metrics": {},
            "redaction_strings": ("John Smith",),
        }

    def test_result_is_only_audit_logged(self):
        result = self._image_llm_text_result()
        processor = PDFProcessor()
        processor.run_metrics = {
            "text_analysis_total_time": 0.0,
            "image_analysis_total_time": 0.0,
        }
        processor._text_redaction_summary = {}

        processor._apply_rule(Mock(redact=Mock(return_value=result)))

        audit_messages = [c.args[0] for c in LoggingUtil.log_audit.call_args_list]
        info_messages = [c.args[0] for c in LoggingUtil.log_info.call_args_list]
        assert any("John Smith" in message for message in audit_messages)
        assert not any("John Smith" in message for message in info_messages)
        assert processor.redaction_results == [result]

    def test_no_decisions_leaves_summary_unchanged(self):
        result = LLMTextRedactionResult(
            rule_name="names", run_metrics={}, redaction_strings=("Joan Brown",)
        )
        processor = PDFProcessor()
        processor.run_metrics = {"text_analysis_total_time": 0.0}
        processor._text_redaction_summary = {}
        processor.redaction_results = []

        processor._apply_rule(Mock(redact=Mock(return_value=result)))

        assert processor._text_redaction_summary["names"] == {
            "redaction_strings": ("Joan Brown",),
            "n_proposed": 1,
            "n_applied": 1,
        }


class TestApplyRedactionRules(TestExamineApplyRedactionsBase):
    def test_includes_redaction_strings_in_text_summary(self):
        image = Image.new("RGB", (100, 100))

        text_redactor_config = Mock()
        text_redactor_config.redactor_type = "llm_text"
        image_redactor_config = Mock()
        image_redactor_config.redactor_type = "image_llm_text"
        mock_redaction_rules = [text_redactor_config, image_redactor_config]

        page_text = "Hello World this is a test document with English text content"

        mock_text_redact = Mock(
            redact=Mock(
                return_value=TextRedactionResult(
                    rule_name="TextRedactor1",
                    run_metrics={"tokens_used": 10},
                    redaction_strings=("Hello", "World"),
                )
            )
        )
        mock_image_llm_text_redact = Mock(
            redact=Mock(
                return_value=ImageLLMTextRedactionResult(
                    rule_name="ImageRedactor1",
                    run_metrics={"total_images_to_analyse": 1},
                    redaction_results=(
                        ImageRedactionResult.Result(
                            image_dimensions=image.size,
                            source_image=image,
                            redaction_boxes=((0, 0, 100, 100),),
                            names=("test_redaction",),
                        ),
                    ),
                    redaction_strings=("test_redaction",),
                )
            )
        )
        redactor_factory_get_side_effect = [
            lambda config, r=mock_text_redact: r,
            lambda config, r=mock_image_llm_text_redact: r,
        ]

        with (
            patch.object(PDFUtil, "extract_unique_pdf_images", return_value=[image]),
            patch(
                "src.redaction.file_processor.RedactorFactory.get",
                side_effect=redactor_factory_get_side_effect,
            ),
        ):
            processor = PDFProcessor()
            processor.redaction_rules = mock_redaction_rules
            processor.pages_metadata = [
                self.create_mock_page_metadata(
                    page_number=0,
                    text_content=page_text,
                    lines=[page_text],
                    y0=[0],
                    x0=[[10 * i for i in range(len(page_text.split()))]],
                    y1=[10],
                    x1=[[10 * (i + 1) for i in range(len(page_text.split()))]],
                )
            ]
            processor.pdf_text = page_text
            processor.pdf_images = [
                PDFImageMetadata(
                    source_image_resolution=(100, 100),
                    file_format="jpeg",
                    image=image,
                    page_number=0,
                    image_transform_in_pdf=(75.0, 0.0, -0.0, 75.0, 0.0, -0.0),
                )
            ]
            processor._text_redaction_summary = {}

            processor._apply_redaction_rules()

        mock_text_redact.redact.assert_called_once()
        mock_image_llm_text_redact.redact.assert_called_once()

        expected_text_redaction_summary = {
            "TextRedactor1": {
                "redaction_strings": ("Hello", "World"),
                "n_proposed": 2,
                "n_applied": 2,
            },
            "ImageRedactor1": {
                "redaction_strings": ("test_redaction",),
                "n_proposed": 1,
                "n_applied": 1,
            },
        }
        assert processor._text_redaction_summary == expected_text_redaction_summary

    def test_image_text_rule_not_given_full_page_with_text(self):
        """
        - Given a PDF page with extractable text and no embedded images
        - When I apply the redaction rules
        - Then the image text rule is given no images, while the image rule
          (face/signature) is still given the full page
        """
        page_image = Image.new("RGB", (100, 100), "white")
        page_metadata = PDFPageMetadata(
            page_number=0,
            raw_text="Hello World",
            rendered_image=PDFImageMetadata(
                source_image_resolution=(100, 100),
                file_format="png",
                image=page_image,
                page_number=0,
                image_transform_in_pdf=(100.0, 0.0, 0.0, 100.0, 0.0, 0.0),
            ),
        )
        image_text_rule = Mock(spec=["name", "redactor_type", "images"])
        image_text_rule.name = "image text rule"
        image_text_rule.redactor_type = "ImageTextRedaction"
        image_rule = Mock(spec=["name", "redactor_type", "images"])
        image_rule.name = "image rule"
        image_rule.redactor_type = "ImageRedaction"
        redactor = Mock(
            redact=Mock(
                return_value=ImageRedactionResult(
                    rule_name="rule", run_metrics={}, redaction_results=()
                )
            )
        )

        with patch(
            "src.redaction.file_processor.RedactorFactory.get",
            return_value=lambda config: redactor,
        ):
            processor = PDFProcessor()
            processor.redaction_rules = [image_text_rule, image_rule]
            processor.pages_metadata = [page_metadata]
            processor.pdf_images = []
            processor._apply_redaction_rules()

        assert image_text_rule.images == []
        assert list(image_rule.images) == [page_image]
    def test_records_page_offsets_per_rule(self):
        """
        - Given a text rule and an image text rule
        - When I apply the redaction rules
        - Then each rule is mapped to the page offsets of the text it was given
        """
        text_rule = Mock(spec=["name", "redactor_type", "text"])
        text_rule.name = "text rule"
        image_text_rule = Mock(
            spec=["name", "redactor_type", "text", "images", "rendered_images"]
        )
        image_text_rule.name = "image text rule"
        redactor = Mock(
            redact=Mock(
                return_value=TextRedactionResult(rule_name="rule", run_metrics={})
            )
        )

        with (
            patch.object(PDFUtil, "extract_unique_pdf_images", return_value=[]),
            patch(
                "src.redaction.file_processor.RedactorFactory.get",
                return_value=lambda config: redactor,
            ),
        ):
            processor = PDFProcessor()
            processor.redaction_rules = [text_rule, image_text_rule]
            processor.pdf_text_page_offsets = [(1, 0)]
            processor.rendered_pdf_text_page_offsets = [(2, 0)]
            processor._apply_redaction_rules()

        assert processor._rule_page_offsets == {
            "text rule": [(1, 0)],
            "image text rule": [(2, 0)],
        }


class TestExtractPDFTextContent(TestExamineApplyRedactionsBase):
    def test_pdf_with_text_returns_text_content(self):
        text_on_page = "Hello World"
        file_bytes = _make_pdf_with_text(text_on_page)

        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content=text_on_page,
            lines=[text_on_page],
            y0=[0],
            y1=[10],
            x0=[[0]],
            x1=[[10]],
        )
        with (
            patch.object(
                PDFUtil,
                "extract_page_content",
                return_value=page_metadata,
            ),
            patch.object(
                AzureImageAnalyser, "detect_text_in_images"
            ) as mock_detect_text_in_images,
        ):
            pdf_processor = PDFProcessor()
            pdf_processor._extract_pdf_text_content(file_bytes)

        mock_detect_text_in_images.assert_not_called()

        result_metadata = pdf_processor.pages_metadata
        assert len(result_metadata) == 1

        actual_page_metadata = result_metadata[0]
        assert actual_page_metadata == page_metadata

    def test_pdf_without_text_returns_printed_text(self):
        doc = pymupdf.open()
        doc.new_page()
        file_bytes = BytesIO()
        doc.save(file_bytes)
        file_bytes.seek(0)
        image = Image.new("RGB", (100, 100), color="white")
        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content="",
            lines=[],
            y0=[],
            y1=[],
            x0=[],
            x1=[],
        )
        page_metadata.rendered_image = PDFImageMetadata(
            source_image_resolution=(100, 100),
            file_format="png",
            image=image,
            page_number=0,
            image_transform_in_pdf=(0.48, 0.0, 0.0, 0.48, 0.0, 0.0),
        )

        printed_text_on_page = "Hello World"
        text_rect_map = (printed_text_on_page, (0, 0, 10, 10))
        detect_text_return_value = (
            (
                image,
                (text_rect_map,),
            ),
        )

        with (
            patch.object(
                PDFUtil,
                "extract_page_content",
                return_value=page_metadata,
            ),
            patch.object(
                AzureImageAnalyser,
                "detect_text_in_images",
                return_value=detect_text_return_value,
            ) as mock_detect_text_in_images,
        ):
            pdf_processor = PDFProcessor()
            pdf_processor._extract_pdf_text_content(file_bytes)

        mock_detect_text_in_images.assert_called_once()

        result_metadata = pdf_processor.pages_metadata
        assert len(result_metadata) == 1

        actual_page_metadata = result_metadata[0]
        assert actual_page_metadata.raw_text == printed_text_on_page

        rendered_image = actual_page_metadata.rendered_image
        assert rendered_image is not None
        assert rendered_image.image == image
        assert len(rendered_image.text_rect_map) == 1
        assert rendered_image.text_rect_map[0].text == printed_text_on_page
        assert rendered_image.text_rect_map[0].rect == (0, 0, 10, 10)


class TestExtractPDFTextAndImages(TestExamineApplyRedactionsBase):
    def test_pdf_with_text_and_images_assigns_text_and_images(self):
        text_on_page = "Hello World"
        file_bytes = _make_pdf_with_text(text_on_page)

        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content=text_on_page,
            lines=[text_on_page],
            y0=[0],
            y1=[10],
            x0=[[0]],
            x1=[[10]],
        )
        image_metadata = PDFImageMetadata(
            source_image_resolution=(100, 100),
            file_format="jpeg",
            image=Image.new("RGB", (100, 100), color="white"),
            page_number=0,
            image_transform_in_pdf=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        )

        with (
            patch.object(
                PDFUtil,
                "extract_page_content",
                return_value=page_metadata,
            ),
            patch.object(
                PDFUtil,
                "extract_pdf_images",
                return_value=[image_metadata],
            ),
            patch("src.redaction.file_processor.is_english_text", return_value=True),
        ):
            pdf_processor = PDFProcessor()
            pdf_processor._extract_pdf_text_and_images(file_bytes)

        result_metadata = pdf_processor.pages_metadata
        assert len(result_metadata) == 1

        actual_page_metadata = result_metadata[0]
        assert actual_page_metadata == page_metadata

        result_images = pdf_processor.pdf_images
        assert len(result_images) == 1
        assert result_images[0] == image_metadata

    def test_extracted_text_is_only_audit_logged(self):
        text_on_page = "Hello World"
        file_bytes = _make_pdf_with_text(text_on_page)
        page_metadata = self.create_mock_page_metadata(
            page_number=0,
            text_content=text_on_page,
            lines=[text_on_page],
            y0=[0],
            y1=[10],
            x0=[[0]],
            x1=[[10]],
        )

        with (
            patch.object(
                PDFUtil,
                "extract_page_content",
                return_value=page_metadata,
            ),
            patch.object(PDFUtil, "extract_pdf_images", return_value=[]),
            patch("src.redaction.file_processor.is_english_text", return_value=True),
        ):
            PDFProcessor()._extract_pdf_text_and_images(file_bytes)

        audit_messages = [c.args[0] for c in LoggingUtil.log_audit.call_args_list]
        info_messages = [c.args[0] for c in LoggingUtil.log_info.call_args_list]
        assert any(text_on_page in message for message in audit_messages)
        assert not any(text_on_page in message for message in info_messages)
    def test_sets_page_offsets(self):
        """
        - Given a two page PDF with embedded text
        - When I extract its text
        - Then the page offsets locate each page's text within the joined text
        """
        file_bytes = _make_pdf_with_pages("Hello there", "General Kenobi")

        with (
            patch.object(PDFUtil, "extract_pdf_images", return_value=[]),
            patch("src.redaction.file_processor.is_english_text", return_value=True),
        ):
            processor = PDFProcessor()
            processor._extract_pdf_text_and_images(file_bytes)

        assert processor.pdf_text == "Hello there\nGeneral Kenobi"
        assert processor.pdf_text_page_offsets == [(1, 0), (2, 12)]
        for page_number, start in processor.pdf_text_page_offsets:
            page_text = processor.pages_metadata[page_number - 1].raw_text
            assert processor.pdf_text[start:].startswith(page_text)
        assert processor.rendered_pdf_text_page_offsets == [(1, 0), (2, 12)]


class TestRedact:
    def test_skips_non_english_raises_exception(self):
        """
        - Given a non-English PDF input
        - When redact() is called
        - Then it should raise NonEnglishContentException and not modify the original bytes
        """
        french_text = (
            "Bonjour, ceci est un document de test. Ce fichier PDF contient du texte en français, "
            "destiné à vérifier la détection de la langue. Il ne doit pas être traité pour la rédaction."
        )
        file_bytes = _make_pdf_with_text(french_text)

        # Sanity check language detection
        doc_text = "\n".join(
            page.get_text() for page in pymupdf.open(stream=file_bytes)
        )
        file_bytes.seek(0)
        assert is_english_text(doc_text) is False

        with pytest.raises(NonEnglishContentException):
            PDFProcessor().redact(file_bytes, {"redaction_rules": []})

        # Ensure original stream still represents a PDF without highlight annotations
        pdf = pymupdf.open(stream=file_bytes)
        annots = [a for p in pdf for a in p.annots(pymupdf.PDF_ANNOT_HIGHLIGHT)]
        assert not annots

    def test_skips_blank_pdf(self):
        file_bytes = _make_pdf_with_text(" \n")
        doc_text = "\n".join(
            page.get_text() for page in pymupdf.open(stream=file_bytes)
        )
        file_bytes.seek(0)
        assert is_english_text(doc_text) is False

        # does not raise exception
        with (
            patch.object(PDFUtil, "extract_pdf_images", return_value=[]),
            patch.object(PDFUtil, "extract_unique_pdf_images", return_value=""),
            patch.object(
                PDFProcessor,
                "_apply_provisional_image_redactions",
                return_value=file_bytes,
            ),
            patch.object(
                PDFProcessor,
                "_apply_provisional_text_redactions",
                return_value=file_bytes,
            ),
        ):
            result = PDFProcessor().redact(file_bytes, {"redaction_rules": []})

        assert result == file_bytes

    def test_run_metrics_saved(self):
        """
        - Given I have a PDF with English text
        - When I call redact with a text redaction rule
        - Then run_metrics should contain all expected timing and summary keys
        """
        file_bytes = _make_pdf_with_text(
            "Hello World this is a test document with English text content"
        )

        mock_text_result = TextRedactionResult(
            rule_name="test_rule",
            run_metrics={"tokens_used": 10},
            redaction_strings=("Hello", "World"),
        )

        mock_redactor = Mock()
        mock_redactor.redact.return_value = mock_text_result

        mock_redaction_config = Mock()
        mock_redaction_config.redactor_type = "llm_text"
        mock_redaction_config.text = None
        mock_redaction_config.images = None

        with (
            patch.object(PDFUtil, "extract_pdf_images", return_value=[]),
            patch.object(PDFUtil, "extract_unique_pdf_images", return_value=[]),
            patch("src.redaction.file_processor.is_english_text", return_value=True),
            patch.object(
                PDFProcessor,
                "_apply_provisional_text_redactions",
                return_value=file_bytes,
            ),
            patch.object(
                PDFProcessor,
                "_apply_provisional_image_redactions",
                return_value=file_bytes,
            ),
            patch(
                "src.redaction.file_processor.RedactorFactory.get",
                return_value=lambda config: mock_redactor,
            ),
        ):
            processor = PDFProcessor()
            processor.redact(file_bytes, {"redaction_rules": [mock_redaction_config]})
            run_metrics = processor.get_run_metrics()

        assert run_metrics is not None
        # Check all expected keys are present
        expected_keys = {
            "pdf_text_extraction_time",
            "pdf_image_extraction_time",
            "text_analysis_total_time",
            "image_analysis_total_time",
            "analysis_total_time",
            "text_redaction_apply_time",
            "image_redaction_apply_time",
            "result_metrics",
            "aggregate_result_metrics",
            "unapplied_text_redaction_terms",
            "text_redaction_summary",
        }
        assert set(run_metrics.keys()) == expected_keys

        # Check timing values are non-negative floats
        timing_keys = {key for key in expected_keys if "time" in key}
        for key in timing_keys:
            assert isinstance(run_metrics[key], float)
            assert run_metrics[key] >= 0

        # Check analysis_total_time is the sum of text and image analysis
        assert run_metrics["analysis_total_time"] == pytest.approx(
            run_metrics["text_analysis_total_time"]
            + run_metrics["image_analysis_total_time"]
        )

        # Check result_metrics contains the rule
        assert "test_rule" in run_metrics["result_metrics"]
        assert run_metrics["result_metrics"]["test_rule"] == {"tokens_used": 10}

        # Check text_redaction_summary
        assert "test_rule" in run_metrics["text_redaction_summary"]
        assert run_metrics["text_redaction_summary"]["test_rule"][
            "redaction_strings"
        ] == (
            "Hello",
            "World",
        )
        assert run_metrics["text_redaction_summary"]["test_rule"]["n_proposed"] == 2


class TestApplyProvisionalImageRedactions:
    def _get_annotated_rects(self, document_bytes):
        rects = []
        for page in pymupdf.open(stream=document_bytes):
            for annotation in page.annots(pymupdf.PDF_ANNOT_HIGHLIGHT):
                rects.append(annotation.rect)
        return rects

    def test_returns_pdf_with_image_highlights(self):
        """
        - Given I have a PDF with a single image, and some redactions to apply to the image
        - When I call _apply_provisional_image_redactions
        - Then the redactions should be correctly applied to the document, and match a pre-baked example
        """
        # Load the test PDF and extract the source image
        doc_bytes = open_pdf(TRANSLATED_IMAGE_PDF)
        pdf = pymupdf.open(stream=doc_bytes)
        source_image = next(
            Image.open(BytesIO(pdf.extract_image(image[0]).get("image")))
            for page in pdf
            for image in page.get_images(full=True)
        )

        # Create a mock redaction result for the image
        redactions = [
            ImageRedactionResult(
                rule_name="test__pdf_processor__apply_provisional_image_redactions",
                run_metrics={},
                redaction_results=(
                    ImageRedactionResult.Result(
                        image_dimensions=(100, 100),
                        source_image=source_image,
                        redaction_boxes=((0, 0, 100, 100),),
                        names=("test_redaction",),
                    ),
                ),
            )
        ]
        pdf_image_metadata = [
            PDFImageMetadata(
                source_image_resolution=(100, 100),
                file_format="jpeg",
                image=source_image,
                page_number=0,
                image_transform_in_pdf=(75.0, 0.0, -0.0, 75.0, 73.5, 88.0462646484375),
            )
        ]

        # Apply provisional image redactions to the PDF
        with (
            patch.object(PDFUtil, "__init__", return_value=None),
            patch.object(
                PDFUtil, "extract_pdf_images", return_value=pdf_image_metadata
            ),
        ):
            pdf_processor = PDFProcessor()
            pdf_processor.pages_metadata = [PDFUtil.extract_page_content(pdf[0])]
            pdf_processor.pdf_images = pdf_image_metadata
            redacted_doc_bytes = pdf_processor._apply_provisional_image_redactions(
                doc_bytes, redactions
            )
        actual_annotated_rects = self._get_annotated_rects(redacted_doc_bytes)

        # Compare with expected redacted PDF with image highlights
        expected_annotation_rects = [
            pymupdf.Rect(
                55.82174301147461,
                83.3587646484375,
                166.17825317382812,
                167.7337646484375,
            )
        ]

        assert expected_annotation_rects == actual_annotated_rects


class TestApply:
    def test_returns_highlighted_scrubbed_pdf(self):
        curated_doc_bytes = open_pdf(TEXT_IMAGE_PROPOSED_PDF)
        expected_redacted_doc_bytes = open_pdf(TEXT_IMAGE_REDACTED_PDF)

        actual_redacted_doc_bytes, redactions_applied = PDFProcessor().apply(
            curated_doc_bytes, {}
        )
        assert redactions_applied is True

        # Compare the text content of the redacted document to the expected redacted document
        expected_redacted_doc = pymupdf.open(stream=expected_redacted_doc_bytes)
        actual_redacted_doc = pymupdf.open(stream=actual_redacted_doc_bytes)
        expected_missing_words = {"Riker)", "Phillipa)"}
        expected_text = "".join(page.get_text() for page in expected_redacted_doc)
        for word_to_remove in expected_missing_words:
            expected_text = expected_text.replace(word_to_remove, "")
        actual_text = "".join(page.get_text() for page in actual_redacted_doc)
        assert expected_text == actual_text

        # Compare the metadata of the redacted document to the expected redacted document
        expected_metadata = {
            "format": "PDF 1.4",
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creationDate": "",
            "modDate": "",
            "trapped": "",
            "encryption": None,
        }
        assert expected_metadata == actual_redacted_doc.metadata, (
            "Expected the metadata in the pdf to have been scrubbed, but it was not. "
            f"Expected: {expected_metadata}, Actual: {actual_redacted_doc.metadata}"
        )

        # Compare the image content of the redacted document to the expected redacted image
        expected_image = open_image(REDACTED_JPG)
        pdf_images = [
            Image.open(BytesIO(actual_redacted_doc.extract_image(xref[0]).get("image")))
            for page in actual_redacted_doc
            for xref in page.get_images(full=True)
        ]
        temp_bytes = BytesIO()
        pdf_images[0].save(temp_bytes, format="JPEG")
        actual_image = Image.open(temp_bytes)
        assert expected_image == actual_image, (
            "Expected the image in the pdf to be redacted, but it did not match the redacted sample"
        )

    def test_scrubs_pdf_with_nothing_to_redact(self):
        raw_doc_bytes = open_pdf(SOURCE_PDF)

        actual_redacted_doc_bytes, redactions_applied = PDFProcessor().apply(
            raw_doc_bytes, {}
        )
        assert redactions_applied is False

        # Compare the text content of the redacted document to the expected redacted document
        actual_redacted_doc = pymupdf.open(stream=actual_redacted_doc_bytes)
        # Compare the metadata of the redacted document to the expected redacted document
        expected_metadata = {
            "format": "PDF 1.4",
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creationDate": "",
            "modDate": "",
            "trapped": "",
            "encryption": None,
        }
        assert expected_metadata == actual_redacted_doc.metadata, (
            "Expected the metadata in the pdf to have been scrubbed, but it was not. "
            f"Expected: {expected_metadata}, Actual: {actual_redacted_doc.metadata}"
        )

    def test_saves_run_metrics(self):
        """
        - Given I have a PDF with highlight annotations (proposed redactions)
        - When I call apply
        - Then run_metrics should contain redaction_time and scrub_time as non-negative floats
        """
        curated_doc_bytes = open_pdf(TEXT_IMAGE_PROPOSED_PDF)

        processor = PDFProcessor()
        processor.apply(curated_doc_bytes, {})
        run_metrics = processor.get_run_metrics()

        assert run_metrics is not None
        assert set(run_metrics.keys()) == {
            "redaction_time",
            "scrub_time",
            "n_highlights",
        }
        assert isinstance(run_metrics["redaction_time"], float)
        assert isinstance(run_metrics["scrub_time"], float)
        assert isinstance(run_metrics["n_highlights"], int)
        assert run_metrics["redaction_time"] >= 0
        assert run_metrics["scrub_time"] >= 0
        assert run_metrics["n_highlights"] >= 0


class TestGetRedactionRationale:
    def test_returns_none_without_decisions(self):
        assert PDFProcessor().get_redaction_rationale() is None

    def test_traces_decisions_to_pages(self):
        """
        - Given an LLM result for two pages of text analysed in two overlapping chunks
        - When I get the redaction rationale
        - Then each occurrence is mapped to the pages of its span, or of its chunk
          when the span was not found, the redactions are sorted by first page then
          text, and unapplied terms are flagged
        """
        # "Joan Brown lives here.\nMary Jones met Joan Brown."
        page_offsets = [(1, 0), (2, 23)]
        result = LLMTextRedactionResult(
            rule_name="names",
            run_metrics={},
            redaction_decisions=(
                {"text": "Ann Smith", "occurrences": [_occurrence(1, [23])]},
                {
                    "text": "Mary Jones",
                    "occurrences": [_occurrence(1, [])],
                },
                {
                    "text": "Joan Brown",
                    "occurrences": [_occurrence(0, [0]), _occurrence(1, [38])],
                },
            ),
            excluded_decisions=(
                {"text": "Cllr Raj Patel", "occurrences": [_occurrence(0, [])]},
            ),
            chunk_summaries=(
                {"chunk_index": 0, "char_start": 0, "char_end": 22},
                {"chunk_index": 1, "char_start": 17, "char_end": 49},
            ),
            clause_legend={"T1": "Names"},
        )
        processor = PDFProcessor()
        processor.redaction_rules = [
            SimpleNamespace(
                name="names", model="gpt-5.6-luna", chunk_size=10, chunk_overlap=2
            )
        ]
        processor._llm_text_results = [result]
        processor._rule_page_offsets = {"names": page_offsets}
        processor.run_metrics = {"unapplied_text_redaction_terms": ["Mary Jones"]}

        [rationale] = processor.get_redaction_rationale()

        assert {
            k: v for k, v in rationale.items() if k not in {"redactions", "exclusions"}
        } == {
            "rule_name": "names",
            "model": "gpt-5.6-luna",
            "chunk_size": 10,
            "chunk_overlap": 2,
            "clauses": {"T1": "Names"},
            "page_offsets": page_offsets,
            "chunks": [
                {"chunk_index": 0, "char_start": 0, "char_end": 22, "pages": [1]},
                {"chunk_index": 1, "char_start": 17, "char_end": 49, "pages": [1, 2]},
            ],
        }
        assert [
            (
                r["text"],
                r["pages"],
                [o["pages"] for o in r["occurrences"]],
                r["applied"],
            )
            for r in rationale["redactions"]
        ] == [
            ("Joan Brown", [1, 2], [[1], [2]], True),
            ("Mary Jones", [1, 2], [[1, 2]], False),
            ("Ann Smith", [2], [[2]], True),
        ]
        assert [(e["text"], e["pages"]) for e in rationale["exclusions"]] == [
            ("Cllr Raj Patel", [1])
        ]
        assert "applied" not in rationale["exclusions"][0]

    def test_redaction_pages_are_the_pages_it_is_highlighted_on(self):
        """
        - Given a term that is part of another word on page 1, and a whole word on page 2
        - When the provisional redactions are applied, and the term's candidate
          positions include the one inside the other word
        - Then the rationale reports the term, and its occurrence, only on the page
          it is highlighted on, and a term that was not highlighted keeps the pages
          its positions are on
        """
        file_bytes = _make_pdf_with_pages(
            "Please ask another neighbour.", "She said her name is Joan."
        )
        processor = PDFProcessor()
        processor._apply_provisional_text_redactions(file_bytes, ["her"])
        assert processor._highlight_pages == {"her": {2}}

        def decision(text, offsets):
            return {
                "text": text,
                "occurrences": [
                    {"chunk_index": 0, "span_offsets": offsets, "pages": []}
                ],
            }

        result = LLMTextRedactionResult(
            rule_name="names",
            run_metrics={},
            redaction_decisions=(
                decision("her", [14, 41]),
                decision("Nobody", [45]),
            ),
            chunk_summaries=({"chunk_index": 0, "char_start": 0, "char_end": 56},),
        )
        processor._llm_text_results = [result]
        processor._rule_page_offsets = {"names": [(1, 0), (2, 30)]}
        processor.run_metrics = {"unapplied_text_redaction_terms": ["Nobody"]}

        [rationale] = processor.get_redaction_rationale()

        pages = {r["text"]: r for r in rationale["redactions"]}
        assert pages["her"]["pages"] == [2]
        assert pages["her"]["occurrences"][0]["pages"] == [2]
        assert pages["her"]["applied"] is True
        assert pages["Nobody"]["pages"] == [2]
        assert pages["Nobody"]["applied"] is False

    def test_redact_keeps_decisions_for_rationale_only(self):
        """
        - Given a two page PDF and an LLM rule that redacts a name on both pages and a
          term that is not in the PDF
        - When I redact the PDF
        - Then the run metrics hold no decisions, and the rationale traces the name
          to both pages and flags the missing term as unapplied
        """
        file_bytes = _make_pdf_with_pages(
            "Joan Brown lives here.", "Mary Jones met Joan Brown."
        )
        rule = SimpleNamespace(
            name="names",
            label=None,
            redactor_type="llm_text",
            text=None,
            model="gpt-5.6-luna",
            chunk_size=1500,
            chunk_overlap=60,
        )

        def redact():
            text = rule.text
            joan_offsets = [m.start() for m in re.finditer("Joan Brown", text)]
            return LLMTextRedactionResult(
                rule_name="names",
                run_metrics={},
                redaction_strings=("Joan Brown", "Nobody Here"),
                redaction_decisions=(
                    {
                        "text": "Joan Brown",
                        "occurrences": [_occurrence(0, joan_offsets)],
                    },
                    {"text": "Nobody Here", "occurrences": [_occurrence(0, [])]},
                ),
                chunk_summaries=(
                    {"chunk_index": 0, "char_start": 0, "char_end": len(text)},
                ),
            )

        with (
            patch.object(PDFUtil, "extract_pdf_images", return_value=[]),
            patch.object(PDFUtil, "extract_unique_pdf_images", return_value=[]),
            patch("src.redaction.file_processor.is_english_text", return_value=True),
            patch(
                "src.redaction.file_processor.RedactorFactory.get",
                return_value=lambda config: Mock(redact=Mock(side_effect=redact)),
            ),
        ):
            processor = PDFProcessor()
            processor.redact(file_bytes, {"redaction_rules": [rule]})

        summary = processor.get_run_metrics()["text_redaction_summary"]["names"]
        assert "redaction_decisions" not in summary
        assert "excluded_decisions" not in summary
        [rationale] = processor.get_redaction_rationale()
        assert rationale["chunks"][0]["pages"] == [1, 2]
        assert [
            (r["text"], r["pages"], r["applied"]) for r in rationale["redactions"]
        ] == [
            ("Joan Brown", [1, 2], True),
            ("Nobody Here", [1, 2], False),
        ]
