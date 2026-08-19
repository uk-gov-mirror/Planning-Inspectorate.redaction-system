import os
from datetime import UTC, datetime
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pymupdf
import pytest
from PIL import Image

from core.redaction.exceptions import NonEnglishContentException
from core.redaction.file_processor import PDFProcessor
from core.redaction.result import (
    ImageRedactionResult,
    TextRedactionResult,
)
from core.util.pdf_util import (
    PDFImageMetadata,
    PDFLineMetadata,
    PDFPageMetadata,
    PDFUtil,
)
from core.util.text_util import get_normalised_words, is_english_text


@pytest.fixture(autouse=True)
def _mock_init():
    def init_side_effect(self):
        self.run_metrics = {}
        self.terms_found = {}

    with patch.object(PDFProcessor, "__init__", init_side_effect):
        yield


def _make_pdf_with_text(text: str) -> BytesIO:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    b = BytesIO()
    doc.save(b)
    b.seek(0)
    return b


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
                "extract_page_text",
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
                "core.redaction.file_processor.RedactorFactory.get",
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
        with open(
            "test/resources/pdf/test__pdf_processor__translated_image.pdf", "rb"
        ) as f:
            doc_bytes = BytesIO(f.read())
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
        with patch.object(
            PDFUtil, "extract_pdf_images", return_value=pdf_image_metadata
        ):
            redacted_doc_bytes = PDFProcessor()._apply_provisional_image_redactions(
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
        with open(
            os.path.join(
                "test",
                "resources",
                "pdf",
                "test__pdf_processor__text_and_image_proposed.pdf",
            ),
            "rb",
        ) as f:
            curated_doc_bytes = BytesIO(f.read())
        with open(
            os.path.join(
                "test",
                "resources",
                "pdf",
                "test__pdf_processor__text_and_image_redacted.pdf",
            ),
            "rb",
        ) as f:
            expected_redacted_doc_bytes = BytesIO(f.read())

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
        with open(
            os.path.join("test", "resources", "image", "image_with_text_redacted.jpg"),
            "rb",
        ) as f:
            expected_image_bytes = BytesIO(f.read())
            expected_image = Image.open(expected_image_bytes)
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
        with open(
            os.path.join(
                "test",
                "resources",
                "pdf",
                "test__pdf_processor__source.pdf",
            ),
            "rb",
        ) as f:
            raw_doc_bytes = BytesIO(f.read())

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
        with open(
            os.path.join(
                "test",
                "resources",
                "pdf",
                "test__pdf_processor__text_and_image_proposed.pdf",
            ),
            "rb",
        ) as f:
            curated_doc_bytes = BytesIO(f.read())

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
