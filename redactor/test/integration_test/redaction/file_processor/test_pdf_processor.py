import os
from io import BytesIO
from math import isclose
from pathlib import Path
from string import punctuation

import pymupdf
import pytest
from pymupdf import Rect

from core.redaction.config import (
    ImageLLMTextRedactionConfig,
    ImageRedactionConfig,
    LLMTextRedactionConfig,
)
from core.redaction.file_processor import PDFProcessor
from core.util.pdf_util import PDFUtil
from test.util.util import (
    assert_instances_to_redact_approx_equal,
    assert_rect_approx_equal,
)

pdf_dir = os.path.join("test", "resources", "pdf")

SOURCE_PDF_PATH = os.path.join(pdf_dir, "test__pdf_processor__source.pdf")
PROPOSED_PDF_PATH = os.path.join(pdf_dir, "test__pdf_processor__proposed.pdf")
SOURCE_IMAGE_PDF_PATH = os.path.join(pdf_dir, "test__pdf_processor__source_image.pdf")
REDACTED_PDF_PATH = os.path.join(pdf_dir, "test__pdf_processor__redacted.pdf")
SIGNATURE_PDF_PATH = os.path.join(pdf_dir, "test__pdf_processor__signature.pdf")


def open_pdf_from_file(file_path: Path) -> BytesIO:
    with open(file_path, "rb") as f:
        document_bytes = BytesIO(f.read())
    return document_bytes


def get_pdf_annotations(pdf: pymupdf.Document, annotation_class):
    return [annotation for page in pdf for annotation in page.annots(annotation_class)]


def extract_annotated_text(document_bytes):
    annotated_text = []
    for page in pymupdf.open(stream=document_bytes):
        for annotation in page.annots(pymupdf.PDF_ANNOT_HIGHLIGHT):
            annotated_text.append(
                " ".join(page.get_textbox(annotation.rect).split())
                .strip(punctuation)
                .lower()
            )
    return annotated_text


def create_config(is_image: bool = False, label: str | None = None):
    llm_text_redaction_args = {
        "name": "config name",
        "model": "gpt-4.1",
        "system_prompt": (
            "You will be sent text to analyse. The text is a quote from Star Trek. "
            "Please find all strings in the text that adhere to the following rules: "
        ),
        "redaction_terms": [
            "The names of characters",
            "Rank",
            "Genders, such as she, her, he, him, they, their",
        ],
    }
    if is_image:
        config = ImageLLMTextRedactionConfig(
            redactor_type="ImageLLMTextRedaction",
            label=label,
            **llm_text_redaction_args,
        )
    else:
        config = LLMTextRedactionConfig(
            redactor_type="LLMTextRedaction", label=label, **llm_text_redaction_args
        )
    return {"redaction_rules": [config]}


class TestExtractPDFAnnotations:
    def test_returns_annotation_list(self):
        """
        Given I have a PDF document with annotations
        When I call _extract_pdf_annotations with the PDF and annotation type
        Then I should receive a list of all annotations of that type in the PDF, with page numbers included in the annotation info
        """
        document_bytes = open_pdf_from_file(SOURCE_PDF_PATH)
        pdf_processor = PDFProcessor()
        annotations = pdf_processor._extract_pdf_annotations(document_bytes)

        expected_annotations = [
            {
                "page_number": 0,
                "annotations": [
                    {
                        "content": "Text Redaction",
                        "subject": "[180.76254272460938, 145.0911865234375, 241.24356079101562, 157.3802490234375]",
                        "type": "Highlight",
                        "rect": Rect(
                            180.76254272460938,
                            145.0911865234375,
                            241.24356079101562,
                            157.3802490234375,
                        ),
                        "text": "Commander",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[244.29502868652344, 145.0911865234375, 267.51654052734375, 157.3802490234375]",
                        "type": "Highlight",
                        "rect": Rect(
                            244.29502868652344,
                            145.0911865234375,
                            267.51654052734375,
                            157.3802490234375,
                        ),
                        "text": "Data",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[72.0, 101.452392578125, 97.65274810791016, 113.741455078125]",
                        "type": "Highlight",
                        "rect": Rect(
                            72.0, 101.452392578125, 97.65274810791016, 113.741455078125
                        ),
                        "text": "Riker",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[164.2420654296875, 101.452392578125, 199.68487548828125, 113.741455078125]",
                        "type": "Highlight",
                        "rect": Rect(
                            164.2420654296875,
                            101.452392578125,
                            199.68487548828125,
                            113.741455078125,
                        ),
                        "text": "Phillipa",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[194.0673065185547, 72.35986328125, 215.45306396484375, 84.64892578125]",
                        "type": "Highlight",
                        "rect": Rect(
                            194.0673065185547,
                            72.35986328125,
                            215.45306396484375,
                            84.64892578125,
                        ),
                        "text": "your",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[470.42864990234375, 101.452392578125, 492.6402282714844, 113.741455078125]",
                        "type": "Highlight",
                        "rect": Rect(
                            470.42864990234375,
                            101.452392578125,
                            492.6402282714844,
                            113.741455078125,
                        ),
                        "text": "Your",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[273.0046081542969, 217.822509765625, 295.2162170410156, 230.111572265625]",
                        "type": "Highlight",
                        "rect": Rect(
                            273.0046081542969,
                            217.822509765625,
                            295.2162170410156,
                            230.111572265625,
                        ),
                        "text": "Your",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[72.0, 115.9986572265625, 108.05452728271484, 128.2877197265625]",
                        "type": "Highlight",
                        "rect": Rect(
                            72.0,
                            115.9986572265625,
                            108.05452728271484,
                            128.2877197265625,
                        ),
                        "text": "Honour",
                    },
                    {
                        "content": "Text Redaction",
                        "subject": "[298.2676696777344, 217.822509765625, 334.3221740722656, 230.111572265625]",
                        "type": "Highlight",
                        "rect": Rect(
                            298.2676696777344,
                            217.822509765625,
                            334.3221740722656,
                            230.111572265625,
                        ),
                        "text": "Honour",
                    },
                ],
            }
        ]

        for expected, actual in zip(expected_annotations, annotations):
            assert expected["page_number"] == actual["page_number"]
            for expected_annot, actual_annot in zip(
                expected["annotations"], actual["annotations"]
            ):
                for key in ["content", "subject", "type", "text"]:
                    assert expected_annot[key] == actual_annot[key]
                assert_rect_approx_equal(expected_annot["rect"], actual_annot["rect"])


class TestExamineProvisionalRedactionsOnPage:
    def test_finds_provisional_redactions_on_page(self):
        """
        Given I have some provisional redaction candidates for a PDF
        I want to examine each candidate and determine which should be kept as a redaction instance
        With multi-part redactions handled correctly
        """
        document_bytes = open_pdf_from_file(SOURCE_PDF_PATH)
        redaction_candidates = [
            (
                Rect(72.0, 101.452392578125, 101.31322479248047, 113.741455078125),
                "Riker",
            ),
            (
                Rect(
                    164.2420654296875,
                    101.452392578125,
                    203.34519958496094,
                    113.741455078125,
                ),
                "Phillipa",
            ),
            (
                Rect(
                    180.76254272460938,
                    145.0911865234375,
                    270.5718994140625,
                    157.3802490234375,
                ),
                "Commander Data",
            ),
        ]
        pdf_processor = PDFProcessor()
        pdf_processor.terms_found = {}
        pdf = pymupdf.open(stream=document_bytes)

        instances_to_redact = pdf_processor._examine_provisional_redactions_on_page(
            [text for _, text in redaction_candidates],
            PDFUtil.extract_page_text(pdf[0]),
        )

        assert_instances_to_redact_approx_equal(
            instances_to_redact,
            [(0, rect, term) for rect, term in redaction_candidates],
        )


class TestApplyProvisionalTextRedactions:
    def test_applies_highlights_to_redaction_strings(self):
        document_bytes = open_pdf_from_file(SOURCE_PDF_PATH)
        redaction_strings = [
            "Your",
            "Honour",
            "Riker",
            "Phillipa",
            "Commander",
            "Data",
        ]
        pdf_processor = PDFProcessor()
        redaction_rules = create_config(is_image=False, label="config label")[
            "redaction_rules"
        ]
        pdf_processor.redaction_rules = redaction_rules
        text_redaction_config = redaction_rules[0]
        pdf_processor._text_redaction_summary = {
            text_redaction_config.name: {
                "redaction_strings": redaction_strings,
                "n_proposed": len(redaction_strings),
                "n_applied": 0,
            }
        }
        redacted_document_bytes = pdf_processor._apply_provisional_text_redactions(
            document_bytes, redaction_strings
        )

        # Generate expected redaction text from the raw document
        expected_provisional_redaction_bytes = open_pdf_from_file(PROPOSED_PDF_PATH)
        expected_annotated_text = extract_annotated_text(
            expected_provisional_redaction_bytes
        )

        # Get the actual redacted text
        actual_annotated_text = extract_annotated_text(redacted_document_bytes)

        # Check all expected redaction strings are present
        matches = {
            expected_text: expected_text in actual_annotated_text
            for expected_text in expected_annotated_text
        }
        valid_match_count = len([x for x in matches.values() if x])

        assert valid_match_count == len(matches)

        # Check that the annotations have the correct label
        for page in pymupdf.open(stream=redacted_document_bytes):
            for annotation in page.annots(pymupdf.PDF_ANNOT_HIGHLIGHT):
                print(annotation.info)
                assert annotation.info["title"] == text_redaction_config.label
                assert annotation.info["content"] in redaction_strings

    def test_does_not_apply_to_partial_matches(self):
        document_bytes = open_pdf_from_file(SOURCE_PDF_PATH)
        redaction_strings = ["it"]

        redacted_document_bytes = PDFProcessor()._apply_provisional_text_redactions(
            document_bytes, redaction_strings
        )

        # Get the actual redacted text
        annotated_text_expanded = []
        for page in pymupdf.open(stream=redacted_document_bytes):
            for annotation in page.annots(pymupdf.PDF_ANNOT_HIGHLIGHT):
                annotation_rect = annotation.rect
                w = annotation_rect.width / 4
                annotated_text_expanded.append(
                    page.get_textbox(annotation_rect + (-w, 0, w, 0)).strip().lower()
                )

        # Find all instances of "it" in the annotated text
        actual_annotated_text = [
            t for text in annotated_text_expanded for t in text.split(" ") if "it" in t
        ]

        for word in ["criteria", "with", "servitude", "sits", "waiting"]:
            assert word not in actual_annotated_text

        assert set(actual_annotated_text) == {"it"}

    def test_applies_to_line_break(self):
        document_bytes = open_pdf_from_file(SOURCE_PDF_PATH)
        redaction_strings = ["all who come after him"]

        redacted_document_bytes = PDFProcessor()._apply_provisional_text_redactions(
            document_bytes, redaction_strings
        )

        # Get the actual redacted text
        actual_annotated_text = extract_annotated_text(redacted_document_bytes)

        assert len(actual_annotated_text) == 2
        assert "all who" in actual_annotated_text
        assert "come after him" in actual_annotated_text

    def test_applies_to_multi_line_breaks(self):
        document_bytes = open_pdf_from_file(SOURCE_PDF_PATH)
        redaction_strings = [
            (
                "It could significantly redefine the boundaries of personal liberty and freedom,"
                " expanding them for some, savagely curtailing them for others."
            )
        ]

        redacted_document_bytes = PDFProcessor()._apply_provisional_text_redactions(
            document_bytes, redaction_strings
        )

        # Get the actual redacted text
        actual_annotated_text = extract_annotated_text(redacted_document_bytes)

        assert len(actual_annotated_text) == 3
        assert "it" in actual_annotated_text
        assert (
            "could significantly redefine the boundaries of personal liberty and freedom,"
            " expanding them" in actual_annotated_text
        )
        assert "for some, savagely curtailing them for others" in actual_annotated_text


class TestRedact:
    def test_returns_annotated_pdf_bytes(self):
        """
        - Given I have a PDF with some content
        - When I call redact() with some config and the pdf content as bytes
        - Then I should receive a new bytes object which contains the PDF with redactions as specified by the input config
        """
        file_bytes = open_pdf_from_file(SOURCE_PDF_PATH)
        expected_redacted_text = {
            "commander",
            "data",
            "you",
            "he",
            "him",
            "he's",
            "them",
        }
        pdf_before = pymupdf.open(stream=file_bytes)
        page_annotations_before = get_pdf_annotations(
            pdf_before, pymupdf.PDF_ANNOT_HIGHLIGHT
        )
        assert not page_annotations_before
        redaction_config = create_config(label="config label")
        redacted_file_bytes = PDFProcessor().redact(file_bytes, redaction_config)
        actual_annotated_text = set(extract_annotated_text(redacted_file_bytes))

        matches = {
            expected_result: any(
                expected_result in redaction_string
                for redaction_string in actual_annotated_text
            )
            for expected_result in expected_redacted_text
        }
        acceptance_threshold = 0.1
        match_percent = float(len(tuple(x for x in matches.values() if x))) / float(
            len(expected_redacted_text)
        )

        error_message = (
            f"Expected a match threshold of at least {acceptance_threshold}, but was {match_percent}."
            f"\nExpected results {expected_redacted_text}\nActual results: {actual_annotated_text}"
        )
        assert match_percent >= acceptance_threshold, error_message

        # Check that the annotations have the correct label
        for page in pymupdf.open(stream=redacted_file_bytes):
            for annotation in page.annots(pymupdf.PDF_ANNOT_HIGHLIGHT):
                print(annotation.info)
                assert (
                    annotation.info["title"]
                    == redaction_config["redaction_rules"][0].label
                )

    def test_returns_annotated_image_pdf_bytes(self):
        """
        - Given I have a PDF with some content
        - When I call redact() with some config and the pdf content as bytes
        - Then I should receive a new bytes object which contains the PDF with redactions as specified by the input config
        """
        file_bytes = open_pdf_from_file(SOURCE_IMAGE_PDF_PATH)
        expected_redacted_text = {
            "commander",
            "data",
            "you",
            "he",
            "him",
            "he's",
            "them",
        }
        pdf_before = pymupdf.open(stream=file_bytes)
        page_annotations_before = get_pdf_annotations(
            pdf_before, pymupdf.PDF_ANNOT_HIGHLIGHT
        )
        assert not page_annotations_before

        pdf_processor = PDFProcessor()
        redacted_file_bytes = pdf_processor.redact(
            file_bytes, create_config(is_image=True)
        )

        pdf_after = pymupdf.open(stream=redacted_file_bytes)

        expected_annotation_rects = [
            Rect(
                448.9051818847656,
                131.69879150390625,
                469.8133850097656,
                144.7745361328125,
            ),
            Rect(
                437.7434387207031,
                203.3111572265625,
                458.2377014160156,
                216.69097900390625,
            ),
            Rect(
                125.64540100097656,
                217.90728759765625,
                145.86898803710938,
                231.28717041015625,
            ),
            Rect(396.392578125, 74.3953857421875, 414.3232727050781, 87.4710693359375),
            Rect(
                325.8874816894531,
                88.68743896484375,
                343.80230712890625,
                102.37127685546875,
            ),
            Rect(
                118.94203186035156,
                174.9466552734375,
                136.60203552246094,
                188.02239990234375,
            ),
            Rect(
                466.6270446777344,
                103.3511962890625,
                493.506103515625,
                115.8187255859375,
            ),
            Rect(
                72.32075500488281,
                117.62628173828125,
                115.28176879882812,
                131.00616455078125,
            ),
            Rect(
                271.08197021484375,
                217.3160400390625,
                298.1842346191406,
                231.60809326171875,
            ),
            Rect(
                294.9021301269531,
                217.3160400390625,
                340.4108581542969,
                231.60809326171875,
            ),
            Rect(
                359.51593017578125,
                131.681884765625,
                386.7773132324219,
                145.06170654296875,
            ),
            Rect(
                179.11331176757812,
                145.9739990234375,
                244.79586791992188,
                159.9619140625,
            ),
            Rect(
                241.30679321289062,
                145.686767578125,
                274.3641052246094,
                159.97882080078125,
            ),
            Rect(
                72.32077026367188, 74.37847900390625, 95.5218505859375, 87.75830078125
            ),
            Rect(
                451.21392822265625,
                88.68743896484375,
                480.4974365234375,
                102.37127685546875,
            ),
            Rect(
                119.76990509033203,
                102.70916748046875,
                149.84971618652344,
                117.00128173828125,
            ),
            Rect(
                221.00552368164062,
                102.70916748046875,
                251.0853271484375,
                117.00128173828125,
            ),
            Rect(
                365.1365661621094,
                102.74298095703125,
                388.7358093261719,
                116.4268798828125,
            ),
            Rect(
                353.4334716796875, 145.9739990234375, 377.16009521484375, 159.9619140625
            ),
            Rect(
                310.3948059082031,
                203.27740478515625,
                334.66278076171875,
                217.265380859375,
            ),
            Rect(
                242.11883544921875,
                231.91217041015625,
                265.4315490722656,
                246.20428466796875,
            ),
            Rect(
                266.6078186035156,
                117.33917236328125,
                286.146728515625,
                131.02301025390625,
            ),
            Rect(
                423.12652587890625,
                117.35601806640625,
                443.3501281738281,
                130.73590087890625,
            ),
            Rect(
                145.27786254882812,
                160.2998046875,
                165.21495056152344,
                174.28778076171875,
            ),
            Rect(502.7712097167969, 160.89111328125, 522.8673095703125, 173.966796875),
            Rect(
                371.09161376953125,
                88.68743896484375,
                380.3445739746094,
                102.37127685546875,
            ),
            Rect(
                72.1933364868164,
                102.72607421875,
                107.01799774169922,
                116.71405029296875,
            ),
            Rect(
                162.80857849121094,
                102.70916748046875,
                208.31735229492188,
                117.00128173828125,
            ),
            Rect(
                339.7560119628906,
                131.681884765625,
                362.68646240234375,
                145.06170654296875,
            ),
            Rect(
                359.51593017578125,
                131.681884765625,
                386.7773132324219,
                145.06170654296875,
            ),
        ]

        actual_annotation_rects = []
        for page in pdf_after:
            actual_annotation_rects.extend(
                [
                    annotation.rect
                    for annotation in page.annots(pymupdf.PDF_ANNOT_HIGHLIGHT)
                ]
            )

        matches = 0
        for actual_rect in actual_annotation_rects:
            for expected_rect in expected_annotation_rects:
                if (
                    isclose(actual_rect.x0, expected_rect.x0, abs_tol=1.0)
                    and isclose(actual_rect.y0, expected_rect.y0, abs_tol=1.0)
                    and isclose(actual_rect.x1, expected_rect.x1, abs_tol=1.0)
                    and isclose(actual_rect.y1, expected_rect.y1, abs_tol=1.0)
                ):
                    matches += 1
                    break

        match_percent = float(matches) / float(len(expected_annotation_rects))
        acceptance_threshold = 0.1
        error_message = (
            f"Expected a match threshold of at least {acceptance_threshold}, but was {match_percent}."
            f"\nExpected results {expected_redacted_text}\nActual results: {actual_annotation_rects}"
        )
        assert match_percent >= acceptance_threshold, error_message

        run_metrics = pdf_processor.get_run_metrics()
        assert run_metrics["unapplied_text_redaction_terms"] == []

        text_redaction_summary = run_metrics["text_redaction_summary"]
        image_text_summary = text_redaction_summary.get("config name")
        assert image_text_summary is not None

        n_proposed = image_text_summary["n_proposed"]
        assert n_proposed > 0
        # All should be applied
        assert image_text_summary["n_applied"] == n_proposed

    @pytest.mark.skip(
        reason="This test will not pass until NRR-248 (analyse flattened/printed PDfs) is implemented"
    )
    def test_returns_annotated_image_with_signature_pdf_bytes(self):
        """
        - Given I have a PDF with some an image of a signature
        - When I call redact() with some config and the pdf content as bytes
        - Then I should receive a new bytes object which contains the PDF with the signature highlighted
        """
        file_bytes = open_pdf_from_file(SIGNATURE_PDF_PATH)
        pdf_before = pymupdf.open(stream=file_bytes)
        page_annotations_before = get_pdf_annotations(
            pdf_before, pymupdf.PDF_ANNOT_HIGHLIGHT
        )
        assert not page_annotations_before

        redacted_file_bytes = PDFProcessor().redact(
            file_bytes,
            {
                "redaction_rules": [
                    ImageRedactionConfig(
                        name="config name",
                        redactor_type="ImageRedaction",
                    )
                ]
            },
        )

        pdf_after = pymupdf.open(stream=redacted_file_bytes)
        actual_annotation_rects = []
        for page in pdf_after:
            actual_annotation_rects.extend(
                [
                    annotation.rect
                    for annotation in page.annots(pymupdf.PDF_ANNOT_HIGHLIGHT)
                ]
            )

        expected_annotation_rects = [
            pymupdf.Rect(
                76.2924575805664,
                446.69781494140625,
                217.73513793945312,
                515.2452392578125,
            )
        ]
        assert actual_annotation_rects == expected_annotation_rects


class TestApply:
    def test_applies_redaction_boxes(self):
        """
        - Given we have a pdf with some provisional redations, and a sample of what a fully-redacted pdf (with the same redactions) should look like
        - When I call apply() with the provisional redaction pdf, and config
        - Then the final redacted output should have the same content as our sample fully-redacted pdf
        """
        # Run the redaction process against the provisional redaction file
        provisional_redaction_file_bytes = open_pdf_from_file(PROPOSED_PDF_PATH)
        provisional_redactions = get_pdf_annotations(
            pymupdf.open(stream=provisional_redaction_file_bytes),
            pymupdf.PDF_ANNOT_HIGHLIGHT,
        )
        assert provisional_redactions, (
            "test__pdf_processor__apply requires a document that has provisional redactions - there were none found in the document"
        )
        redacted_file_bytes, redactions_applied = PDFProcessor().apply(
            provisional_redaction_file_bytes, create_config()
        )

        # Extract text from source and final documents
        expected_redacted_document_bytes = open_pdf_from_file(REDACTED_PDF_PATH)
        expected_redacted_document_text = "\n".join(
            page.get_text()
            for page in pymupdf.open(stream=expected_redacted_document_bytes)
        )

        redacted_document = pymupdf.open(stream=redacted_file_bytes)
        actual_redacted_document_text = "\n".join(
            page.get_text() for page in redacted_document
        )

        # Compare the text of the redacted document to the expected redacted document
        assert expected_redacted_document_text == actual_redacted_document_text
        assert redactions_applied is True

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
        assert expected_metadata == redacted_document.metadata, (
            "Expected the metadata in the pdf to have been scrubbed, but it was not. "
            f"Expected: {expected_metadata}, Actual: {redacted_document.metadata}"
        )
