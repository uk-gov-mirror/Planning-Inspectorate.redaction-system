from io import BytesIO
from math import isclose
from string import punctuation

import pymupdf
from pymupdf import Rect

from src.redaction.config import (
    ImageLLMTextRedactionConfig,
    ImageRedactionConfig,
    LLMTextRedactionConfig,
)
from src.redaction.file_processor import PDFProcessor
from src.redaction.utils.pdf import PDFUtil
from tests.utils import (
    PRINTED_PDF,
    PROPOSED_PDF,
    REDACTED_PDF,
    SIGNATURE_PDF,
    SOURCE_IMAGE_PDF,
    SOURCE_PDF,
    assert_instances_to_redact_approx_equal,
    assert_rect_approx_equal,
    open_pdf,
)


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
        "model": "gpt-5.6-luna",
        "role": "You are an assistant that extracts requested terms from a given text.",
        "task": (
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
        document_bytes = open_pdf(SOURCE_PDF)
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
        document_bytes = open_pdf(SOURCE_PDF)
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
            PDFUtil.extract_page_metadata(pdf[0]),
        )

        assert_instances_to_redact_approx_equal(
            instances_to_redact,
            [(0, rect, term) for rect, term in redaction_candidates],
        )


class TestApplyProvisionalTextRedactions:
    def test_applies_highlights_to_redaction_strings(self):
        document_bytes = open_pdf(SOURCE_PDF)
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
        expected_provisional_redaction_bytes = open_pdf(PROPOSED_PDF)
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
        document_bytes = open_pdf(SOURCE_PDF)
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
        document_bytes = open_pdf(SOURCE_PDF)
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
        document_bytes = open_pdf(SOURCE_PDF)
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


class TestExtractPDFTextContent:
    def test_pdf_with_text_returns_text_content(self):
        document_bytes = open_pdf(SOURCE_PDF)
        pdf_processor = PDFProcessor()
        pdf_processor._extract_pdf_text_content(document_bytes)

        page_metadata = pdf_processor.pages_metadata[0]
        assert len(page_metadata.raw_text) > 0
        assert len(page_metadata.lines) > 0

        # Always rendered for image redaction
        assert page_metadata.rendered_image is not None

    def test_pdf_without_text_returns_printed_text(self):
        document_bytes = open_pdf(PRINTED_PDF)
        pdf_processor = PDFProcessor()
        pdf_processor._extract_pdf_text_content(document_bytes)

        page_metadata = pdf_processor.pages_metadata[0]
        assert len(page_metadata.raw_text) > 0
        assert len(page_metadata.lines) == 0

        rendered_image = page_metadata.rendered_image
        assert rendered_image.image is not None
        assert len(rendered_image.text_rect_map) > 0


class TestRedact:
    def test_returns_annotated_pdf_bytes(self):
        """
        - Given I have a PDF with some content
        - When I call redact() with some config and the pdf content as bytes
        - Then I should receive a new bytes object which contains the PDF with redactions as specified by the input config
        """
        file_bytes = open_pdf(SOURCE_PDF)
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
        file_bytes = open_pdf(SOURCE_IMAGE_PDF)
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
            Rect(395.3376770019531, 79.43072509765625, 412.7724609375, 92.76025390625),
            Rect(
                319.3392333984375,
                94.54766845703125,
                336.7740478515625,
                107.877197265625,
            ),
            Rect(
                141.28866577148438,
                185.27484130859375,
                158.939453125,
                198.14459228515625,
            ),
            Rect(
                363.05877685546875,
                94.54766845703125,
                372.3216857910156,
                107.877197265625,
            ),
            Rect(
                79.68730926513672,
                79.890380859375,
                100.41425323486328,
                92.30059814453125,
            ),
            Rect(
                441.1964416503906,
                94.9818115234375,
                469.0621032714844,
                107.85162353515625,
            ),
            Rect(
                123.62283325195312,
                110.07318115234375,
                152.08966064453125,
                123.40264892578125,
            ),
            Rect(
                220.36325073242188,
                109.63909912109375,
                248.20547485351562,
                123.42816162109375,
            ),
            Rect(362.337890625, 110.0987548828125, 384.0745849609375, 122.968505859375),
            Rect(
                348.8542785644531, 155.44952392578125, 370.9996032714844, 168.3193359375
            ),
            Rect(
                387.57440185546875,
                215.483154296875,
                410.3208923339844,
                228.8126220703125,
            ),
            Rect(
                362.96246337890625,
                245.282958984375,
                384.2672119140625,
                259.072021484375,
            ),
            Rect(
                79.90330505371094,
                110.07318115234375,
                112.04747009277344,
                123.40264892578125,
            ),
            Rect(
                165.20310974121094,
                109.63909912109375,
                208.16329956054688,
                123.42816162109375,
            ),
            Rect(
                457.0119323730469,
                110.58392333984375,
                483.073974609375,
                122.0748291015625,
            ),
            Rect(
                79.37503051757812,
                125.67529296875,
                121.15621948242188,
                137.6258544921875,
            ),
            Rect(394.0155944824219, 230.166015625, 460.67425537109375, 243.955078125),
            Rect(
                269.49078369140625,
                124.78155517578125,
                288.5599670410156,
                138.11102294921875,
            ),
            Rect(421.27166748046875, 125.2412109375, 440.7728576660156, 137.6513671875),
            Rect(
                153.1378936767578,
                170.56646728515625,
                172.0144500732422,
                183.43621826171875,
            ),
            Rect(
                78.96643829345703,
                186.14300537109375,
                98.27497100830078,
                198.09356689453125,
            ),
            Rect(
                451.0990295410156,
                140.35809326171875,
                471.0087585449219,
                152.768310546875,
            ),
            Rect(
                124.9449234008789, 230.62567138671875, 144.638671875, 243.49542236328125
            ),
            Rect(
                254.1802215576172,
                229.73187255859375,
                274.8603820800781,
                243.98065185546875,
            ),
            Rect(
                181.6431427001953,
                155.015380859375,
                272.2162170410156,
                168.34490966796875,
            ),
            Rect(
                194.71812438964844, 79.43072509765625, 221.5506134033203, 92.76025390625
            ),
            Rect(
                457.0119323730469,
                110.58392333984375,
                483.073974609375,
                122.0748291015625,
            ),
            Rect(394.0155944824219, 230.166015625, 421.0406494140625, 243.955078125),
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

    def test_returns_annotated_image_with_signature_pdf_bytes(self):
        """
        - Given I have a PDF with some an image of a signature
        - When I call redact() with some config and the pdf content as bytes
        - Then I should receive a new bytes object which contains the PDF with the signature highlighted
        """
        file_bytes = open_pdf(SIGNATURE_PDF)
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

    def test_returns_annotated_printed_pdf_bytes(self):
        file_bytes = open_pdf(PRINTED_PDF)
        pdf_before = pymupdf.open(stream=file_bytes)
        page_annotations_before = get_pdf_annotations(
            pdf_before, pymupdf.PDF_ANNOT_HIGHLIGHT
        )
        assert not page_annotations_before

        redacted_file_bytes = PDFProcessor().redact(
            file_bytes, create_config(is_image=True)
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

        assert len(actual_annotation_rects) > 0


class TestApply:
    def test_applies_redaction_boxes(self):
        """
        - Given we have a pdf with some provisional redations, and a sample of what a fully-redacted pdf (with the same redactions) should look like
        - When I call apply() with the provisional redaction pdf, and config
        - Then the final redacted output should have the same content as our sample fully-redacted pdf
        """
        # Run the redaction process against the provisional redaction file
        provisional_redaction_file_bytes = open_pdf(PROPOSED_PDF)
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
        expected_redacted_document_bytes = open_pdf(REDACTED_PDF)
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


RATIONALE_PAGES = (
    (
        "Joan Brown of 12 Mill Lane objects to the proposed extension at the rear of "
        "her neighbour's house. She says that the building would block light to her "
        "garden and that she was not consulted by the applicant before the plans were "
        "submitted to the council."
    ),
    (
        "Mary Jones, who lives next door, supports the proposal. She believes that the "
        "extension will improve the street and says that Joan Brown has misunderstood "
        "the plans. Councillor Raj Patel attended the site visit and spoke in support."
    ),
    (
        "In conclusion, Joan Brown asks the planning committee to refuse the "
        "application. Her letter was written on behalf of the residents of Mill Lane, "
        "and she would be happy to meet an officer at the site to discuss the concerns "
        "raised."
    ),
)


def make_pdf_with_pages(*texts: str) -> BytesIO:
    doc = pymupdf.open()
    for text in texts:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 540, 770), text)
    file_bytes = BytesIO()
    doc.save(file_bytes)
    file_bytes.seek(0)
    return file_bytes


def create_rationale_config(**kwargs) -> LLMTextRedactionConfig:
    return LLMTextRedactionConfig(
        name="config name",
        redactor_type="LLMTextRedaction",
        model="gpt-5.6-luna",
        role="You redact personal data from planning representations.",
        task="Find the personal data of members of the public in the text.",
        redaction_terms=["Names of members of the public", "Home addresses"],
        do_not_redact=["Names of elected councillors"],
        include_rationale=True,
        max_tokens=3000,
        **kwargs,
    )


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


class TestGetRedactionRationale:
    def redact_pages(self, **config_kwargs) -> tuple[PDFProcessor, dict]:
        config = create_rationale_config(**config_kwargs)
        processor = PDFProcessor()
        processor.redact(
            make_pdf_with_pages(*RATIONALE_PAGES), {"redaction_rules": [config]}
        )
        [rationale] = processor.get_redaction_rationale()
        return processor, rationale

    def test_traces_decisions_to_chunks_and_pages(self):
        """
        - Given a three page PDF analysed by an LLM rule in several small chunks, with
          include_rationale enabled
        - When I redact the PDF and get the redaction rationale
        - Then the rule's chunks cover the pages, and every redaction's pages are
          exactly the pages whose text contains it
        """
        config = create_rationale_config(chunk_size=60, chunk_overlap=10)
        processor = PDFProcessor()
        processor.redact(
            make_pdf_with_pages(*RATIONALE_PAGES), {"redaction_rules": [config]}
        )
        [rationale] = processor.get_redaction_rationale()

        assert rationale["rule_name"] == "config name"
        assert rationale["model"] == "gpt-5.6-luna"
        assert (rationale["chunk_size"], rationale["chunk_overlap"]) == (60, 10)
        assert rationale["clauses"] == config.get_clause_legend()

        chunks = rationale["chunks"]
        assert len(chunks) > 1
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert all(c["status"] == "ok" for c in chunks)
        assert all(c["pages"] for c in chunks)
        assert {page for c in chunks for page in c["pages"]} == {1, 2, 3}

        assert rationale["redactions"]
        page_text = [normalise_whitespace(text) for text in RATIONALE_PAGES]
        for redaction in rationale["redactions"]:
            assert redaction["pages"]
            assert isinstance(redaction["applied"], bool)
            expected_pages = [
                number
                for number, text in enumerate(page_text, start=1)
                if normalise_whitespace(redaction["text"]) in text
            ]
            if expected_pages:
                assert redaction["pages"] == expected_pages, redaction["text"]
            for occurrence in redaction["occurrences"]:
                assert 0 <= occurrence["chunk_index"] < len(chunks)
                assert occurrence["pages"]
                assert set(occurrence["pages"]) <= set(redaction["pages"])

    def test_redaction_pages_match_the_pages_highlighted_in_the_pdf(self):
        """
        - Given pages where words such as 'her' and 'other' appear inside longer words
          like 'another' and 'there' as well as on their own
        - When I redact the PDF and get the redaction rationale
        - Then each applied redaction is reported on exactly the pages it is
          highlighted on in the PDF, and each occurrence on one of those pages
        """
        pages = (
            (
                "Joan Brown told another neighbour, who lives there, about the plans "
                "for the extension at the rear of the house on Mill Lane."
            ),
            (
                "Mary Jones said her garden would lose light, and that the other "
                "houses were not affected. She asked Joan Brown to write to the council."
            ),
        )
        config = create_rationale_config(chunk_size=40, chunk_overlap=5)
        processor = PDFProcessor()
        redacted = processor.redact(
            make_pdf_with_pages(*pages), {"redaction_rules": [config]}
        )
        [rationale] = processor.get_redaction_rationale()
        highlighted = {}
        for page in PDFProcessor.get_proposed_redactions(redacted):
            for annotation in page["annotations"]:
                highlighted.setdefault(annotation["proposedRedaction"], set()).add(
                    page["pageNumber"]
                )

        applied = [r for r in rationale["redactions"] if r["applied"]]
        assert applied
        for redaction in applied:
            term_pages = highlighted[" ".join(redaction["text"].split("\n"))]
            assert set(redaction["pages"]) == term_pages, redaction["text"]
            for occurrence in redaction["occurrences"]:
                assert occurrence["pages"]
                assert set(occurrence["pages"]) <= term_pages

    def test_redactions_are_sorted_by_first_page(self):
        _, rationale = self.redact_pages(chunk_size=60, chunk_overlap=10)

        first_pages = [r["pages"][0] for r in rationale["redactions"]]
        assert first_pages == sorted(first_pages)

    def test_name_on_several_pages_has_an_occurrence_per_page(self):
        """
        - Given a name that appears on all three pages
        - When each page is analysed as its own chunk
        - Then the name is one redaction with the pages it appears on, and each
          occurrence is traced to a different chunk
        """
        _, rationale = self.redact_pages(chunk_size=80, chunk_overlap=0)

        [redaction] = [
            r
            for r in rationale["redactions"]
            if normalise_whitespace(r["text"]).lower() == "joan brown"
        ]
        assert redaction["pages"] == [1, 2, 3]
        assert redaction["chunk_count"] > 1
        assert len({o["chunk_id"] for o in redaction["occurrences"]}) > 1

    def test_decisions_are_kept_out_of_run_metrics(self):
        """
        - Given include_rationale is enabled and the LLM has returned decisions
        - When I redact the PDF
        - Then the decisions are only in the rationale, not in the run metrics that
          are returned to the caller
        """
        processor, rationale = self.redact_pages(chunk_size=60, chunk_overlap=10)

        assert rationale["redactions"]
        summary = processor.get_run_metrics()["text_redaction_summary"]["config name"]
        assert set(summary) == {"redaction_strings", "n_proposed", "n_applied"}

    def test_returns_none_without_include_rationale(self):
        """
        - Given an LLM rule with include_rationale left off
        - When I redact the PDF
        - Then there is no rationale
        """
        config = create_rationale_config(chunk_size=60, chunk_overlap=10).model_copy(
            update={"include_rationale": False}
        )
        processor = PDFProcessor()
        processor.redact(
            make_pdf_with_pages(*RATIONALE_PAGES), {"redaction_rules": [config]}
        )

        assert processor.get_run_metrics()["text_redaction_summary"]["config name"]
        assert processor.get_redaction_rationale() is None

    def test_returns_none_for_image_rule(self):
        """
        - Given an image text rule with include_rationale enabled
        - When I redact a PDF containing images
        - Then there is no rationale, since image rules do not return decisions
        """
        config = create_config(is_image=True)["redaction_rules"][0].model_copy(
            update={"include_rationale": True, "max_tokens": 3000}
        )
        processor = PDFProcessor()
        processor.redact(open_pdf(SOURCE_IMAGE_PDF), {"redaction_rules": [config]})

        assert processor.get_redaction_rationale() is None
