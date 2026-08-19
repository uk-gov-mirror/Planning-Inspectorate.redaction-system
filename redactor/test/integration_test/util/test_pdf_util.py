from io import BytesIO
from math import isclose

import pymupdf
from pymupdf import Rect

from core.util.pdf_util import PDFUtil
from test.util.util import assert_instances_to_redact_approx_equal


class TestPDFUtil:
    def test_examine_provisional_text_redaction(self):
        """
        Given I have a provisional redaction candidate for a PDF
        I want to determine whether it exactly matches the text on the page
        If if is a multi-part redaction, I want to capture all parts of the redaction
        """
        with open("test/resources/pdf/test__pdf_processor__source.pdf", "rb") as f:
            document_bytes = BytesIO(f.read())
        terms_to_redact = [
            "Riker",  # Single word redaction
            "Commander Data",  # Multi-word redaction
            "Your Honour",  # Multi-word redaction across line break
        ]
        pdf = pymupdf.open(stream=document_bytes)

        instances_to_redact = []
        for term in terms_to_redact:
            instances_to_redact.extend(
                PDFUtil.examine_provisional_text_redaction(
                    term,
                    PDFUtil.extract_page_metadata(pdf[0]),
                )
            )

        expected_result = [
            (
                0,
                Rect(72.0, 101.452392578125, 101.31330871582031, 113.741455078125),
                "Riker",
            ),
            (
                0,
                Rect(
                    180.76254272460938,
                    145.0911865234375,
                    270.5700378417969,
                    157.3802490234375,
                ),
                "Commander Data",
            ),
            (
                0,
                Rect(
                    470.42864990234375,
                    101.452392578125,
                    492.64031982421875,
                    113.741455078125,
                ),
                "Your Honour",
            ),
            (
                0,
                Rect(72.0, 115.9986572265625, 110.50122833251953, 128.2877197265625),
                "Your Honour",
            ),
            (
                0,
                Rect(
                    273.0046081542969,
                    217.822509765625,
                    336.7688903808594,
                    230.111572265625,
                ),
                "Your Honour",
            ),
        ]

        assert_instances_to_redact_approx_equal(instances_to_redact, expected_result)

    def test_add_provisional_redaction(self):
        doc = pymupdf.open()
        page = doc.new_page()
        rect = Rect(0, 0, 10, 10)
        term = "Hello"
        PDFUtil.add_provisional_redaction(page, rect, term, title="Greeting")
        annotations = list(page.annots())
        assert len(annotations) == 1
        annot = annotations[0]

        info = annot.info
        assert info["title"] == "Greeting"
        assert info["content"] == "Hello"
        assert info["creationDate"] is not None
        assert annot.vertices == [(0.0, 0.0), (10.0, 0.0), (0, 10.0), (10.0, 10.0)]
        assert annot.type == (8, "Highlight")

        # Save and reopen to verify persisted colours (pymupdf requires reload to read back)
        pdf_bytes = BytesIO()
        doc.save(pdf_bytes)
        pdf_bytes.seek(0)
        reopened = pymupdf.open(stream=pdf_bytes)
        annot = next(iter(reopened[0].annots()))

        reopened_info = annot.info
        assert reopened_info["title"] == "Greeting", (
            f"Title not persisted after save/reopen: got '{reopened_info['title']}'"
        )
        assert reopened_info["content"] == "Hello"

        expected_colours = [0.2157, 0.898, 1.0]
        actual_colours = annot.colors["stroke"]
        assert len(actual_colours) == 3, (
            f"Expected highlight colour to have 3 components, got {actual_colours}"
        )
        assert all(
            isclose(actual_colours[i], expected_colours[i], abs_tol=1e-2)
            for i in range(3)
        ), f"Expected {expected_colours}, got {list(actual_colours)}"
