import json

import pymupdf

from tests.perf.corpus_pdfs import (
    embed_images,
    expected_strings,
    make_scanned,
    render_text_pdf,
    write_fixture,
)

DOCUMENT = {
    "id": "doc-1",
    "text": "Joan Brown lives at 4 High Street.\n\nCllr James Brown replied.",
    "entities": [
        {"text": "Joan Brown", "redact": True},
        {"text": "Cllr James Brown", "redact": False},
        {"text": "Joan Brown", "redact": True},
        {"text": "4 High Street", "redact": True},
    ],
}


def test_render_text_pdf_has_the_text_and_no_highlights():
    pdf = render_text_pdf(DOCUMENT["text"])

    assert "Joan Brown lives" in pdf[0].get_text()
    assert not any(page.first_annot for page in pdf)


def test_make_scanned_keeps_the_pages_as_pictures_without_text():
    scanned = make_scanned(render_text_pdf(DOCUMENT["text"]))

    assert scanned.page_count == 1
    assert scanned[0].get_text().strip() == ""
    assert len(scanned[0].get_images()) == 1


def test_embed_images_adds_an_image_to_every_other_page():
    pdf = render_text_pdf("\n\n".join(["A paragraph. " * 60] * 80))
    assert pdf.page_count > 2

    embed_images(pdf, every=2)

    with_images = [page.number for page in pdf if page.get_images()]
    assert with_images == list(range(0, pdf.page_count, 2))


def test_expected_strings_are_the_redacted_entities_without_repeats():
    assert expected_strings(DOCUMENT) == ["Joan Brown", "4 High Street"]


def test_write_fixture_writes_the_pdf_and_its_expected_strings(tmp_path):
    path = write_fixture(DOCUMENT, tmp_path, "scanned")

    assert path.name == "doc-1_scanned.pdf"
    assert pymupdf.open(path).page_count == 1
    expected = json.loads((tmp_path / "doc-1_scanned.expected.json").read_text("utf-8"))
    assert expected == {"redact": ["Joan Brown", "4 High Street"]}
