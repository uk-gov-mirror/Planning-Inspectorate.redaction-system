import json

import pymupdf
import pytest
from eval import corpus_to_pdf
from eval.corpus import RULE_NAMES, Document, Entity, write_corpus
from eval.corpus_to_pdf import KEEP_HIGHLIGHT_COLOR, convert_document

from src.redaction.utils.pdf import ANNOT_HIGHLIGHT_COLOR

TEXT = (
    "My neighbour Ms Joan Whitcombe objects to the terms of the scheme.\n\n"
    "Cllr Raj Patel agreed. Email joan.whitcombe@example.com or call 07700 900123."
)


def _document(entities=None, text=TEXT) -> Document:
    return Document(
        id="doc-1",
        doc_type="relevant_representation",
        text=text,
        split="test",
        entities=tuple(
            entities
            if entities is not None
            else [
                Entity("Joan Whitcombe", "names", "third_party", True, "neighbour"),
                Entity("Ms", "names", "title", True),
                Entity("Cllr Raj Patel", "names", "councillor", False, "councillor"),
                Entity(
                    "joan.whitcombe@example.com", "contact_identity_info", "email", True
                ),
            ]
        ),
    )


def _highlights(pdf: pymupdf.Document) -> list[dict]:
    """Each highlight's metadata, colour and the words whose centres it covers"""
    highlights = []
    for page in pdf:
        words = page.get_text("words")
        for annotation in page.annots():
            vertices = annotation.vertices
            rect = pymupdf.Rect(*vertices[0], *vertices[-1])
            text = " ".join(
                w[4]
                for w in words
                if rect.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
            )
            highlights.append(
                {
                    **annotation.info,
                    "stroke": annotation.colors["stroke"],
                    "text": text,
                }
            )
    return highlights


def test_convert_document__highlights_redactions_with_metadata():
    """
    - Given a document with redact and keep labels
    - When I convert it
    - Then each redact entity is highlighted in the redaction colour, carrying its
      rule, text and label, and the hard negative is not highlighted
    """
    pdf, counts = convert_document(_document(), list(RULE_NAMES))

    assert counts == {
        "Joan Whitcombe": 1,
        "Ms": 1,
        "joan.whitcombe@example.com": 1,
    }
    highlights = {h["content"]: h for h in _highlights(pdf)}
    assert set(highlights) == set(counts)
    joan = highlights["Joan Whitcombe"]
    assert joan["stroke"] == pytest.approx(ANNOT_HIGHLIGHT_COLOR, abs=1e-3)
    assert joan["title"] == "names"
    assert json.loads(joan["subject"]) == {
        "rule": "names",
        "category": "third_party",
        "redact": True,
        "rationale": "neighbour",
    }
    assert joan["text"] == "Joan Whitcombe"
    assert pdf.metadata["title"] == "doc-1"
    assert pdf.metadata["subject"] == "relevant_representation"


def test_convert_document__matches_whole_words_only():
    """'Ms' is highlighted once, as a title, and not inside 'terms'"""
    pdf, _ = convert_document(_document(), ["names"])

    ms = [h for h in _highlights(pdf) if h["content"] == "Ms"]
    assert len(ms) == 1
    assert ms[0]["text"] == "Ms"


def test_convert_document__include_keep_highlights_hard_negatives():
    pdf, counts = convert_document(_document(), ["names"], include_keep=True)

    assert counts["Cllr Raj Patel"] == 1
    keep = next(h for h in _highlights(pdf) if h["content"] == "Cllr Raj Patel")
    assert keep["stroke"] == pytest.approx(KEEP_HIGHLIGHT_COLOR, abs=1e-3)
    assert keep["title"] == "names (keep)"
    assert json.loads(keep["subject"])["redact"] is False


def test_convert_document__filters_by_rule():
    pdf, counts = convert_document(_document(), ["contact_identity_info"])

    assert set(counts) == {"joan.whitcombe@example.com"}
    assert len(_highlights(pdf)) == 1


def test_convert_document__reports_entities_not_found():
    document = _document([Entity("Mary Jones", "names", "third_party", True)])

    pdf, counts = convert_document(document, ["names"])

    assert counts == {"Mary Jones": 0}
    assert _highlights(pdf) == []


def test_convert_document__span_across_lines_and_pages():
    """A long document spans pages, and an entity wrapping onto a new line is still
    highlighted, one highlight per line"""
    filler = "The traffic assessment considered peak hour flows on the road. " * 12
    name = "Joan Whitcombe of Willow Tree Farm Cottage"
    text = "\n\n".join([filler] * 6 + [f"{filler}{name} {filler}"] + [filler] * 6)

    pdf, counts = convert_document(
        _document([Entity(name, "names", "third_party", True)], text), ["names"]
    )

    assert pdf.page_count > 1
    assert counts[name] >= 1
    highlighted = " ".join(h["text"] for h in _highlights(pdf))
    assert highlighted.split() == name.split()


def test_main__writes_a_pdf_per_document(tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(
        corpus,
        [
            _document(),
            Document(
                id="doc-2",
                doc_type="x",
                text="Mary Jones is in the dev split.",
                entities=(Entity("Mary Jones", "names", "third_party", True),),
                split="dev",
            ),
        ],
    )

    corpus_to_pdf.main(
        [
            "--corpus",
            str(corpus),
            "--out-dir",
            str(tmp_path / "pdfs"),
            "--split",
            "test",
        ]
    )

    assert sorted(p.name for p in (tmp_path / "pdfs").glob("*.pdf")) == ["doc-1.pdf"]
    assert "3 highlights for 3 of 3 entities" in capsys.readouterr().out
    saved = pymupdf.open(tmp_path / "pdfs" / "doc-1.pdf")
    assert len(_highlights(saved)) == 3


def test_convert_document__highlights_only_the_name_part_of_an_email():
    """
    - Given an entity that is only the name part of an email address
    - When I convert the document
    - Then only that part of the address is highlighted, and the domain is not matched
    """
    text = "Email Ada.Merrin@post.invalid or write to the post office."
    document = _document(
        [
            Entity("ada.merrin", "contact_identity_info", "email_name", True),
            Entity("post", "contact_identity_info", "domain", True),
        ],
        text,
    )

    pdf, counts = convert_document(document, list(RULE_NAMES))

    assert counts == {"ada.merrin": 1, "post": 1}
    page = pdf[0]
    annotations = {a.info["content"]: a for a in page.annots()}
    vertices = annotations["ada.merrin"].vertices
    name_part = pymupdf.Rect(*vertices[0], *vertices[-1])
    expected = page.search_for("Ada.Merrin")[0]
    assert (name_part.x0, name_part.x1) == pytest.approx(
        (expected.x0, expected.x1), abs=0.5
    )
    # "post" only matches the word in "post office", not the email domain
    highlights = {h["content"]: h["text"] for h in _highlights(pdf)}
    assert highlights["post"] == "post"
