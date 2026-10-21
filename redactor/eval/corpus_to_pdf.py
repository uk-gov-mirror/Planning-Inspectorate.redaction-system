"""Write a labelled corpus to PDFs, highlighting each labelled span with its metadata.

Each document becomes one PDF. Every occurrence of a labelled entity is highlighted:

- ``redact: true`` entities get the same light blue highlight the redactor uses for
  proposed redactions, so the PDFs can be used as input to tools that read them.
- ``redact: false`` entities (hard negatives) are only highlighted, in green, with
  ``--include-keep``.

The annotation's title is the entity's rule and its content is the entity text, as in the
redactor's own annotations. Its subject holds the label as JSON (rule, category, redact
and rationale). Spans are found like the redactor finds them: on normalised words, so
case and punctuation are ignored and "Ms" does not match inside "terms". As in the
redactor, the name part of an email address ("ada.merrin") matches the address it is in,
and only that part of the address is highlighted.

Example (run from the `redactor` directory):

    python -m eval.corpus_to_pdf --corpus eval/data/prompt_corpus.jsonl \\
        --split dev --out-dir eval/results/prompt_corpus_pdfs
"""

import argparse
import html
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pymupdf

from src.redaction.utils.pdf import ANNOT_HIGHLIGHT_COLOR, PDFUtil
from src.redaction.utils.text import (
    email_name_part_span,
    get_normalised_words,
    normalise_text,
)

from .corpus import RULE_NAMES, Document, Entity, load_corpus
from .scoring import find_subsequence, normalise_words

KEEP_HIGHLIGHT_COLOR = (0.6, 0.9, 0.6)  # light green
MARGIN = 50
CSS = "body { font-family: sans-serif; font-size: 11pt; } p { margin: 0 0 8pt 0; }"

Token = tuple[str, int, pymupdf.Rect, tuple[int, int, int]]
"""(normalised word, page number, word rect, (page number, block, line))"""


def render_pdf(text: str) -> pymupdf.Document:
    """Lay the text out as A4 pages. Blank lines start paragraphs, single newlines
    are kept as line breaks"""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    body = "".join(
        "<p>" + html.escape(paragraph.strip()).replace("\n", "<br>") + "</p>"
        for paragraph in paragraphs
    )
    story = pymupdf.Story(html=body, user_css=CSS)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (MARGIN, MARGIN, -MARGIN, -MARGIN)
    buffer = BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return pymupdf.open(stream=buffer.getvalue())


def index_tokens(pdf: pymupdf.Document) -> list[Token]:
    """Normalised words of the whole PDF in reading order, with where each one is"""
    tokens = []
    for page in pdf:
        for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
            tokens.extend(
                (
                    token,
                    page.number,
                    pymupdf.Rect(x0, y0, x1, y1),
                    (page.number, block, line),
                )
                for token in normalise_words(word)
            )
    return tokens


def index_email_words(pdf: pymupdf.Document) -> list[tuple[str, int, pymupdf.Rect]]:
    """Email addresses in the PDF, as (normalised address, page number, rect), so a
    term that is only the name part of an address can be matched to it"""
    return [
        (normalise_text(word), page.number, pymupdf.Rect(x0, y0, x1, y1))
        for page in pdf
        for x0, y0, x1, y1, word, *_ in page.get_text("words")
        if "@" in normalise_text(word)
    ]


def locate(
    tokens: list[Token],
    text: str,
    email_words: list[tuple[str, int, pymupdf.Rect]] = (),
    pdf: pymupdf.Document | None = None,
) -> list[tuple[int, pymupdf.Rect]]:
    """Find every occurrence of the text, as one (page number, rect) per line it spans.
    A single-word text also matches the email address it is the name part of, narrowed
    to that part if the PDF is given."""
    needle = normalise_words(text)
    haystack = tuple(token[0] for token in tokens)
    found = []
    words = get_normalised_words(text)
    if len(words) == 1:
        for address, page_number, rect in email_words:
            if span := email_name_part_span(address, words[0].casefold()):
                edges = pdf and PDFUtil.get_character_edges(
                    pdf[page_number], rect, address, span
                )
                if edges:
                    rect = pymupdf.Rect(edges[0], rect.y0, edges[1], rect.y1)
                found.append((page_number, rect))
    for start in find_subsequence(haystack, needle):
        lines: dict[tuple[int, int, int], list[Token]] = defaultdict(list)
        for token in tokens[start : start + len(needle)]:
            lines[token[3]].append(token)
        for line_tokens in lines.values():
            rect = pymupdf.Rect(line_tokens[0][2])
            for token in line_tokens[1:]:
                rect |= token[2]
            found.append((line_tokens[0][1], rect))
    return found


def add_highlight(page: pymupdf.Page, rect: pymupdf.Rect, entity: Entity):
    """Highlight the rect, with the entity's label in the annotation's metadata"""
    annotation = page.add_highlight_annot(rect)
    annotation.set_info(
        {
            "title": entity.rule if entity.redact else f"{entity.rule} (keep)",
            "content": entity.text,
            "subject": json.dumps(
                {
                    "rule": entity.rule,
                    "category": entity.category,
                    "redact": entity.redact,
                    "rationale": entity.rationale,
                },
                ensure_ascii=False,
            ),
            "creationDate": pymupdf.get_pdf_now(),
        }
    )
    annotation.set_colors(
        stroke=ANNOT_HIGHLIGHT_COLOR if entity.redact else KEEP_HIGHLIGHT_COLOR
    )
    annotation.update()


def convert_document(
    document: Document, rules: list[str], include_keep: bool = False
) -> tuple[pymupdf.Document, dict[str, int]]:
    """
    Render a document and highlight its labelled spans.

    :param rules: Only highlight entities of these rules.
    :param include_keep: Also highlight the hard negatives (``redact: false``).
    :return: The PDF, and the number of highlights made for each entity text (0 if
    the entity was not found in the PDF).
    """
    pdf = render_pdf(document.text)
    pdf.set_metadata(
        {
            "title": document.id,
            "subject": document.doc_type,
            "keywords": f"split={document.split}; source={document.source}",
            "creator": "redactor eval corpus_to_pdf",
        }
    )
    tokens = index_tokens(pdf)
    email_words = index_email_words(pdf)
    counts = {}
    for entity in document.entities:
        if entity.rule not in rules or not (entity.redact or include_keep):
            continue
        matches = locate(tokens, entity.text, email_words, pdf)
        for page_number, rect in matches:
            add_highlight(pdf[page_number], rect, entity)
        counts[entity.text] = len(matches)
    return pdf, counts


def main(argv=None):
    """Command line entry point: write one annotated PDF per document in the corpus"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, help="Path to a JSONL corpus")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["dev", "test"])
    parser.add_argument("--rules", nargs="+", default=list(RULE_NAMES))
    parser.add_argument("--ids", nargs="+", help="Only convert these document ids")
    parser.add_argument("--limit", type=int, help="Only convert the first N documents")
    parser.add_argument(
        "--include-keep",
        action="store_true",
        help="Also highlight, in green, the spans that must not be redacted",
    )
    args = parser.parse_args(argv)

    documents = load_corpus(args.corpus, split=args.split)
    if args.ids:
        documents = [d for d in documents if d.id in args.ids]
    if args.limit:
        documents = documents[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        pdf, counts = convert_document(document, args.rules, args.include_keep)
        pdf.save(out_dir / f"{document.id}.pdf", garbage=3, deflate=True)
        unmatched = [text for text, count in counts.items() if not count]
        print(
            f"{document.id}: {pdf.page_count} pages, {sum(counts.values())} highlights "
            f"for {len(counts) - len(unmatched)} of {len(counts)} entities"
        )
        for text in unmatched:
            print(f"  not found in the PDF: {text!r}")
        pdf.close()


if __name__ == "__main__":
    main()
