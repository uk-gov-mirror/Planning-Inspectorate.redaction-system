"""Render corpus documents as PDFs for the perf test, without highlights.

The redactor adds its own highlights and reads any it finds as proposed redactions, so
the labels go in a JSON file next to each PDF instead of in the PDF itself.

Run from the `redactor` directory:

    python -m tests.perf.corpus_pdfs --corpus <corpus.jsonl> \\
        --out-dir tests/resources/pdf/perf --ids <id> ... \\
        --scanned <id> ... --images <id> ...
"""

import argparse
import html
import json
from io import BytesIO
from pathlib import Path

import pymupdf

MARGIN = 50
CSS = "body { font-family: sans-serif; font-size: 11pt; } p { margin: 0 0 8pt 0; }"
IMAGE_DIR = Path(__file__).resolve().parents[1] / "resources" / "image"
EMBEDDED_IMAGES = ("faces.jpeg", "signature.png", "number_plate.jpg")


def load_corpus(path: Path) -> dict[str, dict]:
    """The documents of a JSONL corpus, by id"""
    with open(path, encoding="utf-8") as corpus_file:
        documents = [json.loads(line) for line in corpus_file if line.strip()]
    return {document["id"]: document for document in documents}


def render_text_pdf(text: str) -> pymupdf.Document:
    """Lay the text out as A4 pages. Blank lines start paragraphs and single newlines
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


def make_scanned(pdf: pymupdf.Document, dpi: int = 150) -> pymupdf.Document:
    """A PDF of each page's picture, so that there is no text layer to read"""
    scanned = pymupdf.open()
    for page in pdf:
        new_page = scanned.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, pixmap=page.get_pixmap(dpi=dpi))
    return scanned


def embed_images(pdf: pymupdf.Document, every: int = 2) -> None:
    """Put a sample image in the bottom margin of every Nth page"""
    for number, page in enumerate(pdf):
        if number % every:
            continue
        image = EMBEDDED_IMAGES[(number // every) % len(EMBEDDED_IMAGES)]
        bottom = page.rect.y1
        rect = pymupdf.Rect(MARGIN, bottom - MARGIN + 5, MARGIN + 80, bottom - 5)
        page.insert_image(rect, filename=str(IMAGE_DIR / image))


def expected_strings(document: dict) -> list[str]:
    """The strings that the document labels as to be redacted, without repeats"""
    return list(
        dict.fromkeys(e["text"] for e in document["entities"] if e.get("redact"))
    )


def write_fixture(document: dict, out_dir: Path, variant: str = "text") -> Path:
    """Write one document as a PDF, with its expected strings in a JSON file beside it"""
    pdf = render_text_pdf(document["text"])
    suffix = ""
    if variant == "scanned":
        pdf, suffix = make_scanned(pdf), "_scanned"
    elif variant == "images":
        embed_images(pdf)
        suffix = "_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{document['id']}{suffix}.pdf"
    pdf.save(path, garbage=3, deflate=True)
    pdf.close()
    path.with_suffix(".expected.json").write_text(
        json.dumps({"redact": expected_strings(document)}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ids", nargs="*", default=[], help="Documents as text PDFs")
    parser.add_argument("--scanned", nargs="*", default=[], help="Documents as scans")
    parser.add_argument("--images", nargs="*", default=[], help="With embedded images")
    args = parser.parse_args(argv)

    documents = load_corpus(args.corpus)
    for variant, ids in (
        ("text", args.ids),
        ("scanned", args.scanned),
        ("images", args.images),
    ):
        for document_id in ids:
            path = write_fixture(documents[document_id], args.out_dir, variant)
            print(f"{path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
