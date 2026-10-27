"""Convert the existing labelled e2e PDFs into held-out corpus documents.

- PINS_anon_samples.pdf + anon_samples_terms.json: one document per "Sample N:"
  section. Every term is a positive, but the terms aren't attributed to a rule, so they
  use the `unspecified` rule and only count towards the combined (all-rules) score.
- eval_dataset.pdf + eval_dataset_expected.json: Group A strings are positives and
  Group B strings are hard negatives. The "Group A"/"Group B" headings are removed so
  they don't reveal the labels to the model. These labels predate the current
  guidelines, so review any disagreements before relying on them.

Example (run from the `redactor` directory):

    python -m eval.convert_existing_samples --out eval/data/existing_test.jsonl
"""

import argparse
import json
import re
from pathlib import Path

import pymupdf

from .corpus import UNSPECIFIED_RULE, Document, Entity, validate_document, write_corpus

E2E_DATA = Path(__file__).parents[1] / "tests" / "e2e" / "data"
_SAMPLE_HEADING = re.compile(r"^\s*Sample\s+(\d+)\s*:", re.MULTILINE)
_GROUP_HEADING = re.compile(r"^\s*Group [AB]\s*$\n?", re.MULTILINE)


def extract_pdf_text(path: Path) -> str:
    """Extract the text of every page of a PDF, separating pages with blank lines"""
    with pymupdf.open(path) as pdf:
        return "\n\n".join(page.get_text() for page in pdf)


def find_verbatim(term: str, text: str) -> str | None:
    """Return the text as it appears in the document, tolerating whitespace changes
    introduced by PDF text extraction"""
    if term in text:
        return term
    words = term.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in words), text)
    return match.group(0) if match else None


def split_samples(text: str) -> list[tuple[str, str]]:
    """Split the anon samples text into (sample number, sample text) pairs"""
    matches = list(_SAMPLE_HEADING.finditer(text))
    samples = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        samples.append((match.group(1), text[match.end() : end].strip()))
    return samples


def label_document(
    doc_id: str,
    doc_type: str,
    text: str,
    positives: list[str],
    negatives: list[str],
    source: str,
) -> tuple[Document, list[str]]:
    """
    Build a corpus Document from text and legacy lists of terms to (not) redact.

    Terms are matched verbatim against the text; terms not found are skipped and
    duplicates are only labelled once.

    :param positives: Terms that should be redacted.
    :param negatives: Terms that look sensitive but should not be redacted.
    :param source: Name of the file the text came from.
    :return: The document, and the input terms that were found in the text.
    """
    entities = []
    found = set()
    for terms, redact in ((positives, True), (negatives, False)):
        for term in terms:
            verbatim = find_verbatim(term, text)
            if verbatim and verbatim not in {e.text for e in entities}:
                entities.append(
                    Entity(
                        text=verbatim,
                        rule=UNSPECIFIED_RULE,
                        category="legacy_label",
                        redact=redact,
                    )
                )
                found.add(term)
    document = Document(
        id=doc_id,
        doc_type=doc_type,
        text=text,
        entities=tuple(entities),
        split="test",
        source=source,
    )
    return document, [t for t in (*positives, *negatives) if t in found]


def convert(data_dir: Path = E2E_DATA) -> tuple[list[Document], list[str]]:
    """
    Convert the existing e2e PDFs and term lists into labelled corpus documents.

    Produces one document per anonymisation sample plus one for the evaluation
    dataset, and validates every document's labels.

    :return: The documents, and the terms that could not be found in the text.
    :raises ValueError: If a document's labels are invalid.
    """
    documents = []
    used_terms = set()

    anon_terms = json.loads(
        (data_dir / "anon_samples_terms.json").read_text(encoding="utf-8")
    )
    anon_text = extract_pdf_text(data_dir / "PINS_anon_samples.pdf")
    for number, sample_text in split_samples(anon_text):
        document, used = label_document(
            f"anon-sample-{int(number):03d}",
            "anonymised_representation",
            sample_text,
            anon_terms,
            [],
            "PINS_anon_samples.pdf",
        )
        documents.append(document)
        used_terms.update(used)
    unmatched = [t for t in anon_terms if t not in used_terms]

    expected = json.loads(
        (data_dir / "eval_dataset_expected.json").read_text(encoding="utf-8")
    )
    eval_text = _GROUP_HEADING.sub("", extract_pdf_text(data_dir / "eval_dataset.pdf"))
    document, used = label_document(
        "eval-dataset",
        "evaluation_dataset",
        eval_text,
        expected["group_a_strings"],
        expected["group_b_strings"],
        "eval_dataset.pdf",
    )
    documents.append(document)
    unmatched += [
        t
        for t in (*expected["group_a_strings"], *expected["group_b_strings"])
        if t not in used
    ]

    for document in documents:
        problems = validate_document(document)
        if problems:
            raise ValueError(f"{document.id}: {problems}")
    return documents, unmatched


def main(argv=None):
    """Command line entry point: convert the samples and write them to --out"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    documents, unmatched = convert()
    write_corpus(args.out, documents)
    labelled = sum(len(d.entities) for d in documents)
    print(f"Wrote {len(documents)} documents with {labelled} labels to {args.out}")
    if unmatched:
        print(f"{len(unmatched)} terms were not found in the extracted text:")
        for term in unmatched:
            print(f"  {term!r}")


if __name__ == "__main__":
    main()
