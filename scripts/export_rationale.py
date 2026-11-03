"""
Export the LLM redaction rationale saved by ANALYSE runs to formats that are easier to
read than raw JSON

For each `<job>/rationale.json` in the redactiondata container that was created in the
date range, this writes:
  - `rationale.csv`: one row per occurrence of a term, for filtering in Excel
  - `html/<job>.html`: one page per job, grouped by page of the PDF then by term in the
    order they appear, with each term marked with its rule, and the clauses of each rule

The rationale quotes personal data. Write the output somewhere that is not tracked by git,
and delete it when you are done. The redactiondata container deletes blobs after 7 days,
so export before then. Your IP needs access to the storage account's network rules, and
you must be signed in with `az login`.

# Example usage
`python3 scripts/export_rationale.py --storage-name pinsstredactiondevuks --out-dir ~/rationale --start-date 2026-11-01`
"""

import csv
import html
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import click
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)
from azure.storage.blob import BlobServiceClient, ContainerClient

CONTAINER_NAME = "redactiondata"
RATIONALE_BLOB_NAME = "rationale.json"

CSV_COLUMNS = [
    "job_id",
    "date",
    "file_name",
    "config_name",
    "rule_name",
    "kind",
    "text",
    "applied",
    "conflicts_with_redaction",
    "pages",
    "confidence",
    "data_subject",
    "clause_ids",
    "clauses",
    "reason",
    "chunk_index",
    "chunk_id",
    "span_offsets",
]


def _join(values: list[Any], separator: str = ", ") -> str:
    return separator.join(str(value) for value in values)


def _describe_clauses(clause_ids: list[str], clauses: dict[str, str]) -> str:
    return "; ".join(f"{i}: {clauses.get(i, '?')}" for i in clause_ids)


def flatten_rationale(rationale: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a job's rationale into one row per occurrence of each redacted or
    excluded term, sorted by rule, kind, first page and term"""
    rows = []
    for rule in rationale.get("rules", []):
        clauses = rule.get("clauses", {})
        for kind, key in (("redaction", "redactions"), ("exclusion", "exclusions")):
            for decision in rule.get(key, []):
                for occurrence in decision.get("occurrences", []):
                    clause_ids = occurrence.get("clause_ids", [])
                    rows.append(
                        {
                            "job_id": rationale.get("job_id"),
                            "date": rationale.get("date"),
                            "file_name": rationale.get("file_name"),
                            "config_name": rationale.get("config_name"),
                            "rule_name": rule.get("rule_name"),
                            "kind": kind,
                            "text": decision.get("text"),
                            "applied": decision.get("applied"),
                            "conflicts_with_redaction": decision.get(
                                "conflicts_with_redaction"
                            ),
                            "pages": _join(occurrence.get("pages", [])),
                            "confidence": occurrence.get("confidence"),
                            "data_subject": occurrence.get("data_subject"),
                            "clause_ids": _join(clause_ids),
                            "clauses": _describe_clauses(clause_ids, clauses),
                            "reason": occurrence.get("reason"),
                            "chunk_index": occurrence.get("chunk_index"),
                            "chunk_id": occurrence.get("chunk_id"),
                            "span_offsets": _join(occurrence.get("span_offsets", [])),
                            "_first_page": (decision.get("pages") or [math.inf])[0],
                        }
                    )
    rows.sort(
        key=lambda r: (
            r["rule_name"] or "",
            r["kind"],
            r["_first_page"],
            r["text"] or "",
            r["chunk_index"] if r["chunk_index"] is not None else -1,
        )
    )
    for row in rows:
        del row["_first_page"]
    return rows


HTML_STYLE = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 2rem; color: #222; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.2rem; margin-top: 2rem; }
h3 { font-size: 1.05rem; margin-top: 1.5rem; border-bottom: 1px solid #ccc; }
.meta { color: #555; } .term { margin: 0.8rem 0 0.2rem 0; font-weight: 600; }
.chip { display: inline-block; padding: 0 0.4rem; margin-left: 0.3rem; border-radius: 3px;
        background: #e6eefc; font-size: 0.8rem; font-weight: 400; }
.chip.low { background: #fde8e8; } .chip.medium { background: #fff4d6; }
.chip.high { background: #e3f5e1; } .chip.warn { background: #f8d0d0; }
.chip.excluded { background: #eee; } .chip.rule { background: #ece6fb; }
ul.reasons { margin: 0.2rem 0 0 1.2rem; padding: 0; } .small { color: #555; font-size: 0.85rem; }
ul.clauses { margin: 0.2rem 0 0 1.2rem; padding: 0; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _clause_chips(clause_ids: list[str], clauses: dict[str, str]) -> str:
    return "".join(
        f'<span class="chip" title="{_esc(clauses.get(i, ""))}">{_esc(i)}</span>'
        for i in clause_ids
    )


def _first_offset_on_page(
    occurrences: list[dict[str, Any]],
    page_offsets: list[list[int]],
    page: int | None,
) -> float:
    """Where the earliest of the occurrences' spans that is on the page starts in the
    analysed text, or infinity if none of their positions are known"""
    for i, (page_number, start) in enumerate(page_offsets):
        if page_number == page:
            end = page_offsets[i + 1][1] if i + 1 < len(page_offsets) else math.inf
            offsets = [
                offset
                for occurrence in occurrences
                for offset in occurrence.get("span_offsets", [])
                if start <= offset < end
            ]
            return min(offsets, default=math.inf)
    return math.inf


def _entries_by_page(
    rationale: dict[str, Any],
) -> dict[int | None, list[dict[str, Any]]]:
    """Group every rule's decisions by the pages they appear on, in order of
    appearance in the text. Decisions with no known page are grouped under None"""
    by_page = defaultdict(list)
    for rule in rationale.get("rules", []):
        page_offsets = rule.get("page_offsets", [])
        for kind, key in (("redaction", "redactions"), ("exclusion", "exclusions")):
            for decision in rule.get(key, []):
                for page in decision.get("pages") or [None]:
                    occurrences = [
                        o
                        for o in decision.get("occurrences", [])
                        if page is None or page in o.get("pages", [])
                    ]
                    by_page[page].append(
                        {
                            "rule": rule,
                            "kind": kind,
                            "decision": decision,
                            "occurrences": occurrences,
                            "offset": _first_offset_on_page(
                                occurrences, page_offsets, page
                            ),
                        }
                    )
    for entries in by_page.values():
        entries.sort(
            key=lambda e: (
                e["offset"],
                str(e["decision"].get("text")).lower(),
                e["rule"].get("rule_name") or "",
            )
        )
    return by_page


def _render_entry(entry: dict[str, Any]) -> str:
    rule, kind, decision = entry["rule"], entry["kind"], entry["decision"]
    # Confidence (or kept), then the rule, then the clauses, then any warnings
    if kind == "exclusion":
        badges = ['<span class="chip excluded">kept</span>']
    else:
        badges = [
            (
                f'<span class="chip {_esc(decision.get("confidence"))}">'
                f"{_esc(decision.get('confidence'))}</span>"
            )
        ]
    badges.append(f'<span class="chip rule">{_esc(rule.get("rule_name"))}</span>')
    badges.append(
        _clause_chips(decision.get("clause_ids", []), rule.get("clauses", {}))
    )
    if kind == "exclusion" and decision.get("conflicts_with_redaction"):
        badges.append('<span class="chip warn">also redacted in another chunk</span>')
    if kind == "redaction" and decision.get("applied") is False:
        badges.append('<span class="chip warn">not found in the PDF</span>')
    reasons = "".join(
        "<li>"
        f"{_esc(o.get('reason'))} "
        f'<span class="small">(chunk {_esc(o.get("chunk_index"))}'
        f"{', ' + _esc(o.get('data_subject')) if o.get('data_subject') else ''}"
        f"{', ' + _esc(o.get('confidence')) if kind == 'redaction' else ''})</span>"
        "</li>"
        for o in entry["occurrences"]
    )
    return (
        f'<div class="term">{_esc(decision.get("text"))} {"".join(badges)}</div>'
        f'<ul class="reasons">{reasons}</ul>'
    )


def _render_rule_clauses(rule: dict[str, Any]) -> str:
    chunks = rule.get("chunks", [])
    not_ok = [c for c in chunks if c.get("status") != "ok"]
    clauses = "".join(
        f"<li><b>{_esc(clause_id)}</b> {_esc(wording)}</li>"
        for clause_id, wording in rule.get("clauses", {}).items()
    )
    return (
        f"<h3>{_esc(rule.get('rule_name'))}</h3>"
        '<p class="meta">'
        f"{_esc(rule.get('model'))}, {len(chunks)} chunks"
        f"{f', <b>{len(not_ok)} not analysed</b>' if not_ok else ''}</p>"
        f'<ul class="clauses">{clauses}</ul>'
    )


def render_job_html(rationale: dict[str, Any]) -> str:
    """Render a job's rationale as a page grouped by PDF page, then by term in order
    of appearance, followed by the clauses of each rule"""
    body = [
        f"<h1>Redaction rationale for {_esc(rationale.get('file_name'))}</h1>",
        (
            '<p class="meta">'
            f"Job {_esc(rationale.get('job_id'))} on {_esc(rationale.get('date'))}, "
            f"config {_esc(rationale.get('config_name'))}</p>"
        ),
    ]
    by_page = _entries_by_page(rationale)
    for page in sorted(by_page, key=lambda p: math.inf if p is None else p):
        body.append(f"<h2>{'Page unknown' if page is None else f'Page {page}'}</h2>")
        body.extend(_render_entry(entry) for entry in by_page[page])
    body.append("<h2>Redaction rules</h2>")
    body.extend(_render_rule_clauses(rule) for rule in rationale.get("rules", []))
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Redaction rationale</title><style>{HTML_STYLE}</style></head>"
        f"<body>{''.join(body)}</body></html>"
    )


def collect_rationales(
    container_client: ContainerClient,
    job_prefix: str = "",
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict[str, Any]]:
    """Download every rationale blob created in the date range (inclusive)"""
    rationales = []
    for blob in container_client.list_blobs(name_starts_with=job_prefix or None):
        if blob.name.rsplit("/", 1)[-1] != RATIONALE_BLOB_NAME:
            continue
        created = blob.creation_time.date() if blob.creation_time else None
        if created and (
            (start_date and created < start_date) or (end_date and created > end_date)
        ):
            continue
        try:
            data = container_client.get_blob_client(blob.name).download_blob().readall()
            rationales.append(json.loads(data))
        except (ValueError, OSError) as e:
            click.echo(f"Skipping '{blob.name}': {e}", err=True)
    return rationales


def _safe_file_name(job_id: str | None) -> str:
    return re.sub(r"[^\w.-]", "_", job_id or "unknown")


def write_outputs(
    rationales: list[dict[str, Any]], out_dir: Path, write_html: bool = True
) -> tuple[Path, int]:
    """Write the combined CSV, and an HTML page for each job if requested

    :return tuple[Path, int]: The CSV path and the number of rows in it
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for rationale in rationales for row in flatten_rationale(rationale)]
    csv_path = out_dir / "rationale.csv"
    # utf-8-sig so that Excel detects the encoding
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if write_html:
        html_dir = out_dir / "html"
        html_dir.mkdir(exist_ok=True)
        for rationale in rationales:
            html_path = html_dir / f"{_safe_file_name(rationale.get('job_id'))}.html"
            html_path.write_text(render_job_html(rationale), encoding="utf-8")
    return csv_path, len(rows)


@click.command()
@click.option(
    "--storage-name",
    envvar="STORAGE_NAME",
    required=True,
    help="Name of the redaction storage account",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Where to write the export. It contains personal data",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Only export jobs created on or after this date",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Only export jobs created on or before this date",
)
@click.option("--job-prefix", default="", help="Only export jobs starting with this")
@click.option("--html/--no-html", "write_html", default=True)
def main(storage_name, out_dir, start_date, end_date, job_prefix, write_html):
    service_client = BlobServiceClient(
        f"https://{storage_name}.blob.core.windows.net",
        credential=ChainedTokenCredential(
            ManagedIdentityCredential(), AzureCliCredential()
        ),
    )
    rationales = collect_rationales(
        service_client.get_container_client(CONTAINER_NAME),
        job_prefix=job_prefix,
        start_date=start_date.date() if start_date else None,
        end_date=end_date.date() if end_date else None,
    )
    if not rationales:
        click.echo("No rationale files were found")
        return
    csv_path, n_rows = write_outputs(rationales, out_dir, write_html)
    click.echo(f"Exported {len(rationales)} jobs ({n_rows} rows) to {csv_path.parent}")


if __name__ == "__main__":
    main()
