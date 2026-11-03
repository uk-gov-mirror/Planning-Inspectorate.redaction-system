import csv
import importlib.util
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT_PATH = Path(__file__).parents[4] / "scripts" / "export_rationale.py"
spec = importlib.util.spec_from_file_location("export_rationale", SCRIPT_PATH)
export_rationale = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_rationale)


def create_rationale(job_id="job-1") -> dict:
    return {
        "job_id": job_id,
        "date": "2026-11-03",
        "file_name": "doc.pdf",
        "config_name": "default",
        "rules": [
            {
                "rule_name": "names",
                "model": "gpt-5.6-luna",
                "clauses": {"T1": "Names of the public", "D1": "Councillors"},
                "page_offsets": [[1, 0], [2, 50]],
                "chunks": [
                    {"chunk_index": 0, "status": "ok", "pages": [1]},
                    {"chunk_index": 1, "status": "failed", "pages": [2]},
                ],
                "redactions": [
                    {
                        "text": "Mary <Jones>",
                        "confidence": "high",
                        "clause_ids": ["T1"],
                        "applied": False,
                        "pages": [2],
                        "occurrences": [
                            {
                                "chunk_index": 1,
                                "chunk_id": "id1",
                                "data_subject": "third_party",
                                "clause_ids": ["T1"],
                                "reason": "Neighbour",
                                "confidence": "high",
                                "span_offsets": [60],
                                "pages": [2],
                            }
                        ],
                    },
                    {
                        "text": "Joan Brown",
                        "confidence": "high",
                        "clause_ids": ["T1", "X9"],
                        "applied": True,
                        "pages": [1, 2],
                        "occurrences": [
                            {
                                "chunk_index": 0,
                                "chunk_id": "id0",
                                "data_subject": "author",
                                "clause_ids": ["T1"],
                                "reason": "Author of the letter",
                                "confidence": "high",
                                "span_offsets": [0, 12],
                                "pages": [1],
                            },
                            {
                                "chunk_index": 1,
                                "chunk_id": "id1",
                                "data_subject": "author",
                                "clause_ids": ["T1", "X9"],
                                "reason": "Named again",
                                "confidence": "low",
                                "span_offsets": [],
                                "pages": [2],
                            },
                        ],
                    },
                ],
                "exclusions": [
                    {
                        "text": "Cllr Raj Patel",
                        "clause_ids": ["D1"],
                        "conflicts_with_redaction": True,
                        "pages": [],
                        "occurrences": [
                            {
                                "chunk_index": 0,
                                "chunk_id": "id0",
                                "clause_ids": ["D1"],
                                "reason": "Councillor",
                                "pages": [],
                            }
                        ],
                    }
                ],
            }
        ],
    }


class TestFlattenRationale:
    def test_one_row_per_occurrence_sorted_by_page_then_text(self):
        rows = export_rationale.flatten_rationale(create_rationale())

        assert [(r["kind"], r["text"], r["chunk_index"]) for r in rows] == [
            ("exclusion", "Cllr Raj Patel", 0),
            ("redaction", "Joan Brown", 0),
            ("redaction", "Joan Brown", 1),
            ("redaction", "Mary <Jones>", 1),
        ]
        assert all(set(r) == set(export_rationale.CSV_COLUMNS) for r in rows)

    def test_row_details(self):
        rows = export_rationale.flatten_rationale(create_rationale())

        joan_again = rows[2]
        assert joan_again["pages"] == "2"
        assert joan_again["clause_ids"] == "T1, X9"
        assert joan_again["clauses"] == "T1: Names of the public; X9: ?"
        assert joan_again["reason"] == "Named again"
        assert joan_again["span_offsets"] == ""
        assert joan_again["applied"] is True
        assert rows[1]["span_offsets"] == "0, 12"
        assert rows[0]["conflicts_with_redaction"] is True
        assert rows[0]["data_subject"] is None

    def test_no_rules(self):
        assert export_rationale.flatten_rationale({"job_id": "x"}) == []


class TestRenderJobHtml:
    def test_groups_by_page_then_term(self):
        page = export_rationale.render_job_html(create_rationale())

        headings = [h for h in ("Page 1", "Page 2", "Page unknown") if h in page]
        assert headings == ["Page 1", "Page 2", "Page unknown"]
        assert page.index("Page 1") < page.index("Page 2") < page.index("Page unknown")
        # Joan Brown is on both pages, but each page only shows its own reason
        page_one = page.split("Page 2")[0]
        page_two = page.split("Page 2")[1].split("Page unknown")[0]
        assert "Author of the letter" in page_one
        assert "Named again" not in page_one
        assert "Named again" in page_two
        assert "Author of the letter" not in page_two

    def test_terms_are_in_order_of_appearance_across_rules(self):
        """
        - Given redactions and exclusions from two rules on the same page, and a term
          whose position is not known
        - When I render the job
        - Then the page lists the terms in the order they appear in the text,
          regardless of rule or kind, with the unlocated term last
        """

        def decision(text, offsets):
            return {
                "text": text,
                "confidence": "high",
                "pages": [1],
                "occurrences": [
                    {"chunk_index": 0, "span_offsets": offsets, "pages": [1]}
                ],
            }

        def rule(name, redactions, exclusions=()):
            return {
                "rule_name": name,
                "page_offsets": [[1, 0]],
                "redactions": redactions,
                "exclusions": list(exclusions),
            }

        rationale = {
            "rules": [
                rule(
                    "names",
                    [decision("Zed", [90]), decision("Alice", [10])],
                    [decision("Cllr Bob", [40])],
                ),
                rule("contact", [decision("07700", [20]), decision("Unlocated", [])]),
            ]
        }

        page = export_rationale.render_job_html(rationale)

        terms = ["Alice", "07700", "Cllr Bob", "Zed", "Unlocated"]
        positions = [page.index(f'class="term">{term}') for term in terms]
        assert positions == sorted(positions)

    def test_offsets_are_taken_from_the_page_being_shown(self):
        """
        - Given a term on two pages, and another term only on the second page
        - When I render the job
        - Then on the second page the term is placed by its position on that page, not
          by its earlier position on the first
        """

        def occurrence(offsets, page):
            return {"chunk_index": 0, "span_offsets": offsets, "pages": [page]}

        rationale = {
            "rules": [
                {
                    "rule_name": "names",
                    "page_offsets": [[1, 0], [2, 100]],
                    "redactions": [
                        {
                            "text": "Repeated",
                            "pages": [1, 2],
                            "occurrences": [
                                occurrence([5], 1),
                                occurrence([180], 2),
                            ],
                        },
                        {
                            "text": "Second",
                            "pages": [2],
                            "occurrences": [occurrence([120], 2)],
                        },
                    ],
                }
            ]
        }

        page_two = export_rationale.render_job_html(rationale).split("Page 2")[1]

        assert page_two.index("Second") < page_two.index("Repeated")

    def test_marks_each_term_with_its_rule_between_confidence_and_clauses(self):
        page = export_rationale.render_job_html(create_rationale())

        joan = page.split('class="term">Joan Brown')[1].split("</div>")[0]
        assert (
            joan.index(">high<")
            < joan.index('class="chip rule">names<')
            < joan.index(">T1<")
        )
        councillor = page.split('class="term">Cllr Raj Patel')[1].split("</div>")[0]
        assert (
            councillor.index(">kept<")
            < councillor.index('class="chip rule">names<')
            < councillor.index(">D1<")
        )

    def test_lists_the_clauses_of_each_rule(self):
        rationale = create_rationale()
        other_rule = {
            "rule_name": "contact",
            "model": "gpt-5.6-luna",
            "clauses": {"T1": "Phone numbers", "M1": "Emails <all>"},
            "chunks": [],
            "redactions": [],
            "exclusions": [],
        }
        rationale["rules"].append(other_rule)

        page = export_rationale.render_job_html(rationale)

        rules_section = page.split("<h2>Redaction rules</h2>")[1]
        assert page.index("Page unknown") < page.index("Redaction rules")
        assert rules_section.index("<h3>names</h3>") < rules_section.index(
            "<h3>contact</h3>"
        )
        for expected in (
            "<b>T1</b> Names of the public",
            "<b>D1</b> Councillors",
            "<b>T1</b> Phone numbers",
            "<b>M1</b> Emails &lt;all&gt;",
        ):
            assert expected in rules_section

    def test_flags_and_chunk_status(self):
        page = export_rationale.render_job_html(create_rationale())

        assert "not found in the PDF" in page
        assert "also redacted in another chunk" in page
        assert "1 not analysed" in page
        assert 'title="Names of the public"' in page

    def test_escapes_text_from_the_document(self):
        page = export_rationale.render_job_html(create_rationale())

        assert "Mary &lt;Jones&gt;" in page
        assert "Mary <Jones>" not in page


class TestCollectRationales:
    @staticmethod
    def _blob(name, created):
        blob = Mock(creation_time=datetime(*created, tzinfo=UTC))
        blob.name = name
        return blob

    def test_filters_by_name_and_date(self):
        blobs = [
            self._blob("a-1/rationale.json", (2026, 11, 1)),
            self._blob("a-1/proposed_redactions.json", (2026, 11, 1)),
            self._blob("b-1/rationale.json", (2026, 11, 3)),
            self._blob("c-1/rationale.json", (2026, 11, 9)),
        ]
        container = Mock()
        container.list_blobs.return_value = blobs
        container.get_blob_client.side_effect = lambda name: Mock(
            **{
                "download_blob.return_value.readall.return_value": b'{"job_id": "%s"}'
                % name.encode()
            }
        )

        result = export_rationale.collect_rationales(
            container, start_date=date(2026, 11, 2), end_date=date(2026, 11, 8)
        )

        assert [r["job_id"] for r in result] == ["b-1/rationale.json"]

    def test_passes_prefix_to_listing(self):
        container = Mock()
        container.list_blobs.return_value = []

        export_rationale.collect_rationales(container, job_prefix="abc")

        container.list_blobs.assert_called_once_with(name_starts_with="abc")

    def test_skips_malformed_json(self):
        container = Mock()
        container.list_blobs.return_value = [
            self._blob("a-1/rationale.json", (2026, 11, 1))
        ]
        container.get_blob_client.return_value.download_blob.return_value.readall.return_value = b"{not json"

        assert export_rationale.collect_rationales(container) == []


class TestWriteOutputs:
    def test_writes_csv_and_html(self, tmp_path):
        rationales = [create_rationale("job-1"), create_rationale("job-2")]

        csv_path, n_rows = export_rationale.write_outputs(rationales, tmp_path / "out")

        assert n_rows == 8
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 8
        assert list(rows[0]) == export_rationale.CSV_COLUMNS
        assert {r["job_id"] for r in rows} == {"job-1", "job-2"}
        assert sorted(p.name for p in (tmp_path / "out" / "html").iterdir()) == [
            "job-1.html",
            "job-2.html",
        ]

    def test_no_html(self, tmp_path):
        export_rationale.write_outputs([create_rationale()], tmp_path, write_html=False)

        assert (tmp_path / "rationale.csv").exists()
        assert not (tmp_path / "html").exists()

    @pytest.mark.parametrize(
        ("job_id", "expected"),
        [("a/b\\c:d", "a_b_c_d"), ("../../evil", ".._.._evil"), (None, "unknown")],
    )
    def test_html_file_name_is_safe(self, tmp_path, job_id, expected):
        export_rationale.write_outputs([create_rationale(job_id)], tmp_path)

        assert [p.name for p in (tmp_path / "html").iterdir()] == [f"{expected}.html"]
