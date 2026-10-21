from pathlib import Path

import pytest
import yaml

mlflow = pytest.importorskip("mlflow")

from eval.corpus import RULE_NAMES
from eval.prompt_registry import (
    PromptStructureError,
    export_rule_to_yaml,
    normalise_template,
    parse_prompt_sections,
    prompt_uri,
    register_rule_prompt,
)
from eval.run_chunk_sweep import load_rule_configs

DEFAULT_CONFIG = Path("src/config/default.yaml")


@pytest.fixture
def tracking_uri(tmp_path):
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(previous)


def _detailed_template(rule: str) -> str:
    config = load_rule_configs("default", [rule])[rule]
    return config.model_copy(update={"include_rationale": True}).create_system_prompt()


def test_prompt_uri():
    assert prompt_uri("names", "3") == "prompts:/redaction-names/3"
    assert prompt_uri("names", "baseline") == "prompts:/redaction-names@baseline"


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_parse_prompt_sections__round_trips_default_config(rule):
    """Parsing a rule's numbered prompt gives back exactly its YAML fields"""
    config = load_rule_configs("default", [rule])[rule]

    fields = parse_prompt_sections(_detailed_template(rule))

    assert {field: getattr(config, field) for field in fields} == fields
    assert {"role", "task", "redaction_terms"} <= fields.keys()


def test_parse_prompt_sections__multi_line_clause():
    template = (
        "<Role>\nrole\n</Role>\n\n<Task>\ntask\n</Task>\n\n"
        "<Terms>\n- [T1] first line\n  continued\n- [T2] second\n</Terms>"
    )

    fields = parse_prompt_sections(template)

    assert fields["redaction_terms"] == ["first line\n  continued", "second"]


@pytest.mark.parametrize(
    ("template", "message"),
    [
        (
            "Intro\n<Role>\nr\n</Role>\n<Task>\nt\n</Task>\n<Terms>\n- a\n</Terms>",
            "outside",
        ),
        ("<Role>\nr\n</Role>\n<Terms>\n- a\n</Terms>", "Missing"),
        (
            (
                "<Role>\nr\n</Role>\n<Task>\nt\n</Task>\n<Terms>\n- a\n</Terms>\n"
                "<Notes>\nn\n</Notes>"
            ),
            "Unknown",
        ),
        (
            "<Role>\nr\n</Role>\n<Task>\nt\n</Task>\n<Terms>\nno bullet\n</Terms>",
            "'- '",
        ),
    ],
)
def test_parse_prompt_sections__rejects_unexportable_prompts(template, message):
    with pytest.raises(PromptStructureError, match=message):
        parse_prompt_sections(template)


def test_export_rule_to_yaml__replaces_only_the_rule_prompt(tmp_path):
    """
    - Given new prompt fields for one rule, without a MustRedact section
    - When I export them over the default config
    - Then only that rule's prompt fields change, and the dropped section is removed
    """
    fields = parse_prompt_sections(_detailed_template("names"))
    fields["redaction_terms"] = ["A new term"]
    del fields["must_redact"]
    out = tmp_path / "optimised.yaml"

    export_rule_to_yaml(fields, "names", DEFAULT_CONFIG, out)

    original = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    exported = yaml.safe_load(out.read_text(encoding="utf-8"))
    original_rules = original["redactors"][0]["redaction_rules"]
    exported_rules = exported["redactors"][0]["redaction_rules"]
    assert exported_rules[0]["redaction_terms"] == ["A new term"]
    assert "must_redact" not in exported_rules[0]
    assert exported_rules[0]["task"] == original_rules[0]["task"]
    assert exported_rules[1:] == original_rules[1:]
    assert exported["redactors"][1:] == original["redactors"][1:]


def test_export_rule_to_yaml__unknown_rule(tmp_path):
    fields = parse_prompt_sections(_detailed_template("names"))

    with pytest.raises(KeyError):
        export_rule_to_yaml(fields, "not_a_rule", DEFAULT_CONFIG, tmp_path / "x.yaml")


def test_register_rule_prompt(tracking_uri):
    """The registered version has clause IDs, the chunk settings and the alias"""
    config = load_rule_configs("default", ["names"])["names"].model_copy(
        update={"chunk_size": 2500, "chunk_overlap": 100}
    )

    version = register_rule_prompt(config, "default", "baseline", alias="baseline")

    loaded = mlflow.genai.load_prompt(prompt_uri("names", "baseline"))
    assert loaded.version == version.version
    assert "- [T1] " in loaded.template
    assert loaded.model_config["extra_params"]["chunk_size"] == 2500
    assert loaded.model_config["extra_params"]["chunk_overlap"] == 100
    assert loaded.tags["rule"] == "names"


def test_normalise_template__standard_layout():
    """
    - Given an optimiser's template with irregular spacing and clause labels
    - When I normalise it
    - Then sections are on their own lines, blank-line separated, clauses renumbered,
      and sections the template lacks are left out
    """
    config = load_rule_configs("default", ["names"])["names"]

    messy = (
        "<Role>\nA role.\n</Role>\n\n\n<Task>\n\nDo the task.\n- first\n- second\n\n"
        "</Task>\n<Terms>\n- [T7] one\n\n- [T9] two\n</Terms>\n\n"
        "<DoNotRedact>\n- keep this\n</DoNotRedact>"
    )

    assert normalise_template(messy, config) == (
        "<Role>\nA role.\n</Role>\n\n"
        "<Task>\nDo the task.\n- first\n- second\n</Task>\n\n"
        "<Terms>\n- [T1] one\n- [T2] two\n</Terms>\n\n"
        "<DoNotRedact>\n- [D1] keep this\n</DoNotRedact>"
    )


def test_normalise_template__is_stable_for_default_prompts():
    config = load_rule_configs("default", ["names"])["names"]
    template = _detailed_template("names")

    once = normalise_template(template, config)

    assert normalise_template(once, config) == once
    assert parse_prompt_sections(once)["redaction_terms"] == config.redaction_terms


def test_normalise_template__unparseable():
    config = load_rule_configs("default", ["names"])["names"]

    with pytest.raises(PromptStructureError):
        normalise_template("Just some text", config)
