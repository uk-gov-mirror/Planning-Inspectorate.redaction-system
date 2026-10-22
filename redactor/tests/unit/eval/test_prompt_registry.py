from pathlib import Path

import pytest
import yaml

mlflow = pytest.importorskip("mlflow")

from eval.corpus import RULE_NAMES
from eval.prompt_registry import (
    PromptStructureError,
    export_rule_to_yaml,
    export_rules_to_yaml,
    normalise_template,
    parse_prompt_sections,
    prompt_uri,
    register_rule_prompt,
)
from eval.run_chunk_sweep import load_rule_configs

from src.redaction.config import LLMTextRedactionConfig
from src.redaction.config_processor import ConfigProcessor

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
                "<notes>\nn\n</notes>"
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


def test_export_rules_to_yaml__replaces_each_rule(tmp_path):
    names = parse_prompt_sections(_detailed_template("names"))
    names["redaction_terms"] = ["A new names term"]
    sensitive = parse_prompt_sections(_detailed_template("sensitive_info"))
    sensitive["redaction_terms"] = ["A new sensitive term"]
    out = tmp_path / "optimised.yaml"

    export_rules_to_yaml(
        {"names": names, "sensitive_info": sensitive}, DEFAULT_CONFIG, out
    )

    exported = yaml.safe_load(out.read_text(encoding="utf-8"))
    original = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    rules = {r["name"]: r for r in exported["redactors"][0]["redaction_rules"]}
    original_rules = {r["name"]: r for r in original["redactors"][0]["redaction_rules"]}
    assert rules["names"]["redaction_terms"] == ["A new names term"]
    assert rules["sensitive_info"]["redaction_terms"] == ["A new sensitive term"]
    assert rules["contact_identity_info"] == original_rules["contact_identity_info"]


def test_export_rules_to_yaml__writes_nothing_if_any_rule_fails(tmp_path):
    fields = parse_prompt_sections(_detailed_template("names"))
    out = tmp_path / "x.yaml"

    with pytest.raises(KeyError):
        export_rules_to_yaml(
            {"names": fields, "not_a_rule": fields}, DEFAULT_CONFIG, out
        )

    assert not out.exists()


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


_WITH_EXTRA_SECTION = (
    "<Role>\nr\n</Role>\n\n<Task>\nt\n</Task>\n\n<Terms>\n- [T1] a\n</Terms>\n\n"
    "<EdgeCases>\n- first case\n  continued\n- [X2] second case\n</EdgeCases>"
)


def test_parse_prompt_sections__extra_sections():
    """Non-standard CamelCase sections are parsed, without bullet labels, by tag"""
    fields = parse_prompt_sections(_WITH_EXTRA_SECTION)

    assert fields["extra_sections"] == {
        "EdgeCases": ["first case\n  continued", "second case"]
    }


def test_parse_prompt_sections__repeated_extra_section():
    template = _WITH_EXTRA_SECTION + "\n<EdgeCases>\n- again\n</EdgeCases>"

    with pytest.raises(PromptStructureError, match="more than once"):
        parse_prompt_sections(template)


def test_normalise_template__keeps_extra_sections_unlabelled():
    config = load_rule_configs("default", ["names"])["names"]

    assert normalise_template(_WITH_EXTRA_SECTION, config) == (
        "<Role>\nr\n</Role>\n\n<Task>\nt\n</Task>\n\n<Terms>\n- [T1] a\n</Terms>\n\n"
        "<EdgeCases>\n- first case\n  continued\n- second case\n</EdgeCases>"
    )


def test_export_rule_to_yaml__extra_sections_round_trip(tmp_path):
    """
    - Given a prompt with an extra section
    - When I export it, and build a prompt from the exported rule
    - Then the exported YAML holds the section and the prompt matches the original
    """
    config = load_rule_configs("default", ["names"])["names"]
    fields = parse_prompt_sections(normalise_template(_WITH_EXTRA_SECTION, config))
    out = tmp_path / "optimised.yaml"

    export_rule_to_yaml(fields, "names", DEFAULT_CONFIG, out)

    exported = yaml.safe_load(out.read_text(encoding="utf-8"))
    rule = exported["redactors"][0]["redaction_rules"][0]
    assert rule["extra_sections"] == {
        "EdgeCases": ["first case\n  continued", "second case"]
    }
    rebuilt = next(
        c
        for c in ConfigProcessor.validate_and_parse_redaction_config(exported)
        if c.name == "names" and type(c) is LLMTextRedactionConfig
    ).model_copy(update={"include_rationale": True})
    assert rebuilt.create_system_prompt() == normalise_template(
        _WITH_EXTRA_SECTION, config
    )


def test_export_rule_to_yaml__drops_removed_extra_sections(tmp_path):
    """Exporting fields without extra sections removes ones the base config had"""
    base = tmp_path / "base.yaml"
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["redactors"][0]["redaction_rules"][0]["extra_sections"] = {"Old": ["x"]}
    base.write_text(yaml.dump(config), encoding="utf-8")
    out = tmp_path / "out.yaml"

    export_rule_to_yaml(
        parse_prompt_sections(_detailed_template("names")), "names", base, out
    )

    rule = yaml.safe_load(out.read_text(encoding="utf-8"))["redactors"][0][
        "redaction_rules"
    ][0]
    assert "extra_sections" not in rule
