"""Bridge between the YAML redaction config and the MLflow Prompt Registry.

The YAML config stays the source of truth. Each LLM rule's system prompt is registered as
``redaction-<rule>``, and an optimised version is parsed back into the rule's YAML fields
and written to a new config file for review.
"""

import re
from pathlib import Path

import mlflow
import yaml
from mlflow.entities.model_registry import PromptVersion
from tiktoken import get_encoding

from src.redaction.config import EXTRA_SECTION_TAG, LLMTextRedactionConfig
from src.types import LLMDetailedRedactionResultFormat

PROMPT_NAME_PREFIX = "redaction-"

# (XML tag, config field, is a list of clauses)
PROMPT_SECTIONS = (
    ("Role", "role", False),
    ("Task", "task", False),
    ("Terms", "redaction_terms", True),
    ("MustRedact", "must_redact", True),
    ("DoNotRedact", "do_not_redact", True),
    ("Examples", "examples", True),
)
REQUIRED_SECTIONS = ("Role", "Task", "Terms")
EXTRA_SECTIONS_FIELD = "extra_sections"
PROMPT_FIELDS = (*(field for _, field, _ in PROMPT_SECTIONS), EXTRA_SECTIONS_FIELD)

CLAUSE_ID = re.compile(r"\[([TMDE]\d+)\]")
_SECTION = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.DOTALL)
_CLAUSE_LABEL = re.compile(r"^\[[A-Z]\d+\]\s*")


class PromptStructureError(ValueError):
    pass


def prompt_name(rule: str) -> str:
    """Registry name of a rule's prompt"""
    return f"{PROMPT_NAME_PREFIX}{rule}"


def rule_from_prompt_name(name: str) -> str:
    """Rule name of a registered prompt"""
    return name.removeprefix(PROMPT_NAME_PREFIX)


def prompt_uri(rule: str, ref: str) -> str:
    """URI of a rule's prompt, where ref is a version number or an alias"""
    name = prompt_name(rule)
    return f"prompts:/{name}/{ref}" if ref.isdigit() else f"prompts:/{name}@{ref}"


def register_rule_prompt(
    config: LLMTextRedactionConfig,
    config_name: str,
    commit_message: str | None = None,
    alias: str | None = None,
    template: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> PromptVersion:
    """
    Register a rule's system prompt, with clause IDs, as a new prompt version.

    The prompt is built from the config unless a template is given, e.g. from an
    optimiser. The chunk settings are stored in the model config, so evaluations of this
    version use the chunking it was registered with.
    """
    detailed = config.model_copy(update={"include_rationale": True})
    template = template or detailed.create_system_prompt()
    encoding = get_encoding(detailed.token_encoding_name)
    model_config = {"model_name": config.model}
    if config.max_tokens:
        model_config["max_tokens"] = config.max_tokens
    model_config["extra_params"] = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "reasoning_effort": config.reasoning_effort,
    }
    version = mlflow.genai.register_prompt(
        name=prompt_name(config.name),
        template=template,
        commit_message=commit_message,
        tags={
            "rule": config.name,
            "config_name": config_name,
            "prompt_tokens": str(len(encoding.encode(template))),
            **(extra_tags or {}),
        },
        response_format=LLMDetailedRedactionResultFormat,
        model_config=model_config,
    )
    if alias:
        mlflow.genai.set_prompt_alias(version.name, alias, version.version)
    return version


def normalise_template(template: str, config: LLMTextRedactionConfig) -> str:
    """
    Re-render a prompt in the standard layout: each section's tags on their own lines,
    a blank line between sections, and one numbered '- [T1] ...' bullet per clause.

    Optimisers rewrite the prompt freely, so this fixes layout and clause numbering.

    :raises PromptStructureError: If the prompt cannot be parsed into sections.
    """
    fields = parse_prompt_sections(template)
    update = {field: _stripped(value) for field, value in fields.items()}
    unset = dict.fromkeys(PROMPT_FIELDS)
    return config.model_copy(
        update=unset | update | {"include_rationale": True}
    ).create_system_prompt()


def _stripped(value):
    """Strip whitespace from a field's text, clauses, or an extra section's clauses"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {key: _stripped(clauses) for key, clauses in value.items()}
    return [clause.strip() for clause in value]


def _parse_clauses(body: str, tag: str) -> list[str]:
    """Split a list section into clauses, dropping bullets and clause ID labels"""
    clauses: list[str] = []
    for line in body.split("\n"):
        if line.startswith("- "):
            clauses.append(_CLAUSE_LABEL.sub("", line[2:]))
        elif not line.strip():
            continue
        elif clauses:
            clauses[-1] += "\n" + line
        else:
            raise PromptStructureError(f"<{tag}> has text before its first '- ' item")
    if not clauses:
        raise PromptStructureError(f"<{tag}> has no '- ' items")
    return clauses


def parse_prompt_sections(template: str) -> dict[str, str | list[str]]:
    """
    Parse a system prompt back into the YAML fields it was built from. Sections other
    than the standard ones become the ``extra_sections`` field, keyed by tag.

    :raises PromptStructureError: If a section has an invalid tag, is repeated or
    missing, or there is text outside the sections, since that text would be lost on
    export.
    """
    known = {tag for tag, _, _ in PROMPT_SECTIONS}
    bodies: dict[str, str] = {}
    extras: dict[str, str] = {}
    for match in _SECTION.finditer(template):
        tag, body = match.group(1), match.group(2)
        if tag in bodies or tag in extras:
            raise PromptStructureError(f"Section <{tag}> appears more than once")
        if tag in known:
            bodies[tag] = body
        elif EXTRA_SECTION_TAG.fullmatch(tag):
            extras[tag] = body
        else:
            raise PromptStructureError(
                f"Unknown section <{tag}>: extra sections need a CamelCase tag"
            )

    leftover = _SECTION.sub("", template).strip()
    if leftover:
        raise PromptStructureError(
            f"Text outside the prompt sections would be lost: {leftover[:200]!r}"
        )
    missing = [tag for tag in REQUIRED_SECTIONS if tag not in bodies]
    if missing:
        raise PromptStructureError(f"Missing required sections: {missing}")

    fields = {
        field: _parse_clauses(bodies[tag], tag) if is_list else bodies[tag]
        for tag, field, is_list in PROMPT_SECTIONS
        if tag in bodies
    }
    if extras:
        fields[EXTRA_SECTIONS_FIELD] = {
            tag: _parse_clauses(body, tag) for tag, body in extras.items()
        }
    return fields


class _BlockStyleDumper(yaml.SafeDumper):
    pass


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_BlockStyleDumper.add_representer(str, _represent_str)


def export_rules_to_yaml(
    fields_by_rule: dict[str, dict[str, str | list[str]]],
    base_path: Path,
    out_path: Path,
):
    """
    Write a copy of the base config with each given LLM rule's prompt fields replaced.
    Nothing is written if any rule fails.

    :raises KeyError: If a rule is not an LLMTextRedaction rule in the base config.
    """
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    for rule, fields in fields_by_rule.items():
        rule_config = next(
            (
                rule_config
                for redactor in config["redactors"]
                if redactor["redactor_type"] == "LLMTextRedaction"
                for rule_config in redactor["redaction_rules"]
                if rule_config["name"] == rule
            ),
            None,
        )
        if rule_config is None:
            raise KeyError(f"LLMTextRedaction rule '{rule}' not found in {base_path}")
        for field in PROMPT_FIELDS:
            if field in fields:
                rule_config[field] = fields[field]
            else:
                rule_config.pop(field, None)

        # Fail before writing if production could not build a prompt from the new fields
        LLMTextRedactionConfig(
            redactor_type="LLMTextRedaction", **rule_config
        ).create_system_prompt()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.dump(
            config,
            Dumper=_BlockStyleDumper,
            sort_keys=False,
            allow_unicode=True,
            width=1_000_000,
        ),
        encoding="utf-8",
    )


def export_rule_to_yaml(
    fields: dict[str, str | list[str]], rule: str, base_path: Path, out_path: Path
):
    """Write a copy of the base config with one LLM rule's prompt fields replaced"""
    export_rules_to_yaml({rule: fields}, base_path, out_path)
