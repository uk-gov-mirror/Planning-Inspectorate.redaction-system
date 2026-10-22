from types import SimpleNamespace
from unittest.mock import patch

import pytest
from azure.core.exceptions import ClientAuthenticationError

mlflow = pytest.importorskip("mlflow")

from eval import optimise_prompt
from eval.corpus import Document, Entity, write_corpus
from eval.prompt_registry import register_rule_prompt


@pytest.fixture(autouse=True)
def tracking_uri(tmp_path):
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    yield
    mlflow.set_tracking_uri(previous)


def _prompt(model_config=None, name="redaction-names", version=1):
    return SimpleNamespace(
        name=name,
        version=version,
        uri=f"prompts:/{name}/{version}",
        model_config=model_config,
        template="",
    )


def test_parse_args__gate_defaults_to_test_split():
    args = optimise_prompt.parse_args(["gate", "--rule", "names"])

    assert args.split == "test"
    assert (args.candidate, args.reference) == ("candidate", "baseline")


@pytest.mark.parametrize(
    ("command", "expected"),
    [("evaluate", "dev"), ("optimise", "dev"), ("gate", "test")],
)
def test_parse_args__split_default_is_per_command(command, expected):
    """Only gate uses the held-out test split by default. Setting that default on
    gate must not change it for the other commands that share its arguments"""
    assert optimise_prompt.parse_args([command, "--rule", "names"]).split == expected
    assert (
        optimise_prompt.parse_args(
            [command, "--rule", "names", "--split", "test"]
        ).split
        == "test"
    )


def test_rule_config__precedence():
    """Command line overrides the prompt's model config, which overrides the YAML"""
    prompt = _prompt(
        {"max_tokens": 3000, "extra_params": {"chunk_size": 2500, "chunk_overlap": 100}}
    )
    args = optimise_prompt.parse_args(
        ["evaluate", "--rule", "names", "--chunk-overlap", "0"]
    )

    config = optimise_prompt.rule_config(args, "names", prompt)

    assert (config.chunk_size, config.chunk_overlap, config.max_tokens) == (
        2500,
        0,
        3000,
    )
    assert optimise_prompt.rule_config(args, "names").chunk_size == 1500


class _FakeRunner:
    """Candidate redacts the labelled name; reference also redacts the councillor"""

    def __init__(self, prompt, config, budget=None):
        self.prompt = prompt
        self.total_cost = 0.0

    def predict_document(self, text):
        predictions = ["Joan Whitcombe"]
        if self.prompt.name == "reference":
            predictions = ["Cllr Raj Patel"]
        return {"redaction_strings": predictions, "chunk_count": 1, "failed_chunks": 0}


def _gate(tmp_path, candidate_name, *extra_args):
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(
        corpus,
        [
            Document(
                id=f"doc-{i}",
                doc_type="relevant_representation",
                text="My neighbour Joan Whitcombe objects. Cllr Raj Patel agreed.",
                entities=(
                    Entity("Joan Whitcombe", "names", "third_party", True),
                    Entity("Cllr Raj Patel", "names", "councillor", False),
                ),
                split="test",
            )
            for i in range(5)
        ],
    )
    prompts = {
        "candidate": _prompt(name=candidate_name, version=2),
        "baseline": _prompt(name="reference"),
    }
    args = optimise_prompt.parse_args(
        ["gate", "--rule", "names", "--corpus", str(corpus), "--repeats", "1"]
        + list(extra_args)
    )
    with (
        patch.object(optimise_prompt, "PromptRunner", _FakeRunner),
        patch.object(
            optimise_prompt, "load_rule_prompt", side_effect=lambda r, ref: prompts[ref]
        ),
        patch.object(mlflow.genai, "set_prompt_alias") as mock_set_alias,
    ):
        optimise_prompt.cmd_gate(args)
    return mock_set_alias


def test_gate__passes_and_promotes(tmp_path, capsys):
    mock_set_alias = _gate(tmp_path, "redaction-names", "--promote")

    assert "Gate passed" in capsys.readouterr().out
    mock_set_alias.assert_called_once_with("redaction-names", "champion", 2)
    run = mlflow.search_runs(experiment_names=["Default"]).iloc[0]
    assert run["metrics.passed"] == 1.0
    assert run["metrics.candidate_f2"] == 1.0


def test_gate__logs_the_corpus_as_a_dataset(tmp_path):
    _gate(tmp_path, "redaction-names")

    run_id = mlflow.search_runs(experiment_names=["Default"]).iloc[0]["run_id"]
    [dataset_input] = mlflow.get_run(run_id).inputs.dataset_inputs
    assert dataset_input.dataset.name == "corpus-test"
    assert dataset_input.dataset.digest
    assert [t.value for t in dataset_input.tags if t.key == "mlflow.data.context"] == [
        "gate"
    ]


def test_gate__fails_when_not_better(tmp_path, capsys):
    """A candidate identical to the reference is not significantly better"""
    mock_set_alias = _gate(tmp_path, "reference", "--promote")

    assert "Gate failed" in capsys.readouterr().out
    mock_set_alias.assert_not_called()


@pytest.mark.parametrize("command", ["evaluate", "optimise", "gate"])
def test_main__stops_before_running_if_azure_auth_fails(command):
    error = ClientAuthenticationError(message="no credentials")

    with (
        patch.object(optimise_prompt, "setup_mlflow"),
        patch.object(optimise_prompt, "ChainedTokenCredential") as mock_credential,
        patch.object(optimise_prompt, f"cmd_{command}") as mock_command,
    ):
        mock_credential.return_value.get_token.side_effect = error
        with pytest.raises(SystemExit, match="Azure authentication failed"):
            optimise_prompt.main([command, "--rule", "names"])

    mock_command.assert_not_called()


def test_main__skips_azure_check_for_commands_without_llm_calls():
    with (
        patch.object(optimise_prompt, "setup_mlflow"),
        patch.object(optimise_prompt, "cmd_export") as mock_export,
        patch.object(optimise_prompt, "ChainedTokenCredential") as mock_credential,
    ):
        optimise_prompt.main(["export", "--rule", "names", "--out", "x.yaml"])

    mock_credential.assert_not_called()
    mock_export.assert_called_once()


def _register_source():
    config = optimise_prompt.load_rule_configs("default", ["names"])["names"]
    return config, register_rule_prompt(config, "default", alias="baseline")


def test_reregister_optimised__adds_metadata_and_layout():
    """
    - Given the bare version optimize_prompts registers: messy layout, no tags and no
      structured output
    - When I re-register it
    - Then the new version has the rule's tags, structured output and model config,
      the standard layout and the alias, and the bare version is deleted
    """
    config, source = _register_source()
    bare = mlflow.genai.register_prompt(
        name=source.name,
        template="<Role>\nR.\n</Role> <Task>\nT.\n</Task>\n<Terms>\n- [T4] x\n</Terms>",
        model_config=source.model_config,
    )

    version, warning = optimise_prompt.reregister_optimised(
        bare, source, config, "default", "meta", "candidate"
    )

    assert warning is None
    loaded = mlflow.genai.load_prompt(f"prompts:/{source.name}@candidate")
    assert loaded.version == version.version
    assert loaded.template == (
        "<Role>\nR.\n</Role>\n\n<Task>\nT.\n</Task>\n\n<Terms>\n- [T1] x\n</Terms>"
    )
    assert loaded.tags["rule"] == "names"
    assert loaded.tags["optimizer"] == "meta"
    assert loaded.tags["source_prompt"] == source.uri
    assert loaded.response_format is not None
    assert loaded.model_config == source.model_config
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow.genai.load_prompt(bare.uri)


def test_reregister_optimised__keeps_unparseable_template():
    config, source = _register_source()
    bare = mlflow.genai.register_prompt(name=source.name, template="Just some text")

    version, warning = optimise_prompt.reregister_optimised(
        bare, source, config, "default", "meta", None
    )

    assert "standard layout" in warning
    loaded = mlflow.genai.load_prompt(version.uri)
    assert loaded.template == "Just some text"
    assert loaded.tags["rule"] == "names"


def _hand_edited_version(source):
    """A version made in the MLflow UI: a template, and no tags or chunk settings"""
    return mlflow.genai.register_prompt(
        name=source.name,
        template="<Role>\nR.\n</Role> <Task>\nT.\n</Task>\n<Terms>\n- x\n</Terms>",
        model_config={"provider": "openai", "model_name": "gpt-5.6-luna"},
    )


def _reregister_args(source, edited, *extra):
    return optimise_prompt.parse_args(
        ["reregister", "--rule", "names", "--prompt", str(edited.version), *extra]
    )


def test_cmd_reregister__copies_the_chunk_settings_of_another_version(capsys):
    """
    - Given a hand-edited version with no chunk settings, and a baseline at 4500/450
    - When I re-register it
    - Then the new version has the baseline's chunk settings, the rule's tags, the
      standard layout and the alias, and the hand-edited version is kept
    """
    config, _ = _register_source()
    baseline = register_rule_prompt(
        config.model_copy(update={"chunk_size": 4500, "chunk_overlap": 450}),
        "default",
        alias="baseline",
    )
    edited = _hand_edited_version(baseline)

    optimise_prompt.cmd_reregister(
        _reregister_args(baseline, edited, "--alias", "candidate-reformatted")
    )

    loaded = mlflow.genai.load_prompt("prompts:/redaction-names@candidate-reformatted")
    assert loaded.version not in (baseline.version, edited.version)
    assert loaded.model_config["extra_params"]["chunk_size"] == 4500
    assert loaded.model_config["extra_params"]["chunk_overlap"] == 450
    assert loaded.tags["rule"] == "names"
    assert loaded.tags["optimizer"] == "manual"
    assert loaded.tags["source_prompt"] == edited.uri
    assert loaded.template == (
        "<Role>\nR.\n</Role>\n\n<Task>\nT.\n</Task>\n\n<Terms>\n- [T1] x\n</Terms>"
    )
    assert mlflow.genai.load_prompt(edited.uri).version == edited.version
    assert "chunk settings of" in capsys.readouterr().out


def test_cmd_reregister__command_line_overrides_and_deletes_the_original():
    config, _ = _register_source()
    baseline = register_rule_prompt(config, "default", alias="baseline")
    edited = _hand_edited_version(baseline)

    optimise_prompt.cmd_reregister(
        _reregister_args(
            baseline,
            edited,
            "--chunk-size",
            "3000",
            "--chunk-overlap",
            "0",
            "--delete-original",
        )
    )

    new = mlflow.genai.load_prompt("prompts:/redaction-names@latest")
    assert new.model_config["extra_params"]["chunk_size"] == 3000
    assert new.model_config["extra_params"]["chunk_overlap"] == 0
    with pytest.raises(mlflow.exceptions.MlflowException):
        mlflow.genai.load_prompt(edited.uri)


def _export_args(tmp_path, *extra_args):
    return optimise_prompt.parse_args(
        ["export", "--out", str(tmp_path / "out.yaml"), *extra_args]
    )


def test_export__exports_every_champion(tmp_path, capsys):
    """
    - Given a champion for names and sensitive_info only
    - When I export without naming rules
    - Then both are written to one file and the other rules are skipped
    """
    for rule in ("names", "sensitive_info"):
        rule_config = optimise_prompt.load_rule_configs("default", [rule])[rule]
        register_rule_prompt(rule_config, "default", alias="champion")

    optimise_prompt.cmd_export(_export_args(tmp_path))

    out = capsys.readouterr().out
    assert "Skipping 'contact_identity_info'" in out
    assert "Wrote 2 rule(s)" in out
    assert (tmp_path / "out.yaml").exists()


def test_export__named_rule_without_champion_fails(tmp_path):
    with pytest.raises(SystemExit, match="No 'champion' prompt for rule 'names'"):
        optimise_prompt.cmd_export(_export_args(tmp_path, "--rule", "names"))

    assert not (tmp_path / "out.yaml").exists()


def test_export__no_champions_fails(tmp_path):
    with pytest.raises(SystemExit, match="No rules have a 'champion' prompt"):
        optimise_prompt.cmd_export(_export_args(tmp_path))
