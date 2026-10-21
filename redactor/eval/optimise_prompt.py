"""Register, evaluate, optimise, gate and export LLM redaction prompts with MLflow.

Each LLMTextRedaction rule's system prompt is registered as ``redaction-<rule>``. The YAML
config stays the source of truth: a prompt that passes the gate is exported to a new config
file for human review. Run from the `redactor` directory, inside the VNet:

    python -m eval.optimise_prompt register --chunk-size 1500 --chunk-overlap 60
    python -m eval.optimise_prompt evaluate --rule names --prompt baseline
    python -m eval.optimise_prompt optimise --rule names --optimizer meta
    python -m eval.optimise_prompt gate --rule names --candidate candidate --promote
    python -m eval.optimise_prompt export --rule names --out src/config/optimised.yaml

View results with ``mlflow ui --backend-store-uri sqlite:///eval/results/mlflow.db``.
Traces include the corpus text, so only point this at synthetic or approved data.
"""

import argparse
import os
from importlib import resources
from pathlib import Path

import litellm
import mlflow
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
)
from mlflow.entities.model_registry import PromptVersion
from mlflow.genai.optimize import GepaPromptOptimizer, MetaPromptOptimizer

from src.redaction.config import LLMTextRedactionConfig

from .corpus import RULE_NAMES, load_corpus
from .mlflow_dataset import (
    PromptRunner,
    chunk_records,
    document_records,
    seeded_sample,
    split_documents,
)
from .mlflow_scorers import SCORERS, make_objective
from .prompt_registry import (
    PromptStructureError,
    export_rule_to_yaml,
    normalise_template,
    parse_prompt_sections,
    prompt_uri,
    register_rule_prompt,
)
from .run_chunk_sweep import load_rule_configs
from .scoring import score_document
from .summarise_chunk_sweep import ScoredRun, bootstrap_f_beta_differences, summarise

EXPERIMENT_NAME = "redaction-prompt-engineering"
RESULTS_DIR = Path("eval/results")
DEFAULT_TRACKING_URI = f"sqlite:///{RESULTS_DIR.as_posix()}/mlflow.db"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"  # nosec: B105
LLM_COMMANDS = ("evaluate", "optimise", "gate")
META_GUIDELINES = (
    "This system prompt instructs an LLM to find text to redact from UK Planning "
    "Inspectorate (PINS) casework published online under UK GDPR. Keep the <Role>, "
    "<Task>, <Terms>, <MustRedact>, <DoNotRedact> and <Examples> XML sections, with one "
    "'- ' bullet per clause, and add no text outside them. Keep the [T1]-style clause "
    "labels, renumbering them in order within each section if clauses are added or "
    "removed. Do not add clauses that contradict a DoNotRedact clause or PINS guidance, "
    "and keep the prompt concise. Do not describe the response format, such as one term "
    "per line or what to omit: a response schema already enforces it."
)


def setup_mlflow(tracking_uri: str):
    """Point MLflow at the tracking store and the prompt-engineering experiment"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(RESULTS_DIR / "mlartifacts").resolve().as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Predict functions are traced explicitly, so skip the check that spends an LLM call
    os.environ["MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"] = "true"
    mlflow.openai.autolog()


def configure_reflection_model(model_uri: str):
    """Let LiteLLM reach an Azure OpenAI reflection model with Entra ID"""
    if not model_uri.startswith("azure:/"):
        return
    if endpoint := os.environ.get("OPENAI_ENDPOINT"):
        os.environ.setdefault("AZURE_API_BASE", endpoint)
    os.environ.setdefault("AZURE_API_VERSION", AZURE_API_VERSION)
    # Uses DefaultAzureCredential, refreshing the token during long optimisation runs
    litellm.enable_azure_ad_token_refresh = True


def rule_config(
    args: argparse.Namespace, rule: str, prompt: PromptVersion | None = None
) -> LLMTextRedactionConfig:
    """The rule's config, with chunk and completion settings taken from the command
    line, then the registered prompt's model config, then the YAML config"""
    config = load_rule_configs(args.config, [rule])[rule]
    model_config = (prompt.model_config or {}) if prompt else {}
    extra = model_config.get("extra_params") or {}
    update = {
        "chunk_size": args.chunk_size or extra.get("chunk_size"),
        "chunk_overlap": args.chunk_overlap
        if args.chunk_overlap is not None
        else extra.get("chunk_overlap"),
        "max_tokens": args.max_completion_tokens or model_config.get("max_tokens"),
        "reasoning_effort": args.reasoning_effort or extra.get("reasoning_effort"),
    }
    return LLMTextRedactionConfig(
        **(config.model_dump() | {k: v for k, v in update.items() if v is not None})
    )


def load_documents(args: argparse.Namespace):
    documents = load_corpus(args.corpus, split=args.split)
    return documents[: args.limit] if args.limit else documents


def load_rule_prompt(rule: str, ref: str) -> PromptVersion:
    return mlflow.genai.load_prompt(prompt_uri(rule, ref))


def log_run_params(
    prompt: PromptVersion, config: LLMTextRedactionConfig, args, **params
):
    mlflow.log_params(
        {
            "rule": config.name,
            "prompt_uri": prompt.uri,
            "model": config.model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "max_tokens": config.max_tokens,
            "reasoning_effort": config.reasoning_effort,
            "corpus": args.corpus,
            "split": args.split,
            **params,
        }
    )


def cmd_register(args: argparse.Namespace):
    for rule in args.rules:
        version = register_rule_prompt(
            rule_config(args, rule), args.config, args.message, args.alias
        )
        alias = f" as @{args.alias}" if args.alias else ""
        print(f"Registered {version.uri}{alias}")


def cmd_evaluate(args: argparse.Namespace):
    prompt = load_rule_prompt(args.rule, args.prompt)
    config = rule_config(args, args.rule, prompt)
    documents = load_documents(args)
    if args.level == "chunk":
        records = chunk_records(documents, args.rule, config, args.empty_chunk_ratio)
    else:
        records = document_records(documents, args.rule)
    runner = PromptRunner(prompt, config, args.budget)
    predict = runner.predict_chunk if args.level == "chunk" else runner.predict_document

    for repeat in range(args.repeats):
        with mlflow.start_run(
            run_name=f"evaluate-{args.rule}-{args.prompt}-{args.level}-r{repeat}"
        ):
            log_run_params(prompt, config, args, level=args.level, records=len(records))
            spent = runner.total_cost
            result = mlflow.genai.evaluate(
                data=records,
                predict_fn=mlflow.trace(predict, name=f"redact_{args.level}"),
                scorers=SCORERS,
            )
            mlflow.log_metric("cost_gbp", runner.total_cost - spent)
        print(f"Repeat {repeat}: {result.metrics}")


def reregister_optimised(
    optimised: PromptVersion,
    source: PromptVersion,
    config: LLMTextRedactionConfig,
    config_name: str,
    optimizer: str,
    alias: str | None,
) -> tuple[PromptVersion, str | None]:
    """
    Replace the version the optimiser registered with one that has the rule's tags,
    structured output and model config, and the template in the standard layout.

    Prompt versions are immutable, so this registers a new version and deletes the
    optimiser's. If the template can't be parsed into sections it is kept as it is, and
    a warning is returned, since such a version cannot be exported to YAML either.

    :return: The new version and a warning, if any.
    """
    template, warning = optimised.template, None
    try:
        template = normalise_template(template, config)
    except PromptStructureError as e:
        warning = f"{optimised.uri} could not be put in the standard layout: {e}"
    version = register_rule_prompt(
        config,
        config_name,
        commit_message=f"{optimizer} optimisation of {source.uri}",
        alias=alias,
        template=template,
        extra_tags={"optimizer": optimizer, "source_prompt": source.uri},
    )
    mlflow.MlflowClient().delete_prompt_version(optimised.name, optimised.version)
    return version, warning


def cmd_optimise(args: argparse.Namespace):
    configure_reflection_model(args.reflection_model)
    prompt = load_rule_prompt(args.rule, args.prompt)
    config = rule_config(args, args.rule, prompt)
    documents = load_documents(args)

    val_records = []
    if args.optimizer == "gepa" and args.val_fraction > 0:
        documents, val_documents = split_documents(documents, args.val_fraction)
        val_records = chunk_records(
            val_documents, args.rule, config, args.empty_chunk_ratio
        )
    train_records = chunk_records(documents, args.rule, config, args.empty_chunk_ratio)
    if args.max_records and len(train_records) > args.max_records:
        train_records = seeded_sample(train_records, args.max_records)

    if args.optimizer == "gepa":
        optimizer = GepaPromptOptimizer(
            reflection_model=args.reflection_model,
            max_metric_calls=args.max_metric_calls,
            display_progress_bar=True,
            gepa_kwargs={"valset": val_records} if val_records else None,
        )
    else:
        optimizer = MetaPromptOptimizer(
            reflection_model=args.reflection_model, guidelines=args.guidelines
        )

    runner = PromptRunner(prompt, config, args.budget)
    with mlflow.start_run(run_name=f"optimise-{args.rule}-{args.optimizer}"):
        log_run_params(
            prompt,
            config,
            args,
            reflection_model=args.reflection_model,
            train_records=len(train_records),
            val_records=len(val_records),
            hard_negative_weight=args.hard_negative_weight,
            unlocatable_weight=args.unlocatable_weight,
        )
        result = mlflow.genai.optimize_prompts(
            predict_fn=mlflow.trace(runner.predict_chunk, name="redact_chunk"),
            train_data=train_records,
            prompt_uris=[prompt.uri],
            optimizer=optimizer,
            scorers=SCORERS,
            aggregation=make_objective(
                args.hard_negative_weight, args.unlocatable_weight
            ),
        )
        optimised = result.optimized_prompts[0]
        # optimize_prompts registers only the template, so replace it with a version
        # that has the rule's metadata and structured output, in the standard layout
        optimised, layout_warning = reregister_optimised(
            optimised, prompt, config, args.config, args.optimizer, args.alias
        )
        mlflow.log_param("optimised_prompt_uri", optimised.uri)
        mlflow.log_metric("cost_gbp", runner.total_cost)

    print(
        f"Registered {optimised.uri}"
        + (f" as @{args.alias}" if args.alias else "")
        + f": score {result.initial_eval_score} -> {result.final_eval_score},"
        f" £{runner.total_cost:.2f} on the task model"
    )
    if layout_warning:
        print(f"Warning: {layout_warning}")


def cmd_gate(args: argparse.Namespace):
    """Compare whole-document results of a candidate and a reference prompt, and pass
    if the candidate's F2 is significantly better without more hard-negative hits"""
    prompts = {
        "reference": load_rule_prompt(args.rule, args.reference),
        "candidate": load_rule_prompt(args.rule, args.candidate),
    }
    # Both prompts use the reference's chunking, so only the prompt differs
    config = rule_config(args, args.rule, prompts["reference"])
    runners = {name: PromptRunner(p, config) for name, p in prompts.items()}
    documents = load_documents(args)

    with mlflow.start_run(
        run_name=f"gate-{args.rule}-{args.candidate}-vs-{args.reference}"
    ):
        log_run_params(
            prompts["candidate"],
            config,
            args,
            reference_prompt_uri=prompts["reference"].uri,
            repeats=args.repeats,
            documents=len(documents),
        )
        scored = []
        for repeat in range(args.repeats):
            for document in documents:
                # Interleaved, so throttling or drift affects both prompts equally
                for name, runner in runners.items():
                    spent = sum(r.total_cost for r in runners.values())
                    if args.budget and spent >= args.budget:
                        raise SystemExit(f"Budget of £{args.budget:.2f} spent")
                    outputs = runner.predict_document(document.text)
                    scored.append(
                        ScoredRun(
                            variant=name,
                            rule=args.rule,
                            repeat=repeat,
                            doc_id=document.id,
                            counts=score_document(
                                document, outputs["redaction_strings"], rule=args.rule
                            ),
                            cost_gbp=0.0,
                            doc_tokens=0,
                            chunk_count=outputs["chunk_count"],
                            failed_chunks=outputs["failed_chunks"],
                            latency_s=0.0,
                        )
                    )

        summaries = summarise(scored)
        low, high = bootstrap_f_beta_differences(scored, args.rule, "reference")[
            "candidate"
        ]
        reference = summaries[("reference", args.rule)].counts
        candidate = summaries[("candidate", args.rule)].counts
        hard_negatives_ok = (
            candidate.hard_negative_fp_rate
            <= reference.hard_negative_fp_rate + args.hard_negative_tolerance
        )
        passed = low > 0 and hard_negatives_ok

        for name, counts in (("reference", reference), ("candidate", candidate)):
            mlflow.log_metrics(
                {
                    f"{name}_f2": counts.f_beta(2.0),
                    f"{name}_recall": counts.recall,
                    f"{name}_precision": counts.precision,
                    f"{name}_hard_negative_fp_rate": counts.hard_negative_fp_rate,
                    f"{name}_cost_gbp": runners[name].total_cost,
                }
            )
        mlflow.log_metrics(
            {"f2_diff_ci_low": low, "f2_diff_ci_high": high, "passed": float(passed)}
        )

    print(
        f"F2 {reference.f_beta(2.0):.3f} -> {candidate.f_beta(2.0):.3f} "
        f"(95% CI of difference {low:+.3f} to {high:+.3f}); hard-negative FP rate "
        f"{reference.hard_negative_fp_rate:.3f} -> {candidate.hard_negative_fp_rate:.3f}"
    )
    print("Gate passed" if passed else "Gate failed")
    if passed and args.promote:
        candidate_prompt = prompts["candidate"]
        mlflow.genai.set_prompt_alias(
            candidate_prompt.name, "champion", candidate_prompt.version
        )
        print(f"Set {candidate_prompt.uri} as @champion")


def cmd_export(args: argparse.Namespace):
    prompt = load_rule_prompt(args.rule, args.prompt)
    try:
        fields = parse_prompt_sections(prompt.template)
    except PromptStructureError as e:
        raise SystemExit(f"{prompt.uri} cannot be exported: {e}") from e
    base = (
        Path(args.base)
        if args.base
        else Path(str(resources.files("src.config") / f"{args.config}.yaml"))
    )
    export_rule_to_yaml(fields, args.rule, base, Path(args.out))
    print(f"Wrote {prompt.uri} as rule '{args.rule}' to {args.out}")


def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse the command line arguments for each subcommand"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    common.add_argument("--config", default="default", help="Redaction config name")

    settings = argparse.ArgumentParser(add_help=False)
    settings.add_argument("--chunk-size", type=int, help="Override the chunk size")
    settings.add_argument("--chunk-overlap", type=int, help="Override the overlap")
    settings.add_argument("--max-completion-tokens", type=int)
    settings.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high"]
    )

    run = argparse.ArgumentParser(add_help=False, parents=[common, settings])
    run.add_argument("--rule", required=True, choices=RULE_NAMES)
    run.add_argument("--corpus", default="eval/data/prompt_corpus.jsonl")
    run.add_argument("--limit", type=int, help="Only use the first N documents")
    run.add_argument("--budget", type=float, help="Stop after this cost in GBP")

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser(
        "register",
        parents=[common, settings],
        help="Register each rule's prompt from the YAML config",
    )
    register.add_argument("--rules", nargs="+", default=list(RULE_NAMES))
    register.add_argument("--alias", default="baseline")
    register.add_argument("--message", help="Commit message for the new versions")
    register.set_defaults(func=cmd_register)

    evaluate = subparsers.add_parser(
        "evaluate", parents=[run], help="Score a prompt version with mlflow.genai"
    )
    evaluate.add_argument("--prompt", default="baseline", help="Alias or version")
    evaluate.add_argument("--split", choices=["dev", "test"], default="dev")
    evaluate.add_argument("--level", choices=["chunk", "document"], default="chunk")
    evaluate.add_argument("--empty-chunk-ratio", type=float, default=0.25)
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.set_defaults(func=cmd_evaluate)

    optimise = subparsers.add_parser(
        "optimise", parents=[run], help="Optimise a prompt version on dev chunks"
    )
    optimise.add_argument("--prompt", default="baseline", help="Alias or version")
    optimise.add_argument("--split", choices=["dev", "test"], default="dev")
    optimise.add_argument("--optimizer", choices=["meta", "gepa"], default="meta")
    optimise.add_argument(
        "--reflection-model",
        default="azure:/gpt-5.6-luna",
        help="MLflow model URI of the model that rewrites the prompt",
    )
    optimise.add_argument("--guidelines", default=META_GUIDELINES)
    optimise.add_argument(
        "--max-metric-calls",
        type=int,
        default=100,
        help="GEPA budget: each call is one chunk sent to the task model",
    )
    optimise.add_argument(
        "--val-fraction",
        type=float,
        default=0.3,
        help="Share of documents GEPA holds out to select candidates",
    )
    optimise.add_argument("--max-records", type=int, help="Sample N training chunks")
    optimise.add_argument("--empty-chunk-ratio", type=float, default=0.25)
    optimise.add_argument("--hard-negative-weight", type=float, default=0.5)
    optimise.add_argument("--unlocatable-weight", type=float, default=0.25)
    optimise.add_argument("--alias", default="candidate")
    optimise.set_defaults(func=cmd_optimise)

    gate = subparsers.add_parser(
        "gate", parents=[run], help="Compare a candidate with a reference prompt"
    )
    gate.add_argument("--split", choices=["dev", "test"], default="test")
    gate.add_argument("--candidate", default="candidate", help="Alias or version")
    gate.add_argument("--reference", default="baseline", help="Alias or version")
    gate.add_argument("--repeats", type=int, default=2)
    gate.add_argument("--hard-negative-tolerance", type=float, default=0.02)
    gate.add_argument(
        "--promote", action="store_true", help="Set @champion if the gate passes"
    )
    gate.set_defaults(func=cmd_gate)

    export = subparsers.add_parser(
        "export",
        parents=[common],
        help="Write a prompt version into a copy of a YAML config",
    )
    export.add_argument("--rule", required=True, choices=RULE_NAMES)
    export.add_argument("--prompt", default="champion", help="Alias or version")
    export.add_argument(
        "--base", help="YAML config to copy. Defaults to the --config file"
    )
    export.add_argument("--out", required=True, help="Path of the YAML file to write")
    export.set_defaults(func=cmd_export)

    return parser.parse_args(argv)


def check_azure_auth():
    """Stop now if no Azure token can be obtained, as MLflow would otherwise catch the
    error for every record and finish the run with all-zero scores"""
    # Same credential chain as LLMTextAnalyser
    credential = ChainedTokenCredential(
        ManagedIdentityCredential(), AzureCliCredential()
    )
    try:
        credential.get_token(AZURE_TOKEN_SCOPE)
    except ClientAuthenticationError as e:
        raise SystemExit(
            f"Azure authentication failed, so no LLM calls can be made:\n{e}"
        )


def main(argv=None):
    args = parse_args(argv)
    setup_mlflow(args.tracking_uri)
    if args.command in LLM_COMMANDS:
        check_azure_auth()
    args.func(args)


if __name__ == "__main__":
    main()
