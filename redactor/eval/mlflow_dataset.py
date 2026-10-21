"""Evaluation records and predict functions for running registered redaction prompts.

Records follow MLflow's ``{"inputs": ..., "expectations": ...}`` format. Every record's
input is ``text``: a chunk for prompt optimisation, or a whole document for the gate.
"""

import hashlib
from dataclasses import asdict

from mlflow.entities.model_registry import PromptVersion

from src.analysis.text import LLMTextAnalyser, merge_redaction_decisions
from src.redaction.config import LLMTextRedactionConfig

from .corpus import Document, Entity
from .prompt_registry import CLAUSE_ID


def seeded_sample(items: list, k: int, seed: int = 0) -> list:
    """Reproducible random sample of k items (hash-ordered; not for security use)"""
    keyed = sorted(
        enumerate(items),
        key=lambda pair: hashlib.sha256(f"{seed}:{pair[0]}".encode()).digest(),
    )
    return [item for _, item in keyed[:k]]


def _expectations(document: Document, entities: list[Entity], rule: str) -> dict:
    return {
        "rule": rule,
        "doc_id": document.id,
        "entities": [asdict(entity) for entity in entities],
    }


def chunk_records(
    documents: list[Document],
    rule: str,
    config: LLMTextRedactionConfig,
    empty_chunk_ratio: float = 0.25,
    seed: int = 0,
) -> list[dict]:
    """
    One record per chunk, labelled with the entities wholly inside it.

    Entities of every rule are kept, so a correct prediction for another rule is not
    scored as a false positive. Chunks with no labels for the rule are sampled down to at
    most ``empty_chunk_ratio`` of the records, so they penalise false positives without
    dominating the score.
    """
    splitter = config.create_text_splitter()
    labelled, empty = [], []
    for document in documents:
        for index, chunk in enumerate(splitter.split_text(document.text)):
            entities = [e for e in document.entities if e.text in chunk]
            record = {
                "inputs": {"text": chunk},
                "expectations": _expectations(document, entities, rule)
                | {"chunk_index": index},
            }
            if any(e.rule == rule for e in entities):
                labelled.append(record)
            else:
                empty.append(record)

    if empty_chunk_ratio >= 1:
        return labelled + empty
    n_empty = min(
        len(empty), round(empty_chunk_ratio / (1 - empty_chunk_ratio) * len(labelled))
    )
    kept = sorted(seeded_sample(list(range(len(empty))), n_empty, seed))
    return labelled + [empty[i] for i in kept]


def document_records(documents: list[Document], rule: str) -> list[dict]:
    """One record per whole document, labelled with all of its entities"""
    return [
        {
            "inputs": {"text": document.text},
            "expectations": _expectations(document, list(document.entities), rule),
        }
        for document in documents
    ]


def split_documents(
    documents: list[Document], val_fraction: float, seed: int = 0
) -> tuple[list[Document], list[Document]]:
    """Split documents (not chunks) into train and validation sets, so no document
    contributes to both"""
    shuffled = seeded_sample(list(documents), len(documents), seed)
    n_val = round(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]


def _outputs(
    system_prompt: str,
    redactions: tuple[dict, ...] = (),
    excluded: tuple[dict, ...] = (),
    **extra,
) -> dict:
    return {
        "redaction_strings": [r["text"] for r in redactions],
        "redactions": list(redactions),
        "excluded": list(excluded),
        "prompt_clause_ids": sorted(set(CLAUSE_ID.findall(system_prompt))),
        **extra,
    }


class PromptRunner:
    """
    Runs a registered prompt through LLMTextAnalyser, keeping production's rate limiting,
    retries and cost tracking. Always asks for detailed (reasoned) output.
    """

    def __init__(
        self,
        prompt: PromptVersion,
        config: LLMTextRedactionConfig,
        budget: float | None = None,
    ):
        self.prompt = prompt
        self.config = config.model_copy(update={"include_rationale": True})
        self.budget = budget
        self.analyser = LLMTextAnalyser(self.config)

    @property
    def total_cost(self) -> float:
        return self.analyser.total_cost

    def _system_prompt(self) -> str:
        if self.budget is not None and self.total_cost >= self.budget:
            raise RuntimeError(
                f"Budget of £{self.budget:.2f} spent (£{self.total_cost:.2f})"
            )
        # Read on every call, since the optimiser swaps candidates in via this property
        return self.prompt.template

    def predict_chunk(self, text: str) -> dict:
        """Redact one chunk with a single LLM call"""
        system_prompt = self._system_prompt()
        result = self.analyser._analyse_text_chunk(system_prompt, text)
        if result is None:
            return _outputs(system_prompt, error="LLM call failed after all retries")
        response, _ = result
        redactions, excluded = merge_redaction_decisions(
            [response.choices[0].message.parsed]
        )
        return _outputs(system_prompt, redactions, excluded)

    def predict_document(self, text: str) -> dict:
        """Redact a whole document, chunked as in production"""
        system_prompt = self._system_prompt()
        chunks = self.config.create_text_splitter().split_text(text)
        result = self.analyser.analyse_text(system_prompt, chunks)
        return _outputs(
            system_prompt,
            result.redaction_decisions,
            result.excluded_decisions,
            failed_chunks=result.metadata.failed_chunk_count,
            chunk_count=len(chunks),
        )
