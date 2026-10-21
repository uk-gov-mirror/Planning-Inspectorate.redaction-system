"""Generate a labelled synthetic NSIP casework corpus with an LLM.

Documents are assembled from sections. Each section is generated with its labels, using
structured outputs. A second "audit" call then looks for personal data the first call
failed to label. Long documents mix sections that contain personal data with
technical sections that contain none. They share a cast of fictional people, so the
same entities recur far apart and fall on different sides of chunk boundaries.

Labels follow the guidelines encoded in the redaction config. Use a different model
(or model family) from the one under test, to limit self-agreement bias, and
human-review a sample of documents before relying on the corpus.

Example (run from the `redactor` directory, with Azure credentials available):

    python -m eval.generate_corpus --model gpt-5.6-sol \
        --documents 120 --out eval/data/corpus.jsonl

A separate corpus for prompt engineering, with mostly medium-length documents and its
own ids so it can never collide with the chunk-sweep corpus:

    python -m eval.generate_corpus --model gpt-5.6-sol --profile prompt \
        --documents 100 --seed 1 --out eval/data/prompt_corpus.jsonl
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from tiktoken import get_encoding

from src.redaction.config import LLMTextRedactionConfig, xml_format
from src.redaction.config_processor import ConfigProcessor

from .corpus import RULE_NAMES, Document, Entity, validate_document

load_dotenv(verbose=True)

SCENARIOS = [
    {
        "doc_type": "relevant_representation",
        "brief": "A member of the public objects to the scheme. They mention "
        "neighbours, family members and their own health.",
    },
    {
        "doc_type": "relevant_representation",
        "brief": "A resident supports the scheme and agrees with other named interested "
        "parties' published representations. The resident has lived in the village a long "
        "time and describes how noise affects their sleep and wellbeing (feelings, not "
        "diagnoses).",
    },
    {
        "doc_type": "agent_submission",
        "brief": "A solicitor or land agent at a named firm writes on behalf of a "
        "landowner affected by compulsory acquisition. Include a letterhead, direct "
        "dial, work email and a signature block with pronouns.",
    },
    {
        "doc_type": "email_correspondence",
        "brief": "An email chain between an interested party, the Planning Inspectorate "
        "case team (a generic project mailbox and named case officers) and a statutory "
        "body employee, with signature blocks and phone numbers.",
    },
    {
        "doc_type": "farm_correspondence",
        "brief": "A farmer writes about the impact on their working farm. The farm "
        "is also the family home. Mention family members living there, the farm "
        "address, and a mobile number.",
    },
    {
        "doc_type": "sole_trader_letter",
        "brief": "A self-employed tradesperson (a sole trader) running a business from home "
        "objects to construction traffic.",
    },
    {
        "doc_type": "parish_council_minutes",
        "brief": "Parish council minutes. They name elected councillors, the clerk "
        "(with a home address), and members of the public attending. They include an "
        "unrelated item about a named resident's planning application.",
    },
    {
        "doc_type": "local_impact_report",
        "brief": "A local authority report citing named Councillors and an MP. It "
        "contains council office addresses, generic council mailboxes and gov.uk links.",
    },
    {
        "doc_type": "expert_evidence",
        "brief": "Written evidence from a named expert witness with a biography and "
        "qualifications. Their personal mobile number and home address are included "
        "by mistake.",
    },
    {
        "doc_type": "ecology_chapter",
        "brief": "An Environmental Statement ecology chapter. It gives precise and vague "
        "locations of badger setts, bat roosts (including a maternity roost), a "
        "barn owl nest, and general survey and mitigation content.",
    },
    {
        "doc_type": "ecology_chapter",
        "brief": "An ES appendix on otters and Schedule 1 birds, including grid "
        "references at different precisions, field signs, and conservation plans.",
    },
    {
        "doc_type": "health_representation",
        "brief": "A representation describing a person's diagnosed conditions and "
        "medication, a child's disability, and general concerns about air quality "
        "affecting wellbeing.",
    },
    {
        "doc_type": "solar_farm_objection",
        "brief": "An objection to a solar farm. It accuses the named applicant of using "
        "forced labour in its supply chain, and makes general statements about "
        "Xinjiang and the war in Ukraine.",
    },
    {
        "doc_type": "financial_submission",
        "brief": "A funding statement with company finances. It also includes an extract "
        "naming an employee's salary, shares and bonus, and a resident's mortgage and "
        "house valuation.",
    },
    {
        "doc_type": "hearing_transcript",
        "brief": "Part of a hearing transcript. The Examining Authority, the applicant's "
        "counsel, interested parties and an affected person speak. One speaker "
        "mentions their age and a past criminal conviction of a neighbour.",
    },
    {
        "doc_type": "child_representation",
        "brief": "A representation written by a 15-year-old pupil. It names their "
        "school and friends.",
    },
]

FILLER_BRIEF = (
    "Technical planning content with no personal data at all, e.g. traffic "
    "modelling, landscape and visual impact, drainage, construction phasing, or "
    "policy compliance. It may name the applicant company, public bodies, places, "
    "roads and policy documents."
)

LENGTH_BUCKETS = [  # (target tokens, weight)
    (400, 3),
    (1500, 3),
    (4000, 2),
    (10000, 2),
    (25000, 1),
]
PROMPT_LENGTH_BUCKETS = [  # Mostly 2-4 chunks at the default chunk size
    (1200, 2),
    (2500, 3),
    (4000, 4),
    (6000, 3),
    (10000, 1),
]
PROFILES = {
    "sweep": {"length_buckets": LENGTH_BUCKETS, "id_prefix": "syn"},
    "prompt": {"length_buckets": PROMPT_LENGTH_BUCKETS, "id_prefix": "pe"},
}
SECTION_TOKENS = 1200
MIN_SECTION_TOKENS = 200
FILLER_FRACTION = 0.4


class GeneratedEntity(BaseModel):
    text: str = Field(description="The exact span, copied verbatim from the text")
    rule: str = Field(description=f"One of: {', '.join(RULE_NAMES)}")
    category: str = Field(description="Short snake_case category, e.g. third_party")
    redact: bool = Field(description="Whether the guidelines require redaction")
    rationale: str = Field(description="One sentence justifying the decision")


class GeneratedSection(BaseModel):
    text: str
    entities: list[GeneratedEntity]
    cast: list[str] = Field(
        description="Fictional people and organisations introduced in this section, "
        "with their role, so later sections can refer to them consistently"
    )


class AuditResult(BaseModel):
    missing_entities: list[GeneratedEntity]


def guideline_summary(config_name: str = "default") -> str:
    """Render each rule's terms and exceptions as labelling guidance"""
    configs = [
        c
        for c in ConfigProcessor.validate_and_parse_redaction_config(
            ConfigProcessor.load_config(config_name)
        )
        if type(c) is LLMTextRedactionConfig and c.name in RULE_NAMES
    ]
    sections = []
    for config in configs:
        parts = [xml_format(config.redaction_terms, "Terms", as_list=True)]
        for name, values in (
            ("MustRedact", config.must_redact),
            ("DoNotRedact", config.do_not_redact),
            ("Examples", config.examples),
        ):
            if values:
                parts.append(xml_format(values, name, as_list=True))
        sections.append(
            f'<Rule name="{config.name}">\n' + "\n".join(parts) + "\n</Rule>"
        )
    return "\n\n".join(sections)


GENERATION_INSTRUCTIONS = """You create realistic but entirely fictional documents \
submitted to the Planning Inspectorate for UK Nationally Significant Infrastructure \
Projects (NSIPs), for testing an automated redaction system. Every person, address, \
phone number, email address and organisation must be invented. Do not use real \
private individuals.

Write in the style of the document type, including the mistakes, informal language and \
formatting of real submissions where appropriate.

Label EVERY span that the redaction guidelines below require to be redacted \
(redact=true). Also label plausible hard negatives: spans that look sensitive but must \
NOT be redacted under the guidelines (redact=false), such as the author's own name, \
Councillors and MPs, generic mailboxes, gov.uk links, feelings rather than diagnoses, \
vague durations and vague species locations. Each label's text must be copied exactly \
from your text. Label each distinct string once. Never give the same string \
different redact values.

<Guidelines>
{guidelines}
</Guidelines>"""

AUDIT_INSTRUCTIONS = """You are checking labels for a redaction test dataset. Given \
a text and its existing labels, list any spans the guidelines require to be redacted \
that are NOT already labelled. Copy each span exactly from the text. Return an empty \
list if the labels are complete.

<Guidelines>
{guidelines}
</Guidelines>"""


class CorpusGenerator:
    def __init__(self, model: str, guidelines: str, audit: bool = True):
        """
        :param model: Name of the Azure OpenAI deployment used to generate text.
        :param guidelines: Labelling guidance, as built by `guideline_summary`.
        :param audit: If True, a second LLM pass looks for entities the generator missed.
        """
        token_provider = get_bearer_token_provider(
            ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential()),
            "https://cognitiveservices.azure.com/.default",
        )
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["OPENAI_ENDPOINT"],
            api_version="2024-12-01-preview",
            azure_ad_token_provider=token_provider,
        )
        self.model = model
        self.guidelines = guidelines
        self.audit = audit

    def _parse(self, instructions: str, prompt: str, response_format):
        """Call the model with a structured-output schema and return the parsed object"""
        response = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
            max_completion_tokens=16000,
        )
        return response.choices[0].message.parsed

    def generate_section(
        self,
        doc_type: str,
        brief: str,
        cast: list[str],
        previous: str,
        section_tokens: int = SECTION_TOKENS,
    ) -> GeneratedSection:
        """
        Generate one section of a document along with its entity labels.

        :param brief: Description of what the section should contain.
        :param cast: People introduced so far, so they can be reused for continuity.
        :param previous: Text generated so far; its tail is used to continue from.
        :param section_tokens: Approximate length of the section in tokens.
        """
        prompt = (
            f"Document type: {doc_type}\n"
            f"Section brief: {brief}\n"
            f"Target length: about {section_tokens * 3 // 4} words.\n"
            "Cast introduced so far (reuse these people and refer to them again where "
            f"natural): {json.dumps(cast) if cast else 'none yet'}\n"
            f"End of the previous section, to continue from: {previous[-800:] or 'n/a'}"
        )
        section = self._parse(
            GENERATION_INSTRUCTIONS.format(guidelines=self.guidelines),
            prompt,
            GeneratedSection,
        )
        if self.audit:
            labels = json.dumps([e.model_dump() for e in section.entities])
            audit = self._parse(
                AUDIT_INSTRUCTIONS.format(guidelines=self.guidelines),
                f"<Text>\n{section.text}\n</Text>\n<Labels>\n{labels}\n</Labels>",
                AuditResult,
            )
            section.entities.extend(audit.missing_entities)
        return section

    def generate_document(
        self, doc_id: str, scenario: dict, target_tokens: int, split: str
    ) -> tuple[Document | None, list[str]]:
        """
        Build a labelled document by generating sections until the target length.

        Labels that use an unknown rule or are not verbatim in the text are dropped
        with a warning. The document is rejected (None) if labels conflict or fail
        validation.

        :param scenario: Dict with the `doc_type` and section `brief`.
        :param target_tokens: Approximate document length in tokens.
        :return: The document (or None) and any warnings or problems found.
        """
        encoding = get_encoding("o200k_base")
        doc_seed = int(hashlib.sha256(doc_id.encode()).hexdigest(), 16)
        rng = np.random.default_rng(doc_seed)
        sections: list[str] = []
        entities: dict[str, Entity] = {}
        cast: list[str] = []
        warnings = []
        tokens = 0
        while tokens < target_tokens:
            use_filler = sections and rng.random() < FILLER_FRACTION
            brief = FILLER_BRIEF if use_filler else scenario["brief"]
            # Size the section to what is left, so documents land near their target
            section_tokens = max(
                MIN_SECTION_TOKENS, min(SECTION_TOKENS, target_tokens - tokens)
            )
            section = self.generate_section(
                scenario["doc_type"],
                brief,
                cast,
                "\n\n".join(sections),
                section_tokens,
            )
            sections.append(section.text)
            cast.extend(c for c in section.cast if c not in cast)
            tokens += len(encoding.encode(section.text))
            for generated in section.entities:
                if generated.rule not in RULE_NAMES:
                    warnings.append(f"dropped unknown rule: {generated.rule}")
                    continue
                if generated.text not in section.text:
                    warnings.append(f"dropped non-verbatim label: {generated.text!r}")
                    continue
                existing = entities.get(generated.text)
                if existing and existing.redact != generated.redact:
                    return None, [
                        *warnings,
                        f"conflicting labels for {generated.text!r}",
                    ]
                entities.setdefault(
                    generated.text,
                    Entity(
                        text=generated.text,
                        rule=generated.rule,
                        category=generated.category,
                        redact=generated.redact,
                        rationale=generated.rationale,
                    ),
                )
        document = Document(
            id=doc_id,
            doc_type=scenario["doc_type"],
            text="\n\n".join(sections),
            entities=tuple(entities.values()),
            split=split,
            source=f"synthetic:{self.model}",
        )
        problems = validate_document(document)
        if problems:
            return None, [*warnings, *problems]
        return document, warnings


def assign_split(doc_id: str, test_fraction: float) -> str:
    """Deterministically assign a document to "test" or "dev" by hashing its id"""
    bucket = int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 1000
    return "test" if bucket < test_fraction * 1000 else "dev"


def parse_length_buckets(values: list[str]) -> list[tuple[int, int]]:
    """Parse TOKENS:WEIGHT pairs, e.g. ["2500:3", "6000:1"]

    :raises ValueError: If a pair is malformed or its tokens or weight are not positive.
    """
    buckets = []
    for value in values:
        try:
            tokens, weight = (int(part) for part in value.split(":"))
        except ValueError:
            raise ValueError(f"Expected TOKENS:WEIGHT, got {value!r}") from None
        if tokens < 1 or weight < 1:
            raise ValueError(f"Tokens and weight must be positive, got {value!r}")
        buckets.append((tokens, weight))
    return buckets


def main(argv=None):
    """Command line entry point: generate documents and append them to --out.

    Documents whose ids are already in the output file are skipped, so an
    interrupted run can be resumed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="Generator model deployment")
    parser.add_argument("--documents", type=int, default=120)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="sweep",
        help="Preset length mix and id prefix: 'sweep' for the chunk-size corpus, "
        "'prompt' for mostly medium-length documents for prompt engineering",
    )
    parser.add_argument(
        "--length-buckets",
        nargs="+",
        metavar="TOKENS:WEIGHT",
        help="Override the profile's length mix, e.g. 2500:3 4000:4 6000:3",
    )
    parser.add_argument(
        "--id-prefix",
        help="Prefix of document ids. Defaults to the profile's. Use a different "
        "prefix for each corpus, since ids also decide the dev/test split",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of documents to generate in parallel",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            existing_ids = {json.loads(line)["id"] for line in f if line.strip()}

    generator = CorpusGenerator(
        args.model, guideline_summary(args.config), audit=not args.no_audit
    )
    rng = np.random.default_rng(args.seed)
    profile = PROFILES[args.profile]
    buckets = (
        parse_length_buckets(args.length_buckets)
        if args.length_buckets
        else profile["length_buckets"]
    )
    id_prefix = args.id_prefix or profile["id_prefix"]
    sizes, weights = zip(*buckets, strict=True)
    probabilities = np.array(weights) / sum(weights)

    tasks = []
    for index in range(args.documents):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        target_tokens = int(rng.choice(sizes, p=probabilities))
        doc_id = f"{id_prefix}-{index:04d}-{scenario['doc_type']}"
        if doc_id in existing_ids:
            continue
        split = assign_split(doc_id, args.test_fraction)
        tasks.append((doc_id, scenario, target_tokens, split))

    write_lock = Lock()
    with out_path.open("a", encoding="utf-8") as out:

        def build_document(doc_id, scenario, target_tokens, split):
            for attempt in range(1, args.max_attempts + 1):
                document, warnings = generator.generate_document(
                    doc_id, scenario, target_tokens, split
                )
                for warning in warnings:
                    print(f"  {doc_id}: {warning}")
                if document:
                    # Serialise writes since multiple threads share the output file
                    with write_lock:
                        out.write(
                            json.dumps(document.to_dict(), ensure_ascii=False) + "\n"
                        )
                        out.flush()
                    print(
                        f"{doc_id}: {split}, ~{target_tokens} tokens, "
                        f"{len(document.positives())} positives, "
                        f"{len(document.negatives())} hard negatives"
                    )
                    return
                print(f"{doc_id}: attempt {attempt} failed, retrying")

        max_workers = min(args.max_workers, len(tasks)) or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(build_document, *task) for task in tasks]
            for future in as_completed(futures):
                future.result()
                time.sleep(0.1)  # Small delay to prevent busy waiting


if __name__ == "__main__":
    main()
