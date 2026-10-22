# Chunk-size evaluation

Tools for choosing the LLM text redaction chunk size (`chunk_size` / `chunk_overlap` on each
`LLMTextRedaction` rule, in **tokens**). The goal is to maximise true positives, keep false
positives and false negatives low, and keep token costs down. Everything here calls the LLM
directly on plain text. PDFs and the Function App are not involved, so the effect of chunking
is isolated. None of it runs in CI.

See [METHODOLOGY.md](METHODOLOGY.md) for the statistical design: metrics, variance, sample
sizes, the bootstrap and the prompt-optimisation gate.

The Azure OpenAI resource is private-endpoint only, so anything that calls the LLM must run
from inside the VNet (e.g. a pipeline agent or a VM on the network).

## Files

| File | Purpose |
| --- | --- |
| `corpus.py` | JSONL corpus schema (`Document`, `Entity`) and validation |
| `scoring.py` | Entity-level matching and TP/FP/FN, precision, recall, F-beta |
| `generate_corpus.py` | Generates a labelled synthetic corpus with an LLM |
| `corpus_to_pdf.py` | Writes a corpus to PDFs, highlighting each labelled span with its label as metadata |
| `convert_existing_samples.py` | Converts the labelled e2e PDFs into held-out documents |
| `run_chunk_sweep.py` | Runs every rule over the corpus at each chunk size, and records predictions and token usage |
| `summarise_chunk_sweep.py` | Scores sweep results, bootstraps confidence intervals and recommends a size |
| `data/seed_dev.jsonl` | 8 hand-labelled documents covering the main guideline edge cases (smoke tests, generator reference) |
| `data/existing_test.jsonl` | Output of `convert_existing_samples.py` (51 held-out documents) |

## Corpus format

One JSON document per line:

```json
{"id": "rep-001", "doc_type": "relevant_representation", "split": "dev", "source": "synthetic",
 "text": "...",
 "entities": [
   {"text": "Joan Brown", "rule": "names", "category": "third_party", "redact": true},
   {"text": "Cllr James Brown", "rule": "names", "category": "councillor", "redact": false}
 ]}
```

- `redact: false` entities are **hard negatives**: spans that look sensitive but must not be
  redacted under the guidelines. Examples are authors, Councillors/MPs, generic mailboxes,
  health "feelings", "40+ years" and vague species locations.
- Every entity must occur verbatim in `text`.
- A string can't be both redacted and kept in the same document, because the redactor redacts
  every occurrence of a string.
- `rule: "unspecified"` is for legacy labels without a rule. They count only towards the
  combined score.

## Process

Run all commands from the `redactor` directory.

1. **Build the corpus.** Use a generator model different from the one under test, then
   human-review a stratified sample (by `doc_type` and length) and fix or drop bad labels.

   ```
   python -m tests.eval.generate_corpus --model <generator-deployment> --documents 120 \
       --out tests/eval/data/corpus.jsonl
   python -m tests.eval.convert_existing_samples --out tests/eval/data/existing_test.jsonl
   ```

   About 30% of synthetic documents are assigned to the held-out `test` split. Documents range
   from about 400 to 25,000 tokens and mix personal-data sections with technical filler. People
   are reused across sections, so entities recur and fall on chunk boundaries at every size.

2. **Check the completion budget.** gpt-5.x counts reasoning tokens against
   `max_completion_tokens`. Run a few documents at the largest size and check `failed_chunks`
   and `reasoning_tokens`. If chunks fail, pass a higher `--max-completion-tokens` and keep it
   fixed for every later run. Also check that `cached_input_tokens` is non-zero: the system
   prompt is a static prefix of 1,024 tokens or more, so prompt caching should apply.

3. **Coarse sweep.** Use the dev split, 2 repeats and a wide range of sizes:

   ```
   python -m tests.eval.run_chunk_sweep --corpus tests/eval/data/corpus.jsonl --split dev \
       --chunk-sizes 500 1000 1500 2500 4000 6000 8000 12000 --repeats 2 \
       --out tests/eval/results/coarse.jsonl --budget 50
   python -m tests.eval.summarise_chunk_sweep --corpus tests/eval/data/corpus.jsonl \
       --results tests/eval/results/coarse.jsonl --out-dir tests/eval/results/coarse_summary
   ```

   Runs are resumable. Re-running the same command skips rows already in `--out`.

4. **Fine sweep.** Take about 3 sizes around the knee of the F2-vs-cost curve, use 5 repeats,
   and run the full dev split. Then, at the chosen size, compare overlap with
   `--overlap-ratios 0 0.05 0.1`.

5. **Select.** `recommendation.json` gives the recommended variant overall (`__all__`) and per
   rule:
   - **Primary metric:** F2 (recall-weighted, in line with "if in doubt, take it out").
   - **Candidates:** variants not significantly worse than the best F2. This uses a paired
     bootstrap over documents, so every variant is compared on the same resampled documents.
   - **Guardrails:** per-rule recall floors (`--recall-floor`), the hard-negative false
     positive ceiling (`--hard-negative-fp-ceiling`) and the failed-chunk rate.
   - **Choice:** the cheapest remaining variant by cost per 1,000 document tokens. Cost includes
     the cached-input discount (rates come from `LLMTextAnalyser.OPENAI_MODELS` — check current
     Azure pricing).

   If rules recommend clearly different sizes, set `chunk_size` per rule in
   `src/config/default.yaml` rather than one shared value.

6. **Validate.** Run the chosen configuration on `--split test` of the synthetic corpus and on
   `existing_test.jsonl`, and confirm it isn't worse than the current default. Then run the e2e
   tests (`test_anon_samples`, `test_apply_eval_dataset`) against a deployed build before
   updating `default.yaml`.

## Reading the summary

- `partial_positives`: entities only partly covered. These are counted as false negatives.
- `cross_rule_predictions`: correct redactions found by a different rule than the label's.
  They count as false positives only if no rule's label matches.
- `unlocatable_predictions`: strings that don't occur in the text and so can't be applied.
  An email address counts as one word, and a prediction that is only its name part
  (`ada.merrin`) is located, since the redactor redacts that part of the address.
- `boundary_recall`: recall on entities that were split across a chunk boundary. Low values
  suggest increasing the overlap.
- `consistency`: fraction of entities with the same outcome in every repeat.
- `within_doc_variance` / `between_doc_variance`: variance of per-document F-beta split into
  noise from repeated runs of the same document (within) vs. spread across documents
  (between). High within-doc variance means more repeats are needed to trust a comparison;
  high between-doc variance means the corpus itself is heterogeneous for that variant.
- `summary.csv` and `recommendation.json` are joined by plots in the same `--out-dir`:
  `{rule}_f_beta_vs_cost.png` (the F-beta/cost trade-off per overlap) and
  `{rule}_metrics.png` (precision, recall, F-beta, hard-negative FP rate, boundary recall,
  consistency, variance and cost against chunk size).

Keep the prompts frozen for the whole sweep. Changing a prompt invalidates the comparison.

## Prompt engineering with MLflow

`optimise_prompt.py` versions, scores and optimises each rule's system prompt with MLflow
(install the `eval` extra). It stores everything in `eval/results/mlflow.db`; view it with
`mlflow ui --backend-store-uri sqlite:///eval/results/mlflow.db`. Traces contain corpus text,
so only use synthetic or approved data. The YAML config stays the source of truth.

1. `register --chunk-size <chosen> --chunk-overlap <chosen>` registers every rule's prompt as
   `redaction-<rule>@baseline`, with clause IDs (`[T1]`, `[D2]`, ...) and the chunk settings.
2. `evaluate --rule names` scores a version on dev chunks with the scorers in
   `mlflow_scorers.py` (F2, recall, precision, hard-negative hit rate, unlocatable rate,
   clause citation validity). Edit the YAML and re-register to try manual changes.
3. `optimise --rule names --optimizer meta`, then `--optimizer gepa` if needed, registers
   `@candidate`. The reflection model is reached through LiteLLM with Entra ID.
4. `gate --rule names --promote` runs both prompts over whole test documents and sets
   `@champion` only if F2 is significantly better (paired bootstrap) without more
   hard-negative hits.
5. `export --out src/config/<name>.yaml` writes every rule's champion into one copy of the
   config for review against PINS guidance. Rules without a champion are skipped. Use
   `--rules names` to export only some rules, or `--prompt candidate` for another alias.
6. `compare --configs basic default optimised` runs each config in `src/config` over the test
   split (resumable, one row per document, repeat and rule, in `eval/results/compare/`) and
   saves `comparison.png` and `comparison.csv`. They show F2, recall, precision, hard-negative
   hit rate, unlocatable rate, cost per 1,000 document tokens and the objective for each rule
   and for all rules together. Every config runs at the same chunk settings (the config
   defaults, or `--chunk-size` and `--chunk-overlap`), and counts are pooled over documents
   and repeats, as in the gate. `--plot-only` re-scores and re-plots existing results with no
   LLM calls, e.g. after changing `--hard-negative-weight` or `--unlocatable-weight`.
7. `gates-summary` reads the logged gate runs and plots each rule's promoted champion against
   its baseline, with no LLM calls: the F2 difference with its 95% interval, and baseline vs
   champion for F2, recall, precision, hard-negative hit rate and gate cost. For each rule it
   uses the latest passing gate run of the version `@champion` points to. It saves
   `gates_summary.png` and `gates_summary.csv` in `eval/results/gates/`.
