# Redaction engine

This package contains the core logic that inspects files and applies redactions. See the
repo root [readme.md](../../../readme.md) for how to get set up and run things locally.

## Redactors (`redactor.py`)

- `Redactor` is an abstract base class for a single redaction technique. It validates a
  `RedactionConfig` and performs the redaction described by it.
- Concrete implementations: `LLMTextRedactor` (LLM-based text classifier for redaction), `ImageRedactor`
  (face/signature detection in images), `ImageTextRedactor` (OCR'd text in images), and
  `ImageLLMTextRedactor` (LLM-based text classification applied to OCR'd image text).
- `RedactorFactory.REDACTOR_TYPES` is the registry of all `Redactor` subclasses. Each one
  declares a `redactor_type` name (e.g. `LLMTextRedaction`, `ImageRedaction`,
  `ImageTextRedaction`, `ImageLLMTextRedaction`) that matches the `redactor_type` used in the
  [prompt config](#the-prompt-config-configdefaultyaml) below, and a matching config class in
  `config.py` (e.g. `LLMTextRedactionConfig`) used to validate its rules.

Each redactor is paired with a `RedactionConfig` subclass (its input) and a
`RedactionResult` subclass (its output).

## File processors (`file_processor.py`)

- `FileProcessor` is an abstract base class defining the contract every supported file type
  must implement: `redact()` (add provisional redactions), `apply()` (turn provisional
  redactions into permanent ones), and `sanitise()` (strip hidden content/metadata).
- `PDFProcessor` is the only concrete implementation today, built on
  [PyMuPDF](https://pymupdf.readthedocs.io/). Provisional redactions are stored as PDF
  highlight annotations (see `get_proposed_redactions`), which a reviewer can inspect before
  they're converted into permanent redaction annotations, and the content contained within
  the annotations are removed from the file by `apply()`. `get_final_redactions` reads back 
  the applied annotations for verification and metrics.

### How PDFProcessor identifies text redaction boxes

Text redaction is the more involved case, since an LLM text redactor only returns the
*strings* it thinks should be redacted (`redaction_strings`) — not where they sit on the
page. `PDFProcessor._apply_provisional_text_redactions` and
[`PDFUtil`](utils/pdf.py) work together to turn those strings into highlight boxes:

1. `PDFUtil.extract_page_metadata` parses each page into lines, where every line stores its
   words as an array alongside each word's `x0`/`x1` coordinates and the line's `y0`/`y1`.
   This word-level index is what all matching below is based on — there's no per-term text
   search against the raw PDF.
2. For each page, `_examine_provisional_redactions_on_page` first cheaply checks whether a
   candidate string appears at all in the page's (and next page's) joined raw text, to avoid
   running the more expensive line-by-line matching for terms that clearly aren't present.
3. Surviving candidates go to `PDFUtil.examine_provisional_text_redaction`, which normalises
   the term into words and matches them against each line's word array
   (`_find_potential_matches_in_line` / `_match_word_to_redact_in_line`). Matching is
   deliberately fuzzy: it tolerates plurals/possessives (`"Bob's"` matching `"Bob"`), words
   fused together by a missing space (e.g. `"somethingMonica"`), and hyphenated terms.
   Multi-word terms are matched by walking consecutive words (`_check_subsequent_words`).
4. Once a match's start/end word indices on a line are known, `_construct_pdf_rect` builds
   the highlight's bounding box directly from those words' stored coordinates.
5. If a term is only partially matched at the end of a line, `_check_partial_redaction_across_line_breaks`
   checks whether the remainder continues onto the next line, or the first line of the next
   page, so terms that wrap across a line/page boundary still get a (possibly two-part)
   redaction.
6. Every valid match becomes a highlight annotation via `PDFUtil.add_provisional_redaction`,
   labelled with the original term and the redactor/rule that proposed it.

### How PDFProcessor applies image redaction boxes

Unlike text, `PDFProcessor` does not work out *where* to redact within an image itself —
that's delegated entirely to the `ImageRedactor` family (`ImageRedactor`, `ImageTextRedactor`,
`ImageLLMTextRedactor`), which return `ImageRedactionResult` objects containing
`redaction_boxes` in the image's own local pixel space (see
[analysis](../analysis)). `PDFProcessor` only has to:

1. Extract the images embedded in the PDF (`PDFUtil.extract_pdf_images` /
   `extract_unique_pdf_images`) and match each `ImageRedactionResult` back to the image it
   came from.
2. Convert each local-space redaction box into page coordinates with
   `PDFUtil.transform_bounding_box_to_global_space`, using the image's transform matrix
   within the PDF.
3. Highlight the transformed box the same way as text redactions, via
   `PDFUtil.add_provisional_redaction`.

## Config processing (`config_processor.py`)

- `ConfigProcessor.load_config(config_name)` loads `{config_name}.yaml` from the `src.config`
  package (defaults to `default`, i.e. [../config/default.yaml](../config/default.yaml)), via
  `importlib.resources` so it resolves the same way whether running from source or an installed
  wheel/site-packages copy.
- `ConfigProcessor.validate_and_parse_redaction_config` flattens and validates that yaml into
  `RedactionConfig` objects the redactors above understand, filtering out any rules that
  don't apply to the current `FileProcessor`.

## The prompt config (`config/default.yaml`)

This yaml file defines *what* gets redacted, without needing a code change. It's a list of
`redactors`, each with a `redactor_type` and a list of `redaction_rules`.

For `LLMTextRedaction` rules, each rule has:
- `name` / `label` — an identifier and display label for the rule, which titles the redaction
  highlights in the PDF to allow easy filtering
- `model` — the LLM to use
- `system_prompt` — the instruction given to the LLM (e.g. "extract all requested terms")
- `redaction_terms` — the categories of information to find (e.g. "People's names")
- `constraints` — things to explicitly exclude (e.g. "Do not include locations")
- `chunk_size` / `chunk_overlap` *(optional)* — how the text is split before being sent to the
  LLM, in **tokens** (default 1500 / 60). Image rules inherit these from the text rule they
  reference. See [tests/eval](../../tests/eval/README.md) for how to choose them.

`LLMTextRedactionConfig.create_system_prompt()` (in `config.py`) compiles these fields into
the actual system prompt sent to the LLM, wrapping `system_prompt`, `redaction_terms`,
`constraints`, and a fixed output format instruction in XML tags.

Image-based rule types (e.g. `ImageLLMTextRedaction`) reference an LLM text rule by name via
`text_redaction_rule`, reusing its `redaction_terms` against OCR'd text found in images.

To change what the system redacts, edit this yaml — no code changes required.

