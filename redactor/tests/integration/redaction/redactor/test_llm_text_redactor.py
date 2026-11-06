from src.analysis.text import get_chunk_id
from src.redaction.config import (
    LLMTextRedactionConfig,
)
from src.redaction.redactor import LLMTextRedactor
from src.types import LLMTextRedactionResult


class TestRedact:
    def test_returns_redaction_strings(self):
        """
        Test that the LLMTextRedactor can generate redaction strings from input redaction config

        - Given we have some config for identifying redaction strings in text, and some text to analyse
        - When we pass the config to the LLMTextRedactor
        - Then the LLMTextRedactor should respond with a LLMTextRedactionResult that contains some expected result strings
        """
        # To boost the overall quality of the result, and to make the test stricter, extra context has been added to the prompt here
        config = {
            "name": "config name",
            "redactor_type": "LLMTextRedaction",
            "model": "gpt-5.6-luna",
            "role": "You are an assistant that extracts requested terms from a given text.",
            "task": (
                "You will be sent text to analyse. The text is a quote from Star Wars. "
                "Please find all strings in the text that adhere to the following rules: "
            ),
            "redaction_terms": [
                "The names of characters",
                "Religions",
                "Genders, such as she, her, he, him, they, their",
            ],
        }
        source_text = (
            "Did you ever hear the tragedy of Darth Plagueis the wise? I thought not, it's not a story the Jedi would tell you. It's a Sith legend. "
            "Darth Plagueis was a dark lord of the Sith, so powerful and so wise, that he could use the Force to influence the midichlorians to "
            "create ... life. He had such a knowledge of the dark side, he could even keep the ones he cared about from dying."
            "The dark side of the force is a pathway to many abilities some consider to be unnatural. He became so powerful, the only thing he cared "
            "about was losing his power, which, eventually of course he did. Unfortunately, he taught his apprentice everything he knew, then his "
            "apprentice killed him in his sleep. It's ironic, he could save others from death, but not himself."
        )
        redactor_inst = LLMTextRedactor(
            LLMTextRedactionConfig(text=source_text, **config)
        )
        redaction_result = redactor_inst.redact()
        redaction_strings_cleaned = [
            x.lower() for x in redaction_result.redaction_strings
        ]
        expected_results = {
            "darth plagueis",
            "plagueis",
            "sith",
            "jedi",
            "he",
            "his",
            "himself",
        }
        matches = {
            expected_result: any(
                expected_result in redaction_string
                for redaction_string in redaction_strings_cleaned
            )
            for expected_result in expected_results
        }
        acceptance_threshold = 0.1
        match_percent = float(len(tuple(x for x in matches.values() if x))) / float(
            len(expected_results)
        )
        error_message = (
            f"Expected a match threshold of at least {acceptance_threshold}, but was {match_percent}."
            f"\nExpected results {expected_results}\nActual results: {redaction_strings_cleaned}"
        )
        assert match_percent >= acceptance_threshold, error_message

    def test_include_rationale_traces_spans_to_the_analysed_text(self):
        """
        - Given config with include_rationale enabled and a chunk size small enough
          to split the text into overlapping chunks
        - When the LLMTextRedactor redacts the text
        - Then every chunk summary covers exactly its chunk, and every span offset
          points at the decision's text inside its chunk's range
        """
        source_text = (
            "Joan Brown of 12 Mill Lane objects to the proposal. Her neighbour, "
            "Mary Jones, says the extension would block light to her garden.\n\n"
            "Cllr Raj Patel spoke in support. Joan Brown replied that the hedge "
            "on Mill Lane is protected and asked that Mary Jones be consulted."
        )
        config = LLMTextRedactionConfig(
            name="config name",
            redactor_type="LLMTextRedaction",
            model="gpt-5.6-luna",
            text=source_text,
            role="You redact personal data from planning representations.",
            task="Find the personal data of members of the public in the text.",
            redaction_terms=["Names of members of the public", "Home addresses"],
            do_not_redact=["Names of elected councillors"],
            include_rationale=True,
            max_tokens=3000,
            chunk_size=30,
            chunk_overlap=5,
        )
        chunks = config.create_text_splitter().split_text(source_text)

        result = LLMTextRedactor(config).redact()

        assert len(chunks) > 1
        assert result.clause_legend == config.get_clause_legend()
        for summary, chunk in zip(result.chunk_summaries, chunks, strict=True):
            assert summary["chunk_id"] == get_chunk_id(chunk)
            assert source_text[summary["char_start"] : summary["char_end"]] == chunk

        span_offsets = []
        for decision in result.redaction_decisions + result.excluded_decisions:
            for occurrence in decision["occurrences"]:
                summary = result.chunk_summaries[occurrence["chunk_index"]]
                for offset in occurrence["span_offsets"]:
                    end = offset + len(decision["text"])
                    assert source_text[offset:end] == decision["text"]
                    assert summary["char_start"] <= offset
                    assert end <= summary["char_end"]
                    span_offsets.append(offset)
        assert span_offsets, result.redaction_decisions

    def test_empty_text_returns_empty_result(self):
        """
        - Given we have config but empty text to analyse
        - When we pass the config to the LLMTextRedactor with empty text
        - Then the LLMTextRedactor should return an empty result without calling the LLM
        """
        config = LLMTextRedactionConfig(
            name="config name",
            redactor_type="LLMTextRedaction",
            model="gpt-5.6-luna",
            text="",
            role="some role",
            task="Identify redaction strings",
            redaction_terms=["Names"],
        )
        redactor_inst = LLMTextRedactor(config)
        result = redactor_inst.redact()

        assert isinstance(result, LLMTextRedactionResult)
        assert result.redaction_strings == ()
        assert result.run_metrics == {}

    def test_null_text_returns_empty_result(self):
        """
        - Given we have config but None as the text
        - When we pass the config to the LLMTextRedactor
        - Then the LLMTextRedactor should return an empty result without calling the LLM
        """
        config = LLMTextRedactionConfig(
            name="config name",
            redactor_type="LLMTextRedaction",
            model="gpt-5.6-luna",
            text=None,
            role="some role",
            task="Identify redaction strings",
            redaction_terms=["Names"],
        )
        redactor_inst = LLMTextRedactor(config)
        result = redactor_inst.redact()

        assert isinstance(result, LLMTextRedactionResult)
        assert result.redaction_strings == ()
        assert result.run_metrics == {}
