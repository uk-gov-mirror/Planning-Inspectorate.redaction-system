import pytest

from src.redaction.utils.text import (
    email_name_part_span,
    get_normalised_words,
    is_email_name_part,
    is_english_text,
    normalise_punctuation_unidecode,
    normalise_text,
    normalised_character_origins,
)


@pytest.mark.parametrize(
    ("token", "word", "expected"),
    [
        ("ada.merrin@post.invalid", "ada.merrin", True),
        ("ada.merrin@post.invalid", "ada", True),
        ("ada.merrin@post.invalid", "merrin", True),
        ("ada_merrin+news@post.invalid", "merrin", True),
        ("j.smith@hmrc.gov.uk", "j.smith", True),
        # Part of a component, or the domain
        ("ada.merrin@post.invalid", "errin", False),
        ("adam@post.invalid", "ada", False),
        ("ada.merrin@post.invalid", "post", False),
        ("ada.merrin@post.invalid", "invalid", False),
        # The full address is an exact match, not an email name part
        ("ada.merrin@post.invalid", "ada.merrin@post.invalid", False),
        # Not an email address
        ("ada.merrin", "ada", False),
        ("ada.merrin@post.invalid", "", False),
    ],
)
def test__is_email_name_part(token, word, expected):
    assert is_email_name_part(token, word) is expected


def test__is_english_text__detects_english():
    # Clear English sentence should be detected as English
    text = "This is a plain English sentence with multiple words used for detection."
    assert is_english_text(text) is True


def test__is_english_text__detects_non_english():
    # Clear non-English (French) sentence should not be detected as English
    text = (
        "C'est une phrase en français avec plusieurs mots et des accents comme élève."
    )
    assert is_english_text(text) is False


def test__is_english_text__whitespace_and_empty_returns_false():
    assert is_english_text("") is False
    assert is_english_text("    \n\t   ") is False


def test__is_english_text__short_or_ambiguous_returns_false():
    # Very short strings or those likely to cause detection issues should return False
    assert is_english_text("a") is False
    assert is_english_text("😀") is False
    assert is_english_text("12345") is False


def test__normalise_punctuation_unidecode__replaces_punctuation():
    input_text = "“quotes” and ‘apostrophes’."
    expected_output = "\"quotes\" and 'apostrophes'."
    assert normalise_punctuation_unidecode(input_text) == expected_output


def test__get_normalised_words__normalises_text():
    input_text = "Hello, World! This is a test."
    expected_output = ["hello", "world", "this", "is", "a", "test"]
    assert get_normalised_words(input_text) == expected_output


def test__get_normalised_words__handles_multiple_spaces_and_punctuation():
    input_text = "Multiple   spaces... and punctuation!!!"
    expected_output = ["multiple", "spaces", "and", "punctuation"]
    assert get_normalised_words(input_text) == expected_output


@pytest.mark.parametrize(
    ("token", "word", "expected"),
    [
        ("ada.merrin@post.invalid", "ada.merrin", (0, 10)),
        ("ada.merrin@post.invalid", "merrin", (4, 10)),
        ("ada.merrin@post.invalid", "ada", (0, 3)),
        # From the first match to the last
        ("ada.ada@post.invalid", "ada", (0, 7)),
        ("ada.merrin@post.invalid", "post", None),
        ("ada.merrin", "ada", None),
    ],
)
def test__email_name_part_span(token, word, expected):
    assert email_name_part_span(token, word) == expected


@pytest.mark.parametrize(
    ("text", "expected_origins"),
    [
        ("Hello", [0, 1, 2, 3, 4]),
        ("  Hello!  ", [2, 3, 4, 5, 6]),
        # Punctuation inside the word is kept, and replaced by its ASCII equivalent
        ("Bob\u2019s", [0, 1, 2, 3, 4]),
        ("\u201cada@post.invalid\u201d,", list(range(1, 17))),
        ("", []),
    ],
)
def test__normalised_character_origins(text, expected_origins):
    """Each normalised character maps back to the character it came from"""
    assert normalised_character_origins(text) == expected_origins


def test__normalised_character_origins__cannot_reproduce_normalisation():
    """A final sigma is only lower-cased differently in a whole word, so the
    characters can't be mapped one by one"""
    assert normalise_text("\u039f\u0394\u039f\u03a3") != "".join(
        c.lower() for c in "\u039f\u0394\u039f\u03a3"
    )
    assert normalised_character_origins("\u039f\u0394\u039f\u03a3") is None
