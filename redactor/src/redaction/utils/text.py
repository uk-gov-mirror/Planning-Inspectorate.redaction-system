import re
import threading
from string import punctuation
from unicodedata import category

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException
from unidecode import unidecode

_LANGDETECT_LOCK = threading.Lock()

DetectorFactory.seed = 0


def is_english_text(
    text: str,
    threshold: float = 0.90,
    margin: float = 0.20,
    min_chars: int = 20,
    min_words: int = 3,
) -> bool:
    """
    Detect whether text is English.

    Returns True only if both conditions hold:
    - English probability >= threshold; and
    - English probability exceeds the next-highest language by at least margin.

    Notes:
    - Whitespace is normalised; empty/whitespace-only returns False.
    - The detector is seeded for deterministic results.
    - On detection errors (e.g., too short/ambiguous text), returns False.
    """
    normalised = " ".join(text.split())
    if not normalised:
        return False

    if len(normalised) < min_chars or len(normalised.split()) < min_words:
        return False

    try:
        with _LANGDETECT_LOCK:
            langs = detect_langs(normalised)

        en_prob = next((lp.prob for lp in langs if lp.lang == "en"), 0.0)
        max_other = max((lp.prob for lp in langs if lp.lang != "en"), default=0.0)
        return (en_prob >= threshold) and ((en_prob - max_other) >= margin)
    except LangDetectException:
        return False


def normalise_punctuation_unidecode(text: str) -> str:
    """
    Replace punctuation characters with their closest ASCII equivalent.

    :param str text: Input text
    :return str: Text with punctuation replaced by closest ASCII equivalent
    """
    return "".join(
        c if not category(c).startswith("P") else unidecode(c) or c for c in text
    )


def normalise_text(text: str) -> str:
    """
    Normalise text by normalising punctuation and converting to lowercase.

    :param str text: The text to normalise
    :return str: The normalised text
    """
    return normalise_punctuation_unidecode(text).lower().strip().strip(punctuation)


def get_normalised_words(text: str) -> list[str]:
    """
    Normalise the given text into a list of words for redaction matching

    :param str text: The text to normalise
    :return List[str]: The list of normalised words
    """
    text_normalised = normalise_text(text).split(" ")
    words_normalised = [
        word.strip().strip(  # Remove leading/trailing whitespace
            punctuation  # Remove punctuation around the word
        )
        for word in (text_normalised)
    ]
    return [word for word in words_normalised if word]  # Remove empty words


def email_name_part_span(token: str, word: str) -> tuple[int, int] | None:
    """
    Find where a word is in the name part (before the "@") of an email address, as a
    whole component, e.g. "ada.merrin" or "merrin" in "ada.merrin@post.invalid". The
    domain never matches, and neither does a part of a component (e.g. "errin"). The
    full address is an exact match, so is not matched here.

    :param str token: A normalised word from the document
    :param str word: A normalised word to redact
    :return tuple[int, int] | None: The start and end index in the token, from the
        first match to the last if the word is repeated, or None if there is no match
    """
    name_part, at_sign, _ = token.partition("@")
    if not at_sign or not word or "@" in word:
        return None
    matches = list(re.finditer(rf"(?<![^\W_]){re.escape(word)}(?![^\W_])", name_part))
    if not matches:
        return None
    return matches[0].start(), matches[-1].end()


def is_email_name_part(token: str, word: str) -> bool:
    """Return True if the word is in the name part of the token, an email address"""
    return email_name_part_span(token, word) is not None


def normalised_character_origins(text: str) -> list[int] | None:
    """
    Find which character of the text each character of `normalise_text(text)` came
    from, so a part of a normalised word can be found in the original text.

    :return list[int] | None: The index in the text of each normalised character, or
        None if the normalisation can't be reproduced character by character.
    """
    characters = []
    for index, char in enumerate(text):
        replacement = (
            unidecode(char) or char if category(char).startswith("P") else char
        )
        characters.extend((c, index) for c in replacement.lower())

    # Mirror normalise_text: strip whitespace, then strip punctuation
    start, end = 0, len(characters)
    while start < end and characters[start][0].isspace():
        start += 1
    while end > start and characters[end - 1][0].isspace():
        end -= 1
    while start < end and characters[start][0] in punctuation:
        start += 1
    while end > start and characters[end - 1][0] in punctuation:
        end -= 1

    kept = characters[start:end]
    if "".join(char for char, _ in kept) != normalise_text(text):
        return None
    return [index for _, index in kept]
