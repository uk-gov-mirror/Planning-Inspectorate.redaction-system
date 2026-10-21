"""Entity-level scoring of redaction predictions against a labelled document.

Matching mirrors the production PDF matcher's leniency: comparison is on normalised
words (case-insensitive, punctuation and possessives ignored), and a prediction only
needs to cover part of an entity's words for that coverage to count. Like production,
predictions are treated as strings to redact wherever they occur. An email address is
one word, and a prediction that is only the name part of an address ("ada.merrin" in
"ada.merrin@post.invalid") stands for the whole address, since that is what the redactor
redacts.
"""

import re
from dataclasses import dataclass, fields

from src.redaction.utils.text import get_normalised_words, is_email_name_part

from .corpus import Document, Entity

_POSSESSIVE = re.compile(r"['\u2019]s\b", re.IGNORECASE)
_EMAIL = r"[\w.+\-]+@[^\W_]+(?:-[^\W_]+)*(?:\.[^\W_]+(?:-[^\W_]+)*)+"  # name@domain.tld
_WORD = re.compile(rf"{_EMAIL}|[^\W_]+(?:[@][^\W_]+)?", re.UNICODE)


def normalise_words(text: str) -> tuple[str, ...]:
    """Lower-case text, strip possessives and split it into a tuple of words. An email
    address is a single word"""
    return tuple(_WORD.findall(_POSSESSIVE.sub("", text).casefold()))


class DocumentWords:
    """The normalised words of a document, and of strings compared against it"""

    def __init__(self, text: str):
        """Normalise the document text and list its email addresses"""
        self.words = normalise_words(text)
        self.emails = tuple(dict.fromkeys(w for w in self.words if "@" in w))

    def of(self, text: str) -> tuple[str, ...]:
        """
        Normalise a string to compare with the document. A single term that is not a
        word of the document, but is the name part of an email address in it, stands
        for the whole address.
        """
        words = normalise_words(text)
        terms = get_normalised_words(text)
        if words and len(terms) == 1 and not find_subsequence(self.words, words):
            for email in self.emails:
                if is_email_name_part(email, terms[0].casefold()):
                    return (email,)
        return words


def find_subsequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> list[int]:
    """Return every start index at which needle occurs contiguously in haystack"""
    if not needle or len(needle) > len(haystack):
        return []
    n = len(needle)
    return [i for i in range(len(haystack) - n + 1) if haystack[i : i + n] == needle]


def overlaps(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """True if either word sequence contains the other"""
    return bool(find_subsequence(a, b) or find_subsequence(b, a))


def coverage(entity: tuple[str, ...], predictions: list[tuple[str, ...]]) -> float:
    """Fraction of an entity's words covered by the union of the predictions"""
    if not entity:
        return 0.0
    covered = [False] * len(entity)
    for prediction in predictions:
        if find_subsequence(prediction, entity):
            return 1.0
        for start in find_subsequence(entity, prediction):
            for i in range(start, start + len(prediction)):
                covered[i] = True
    return sum(covered) / len(entity)


@dataclass
class ScoreCounts:
    positives: int = 0
    true_positives: int = 0
    partial_positives: int = 0
    """Positives with some, but not enough, of their words covered"""
    false_negatives: int = 0
    """Positives below the coverage threshold (includes partial positives)"""
    predictions: int = 0
    correct_predictions: int = 0
    false_positives: int = 0
    cross_rule_predictions: int = 0
    """Predictions matching a positive belonging to a different rule"""
    unlocatable_predictions: int = 0
    """Predictions that do not occur in the document, so cannot be applied"""
    hard_negatives: int = 0
    hard_negative_hits: int = 0
    boundary_positives: int = 0
    """Positives that are not wholly contained in any single chunk"""
    boundary_true_positives: int = 0

    def __add__(self, other: "ScoreCounts") -> "ScoreCounts":
        """Sum every counter, so counts from several documents can be combined"""
        return ScoreCounts(
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            }
        )

    @property
    def precision(self) -> float:
        """Share of predictions that were correct (1.0 if there are none)"""
        denominator = self.correct_predictions + self.false_positives
        return self.correct_predictions / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        """Share of positive entities that were redacted (1.0 if there are none)"""
        return self.true_positives / self.positives if self.positives else 1.0

    @property
    def hard_negative_fp_rate(self) -> float:
        """Share of hard negatives (look sensitive, must be kept) wrongly redacted"""
        return (
            self.hard_negative_hits / self.hard_negatives
            if self.hard_negatives
            else 0.0
        )

    @property
    def boundary_recall(self) -> float | None:
        """Recall for entities split across chunks, or None if there were none"""
        if not self.boundary_positives:
            return None
        return self.boundary_true_positives / self.boundary_positives

    def f_beta(self, beta: float = 2.0) -> float:
        """F-beta score of these counts' precision and recall"""
        return f_beta(self.precision, self.recall, beta)


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    """F-beta score. beta > 1 weights recall more heavily than precision"""
    b2 = beta**2
    denominator = b2 * precision + recall
    return (1 + b2) * precision * recall / denominator if denominator else 0.0


def boundary_split_entities(
    entities: tuple[Entity, ...], chunks: list[str]
) -> set[str]:
    """Return the texts of entities that do not fit wholly inside any single chunk"""
    return {
        entity.text
        for entity in entities
        if not any(entity.text in chunk for chunk in chunks)
    }


def positive_outcomes(
    document: Document,
    predictions: list[str],
    rule: str | None = None,
    coverage_threshold: float = 0.8,
) -> dict[str, bool]:
    """Map each positive entity's text to whether the predictions redacted it"""
    document_words = DocumentWords(document.text)
    located = [
        words
        for words in (document_words.of(p) for p in predictions)
        if words and find_subsequence(document_words.words, words)
    ]
    return {
        entity.text: coverage(document_words.of(entity.text), located)
        >= coverage_threshold
        for entity in document.positives(rule)
    }


def score_document(
    document: Document,
    predictions: list[str],
    rule: str | None = None,
    coverage_threshold: float = 0.8,
    boundary_split: set[str] | None = None,
) -> ScoreCounts:
    """
    Score predictions for one document.

    :param rule: Score against a single rule's labels. If None, score against all
    labels, i.e. the combined output of every rule.
    :param coverage_threshold: Fraction of an entity's words that must be covered for
    it to count as a true positive.
    :param boundary_split: Texts of entities that were split across chunk boundaries.
    """
    boundary_split = boundary_split or set()
    document_words = DocumentWords(document.text)
    positives = document.positives(rule)
    other_positives = (
        tuple(e for e in document.positives() if e.rule != rule) if rule else ()
    )
    negatives = document.negatives(rule)

    prediction_words = list(
        dict.fromkeys(w for w in (document_words.of(p) for p in predictions) if w)
    )
    located = [p for p in prediction_words if find_subsequence(document_words.words, p)]

    counts = ScoreCounts(
        positives=len(positives),
        predictions=len(prediction_words),
        unlocatable_predictions=len(prediction_words) - len(located),
        hard_negatives=len(negatives),
    )

    for entity in positives:
        entity_coverage = coverage(document_words.of(entity.text), located)
        is_split = entity.text in boundary_split
        counts.boundary_positives += is_split
        if entity_coverage >= coverage_threshold:
            counts.true_positives += 1
            counts.boundary_true_positives += is_split
        else:
            counts.false_negatives += 1
            counts.partial_positives += entity_coverage > 0

    positive_words = [document_words.of(e.text) for e in positives]
    other_positive_words = [document_words.of(e.text) for e in other_positives]
    false_positive_words = []
    for prediction in located:
        if any(overlaps(prediction, words) for words in positive_words):
            counts.correct_predictions += 1
        elif any(overlaps(prediction, words) for words in other_positive_words):
            counts.cross_rule_predictions += 1
        else:
            counts.false_positives += 1
            false_positive_words.append(prediction)

    counts.hard_negative_hits = sum(
        any(
            overlaps(prediction, document_words.of(entity.text))
            for prediction in false_positive_words
        )
        for entity in negatives
    )
    return counts
