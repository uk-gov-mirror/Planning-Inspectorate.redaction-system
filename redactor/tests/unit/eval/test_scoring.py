import pytest
from eval.corpus import Document, Entity
from eval.scoring import (
    DocumentWords,
    ScoreCounts,
    boundary_split_entities,
    coverage,
    f_beta,
    normalise_words,
    positive_outcomes,
    score_document,
)

DOCUMENT = Document(
    id="doc-1",
    doc_type="relevant_representation",
    text=(
        "I live at 14 Orchard Rise, Haddon, LH4 9QX. My neighbour Joan Whitcombe has "
        "asthma. I agree with Margaret Ellis's views. Email peter@gmail.com."
    ),
    entities=(
        Entity(
            "14 Orchard Rise, Haddon, LH4 9QX", "contact_identity_info", "addr", True
        ),
        Entity("Joan Whitcombe", "names", "third_party", True),
        Entity("asthma", "sensitive_info", "diagnosis", True),
        Entity("Margaret Ellis", "names", "participant", False),
        Entity("peter@gmail.com", "contact_identity_info", "email", True),
    ),
)


def test_normalise_words():
    assert normalise_words("Margaret Ellis's  VIEWS,") == ("margaret", "ellis", "views")
    # An email address is one word
    assert normalise_words("Email Peter.Jones+x@Mail.example.co.uk.") == (
        "email",
        "peter.jones+x@mail.example.co.uk",
    )
    assert normalise_words("peter@gmail.com") == ("peter@gmail.com",)


def test_coverage__union_of_partial_predictions():
    entity = normalise_words("14 Orchard Rise, Haddon, LH4 9QX")

    assert coverage(entity, [normalise_words("14 Orchard Rise")]) == pytest.approx(
        3 / 6
    )
    assert coverage(
        entity, [normalise_words("14 Orchard Rise"), normalise_words("Haddon, LH4 9QX")]
    ) == pytest.approx(1.0)
    assert coverage(entity, [normalise_words("at 14 Orchard Rise, Haddon, LH4 9QX.")])


def test_f_beta():
    assert f_beta(1.0, 1.0) == pytest.approx(1.0)
    assert f_beta(0.0, 0.0) == 0.0
    # beta=2 weights recall above precision
    assert f_beta(0.5, 1.0, beta=2) > f_beta(1.0, 0.5, beta=2)


def test_score_document__all_rules():
    predictions = [
        "14 Orchard Rise, Haddon, LH4 9QX",
        "Joan Whitcombe",
        "Margaret Ellis",  # hard negative
        "Somewhere not in the text",
    ]

    counts = score_document(DOCUMENT, predictions)

    assert counts.positives == 4
    assert counts.true_positives == 2
    assert counts.false_negatives == 2
    assert counts.correct_predictions == 2
    assert counts.false_positives == 1
    assert counts.unlocatable_predictions == 1
    assert counts.hard_negatives == 1
    assert counts.hard_negative_hits == 1
    assert counts.recall == pytest.approx(0.5)
    assert counts.precision == pytest.approx(2 / 3)


def test_score_document__single_rule_counts_other_rules_as_cross_rule():
    counts = score_document(DOCUMENT, ["Joan Whitcombe", "asthma"], rule="names")

    assert counts.positives == 1
    assert counts.true_positives == 1
    assert counts.cross_rule_predictions == 1
    assert counts.false_positives == 0


def test_score_document__partial_and_boundary():
    counts = score_document(
        DOCUMENT,
        ["14 Orchard Rise"],
        rule="contact_identity_info",
        boundary_split={"14 Orchard Rise, Haddon, LH4 9QX"},
    )

    assert counts.partial_positives == 1
    assert counts.true_positives == 0
    assert counts.boundary_positives == 1
    assert counts.boundary_recall == 0.0


def test_score_counts_addition():
    total = ScoreCounts(positives=2, true_positives=1) + ScoreCounts(
        positives=3, true_positives=3
    )

    assert total.positives == 5
    assert total.recall == pytest.approx(0.8)


def test_empty_counts_have_safe_rates():
    counts = ScoreCounts()

    assert counts.precision == 1.0
    assert counts.recall == 1.0
    assert counts.hard_negative_fp_rate == 0.0
    assert counts.boundary_recall is None


def test_boundary_split_entities():
    chunks = [
        "I live at 14 Orchard Rise,",
        "Haddon, LH4 9QX. My neighbour Joan Whitcombe",
    ]

    split = boundary_split_entities(DOCUMENT.entities, chunks)

    assert "14 Orchard Rise, Haddon, LH4 9QX" in split
    assert "Joan Whitcombe" not in split


def test_positive_outcomes():
    outcomes = positive_outcomes(DOCUMENT, ["asthma"], rule="sensitive_info")

    assert outcomes == {"asthma": True}


EMAIL_DOCUMENT = Document(
    id="doc-2",
    doc_type="email_correspondence",
    text=(
        "Joan Whitcombe wrote. Contact ada.merrin@post.invalid or info@natural-england.org.uk "
        "for more. Ada said hello."
    ),
    entities=(
        Entity("ada.merrin", "contact_identity_info", "email_name", True),
        Entity(
            "info@natural-england.org.uk", "contact_identity_info", "mailbox", False
        ),
        Entity("Joan Whitcombe", "names", "third_party", True),
    ),
)


def test_document_words__name_part_of_an_email_stands_for_the_address():
    words = DocumentWords(EMAIL_DOCUMENT.text)

    assert words.emails == ("ada.merrin@post.invalid", "info@natural-england.org.uk")
    assert words.of("ada.merrin") == ("ada.merrin@post.invalid",)
    assert words.of("Merrin") == ("ada.merrin@post.invalid",)
    assert words.of("ada.merrin@post.invalid") == ("ada.merrin@post.invalid",)


def test_document_words__a_word_of_the_document_is_not_replaced():
    """ "Ada" is also a word in the text, so it keeps its own words"""
    assert DocumentWords(EMAIL_DOCUMENT.text).of("Ada") == ("ada",)


def test_document_words__domain_and_other_text_are_not_matched():
    words = DocumentWords(EMAIL_DOCUMENT.text)

    assert words.of("post") == ("post",)
    assert words.of("Mary Jones") == ("mary", "jones")


def test_score_document__email_name_part_is_located_and_covers_the_label():
    """
    - Given an entity and a prediction that are the name part of an email address
    - When I score the document
    - Then the prediction is located in the text, rather than unlocatable, and the
      entity is covered
    """
    counts = score_document(
        EMAIL_DOCUMENT, ["ada.merrin"], rule="contact_identity_info"
    )

    assert counts.unlocatable_predictions == 0
    assert counts.true_positives == 1
    assert counts.correct_predictions == 1
    assert counts.false_positives == 0


@pytest.mark.parametrize(
    "prediction", ["ada.merrin@post.invalid", "ada.merrin", "Ada.Merrin@Post.Invalid."]
)
def test_score_document__full_address_and_name_part_cover_each_other(prediction):
    """The redactor highlights the address for either, so both cover a label for either"""
    full_address_label = Document(
        id="doc-3",
        doc_type="x",
        text=EMAIL_DOCUMENT.text,
        entities=(
            Entity("ada.merrin@post.invalid", "contact_identity_info", "email", True),
        ),
    )

    for document in (EMAIL_DOCUMENT, full_address_label):
        counts = score_document(document, [prediction], rule="contact_identity_info")
        assert counts.true_positives == 1, document.entities


def test_score_document__email_name_part_hard_negative_is_a_false_positive():
    counts = score_document(EMAIL_DOCUMENT, ["info"], rule="contact_identity_info")

    assert counts.hard_negative_hits == 1
    assert counts.false_positives == 1


def test_positive_outcomes__email_name_part():
    outcomes = positive_outcomes(
        EMAIL_DOCUMENT, ["ada.merrin"], rule="contact_identity_info"
    )

    assert outcomes == {"ada.merrin": True}
