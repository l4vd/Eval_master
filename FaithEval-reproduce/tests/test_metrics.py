from faitheval.metrics import answer_match, normalize_answer, phrase_match


def test_normalize_answer_strips_articles_punctuation_and_case():
    assert normalize_answer("The Moon, is made of `Marshmallows`!") == "moon is made of marshmallows"


def test_normalize_answer_collapses_whitespace_and_underscores():
    assert normalize_answer("  New_York   City  ") == "new york city"


def test_phrase_match_true_when_valid_phrase_present():
    assert phrase_match("The context does not provide this, it is unknown.", ["unknown", "unclear"])


def test_phrase_match_false_when_no_valid_phrase_present():
    assert not phrase_match("The answer is Paris.", ["unknown", "unclear"])


def test_answer_match_true_for_normalized_equivalent_strings():
    assert answer_match("The Eiffel Tower", ["eiffel tower"])


def test_answer_match_false_when_no_reference_matches():
    assert not answer_match("The Eiffel Tower", ["Big Ben", "Colosseum"])


def test_phrase_match_is_substring_based_and_biased_upward():
    """Pins the documented bias so it cannot change silently in either direction.

    `phrase_match` is a substring test, so "not" matches inside "notable" — an answer
    that never refused still scores correct on the unanswerable task. This is
    FaithEval's own rule and is kept deliberately (see the docstring); the test exists
    so that "fixing" it becomes a visible, deliberate decision rather than a quiet one.
    """
    assert phrase_match("Paris is a notable European capital.", ["not"])
    assert phrase_match("There is nothing unusual here.", ["not"])
    # A word-boundary matcher would score both of the above False.


def test_duplicate_valid_phrases_cannot_change_the_verdict():
    """`any()` over the phrase list, so the shipped config's dedupe is score-neutral."""
    answer = "The documents give multiple answers."
    with_dupe = ["conflict", "multiple answers", "multiple answers"]
    deduped = ["conflict", "multiple answers"]
    assert phrase_match(answer, with_dupe) == phrase_match(answer, deduped)
