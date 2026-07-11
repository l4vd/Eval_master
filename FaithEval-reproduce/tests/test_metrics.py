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
