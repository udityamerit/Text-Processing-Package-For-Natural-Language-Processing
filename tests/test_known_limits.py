"""Regression tests for limits that were fixed after 0.4.0."""
import pytest

import nlp_text_preprocessing as tp

_report = tp.check_resources()
needs_spacy = pytest.mark.skipif(not _report["spacy_model"], reason="spaCy model not installed")


@pytest.mark.parametrize("text, expected", [
    ("Hello world. How are you? Fine!", 3),
    ("I saw Dr. Smith today. He was fine.", 2),
    ("Mr. and Mrs. Jones left.", 1),
    ("J. K. Rowling wrote it. Yes.", 2),
    ("Fruits, e.g. apples, are good. Eat.", 2),
    ("It costs $5 vs. $6. Fine.", 2),
    ("Wait... what?! Really.", 3),
    ("Pi is 3.14 and that is ok.", 1),
    ("Ends with Dr.", 1),
    ("No punctuation here", 1),
    ("", 0),
])
def test_sentence_count_abbreviations(text, expected):
    assert tp.sentence_count(text) == expected


def test_avg_sentence_len_uses_new_count():
    assert tp.avg_sentence_len("I saw Dr. Smith today. He was fine.") == pytest.approx(4.0)


@needs_spacy
def test_lemmatize_keeps_pronouns():
    out = tp.lemmatize("She gave me the books and told them to us.")
    words = out.rstrip(".").split()
    for pronoun in ("me", "them", "us"):
        assert pronoun in words
    assert "book" in out.split()
    assert " I " not in f" {out} "
