"""readability_scores and mask_pii(names=True)."""
import pytest

import nlp_text_preprocessing as tp

KEYS = {'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog',
        'automated_readability_index', 'coleman_liau'}


def test_readability_scores_keys_and_empty():
    assert set(tp.readability_scores("The cat sat on the mat.")) == KEYS
    for text in ("", None, "   "):
        assert tp.readability_scores(text) == {k: 0.0 for k in KEYS}


def test_flesch_matches_existing_readability_score():
    text = "The cat sat on the mat. It was a sunny day."
    assert tp.readability_scores(text)['flesch_reading_ease'] == tp.readability_score(text)


def test_harder_text_scores_as_harder():
    easy = "The cat sat on the mat. It was a big red cat."
    hard = ("Notwithstanding considerable organisational complexity, international "
            "regulatory harmonisation necessitates comprehensive interdepartmental collaboration.")
    e, h = tp.readability_scores(easy), tp.readability_scores(hard)
    assert e['flesch_reading_ease'] > h['flesch_reading_ease']
    for key in ('flesch_kincaid_grade', 'gunning_fog', 'automated_readability_index', 'coleman_liau'):
        assert e[key] < h[key], key


def test_readability_scores_in_all():
    assert 'readability_scores' in tp.__all__


def test_mask_pii_default_leaves_names():
    assert tp.mask_pii("Barack Obama called me on 555-123-4567.").startswith("Barack Obama")


def test_mask_pii_names_requires_opt_in_and_masks_person():
    pytest.importorskip("spacy")
    if not tp.check_resources()["spacy_model"]:
        pytest.skip("spaCy model not installed")
    out = tp.mask_pii("Barack Obama emailed me at obama@example.com yesterday.", names=True)
    assert "Obama" not in out and "example.com" not in out
    assert out.count("[REDACTED]") == 2
    assert "yesterday" in out


def test_mask_pii_names_custom_token_and_empty():
    pytest.importorskip("spacy")
    if not tp.check_resources()["spacy_model"]:
        pytest.skip("spaCy model not installed")
    assert "<NAME>" in tp.mask_pii("Angela Merkel met Emmanuel Macron.", token="<NAME>", names=True)
    assert tp.mask_pii("", names=True) == ""
    assert tp.mask_pii(None, names=True) == ""
