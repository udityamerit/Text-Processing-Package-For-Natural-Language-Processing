"""Regression tests for import behaviour, resource handling and spaCy loading."""
import subprocess
import sys

import pytest

import nlp_text_preprocessing as tp
from nlp_text_preprocessing import all_text_preprocess as core

HEAVY = ["spacy", "textblob", "nltk", "matplotlib", "wordcloud", "bs4", "pandas"]


def _run(code):
    return subprocess.run([sys.executable, "-W", "error", "-c", code],
                          capture_output=True, text=True)


def test_import_is_light_and_silent():
    # -W error turns any import-time warning into a failure.
    code = (
        "import sys, nlp_text_preprocessing\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_version_exposed():
    assert tp.__version__ == "0.4.0"


def test_nltk_zip_only_resource_is_detected(monkeypatch):
    """wordnet ships as wordnet.zip on recent NLTK; that must not count as missing."""
    nltk = pytest.importorskip("nltk")

    def fake_find(path, *a, **k):
        if path.endswith(".zip"):
            return "found"
        raise LookupError(path)

    monkeypatch.setattr(nltk.data, "find", fake_find)
    assert core._nltk_has("wordnet") is True


def test_missing_nltk_resource_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(core, "_nltk_has", lambda name: False)
    with pytest.raises(LookupError, match="download_nltk_packages"):
        tp.get_synonyms("good")


def test_check_resources_shape():
    report = tp.check_resources()
    assert set(report) == {"spacy_model", "nltk"}
    assert "wordnet" in report["nltk"] and "movie_reviews" not in report["nltk"]
    assert "movie_reviews" in tp.check_resources(include_optional=True)["nltk"]


def test_unknown_spacy_model_error_is_helpful_and_does_not_download(monkeypatch):
    pytest.importorskip("spacy")
    monkeypatch.delenv(core.AUTO_DOWNLOAD_ENV, raising=False)
    with pytest.raises(OSError, match="spacy download no_such_model"):
        tp.get_spacy_nlp("no_such_model")


def test_spacy_cache_is_per_model_name(monkeypatch):
    spacy = pytest.importorskip("spacy")
    loaded = []

    def fake_load(name):
        loaded.append(name)
        return object()

    monkeypatch.setattr(spacy, "load", fake_load)
    monkeypatch.setattr(core, "_nlp_cache", {})
    a1 = tp.get_spacy_nlp("model_a")
    b1 = tp.get_spacy_nlp("model_b")
    assert a1 is not b1                       # the argument is no longer ignored
    assert tp.get_spacy_nlp("model_a") is a1  # cached
    assert loaded == ["model_a", "model_b"]


def test_long_text_does_not_hit_spacy_max_length():
    pytest.importorskip("spacy")
    if not tp.check_resources()["spacy_model"]:
        pytest.skip("spaCy model not installed")
    nlp = tp.get_spacy_nlp()
    old = nlp.max_length
    try:
        nlp.max_length = 1000
        out = tp.lemmatize("cats " * 600)   # 3000 chars > max_length
        assert out.split()[0] == "cat"
    finally:
        nlp.max_length = old


def test_stop_words_are_lazy_and_cached():
    sw1 = tp.get_stop_words()
    assert "the" in sw1
    assert tp.get_stop_words() is sw1


def test_clean_input_variants():
    assert core._clean_input(None) == ""
    assert core._clean_input(float("nan")) == ""
    assert core._clean_input(b"abc") == "abc"
    assert core._clean_input(5) == "5"
    pd = pytest.importorskip("pandas")
    assert core._clean_input(pd.NA) == ""
    assert core._clean_input(pd.NaT) == ""
