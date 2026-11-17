"""run_batch / run_series must match per-text results while using spaCy's nlp.pipe."""
import pytest

import nlp_text_preprocessing as tp
from nlp_text_preprocessing import all_text_preprocess as core

pytest.importorskip("spacy")
if not tp.check_resources()["spacy_model"]:
    pytest.skip("spaCy model not installed", allow_module_level=True)

TEXTS = [
    "The cats were running quickly.",
    "",
    None,
    "Give it to me and him, please.",
    "Dogs bark. Birds sing!",
]


@pytest.mark.parametrize("step", ["lemmatize", "convert_to_base", "singularize_words", "pluralize_words"])
def test_batch_matches_single(step):
    pipe = tp.Pipeline([step])
    assert pipe.run_batch(TEXTS) == [pipe.run(t) for t in TEXTS]


def test_multi_step_batch_matches_single():
    pipe = tp.Pipeline(["remove_urls", "to_lower_case", "lemmatize", "remove_special_chars"])
    texts = ["Visit https://x.org NOW, the cats were running!", "", "Plain words"]
    assert pipe.run_batch(texts) == [pipe.run(t) for t in texts]


def test_batch_uses_nlp_pipe(monkeypatch):
    nlp = tp.get_spacy_nlp()
    calls = {"pipe": 0, "call": 0}
    real_pipe = nlp.pipe

    def spy_pipe(*a, **k):
        calls["pipe"] += 1
        return real_pipe(*a, **k)

    monkeypatch.setattr(nlp, "pipe", spy_pipe)
    tp.Pipeline(["lemmatize"]).run_batch(["cats run", "dogs bark", "birds sing"])
    assert calls["pipe"] == 1               # one pipe() call for the whole list


def test_step_with_arguments_falls_back_to_per_text():
    pipe = tp.Pipeline([("remove_stop_words", {"keep_negations": False})])
    texts = ["this is not good", "the end"]
    assert pipe.run_batch(texts) == [pipe.run(t) for t in texts]


def test_batch_empty_list():
    assert tp.Pipeline(["lemmatize"]).run_batch([]) == []


def test_batch_all_empty_texts_skips_spacy():
    assert tp.Pipeline(["lemmatize"]).run_batch(["", None]) == ["", ""]


def test_batch_non_string_in_middle_raises():
    pipe = tp.Pipeline(["lemmatize", "word_count", "to_lower_case"])
    with pytest.raises(TypeError, match="word_count"):
        pipe.run_batch(["a b c"])


def test_batch_last_step_may_return_non_string():
    assert tp.Pipeline(["lemmatize", "word_count"]).run_batch(["cats run", ""]) == [2, 0]


def test_long_text_in_batch_raises_max_length():
    nlp = tp.get_spacy_nlp()
    old = nlp.max_length
    try:
        nlp.max_length = 500
        out = tp.Pipeline(["lemmatize"]).run_batch(["cats " * 400, "dogs"])
        assert out[0].split()[0] == "cat" and out[1] == "dog"
    finally:
        nlp.max_length = old


def test_run_series_keeps_index_and_name():
    pd = pytest.importorskip("pandas")
    s = pd.Series(["cats run", "dogs bark"], index=[10, 20], name="txt")
    out = tp.Pipeline(["lemmatize"]).run_series(s)
    assert list(out.index) == [10, 20] and out.name == "txt"
    assert list(out) == [tp.lemmatize(t) for t in s]


def test_spacy_docs_order_and_none():
    docs = core._spacy_docs(["a b", "", "c"], disable=("parser", "ner"))
    assert docs[1] is None and docs[0].text == "a b" and docs[2].text == "c"
