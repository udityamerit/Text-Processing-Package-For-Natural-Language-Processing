import json

import pytest

import nlp_text_preprocessing as tp
from nlp_text_preprocessing import Pipeline


def test_step_with_arguments():
    pipe = Pipeline(["to_lower_case", ("remove_stop_words", {"keep_negations": False})])
    assert "not" not in pipe.run("This is NOT good").split()
    keep = Pipeline(["to_lower_case", ("remove_stop_words", {"keep_negations": True})])
    assert "not" in keep.run("This is NOT good").split()


def test_bad_arguments_fail_when_building():
    with pytest.raises(TypeError):
        Pipeline([("remove_stop_words", {"nonexistent": 1})])
    with pytest.raises(TypeError):
        Pipeline([("remove_urls", "not-a-dict")])


def test_unknown_step_fails_early():
    with pytest.raises(ValueError):
        Pipeline(["no_such_step"])


def test_non_string_in_middle_is_an_error_but_last_step_may_return_anything():
    with pytest.raises(TypeError):
        Pipeline(["text_stats", "to_lower_case"]).run("hello world")
    result = Pipeline(["to_lower_case", "text_stats"]).run("Hello World")
    assert isinstance(result, dict)


def test_repr_shows_arguments():
    pipe = Pipeline(["remove_urls", ("remove_stop_words", {"keep_negations": False})])
    text = repr(pipe)
    assert "remove_urls" in text and "keep_negations" in text


def test_save_and_load_roundtrip(tmp_path):
    pipe = Pipeline(["remove_urls", ("remove_stop_words", {"keep_negations": False}), "to_lower_case"])
    path = tmp_path / "pipe.json"
    pipe.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    loaded = Pipeline.load(path)
    sample = "See https://x.com The NOT good Idea"
    assert loaded.run(sample) == pipe.run(sample)


def test_saving_unregistered_callable_is_refused():
    with pytest.raises(ValueError):
        Pipeline([lambda s: s]).to_config()


def test_registered_callable_can_be_saved(tmp_path):
    @tp.register_step("shout_test_step")
    def shout(x):
        return x.upper()

    pipe = Pipeline(["shout_test_step"])
    pipe.save(tmp_path / "p.json")
    assert Pipeline.load(tmp_path / "p.json").run("hi") == "HI"


def test_fit_transform_and_params():
    pipe = Pipeline(["to_lower_case"])
    assert pipe.fit_transform(["A", "B"]) == ["a", "b"]
    assert pipe.get_params() == {"steps": ["to_lower_case"]}
    pipe.set_params(steps=["remove_urls"])
    assert pipe.run("a http://x.com") .strip() == "a"
    with pytest.raises(ValueError):
        pipe.set_params(bogus=1)


def test_add_step_and_len():
    pipe = Pipeline().add_step("to_lower_case")
    assert len(pipe) == 1 and pipe.run("AB") == "ab"


sklearn = pytest.importorskip("sklearn")


def test_sklearn_transformer_in_pipeline_and_clone():
    from sklearn.base import clone
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    TextPreprocessor = tp.TextPreprocessor
    tr = TextPreprocessor(steps=["to_lower_case", "remove_special_chars"])
    assert clone(tr).steps == tr.steps
    model = make_pipeline(tr, TfidfVectorizer())
    matrix = model.fit_transform(["Hello, World!", "HELLO there"])
    assert matrix.shape[0] == 2
    assert "hello" in model.steps[1][1].vocabulary_


def test_sklearn_transformer_default_steps():
    out = tp.TextPreprocessor().fit_transform(["<b>Hi</b> THERE http://x.com"])
    assert out == ["hi there"]
