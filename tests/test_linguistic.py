"""Regression tests for the linguistic, sentiment, safety and pipeline helpers."""
import pytest

import nlp_text_preprocessing as tp

_report = tp.check_resources()
needs_spacy = pytest.mark.skipif(not _report["spacy_model"], reason="spaCy model not installed")
needs_wordnet = pytest.mark.skipif(not _report["nltk"]["wordnet"], reason="NLTK wordnet missing")
needs_punkt = pytest.mark.skipif(not _report["nltk"]["punkt_tab"], reason="NLTK punkt_tab missing")


# --- spaCy based -------------------------------------------------------------------------

@needs_spacy
def test_lemmatize_keeps_spacing_and_punctuation():
    assert tp.lemmatize("The cats were running, quickly!") == "the cat be run, quickly!"


@needs_spacy
def test_convert_to_base_only_touches_nouns_and_verbs():
    assert tp.convert_to_base("The cats were running, quickly!") == "The cat were run, quickly!"


@needs_spacy
def test_noun_chunks_have_exact_offsets_even_for_repeated_or_capitalised_phrases():
    text = "Apple is buying a small startup. The Apple store is big."
    chunks = tp.chunk_noun_phrases(text)
    assert chunks, "expected some noun chunks"
    for c in chunks:
        assert text[c["start"]:c["end"]] == c["phrase"]
    starts = [c["start"] for c in chunks if c["phrase"].endswith("Apple store")]
    assert starts == [text.index("The Apple store")]


@needs_spacy
def test_inflection_keeps_punctuation():
    assert tp.singularize_words("The dogs, and the cats.") == "The dog, and the cat."
    assert tp.pluralize_words("a dog.") .endswith("dogs.")


@needs_spacy
def test_empty_inputs():
    for fn in (tp.lemmatize, tp.convert_to_base, tp.singularize_words, tp.pluralize_words):
        assert fn("") == "" and fn(None) == ""
    for fn in (tp.extract_entities, tp.extract_pos_tags, tp.get_dependency_tree, tp.chunk_noun_phrases):
        assert fn(None) == []


@needs_spacy
def test_clean_text_end_to_end():
    out = tp.clean_text("<p>Don't miss http://x.com - it's GREAT!!! mail a@b.com</p>")
    assert "http" not in out and "@" not in out and "<" not in out
    assert out == out.lower() and "!" not in out
    assert out.startswith("do not miss")
    assert "not" in tp.clean_text("I do not like it", remove_stopwords=True).split()


# --- language -----------------------------------------------------------------------------

@pytest.mark.parametrize("text,code", [
    ("The quick brown fox jumps over the lazy dog", "en"),
    ("Bonjour tout le monde, comment ça va ?", "fr"),
    ("नमस्ते आप कैसे हैं", "hi"),
])
def test_detect_language(text, code):
    assert tp.detect_language(text) == code


@pytest.mark.parametrize("text", ["", None, "12345 😂", "!!!"])
def test_detect_language_unknown_for_undetectable(text):
    assert tp.detect_language(text) == "unknown"


# --- keywords / similarity / n-grams -----------------------------------------------------------

def test_keyword_extraction_is_ordered_and_unicode_aware():
    assert tp.keyword_extraction("apple apple banana café café café the", top_n=3) == ["café", "apple", "banana"]


def test_text_similarity_ignores_case_and_punctuation():
    assert tp.text_similarity("Hello, world!", "hello world") == 1.0
    assert tp.text_similarity("", "") == 0.0


def test_n_grams_are_tuples_without_punctuation():
    assert tp.n_grams("Hello, big world today", 2) == [("Hello", "big"), ("big", "world"), ("world", "today")]
    assert tp.n_grams("a b", 3) == []
    with pytest.raises(ValueError):
        tp.n_grams("a b", 0)


# --- WordNet ----------------------------------------------------------------------------------

@needs_wordnet
def test_synonyms_and_antonyms_are_sorted_and_complete():
    syn = tp.get_synonyms("good")
    assert syn == sorted(syn) and "good" not in syn and "beneficial" in syn
    ant = tp.get_antonyms("good")
    assert "bad" in ant and "evil" in ant      # not only the first antonym of each lemma


# --- spelling --------------------------------------------------------------------------------

@needs_punkt
def test_correct_spelling_keeps_everything_but_misspelled_words():
    out = tp.correct_spelling("I havv a speling mistke, see http://exmple.com and NASA, don't 3rd")
    assert out == "I have a spelling mistake, see http://exmple.com and NASA, don't 3rd"
    assert tp.correct_spelling("Havv") == "Have"


# --- sentiment ---------------------------------------------------------------------------------

@needs_punkt
def test_sentiment_default_is_fast_three_way():
    assert tp.sentiment_analysis("I love this, it is great") == "pos"
    assert tp.sentiment_analysis("this is terrible and awful") == "neg"
    assert tp.sentiment_analysis("the table is in the room") == "neu"
    assert tp.sentiment_analysis("") == "neu"
    assert tp.polarity_subjectivity("") == {"polarity": 0.0, "subjectivity": 0.0}


def test_sentiment_rejects_unknown_method():
    with pytest.raises(ValueError):
        tp.sentiment_analysis("x", method="magic")


def test_naive_bayes_needs_corpus_and_says_so(monkeypatch):
    from nlp_text_preprocessing import all_text_preprocess as core
    monkeypatch.setattr(core, "_naive_bayes_analyzer", None)
    monkeypatch.setattr(core, "_nltk_has", lambda name: False)
    with pytest.raises(LookupError, match="movie_reviews"):
        tp.sentiment_analysis("good", method="naive_bayes")


# --- safety -----------------------------------------------------------------------------------

def test_censor_profanity_preserves_text_layout():
    assert tp.censor_profanity("What the  damn   hell, Hello!") == "What the  ****   ****, Hello!"
    assert tp.detect_profanity("Hello there") is False


def test_spam_signals():
    s = tp.detect_spam_signals("BUY NOW!!! CLICK HERE https://spam.com VERY CHEAP")
    assert s["is_spam_candidate"] and s["has_url"]
    assert not tp.detect_spam_signals("A normal sentence about the weather.")["is_spam_candidate"]


# --- pipeline ---------------------------------------------------------------------------------

def test_pipeline_default_and_custom_steps():
    pipe = tp.Pipeline.default()
    out = pipe.run("<p>Check out https://test.com! Email me at user@test.com</p>")
    assert "test.com" not in out and "<" not in out

    @tp.register_step("shout_for_test")
    def shout(t):
        return t.upper()

    assert tp.Pipeline(["to_lower_case", "shout_for_test"]).run("Ab") == "AB"
    assert tp.Pipeline([str.strip, "remove_numbers"]).run(" a 1 ") == "a "


def test_pipeline_validates_steps_early_and_copies_input_list():
    with pytest.raises(ValueError):
        tp.Pipeline(["no_such_step"])
    with pytest.raises(ValueError):
        tp.Pipeline(["_clean_input"])          # private helpers are not steps
    with pytest.raises(ValueError):
        tp.Pipeline(["Pipeline"])              # classes are not steps
    with pytest.raises(TypeError):
        tp.Pipeline([42])
    steps = ["to_lower_case"]
    pipe = tp.Pipeline(steps)
    pipe.add_step("remove_numbers")
    assert steps == ["to_lower_case"]
    assert "to_lower_case" in repr(pipe)


def test_diff_report_checks_lengths():
    rep = tp.diff_report(["a http://x.com b@c.io"], ["a"])
    assert rep["urls_removed"] == 1 and rep["emails_removed"] == 1
    with pytest.raises(ValueError):
        tp.diff_report(["a", "b"], ["a"])
