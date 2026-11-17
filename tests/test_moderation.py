import pytest

import nlp_text_preprocessing as tp


# ---- profanity ---------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "you are a shitty person", "sh1t happens", "@sshole move", "bitches be crazy",
    "fucking great", "This is a damn test.",
])
def test_profanity_detected_with_endings_and_leetspeak(text):
    assert tp.detect_profanity(text)


@pytest.mark.parametrize("text", [
    "hello from Hell's Kitchen", "Hello there, assess the class", "the pass is closed",
    "Scunthorpe and Dickens", "I like Mississippi", "",
])
def test_profanity_has_no_false_positives(text):
    assert not tp.detect_profanity(text)


def test_profanity_options():
    assert not tp.detect_profanity("damn it", include_mild=False)
    assert tp.detect_profanity("fuck it", include_mild=False)
    assert tp.detect_profanity("what a darn mess", extra_words=["darn"])
    assert not tp.detect_profanity("fuck", allow_words=["fuck"])


def test_censor_keeps_length_and_layout():
    assert tp.censor_profanity("You  sh1t,  Hell's Kitchen") == "You  ****,  Hell's Kitchen"
    assert tp.censor_profanity("damn", mask_char="#") == "####"
    assert tp.censor_profanity(None) == ""


# ---- toxicity ----------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "kill the process", "I do not hate you", "hello from Hell's Kitchen", "take out the trash",
    "that was a nasty storm", "I don't want to hurt you", "you are not stupid",
    "shoot you a message", "",
])
def test_toxicity_ignores_harmless_text(text):
    assert tp.detect_toxicity(text) == 0.0


@pytest.mark.parametrize("text", [
    "You are a stupid idiot loser.", "I will kill you", "you are ugly", "fuck you",
])
def test_toxicity_flags_real_insults_and_threats(text):
    assert tp.detect_toxicity(text) > 0.4


def test_toxicity_grows_with_more_hits_and_stays_below_one():
    one = tp.detect_toxicity("you idiot")
    many = tp.detect_toxicity("you idiot, you moron, you loser, you fool")
    assert 0 < one < many < 1.0


def test_toxicity_report_explains_itself():
    report = tp.toxicity_report("I do not hate you but you are an idiot")
    assert [m["word"] for m in report["matches"]] == ["idiot"]
    assert [m["word"] for m in report["ignored"]] == ["hate"]
    assert report["matches"][0]["targeted"] is True


# ---- spam --------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "OK", "Hi!", "Thanks!!! see you soon", "A normal sentence about the weather.",
    "Meeting at 3pm, here is the link https://x.com/doc", "",
])
def test_spam_has_no_false_positives(text):
    assert not tp.detect_spam_signals(text)["is_spam_candidate"]


def test_spam_detected_with_phrases():
    result = tp.detect_spam_signals("BUY NOW!!! CLICK HERE https://spam.com VERY CHEAP")
    assert result["is_spam_candidate"] and result["has_url"]
    assert "buy now" in result["spam_phrases"]
    assert tp.detect_spam_signals("You have won! Claim your cash prize")["is_spam_candidate"]
    assert tp.detect_spam_signals("http://a.com http://b.com http://c.com")["is_spam_candidate"]


# ---- emotion -----------------------------------------------------------------
def test_emotion_handles_word_endings():
    assert tp.emotion_detection("I loved it")["joy"] == 1
    assert tp.emotion_detection("happier than ever")["joy"] == 1
    assert tp.emotion_detection("She is worrying")["fear"] == 1
    assert tp.emotion_detection("I am very happy and excited!")["joy"] == 2


@pytest.mark.parametrize("text", ["not happy", "I am not very sad", "I never smile", "I don't love this"])
def test_emotion_ignores_negated_words(text):
    assert sum(tp.emotion_detection(text).values()) == 0


def test_emotion_keeps_all_four_keys_and_handles_empty():
    assert tp.emotion_detection(None) == {"joy": 0, "anger": 0, "sadness": 0, "fear": 0}


def test_moderation_functions_work_as_pipeline_steps_with_arguments():
    pipe = tp.Pipeline([("censor_profanity", {"include_mild": False})])
    assert pipe.run("damn and fuck") == "damn and ****"
