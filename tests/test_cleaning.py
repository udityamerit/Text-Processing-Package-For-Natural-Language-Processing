"""Regression tests for the text-cleaning and statistics helpers."""
import pytest

import nlp_text_preprocessing as tp


# --- URLs / emails / mentions / RT -------------------------------------------------

def test_urls_need_a_scheme_or_www_and_keep_trailing_punctuation():
    assert tp.remove_urls("Visit http://a.com/x. Also www.b.org, httpd is a server") == \
        "Visit . Also , httpd is a server"
    assert tp.count_urls("see https://x.io/a?b=1) and httpie") == 1


def test_emails_with_plus_subdomains_and_digits():
    assert tp.count_emails("a.b+c@sub.example.co.uk, x1@y2.io") == 2
    assert tp.remove_emails("mail a.b+c@sub.example.co.uk, now") == "mail , now"


def test_email_is_not_a_mention():
    assert tp.mentions_count("hi @bob mail bob@x.com") == 1
    assert tp.remove_mentions("hi @bob mail bob@x.com") == "hi  mail bob@x.com"


def test_remove_rt_takes_the_colon_too():
    assert tp.remove_rt("RT @bob: great news").strip() == "great news"
    assert tp.count_rt("RT @a: x RT @b: y") == 2


# --- contractions -----------------------------------------------------------------

def test_contractions_with_punctuation_curly_quotes_and_newlines():
    out = tp.contraction_to_expansion("I can't go.\nHe didn’t say it, can't you?")
    assert out == "I cannot go.\nHe did not say it, cannot you?"


def test_contraction_keeps_capital():
    assert tp.contraction_to_expansion("Can't stop").startswith("Cannot")


# --- accents / special chars / repeats ---------------------------------------------

def test_accent_removal_keeps_non_latin_scripts():
    assert tp.remove_accented_chars("café naïve") == "cafe naive"
    assert tp.remove_accented_chars("नमस्ते 你好") == "नमस्ते 你好"


def test_special_chars_do_not_glue_hyphenated_words():
    assert tp.remove_special_chars("well-known don't a/b!") == "well known dont a b"


def test_repeated_chars_only_touch_letters():
    assert tp.remove_repeated_chars("soooo goood") == "soo good"
    assert tp.remove_repeated_chars("1000000 ... \n\n\n") == "1000000 ... \n\n\n"


# --- stop words -----------------------------------------------------------------------

def test_stop_words_keep_negations_by_default():
    assert tp.remove_stop_words("I do not like it, but the cat is here.") == "not like cat"
    assert "not" not in tp.remove_stop_words("I do not like it", keep_negations=False).split()


def test_stop_words_ignore_attached_punctuation():
    assert tp.stop_words_count("The cat, the dog.") == 2
    assert tp.remove_stop_words("The cat, the dog.") == "cat, dog."


# --- emoji ----------------------------------------------------------------------------

def test_emoji_sequences_are_whole():
    text = "hi 😂 👍🏽 🇮🇳 ❤️"
    assert tp.extract_emojis(text) == ["😂", "👍🏽", "🇮🇳", "❤️"]
    cleaned = tp.remove_emojis(text)
    assert not any(ch in cleaned for ch in "😂👍🏽🇮🇳❤\ufe0f")   # no leftover modifiers / selectors


def test_emoji_removal_spares_text_symbols_and_math_letters():
    assert tp.remove_emojis("© 2024 𝐀") == "© 2024 𝐀"


def test_emoji_to_text():
    assert tp.emoji_to_text("hi 😂❤️") == "hi face_with_tears_of_joy red_heart"


# --- phones / PII ---------------------------------------------------------------------

@pytest.mark.parametrize("number", [
    "+91 98765 43210", "(555) 123-4567", "555.123.4567", "+1-800-555-0199",
])
def test_phone_formats_detected(number):
    assert tp.count_phone_numbers(f"call {number} now") == 1
    assert tp.remove_phone_numbers(f"call {number} now") == "call  now"


@pytest.mark.parametrize("text", [
    "on 2023-01-15 10:30", "pi is 3.14159265358979", "order id 12345", "year 2024",
])
def test_not_phones(text):
    assert tp.count_phone_numbers(text) == 0
    assert tp.remove_phone_numbers(text) == text


def test_mask_pii_redacts_everything_with_token():
    out = tp.mask_pii("mail a@b.com card 4111 1111 1111 1111 ssn 123-45-6789 tel +1-800-555-0199")
    assert out == "mail [REDACTED] card [REDACTED] ssn [REDACTED] tel [REDACTED]"
    assert tp.mask_pii("order 20240115") == "order 20240115"
    assert tp.mask_pii("a@b.com", token="<X>") == "<X>"


# --- numbers / hashtags / currency / punctuation ----------------------------------------

def test_remove_numbers_only_standalone():
    assert tp.remove_numbers("3rd H2O 1,000.50 and 2026 COVID19") == "3rd H2O  and  COVID19"


def test_expand_hashtags():
    assert tp.expand_hashtags("#MachineLearning2024 #NLPIsFun #snake_case") == \
        "Machine Learning 2024 NLP Is Fun snake case"


def test_currency_symbols_all_unicode():
    assert tp.remove_currency_symbols("$5 €4 ₹3 ₿1 £2") == "5 4 3 1 2"


def test_unicode_punctuation_and_invisible_spaces():
    assert tp.normalize_unicode_punctuation("a\u00a0b\u200bc “q” — x…") == 'a bc "q" - x...'


# --- html ------------------------------------------------------------------------------

def test_html_blocks_separate_words_and_scripts_are_dropped():
    out = tp.remove_html_tag("<p>a</p><p>b</p><script>x=1</script>")
    assert out.split() == ["a", "b"]
    assert tp.remove_html_tag("see http://x.com a < b") == "see http://x.com a < b"


# --- statistics --------------------------------------------------------------------------

def test_sentence_count_ignores_decimal_points():
    assert tp.sentence_count("Pi is 3.14 today. Done!") == 2
    assert tp.sentence_count("   ") == 0


def test_readability_uses_real_syllables():
    easy = tp.readability_score("The cat sat on the mat.")
    hard = tp.readability_score("Unquestionably extraordinary organizational characteristics.")
    assert easy > 80 > 0 > hard


def test_upper_case_count_ignores_single_letters():
    assert tp.upper_case_count("I am A NASA fan") == 1
    assert tp.capital_ratio("I am A NASA fan") == pytest.approx(1 / 5)


# --- None / NaN safety across the board ---------------------------------------------------

@pytest.mark.parametrize("name", [
    "remove_urls", "remove_emails", "remove_rt", "remove_mentions", "remove_special_chars",
    "remove_repeated_chars", "remove_stop_words", "remove_emojis", "emoji_to_text",
    "remove_phone_numbers", "mask_pii", "expand_hashtags", "remove_numbers",
    "remove_currency_symbols", "contraction_to_expansion", "remove_accented_chars",
    "normalize_unicode_punctuation", "remove_html_tag", "remove_extra_whitespace",
])
def test_none_and_nan_return_empty_string(name):
    fn = getattr(tp, name)
    assert fn(None) == ""
    assert fn(float("nan")) == ""
