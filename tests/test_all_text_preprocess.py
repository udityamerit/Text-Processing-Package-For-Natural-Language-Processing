import unittest

from nlp_text_preprocessing import (
    download_nltk_packages,
    count_emails,
    remove_emails,
    pluralize_words,
    avg_word_len,
    char_count,
    word_count,
    remove_stop_words,
    stop_words_count,
    to_lower_case,
    count_urls,
    remove_urls,
    count_rt,
    hashtags_count,
    mentions_count,
    numerics_count,
    upper_case_count,
    remove_emojis,
    extract_emojis,
    emoji_to_text,
    remove_phone_numbers,
    count_phone_numbers,
    mask_pii,
    remove_extra_whitespace,
    expand_hashtags,
    normalize_unicode_punctuation,
    remove_numbers,
    remove_currency_symbols,
    normalize_whitespace_and_casing,
    extract_entities,
    extract_pos_tags,
    get_dependency_tree,
    detect_language,
    keyword_extraction,
    text_similarity,
    sentence_count,
    avg_sentence_len,
    punctuation_density,
    unique_word_ratio,
    readability_score,
    text_stats,
    detect_profanity,
    censor_profanity,
    detect_toxicity,
    detect_spam_signals,
    polarity_subjectivity,
    emotion_detection,
    Pipeline,
    register_step,
    diff_report,
)


class TestNLPTextPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            download_nltk_packages()
        except Exception:
            pass

    def test_count_emails_bug2_fix(self):
        text = "Please reach out to support@example.com or sales@company.org for help."
        self.assertEqual(count_emails(text), 2)
        self.assertEqual(count_emails("No email address present here."), 0)
        self.assertEqual(remove_emails(text), "Please reach out to  or  for help.")

    def test_pluralize_words_bug3_fix(self):
        text = "the dog is barking"
        result = pluralize_words(text)
        self.assertIn("dogs", result)
        self.assertEqual(result, "the dogs is barking")

    def test_avg_word_len_bug4_fix(self):
        self.assertEqual(avg_word_len(""), 0.0)
        self.assertEqual(avg_word_len("   "), 0.0)
        self.assertEqual(avg_word_len(None), 0.0)
        self.assertEqual(avg_word_len("test word"), 4.0)

    def test_case_insensitive_stop_words_bug5_fix(self):
        text = "The cat sat on The mat"
        self.assertEqual(stop_words_count(text), 3)
        cleaned = remove_stop_words(text)
        self.assertNotIn("The", cleaned.split())
        self.assertNotIn("on", cleaned.split())
        self.assertEqual(cleaned, "cat sat mat")

    def test_input_null_and_nan_safety(self):
        self.assertEqual(word_count(None), 0)
        self.assertEqual(char_count(float('nan')), 0)
        self.assertEqual(to_lower_case(None), "")
        self.assertEqual(remove_urls(None), "")
        self.assertEqual(count_urls(None), 0)
        self.assertEqual(count_emails(float('nan')), 0)

    def test_general_feature_extraction(self):
        sample = "Hello #NLP world! Contact @uditya with 123 numbers. RT @user WOW"
        self.assertEqual(word_count(sample), 11)
        self.assertEqual(hashtags_count(sample), 1)
        self.assertEqual(mentions_count(sample), 2)
        self.assertEqual(numerics_count(sample), 1)
        self.assertEqual(upper_case_count(sample), 3)
        self.assertEqual(count_rt(sample), 1)

    def test_emoji_and_pii_functions(self):
        sample = "Hello 😂 world! Call +1-800-555-0199 or email user@test.com"
        self.assertEqual(len(extract_emojis(sample)), 1)
        self.assertNotIn("😂", remove_emojis(sample))
        self.assertIn("face_with_tears_of_joy", emoji_to_text(sample))
        self.assertEqual(count_phone_numbers(sample), 1)
        self.assertNotIn("+1-800-555-0199", remove_phone_numbers(sample))
        masked = mask_pii(sample)
        self.assertNotIn("user@test.com", masked)

    def test_normalization_and_formatting(self):
        self.assertEqual(remove_extra_whitespace("   Hello   world!  \n "), "Hello world!")
        self.assertEqual(expand_hashtags("I love #MachineLearning"), "I love Machine Learning")
        self.assertEqual(normalize_unicode_punctuation("“Hello”—world…"), '"Hello"-world...')
        self.assertEqual(remove_numbers("Version 2.0 released in 2026"), "Version  released in ")
        self.assertEqual(remove_currency_symbols("Cost: $50 and €40"), "Cost: 50 and 40")
        self.assertEqual(normalize_whitespace_and_casing("  HELLO   WORLD "), "hello world")

    def test_linguistic_nlp_features(self):
        text = "Apple is looking at buying a startup in London."
        entities = extract_entities(text)
        self.assertIsInstance(entities, list)
        pos_tags = extract_pos_tags(text)
        self.assertIsInstance(pos_tags, list)
        deps = get_dependency_tree(text)
        self.assertIsInstance(deps, list)
        self.assertEqual(detect_language(text), "en")
        keywords = keyword_extraction(text, top_n=3)
        self.assertIsInstance(keywords, list)
        sim = text_similarity("apple machine learning", "apple artificial intelligence")
        self.assertGreaterEqual(sim, 0.0)

    def test_new_stats_and_readability(self):
        sample = "This is sentence one. This is sentence two! Is this sentence three?"
        self.assertEqual(sentence_count(sample), 3)
        self.assertGreater(avg_sentence_len(sample), 0)
        self.assertGreater(punctuation_density(sample), 0)
        self.assertGreaterEqual(unique_word_ratio(sample), 0.0)
        self.assertIsInstance(readability_score(sample), float)
        stats = text_stats(sample)
        self.assertIn('readability_score', stats)

    def test_safety_moderation_and_sentiment(self):
        profane_text = "This is a damn test."
        self.assertTrue(detect_profanity(profane_text))
        self.assertIn("****", censor_profanity(profane_text))

        toxic_text = "You are a stupid idiot loser."
        self.assertGreater(detect_toxicity(toxic_text), 0.0)

        spam_text = "BUY NOW!!! CLICK HERE https://spam.com VERY CHEAP"
        spam_signals = detect_spam_signals(spam_text)
        self.assertTrue(spam_signals['is_spam_candidate'])

        pol_sub = polarity_subjectivity("This is a great product!")
        self.assertIn('polarity', pol_sub)
        emotions = emotion_detection("I am very happy and excited!")
        self.assertGreater(emotions['joy'], 0)

    def test_pipeline_engine_and_diff_report(self):
        pipe = Pipeline.default()
        text = "Check out https://test.com! Email me at user@test.com"
        cleaned = pipe.run(text)
        self.assertNotIn("https://test.com", cleaned)
        self.assertNotIn("user@test.com", cleaned)

        # Custom step registration
        @register_step("custom_exclamation_remover")
        def remove_exclamations(t):
            return t.replace("!", "")

        pipe.add_step("custom_exclamation_remover")
        res = pipe.run("Hello world!")
        self.assertNotIn("!", res)

        diff = diff_report(["Check http://test.com", "Contact user@test.com"], ["Check", "Contact"])
        self.assertIsInstance(diff, dict)
        self.assertIn('urls_removed', diff)
        self.assertIn('emails_removed', diff)


if __name__ == '__main__':
    unittest.main()
