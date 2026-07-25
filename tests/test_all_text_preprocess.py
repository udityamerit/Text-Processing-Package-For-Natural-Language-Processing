import unittest
import math

from nlp_text_preprocessing import (
    download_nltk_packages,
    count_emails,
    remove_emails,
    pluralize_words,
    singularize_words,
    avg_word_len,
    char_count,
    word_count,
    remove_stop_words,
    stop_words_count,
    convert_to_base,
    lemmatize,
    sentiment_analysis,
    get_wordcloud,
    clean_text,
    extract_features,
    to_lower_case,
    contraction_to_expansion,
    count_urls,
    remove_urls,
    count_rt,
    remove_rt,
    remove_html_tag,
    remove_accented_chars,
    remove_mentions,
    remove_special_chars,
    remove_repeated_chars,
    hashtags_count,
    mentions_count,
    numerics_count,
    upper_case_count,
    correct_spelling,
    get_noun_phrase,
    n_grams,
)


class TestNLPTextPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure NLTK packages are available during testing
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
        self.assertEqual(upper_case_count(sample), 2)
        self.assertEqual(count_rt(sample), 1)

    def test_cleaning_and_normalization(self):
        self.assertEqual(contraction_to_expansion("don't app"), "do not application")
        self.assertEqual(remove_urls("Check http://google.com or www.openai.com"), "Check  or ")
        self.assertEqual(remove_rt("RT @user Great post!"), " Great post!")
        self.assertEqual(remove_html_tag("<h1>Title</h1><p>Hello</p>"), "TitleHello")
        self.assertEqual(remove_accented_chars("café résumé"), "cafe resume")
        self.assertEqual(remove_mentions("Hello @alice and @bob!"), "Hello  and !")
        self.assertEqual(remove_special_chars("Hello, world!"), "Hello world")
        self.assertEqual(remove_repeated_chars("gooooood dayyyyy"), "good dayy")

    def test_spacy_lemmatization(self):
        text = "running dogs"
        lem = lemmatize(text)
        self.assertIn("run", lem)
        self.assertIn("dog", lem)
        base = convert_to_base(text)
        self.assertIn("run", base)
        self.assertIn("dog", base)

    def test_textblob_linguistics(self):
        self.assertEqual(singularize_words("the dogs are barking"), "the dog are barking")
        self.assertEqual(len(n_grams("natural language processing", n=2)), 2)
        self.assertIsInstance(get_noun_phrase("beautiful sunny day"), list)

    def test_sentiment_analysis(self):
        sentiment = sentiment_analysis("This product is amazing and wonderful!")
        self.assertIn(sentiment, ["pos", "neg"])

    def test_get_wordcloud(self):
        cloud = get_wordcloud("apple banana cherry apple banana", show=False)
        self.assertIsNotNone(cloud)

    def test_extract_features_and_clean_text(self):
        sample = "Check out https://example.com! Contact us at info@test.com."
        features = extract_features(sample)
        self.assertIsInstance(features, dict)
        self.assertIn('word_count', features)
        self.assertIn('sentiment_analysis', features)

        cleaned = clean_text(sample)
        self.assertIsInstance(cleaned, str)


if __name__ == '__main__':
    unittest.main()
