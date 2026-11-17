# entry level code and import text preprocess

__version__ = "0.3.0"

from .all_text_preprocess import *

__all__ = [
    '__version__',
    'download_nltk_packages',
    'check_resources',
    'get_spacy_nlp',
    'get_stop_words',
    'word_count',
    'char_count',
    'avg_word_len',
    'stop_words_count',
    'hashtags_count',
    'mentions_count',
    'numerics_count',
    'upper_case_count',
    'sentence_count',
    'avg_sentence_len',
    'punctuation_density',
    'capital_ratio',
    'unique_word_ratio',
    'readability_score',
    'text_stats',
    'text_stats_batch',
    'to_lower_case',
    'contraction_to_expansion',
    'remove_emails',
    'count_emails',
    'count_urls',
    'remove_urls',
    'count_rt',
    'remove_rt',
    'remove_html_tag',
    'remove_accented_chars',
    'remove_mentions',
    'remove_special_chars',
    'remove_repeated_chars',
    'remove_stop_words',
    'remove_emojis',
    'extract_emojis',
    'emoji_to_text',
    'remove_phone_numbers',
    'count_phone_numbers',
    'mask_pii',
    'remove_extra_whitespace',
    'expand_hashtags',
    'normalize_unicode_punctuation',
    'remove_numbers',
    'remove_currency_symbols',
    'normalize_whitespace_and_casing',
    'convert_to_base',
    'lemmatize',
    'extract_entities',
    'extract_pos_tags',
    'get_dependency_tree',
    'detect_language',
    'keyword_extraction',
    'text_similarity',
    'get_synonyms',
    'get_antonyms',
    'chunk_noun_phrases',
    'get_wordcloud',
    'correct_spelling',
    'get_noun_phrase',
    'n_grams',
    'singularize_words',
    'pluralize_words',
    'detect_profanity',
    'censor_profanity',
    'detect_toxicity',
    'detect_spam_signals',
    'sentiment_analysis',
    'polarity_subjectivity',
    'emotion_detection',
    'Pipeline',
    'register_step',
    'diff_report',
    'extract_features',
    'clean_text',
]


def extract_features(x):
    """
    Extracts a dictionary of text features from text x.
    """
    return {
        'word_count': word_count(x),
        'char_count': char_count(x),
        'avg_word_len': avg_word_len(x),
        'stop_words_count': stop_words_count(x),
        'hashtags_count': hashtags_count(x),
        'mentions_count': mentions_count(x),
        'numerics_count': numerics_count(x),
        'upper_case_count': upper_case_count(x),
        'noun_phrases': get_noun_phrase(x),
        'sentiment_analysis': sentiment_analysis(x)
    }


def clean_text(text, remove_stopwords=False):
    """
    End-to-end cleaning: HTML, URLs and emails are removed, text is lower-cased,
    contractions are expanded, punctuation is dropped and words are lemmatized.

    Set remove_stopwords=True to also drop stop words (negations are kept).
    """
    text = remove_html_tag(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = to_lower_case(text)
    text = contraction_to_expansion(text)
    text = remove_special_chars(text)
    text = remove_extra_whitespace(text)
    text = lemmatize(text)
    if remove_stopwords:
        text = remove_stop_words(text)
    return text
