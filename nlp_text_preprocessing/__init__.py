# entry level code and import text preprocess

from .all_text_preprocess import *

__all__ = [
    'download_nltk_packages',
    'word_count',
    'char_count',
    'avg_word_len',
    'stop_words_count',
    'hashtags_count',
    'mentions_count',
    'numerics_count',
    'upper_case_count',
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
    'convert_to_base',
    'lemmatize',
    'get_wordcloud',
    'correct_spelling',
    'get_noun_phrase',
    'n_grams',
    'singularize_words',
    'pluralize_words',
    'sentiment_analysis',
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


def clean_text(text):
    """
    Applies a sequence of cleaning and normalization steps on text.
    """
    text = to_lower_case(text)
    text = contraction_to_expansion(text)
    text = remove_emails(text)
    text = remove_urls(text)
    text = remove_html_tag(text)
    text = remove_special_chars(text)
    text = lemmatize(text)
    text = convert_to_base(text)
    return text