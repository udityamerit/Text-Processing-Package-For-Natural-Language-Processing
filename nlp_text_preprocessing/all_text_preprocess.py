# All text preprocess code is here

import os
import json
import re
import warnings
import unicodedata

import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer
from spacy.lang.en.stop_words import STOP_WORDS as sw
import nltk

fpath = os.path.join(os.path.dirname(__file__), 'data', 'contractions.json')
try:
    with open(fpath, encoding='utf-8') as f:
        contractions = json.load(f)
except Exception as e:
    warnings.warn(f"Could not load contractions.json from {fpath}: {e}")
    contractions = {}

# Lazy-loaded spaCy model container
_nlp = None

def get_spacy_nlp(model_name='en_core_web_sm'):
    """
    Lazy-loads and caches the spaCy language model.
    Downloads the model automatically if missing.
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(model_name)
        except OSError:
            warnings.warn(f"Downloading spaCy model '{model_name}'...")
            from spacy.cli import download
            download(model_name)
            _nlp = spacy.load(model_name)
    return _nlp

# List of NLTK packages and their corresponding resource paths for checking
nltk_packages = [
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('stopwords', 'corpora/stopwords'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    ('tagsets_json', 'help/tagsets_json'),
    ('wordnet', 'corpora/wordnet'),
    ('words', 'corpora/words'),
    ('maxent_ne_chunker_tab', 'chunkers/maxent_ne_chunker_tab'),
    ('movie_reviews', 'corpora/movie_reviews'),
    ('brown', 'corpora/brown'),
]

def download_nltk_packages():
    """
    Downloads the required NLTK packages.
    """
    for package, _ in nltk_packages:
        nltk.download(package)

# Check if NLTK packages are present; if not, instruct the user to download them
missing_packages = []
for package, resource in nltk_packages:
    try:
        nltk.data.find(resource)
    except LookupError:
        missing_packages.append(package)

if missing_packages:
    warnings.warn(
        f"The following NLTK packages are missing: {', '.join(missing_packages)}. "
        "Please run 'download_nltk_packages()' to download them."
    )

# Lazy-loaded NaiveBayesAnalyzer instance for sentiment analysis
_naive_bayes_analyzer = None

def _get_naive_bayes_analyzer():
    global _naive_bayes_analyzer
    if _naive_bayes_analyzer is None:
        _naive_bayes_analyzer = NaiveBayesAnalyzer()
    return _naive_bayes_analyzer

def _clean_input(x):
    """
    Helper function to safely handle non-string, None, or NaN inputs.
    """
    if x is None:
        return ""
    if isinstance(x, float):
        import math
        if math.isnan(x):
            return ""
    try:
        import pandas as pd
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

# General Feature Extraction

def word_count(x):
    """Returns the total number of words in text x."""
    text = _clean_input(x)
    return len(text.split())

def char_count(x):
    """Returns total characters excluding whitespace in text x."""
    text = _clean_input(x)
    pattern = r'\s'
    return len(re.sub(pattern, '', text))

def avg_word_len(x):
    """Returns average word length in text x, returning 0.0 for empty text."""
    wc = word_count(x)
    return char_count(x) / wc if wc > 0 else 0.0

def stop_words_count(x):
    """Returns the count of stopwords in text x."""
    text = _clean_input(x)
    return len([word for word in text.lower().split() if word in sw])

def hashtags_count(x):
    """Returns count of hashtags (#tag) in text x."""
    text = _clean_input(x)
    return len(re.findall(r'#\w+', text))

def mentions_count(x):
    """Returns count of user mentions (@user) in text x."""
    text = _clean_input(x)
    return len(re.findall(r'@\w+', text))

def numerics_count(x):
    """Returns count of standalone numeric tokens in text x."""
    text = _clean_input(x)
    return len(re.findall(r'\b\d+\b', text))

def upper_case_count(x):
    """Returns count of uppercase words in text x."""
    text = _clean_input(x)
    return len([word for word in text.split() if word.isupper()])

# Preprocessing and Cleaning

def to_lower_case(x):
    """Converts text x to lowercase."""
    text = _clean_input(x)
    return text.lower()

def contraction_to_expansion(x):
    """Expands contractions in text x using contractions dictionary."""
    text = _clean_input(x)
    return " ".join([contractions.get(word.lower(), word) for word in text.split()])

def remove_emails(x):
    """Removes email addresses from text x."""
    text = _clean_input(x)
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(pattern, '', text)

def count_emails(x):
    """Returns count of email addresses in text x."""
    text = _clean_input(x)
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z.-]+\.[A-Z|a-z]{2,}\b'
    return len(re.findall(pattern, text))

def count_urls(x):
    """Returns count of URLs in text x."""
    text = _clean_input(x)
    pattern = r'http\S+|www\.\S+'
    return len(re.findall(pattern, text))

def remove_urls(x):
    """Removes URLs from text x."""
    text = _clean_input(x)
    pattern = r'http\S+|www\.\S+'
    return re.sub(pattern, '', text)

def count_rt(x):
    """Returns count of retweets (RT @user) in text x."""
    text = _clean_input(x)
    pattern = r'\bRT @\w+'
    return len(re.findall(pattern, text))

def remove_rt(x):
    """Removes retweet prefixes (RT @user) from text x."""
    text = _clean_input(x)
    pattern = r'\bRT @\w+'
    return re.sub(pattern, '', text)

def remove_html_tag(x):
    """Strips HTML tags from text x using BeautifulSoup."""
    text = _clean_input(x)
    if not text:
        return ""
    return BeautifulSoup(text, 'lxml').get_text()

def remove_accented_chars(x):
    """Removes accented characters from text x."""
    text = _clean_input(x)
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def remove_mentions(x):
    """Removes user mentions (@user) from text x."""
    text = _clean_input(x)
    pattern = r'@\w+'
    return re.sub(pattern, "", text).strip()

def remove_special_chars(x):
    """Removes special/punctuation characters from text x."""
    text = _clean_input(x)
    pattern = r'[^\w\s]'
    return re.sub(pattern, '', text)

def remove_repeated_chars(x):
    """Truncates character repetitions beyond 2 consecutive occurrences in text x."""
    text = _clean_input(x)
    pattern = r'(.)\1+'
    return re.sub(pattern, r'\1\1', text)

def remove_stop_words(x):
    """Removes stop words from text x (case-insensitive check)."""
    text = _clean_input(x)
    return " ".join([word for word in text.split() if word.lower() not in sw])

def convert_to_base(x):
    """Lemmatizes nouns and verbs in text x using spaCy while leaving other POS tags as-is."""
    text = _clean_input(x)
    if not text:
        return ""
    nlp_model = get_spacy_nlp()
    doc = nlp_model(text)
    tokens = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB']:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)
    res = ' '.join(tokens)
    pattern = r'\s\.'
    return re.sub(pattern, '.', res)

def lemmatize(x):
    """Lemmatizes all tokens in text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return ""
    nlp_model = get_spacy_nlp()
    doc = nlp_model(text)
    return ' '.join([token.lemma_ for token in doc])

def get_wordcloud(x, save_path=None, show=True):
    """Generates and optionally displays or saves a WordCloud for text x."""
    text = _clean_input(x)
    cloud = WordCloud(width=800, height=500).generate(text if text else "empty")
    if save_path:
        cloud.to_file(save_path)
    if show:
        plt.figure(figsize=(15, 7), dpi=100)
        plt.imshow(cloud.to_image(), interpolation='bilinear')
        plt.axis('off')
        plt.show()
    return cloud

def correct_spelling(x):
    """Corrects spelling of words in text x using TextBlob."""
    text = _clean_input(x)
    words = []
    for word in text.split():
        w = Word(word)
        words.append(str(w.correct()))
    return ' '.join(words)

def get_noun_phrase(x):
    """Extracts noun phrases from text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return []
    blob = TextBlob(text)
    return list(blob.noun_phrases)

def n_grams(x, n=2):
    """Generates n-grams from text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return []
    return list(TextBlob(text).ngrams(n))

def singularize_words(x):
    """Converts plural nouns (NNS) to singular in text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return ""
    blob = TextBlob(text)
    return ' '.join([word.singularize() if tag in ['NNS'] else word for word, tag in blob.tags])

def pluralize_words(x):
    """Converts singular nouns (NN) to plural in text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return ""
    blob = TextBlob(text)
    return ' '.join([word.pluralize() if tag in ['NN'] else word for word, tag in blob.tags])

def sentiment_analysis(x):
    """Performs sentiment analysis on text x using TextBlob NaiveBayesAnalyzer."""
    text = _clean_input(x)
    if not text:
        return "pos"
    analyzer = _get_naive_bayes_analyzer()
    return TextBlob(text, analyzer=analyzer).sentiment.classification