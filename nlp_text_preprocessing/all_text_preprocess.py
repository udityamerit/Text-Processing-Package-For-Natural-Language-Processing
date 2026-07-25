# All text preprocess code is here

import os
import json
import re
import math
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
        if math.isnan(x):
            return ""
    try:
        import pandas as pd
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

# EMOJI REGEX PATTERN & MAPPING
EMOJI_PATTERN = r'[\U00010000-\U0010ffff\u2600-\u27ff\u2300-\u23ff\u2B00-\u2BFF\u2190-\u21FF]'
EMOJI_NAME_MAP = {
    '😂': 'face_with_tears_of_joy',
    '❤️': 'red_heart',
    '😍': 'smiling_face_with_heart_eyes',
    '🤣': 'rolling_on_the_floor_laughing',
    '😊': 'smiling_face_with_smiling_eyes',
    '😭': 'loudly_crying_face',
    '🙏': 'folded_hands',
    '😘': 'face_blowing_a_kiss',
    '👍': 'thumbs_up',
    '🥺': 'pleading_face',
    '🔥': 'fire',
    '😁': 'beaming_face_with_smiling_eyes',
    '🎉': 'party_popper',
    '😎': 'smiling_face_with_sunglasses',
    '✨': 'sparkles',
    '💯': 'hundred_points',
}

PROFANITY_WORDS = {'damn', 'hell', 'crap', 'bastard', 'bitch', 'fuck', 'shit', 'asshole', 'dick'}

EMOTION_LEXICON = {
    'joy': {'happy', 'joy', 'love', 'delight', 'wonderful', 'fantastic', 'great', 'glad', 'excited', 'blessed', 'cheerful', 'amazing'},
    'anger': {'angry', 'rage', 'hate', 'furious', 'annoyed', 'mad', 'hostile', 'disgusted', 'outraged', 'irritated'},
    'sadness': {'sad', 'depressed', 'unhappy', 'sorrow', 'grief', 'miserable', 'heartbroken', 'gloomy', 'disappointed'},
    'fear': {'fear', 'scared', 'afraid', 'terrified', 'anxious', 'panic', 'nervous', 'frightened', 'dread'}
}

TOXICITY_WORDS = {'hate', 'stupid', 'idiot', 'fool', 'loser', 'kill', 'ugly', 'trash', 'dumb', 'scum', 'nasty', 'horrible'}


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

def sentence_count(x):
    """Returns the number of sentences in text x."""
    text = _clean_input(x)
    if not text:
        return 0
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

def avg_sentence_len(x):
    """Returns average sentence length in words."""
    sc = sentence_count(x)
    return word_count(x) / sc if sc > 0 else 0.0

def punctuation_density(x):
    """Returns proportion of punctuation characters in text x."""
    text = _clean_input(x)
    if not text:
        return 0.0
    punct = len(re.findall(r'[^\w\s]', text))
    return punct / len(text)

def capital_ratio(x):
    """Returns proportion of uppercase words relative to total words."""
    wc = word_count(x)
    return upper_case_count(x) / wc if wc > 0 else 0.0

def unique_word_ratio(x):
    """Returns proportion of unique words relative to total words."""
    text = _clean_input(x)
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0

def readability_score(x):
    """Calculates Flesch Reading Ease score for text x."""
    text = _clean_input(x)
    wc = word_count(x)
    sc = sentence_count(x)
    if wc == 0 or sc == 0:
        return 0.0
    vowels = len(re.findall(r'[aeiouAEIOU]', text))
    syllables = max(vowels, wc)
    score = 206.835 - (1.015 * (wc / sc)) - (84.6 * (syllables / wc))
    return round(score, 2)

def text_stats(x):
    """Returns a dictionary of all summary feature stats for text x."""
    return {
        'word_count': word_count(x),
        'char_count': char_count(x),
        'avg_word_len': avg_word_len(x),
        'sentence_count': sentence_count(x),
        'avg_sentence_len': avg_sentence_len(x),
        'stop_words_count': stop_words_count(x),
        'hashtags_count': hashtags_count(x),
        'mentions_count': mentions_count(x),
        'numerics_count': numerics_count(x),
        'upper_case_count': upper_case_count(x),
        'punctuation_density': punctuation_density(x),
        'capital_ratio': capital_ratio(x),
        'unique_word_ratio': unique_word_ratio(x),
        'readability_score': readability_score(x)
    }

def text_stats_batch(df, column):
    """Calculates a DataFrame of feature statistics for a pandas DataFrame column."""
    import pandas as pd
    return df[column].apply(text_stats).apply(pd.Series)

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

def remove_emojis(x):
    """Removes emoji characters from text x."""
    text = _clean_input(x)
    return re.sub(EMOJI_PATTERN, '', text)

def extract_emojis(x):
    """Returns a list of emojis present in text x."""
    text = _clean_input(x)
    return re.findall(EMOJI_PATTERN, text)

def emoji_to_text(x):
    """Converts emojis in text x to descriptive text representations."""
    text = _clean_input(x)
    for char, name in EMOJI_NAME_MAP.items():
        text = text.replace(char, f" {name} ")
    return remove_extra_whitespace(text)

def remove_phone_numbers(x):
    """Removes phone numbers from text x."""
    text = _clean_input(x)
    pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    return re.sub(pattern, '', text)

def count_phone_numbers(x):
    """Counts phone numbers in text x."""
    text = _clean_input(x)
    pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    return len(re.findall(pattern, text))

def mask_pii(x):
    """Redacts emails, phone numbers, and card/ID numbers with [REDACTED]."""
    text = _clean_input(x)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[REDACTED]', text)
    return text

def remove_extra_whitespace(x):
    """Collapses consecutive spaces, tabs, and newlines into a single space."""
    text = _clean_input(x)
    return re.sub(r'\s+', ' ', text).strip()

def expand_hashtags(x):
    """Splits camelCase or PascalCase hashtags (#MachineLearning -> Machine Learning)."""
    text = _clean_input(x)
    def _split_tag(match):
        tag = match.group(1)
        split = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        return split
    return re.sub(r'#(\w+)', _split_tag, text)

def normalize_unicode_punctuation(x):
    """Replaces smart quotes, em-dashes, and ellipsis with ASCII equivalents."""
    text = _clean_input(x)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-').replace('…', '...')
    return text

def remove_numbers(x):
    """Strips all numeric digits from text x."""
    text = _clean_input(x)
    return re.sub(r'\d+', '', text)

def remove_currency_symbols(x):
    """Strips common currency symbols ($, €, ₹, £, ¥) from text x."""
    text = _clean_input(x)
    return re.sub(r'[$€₹£¥฿₱₩₫]', '', text)

def normalize_whitespace_and_casing(x):
    """Canonical combination of lowercasing and extra whitespace removal."""
    text = _clean_input(x)
    return remove_extra_whitespace(to_lower_case(text))

# Linguistic and Morphological Processing

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

def extract_entities(x):
    """Extracts named entities (text, label) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    nlp_model = get_spacy_nlp()
    doc = nlp_model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_pos_tags(x):
    """Extracts POS tag breakdown (text, pos, tag) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    nlp_model = get_spacy_nlp()
    doc = nlp_model(text)
    return [(token.text, token.pos_, token.tag_) for token in doc]

def get_dependency_tree(x):
    """Extracts dependency parse tree (token, dep, head) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    nlp_model = get_spacy_nlp()
    doc = nlp_model(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

def detect_language(x):
    """Detects language code of text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return "en"
    try:
        return str(TextBlob(text).detect_language())
    except Exception:
        return "en"

def keyword_extraction(x, top_n=5):
    """Extracts top_n keyword tokens based on term frequency."""
    text = _clean_input(x)
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) if w.lower() not in sw]
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    sorted_words = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]

def text_similarity(x1, x2):
    """Calculates Jaccard token similarity between text x1 and x2."""
    words1 = set(to_lower_case(_clean_input(x1)).split())
    words2 = set(to_lower_case(_clean_input(x2)).split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return round(len(intersection) / max(len(union), 1), 4)

def get_synonyms(word):
    """Returns a list of synonyms for a word using NLTK WordNet."""
    try:
        from nltk.corpus import wordnet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)
    except Exception:
        return []

def get_antonyms(word):
    """Returns a list of antonyms for a word using NLTK WordNet."""
    try:
        from nltk.corpus import wordnet
        antonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name().replace('_', ' '))
        return list(antonyms)
    except Exception:
        return []

def chunk_noun_phrases(x):
    """Extracts noun phrases with character start/end spans."""
    text = _clean_input(x)
    if not text:
        return []
    blob = TextBlob(text)
    res = []
    for np_item in blob.noun_phrases:
        start = text.find(np_item)
        end = start + len(np_item) if start != -1 else len(text)
        res.append({'phrase': np_item, 'start': start, 'end': end})
    return res

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

# Safety and Moderation

def detect_profanity(x):
    """Returns True if text x contains profanity words."""
    text = _clean_input(x).lower()
    words = set(re.findall(r'\b\w+\b', text))
    return len(words.intersection(PROFANITY_WORDS)) > 0

def censor_profanity(x):
    """Censors profanity words in text x with asterisks (****)."""
    text = _clean_input(x)
    words = text.split()
    res = []
    for w in words:
        clean_w = re.sub(r'[^\w]', '', w).lower()
        if clean_w in PROFANITY_WORDS:
            res.append('*' * len(w))
        else:
            res.append(w)
    return ' '.join(res)

def detect_toxicity(x):
    """Calculates a baseline toxicity score (0.0 to 1.0) based on insult/hate keywords."""
    text = _clean_input(x).lower()
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    toxic_count = sum(1 for w in words if w in TOXICITY_WORDS)
    return round(toxic_count / len(words), 4)

def detect_spam_signals(x):
    """Detects spam signals (excessive caps, punctuation, or URL density)."""
    return {
        'excessive_caps': capital_ratio(x) > 0.4,
        'excessive_punctuation': punctuation_density(x) > 0.2,
        'has_url': count_urls(x) > 0,
        'is_spam_candidate': capital_ratio(x) > 0.4 or punctuation_density(x) > 0.2 or count_urls(x) > 2
    }

# Sentiment & Classification

def sentiment_analysis(x):
    """Performs sentiment analysis on text x using TextBlob NaiveBayesAnalyzer."""
    text = _clean_input(x)
    if not text:
        return "pos"
    analyzer = _get_naive_bayes_analyzer()
    return TextBlob(text, analyzer=analyzer).sentiment.classification

def polarity_subjectivity(x):
    """Returns pattern-based sentiment polarity (-1.0 to 1.0) and subjectivity (0.0 to 1.0)."""
    text = _clean_input(x)
    blob = TextBlob(text)
    return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}

def emotion_detection(x):
    """Calculates lexicon-based emotion counts (joy, anger, sadness, fear)."""
    text = _clean_input(x).lower()
    words = re.findall(r'\b\w+\b', text)
    res = {'joy': 0, 'anger': 0, 'sadness': 0, 'fear': 0}
    for w in words:
        for emotion, word_set in EMOTION_LEXICON.items():
            if w in word_set:
                res[emotion] += 1
    return res

# Pipeline Automation Engine

_REGISTERED_STEPS = {}

def register_step(name):
    """
    Decorator to register a custom function as a pipeline step.
    """
    def decorator(func):
        _REGISTERED_STEPS[name] = func
        return func
    return decorator

class Pipeline:
    """
    Pipeline engine to chain preprocessing and cleaning steps.
    """
    def __init__(self, steps=None):
        self.steps = steps or []

    def add_step(self, step):
        self.steps.append(step)
        return self

    def run(self, text):
        res = text
        for step in self.steps:
            if isinstance(step, str):
                if step in _REGISTERED_STEPS:
                    func = _REGISTERED_STEPS[step]
                elif step in globals() and callable(globals()[step]):
                    func = globals()[step]
                else:
                    raise ValueError(f"Pipeline step '{step}' is not available.")
            elif callable(step):
                func = step
            else:
                raise TypeError(f"Invalid pipeline step type: {type(step)}")
            res = func(res)
        return res

    def run_batch(self, texts):
        return [self.run(t) for t in texts]

    def run_series(self, series):
        return series.apply(self.run)

    @classmethod
    def default(cls):
        return cls(['to_lower_case', 'contraction_to_expansion', 'remove_emails', 'remove_urls', 'remove_html_tag', 'remove_special_chars', 'remove_extra_whitespace'])

def diff_report(before_list, after_list):
    """
    Generates a summary diff report comparing text before and after pipeline runs.
    """
    total_items = len(before_list)
    chars_before = sum(len(_clean_input(t)) for t in before_list)
    chars_after = sum(len(_clean_input(t)) for t in after_list)
    words_before = sum(len(_clean_input(t).split()) for t in before_list)
    words_after = sum(len(_clean_input(t).split()) for t in after_list)
    emails_removed = sum(count_emails(t) for t in before_list) - sum(count_emails(t) for t in after_list)
    urls_removed = sum(count_urls(t) for t in before_list) - sum(count_urls(t) for t in after_list)

    return {
        'total_items': total_items,
        'chars_before': chars_before,
        'chars_after': chars_after,
        'char_reduction_pct': round((1 - chars_after / max(chars_before, 1)) * 100, 2),
        'words_before': words_before,
        'words_after': words_after,
        'word_reduction_pct': round((1 - words_after / max(words_before, 1)) * 100, 2),
        'emails_removed': max(emails_removed, 0),
        'urls_removed': max(urls_removed, 0)
    }