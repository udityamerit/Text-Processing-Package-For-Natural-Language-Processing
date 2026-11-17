# All text preprocess code is here
#
# Design notes
# ------------
# * Importing this module is cheap: heavy libraries (spaCy, TextBlob, NLTK,
#   matplotlib, wordcloud, BeautifulSoup, pandas) are imported lazily, only
#   when a function that needs them is called.
# * Nothing is downloaded or installed behind your back. Run
#   ``download_nltk_packages()`` once, or ``check_resources()`` to see what is
#   missing.

import os
import sys
import json
import re
import math
import warnings
import unicodedata

fpath = os.path.join(os.path.dirname(__file__), 'data', 'contractions.json')
try:
    with open(fpath, encoding='utf-8') as f:
        contractions = json.load(f)
except Exception as e:
    warnings.warn(f"Could not load contractions.json from {fpath}: {e}")
    contractions = {}

DEFAULT_SPACY_MODEL = 'en_core_web_sm'
AUTO_DOWNLOAD_ENV = 'NLP_TEXT_PREPROCESSING_AUTO_DOWNLOAD'


# ---------------------------------------------------------------------------
# Lazy dependency helpers
# ---------------------------------------------------------------------------

def _missing_dependency(package, extra=None, purpose=None):
    hint = f"pip install {package}"
    if extra:
        hint = f"pip install 'nlp-text-preprocessing[{extra}]'  (or: pip install {package})"
    what = f" for {purpose}" if purpose else ""
    return ImportError(f"'{package}' is required{what} but is not installed. Install it with: {hint}")


def _import_spacy():
    try:
        import spacy
        return spacy
    except ImportError as e:
        raise _missing_dependency('spacy', purpose='linguistic features') from e


def _import_textblob():
    try:
        import textblob
        return textblob
    except ImportError as e:
        raise _missing_dependency('textblob', purpose='this function') from e


# ---------------------------------------------------------------------------
# Resource management (spaCy model + NLTK corpora)
# ---------------------------------------------------------------------------

# resource name -> (list of acceptable nltk.data.find() paths, why it is needed)
# Both the unpacked and the zipped location are accepted: NLTK >= 3.9 keeps
# some corpora (e.g. wordnet) only as .zip, which a plain find('corpora/wordnet')
# reports as missing even though it loads fine.
NLTK_RESOURCES = {
    'punkt_tab': (['tokenizers/punkt_tab'], 'TextBlob tokenization (noun phrases, tags, sentiment)'),
    'wordnet': (['corpora/wordnet', 'corpora/wordnet.zip'], 'get_synonyms / get_antonyms'),
    'omw-1.4': (['corpora/omw-1.4', 'corpora/omw-1.4.zip'], 'WordNet lookups on recent NLTK versions'),
    'movie_reviews': (['corpora/movie_reviews', 'corpora/movie_reviews.zip'],
                      "sentiment_analysis(method='naive_bayes') only"),
}
# Resources installed by download_nltk_packages() by default.
_DEFAULT_NLTK_DOWNLOADS = ('punkt_tab', 'wordnet', 'omw-1.4')


def _nltk_has(resource):
    try:
        import nltk
    except ImportError:
        return False
    for path in NLTK_RESOURCES[resource][0]:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False


def check_resources(include_optional=False, verbose=False):
    """
    Reports which models / corpora are missing (nothing is downloaded).

    Returns a dict ``{'spacy_model': bool, 'nltk': {name: bool}}``. By default
    the optional ``movie_reviews`` corpus (only used by the legacy Naive Bayes
    sentiment method) is not included.
    """
    try:
        import spacy.util
        spacy_ok = spacy.util.is_package(DEFAULT_SPACY_MODEL)
    except ImportError:
        spacy_ok = False
    names = list(_DEFAULT_NLTK_DOWNLOADS) + (['movie_reviews'] if include_optional else [])
    report = {'spacy_model': spacy_ok, 'nltk': {n: _nltk_has(n) for n in names}}
    if verbose:
        print(f"spaCy model '{DEFAULT_SPACY_MODEL}': {'OK' if spacy_ok else 'MISSING'}")
        for n, ok in report['nltk'].items():
            print(f"NLTK '{n}' ({NLTK_RESOURCES[n][1]}): {'OK' if ok else 'MISSING'}")
        if not spacy_ok or not all(report['nltk'].values()):
            print("Run nlp_text_preprocessing.download_nltk_packages() to install what is missing.")
    return report


def download_nltk_packages(spacy_model=DEFAULT_SPACY_MODEL, include_sentiment_corpus=False):
    """
    One-time setup: downloads the NLTK data this package needs and, unless
    ``spacy_model=None``, the spaCy model.

    Set ``include_sentiment_corpus=True`` to also fetch ``movie_reviews``
    (only needed for ``sentiment_analysis(method='naive_bayes')``).
    """
    import nltk
    names = list(_DEFAULT_NLTK_DOWNLOADS) + (['movie_reviews'] if include_sentiment_corpus else [])
    for package in names:
        if not _nltk_has(package):
            nltk.download(package)
    if spacy_model:
        spacy = _import_spacy()
        import spacy.util
        if not spacy.util.is_package(spacy_model):
            from spacy.cli import download
            download(spacy_model)


def _require_nltk(resource):
    if not _nltk_has(resource):
        raise LookupError(
            f"NLTK resource '{resource}' is not installed (needed for {NLTK_RESOURCES[resource][1]}). "
            "Run: python -c \"import nlp_text_preprocessing as tp; tp.download_nltk_packages()\""
        )


# spaCy models, cached per model name.
_nlp_cache = {}


def get_spacy_nlp(model_name=DEFAULT_SPACY_MODEL, auto_download=None):
    """
    Lazy-loads and caches a spaCy language model (one cache entry per name).

    If the model is not installed a clear ``OSError`` explains how to get it.
    The model is only downloaded automatically when ``auto_download=True`` or
    the environment variable NLP_TEXT_PREPROCESSING_AUTO_DOWNLOAD=1 is set.
    """
    if model_name in _nlp_cache:
        return _nlp_cache[model_name]
    spacy = _import_spacy()
    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        if auto_download is None:
            auto_download = os.environ.get(AUTO_DOWNLOAD_ENV, '') == '1'
        if not auto_download:
            raise OSError(
                f"spaCy model '{model_name}' is not installed. Install it with:\n"
                f"    python -m spacy download {model_name}\n"
                "or call nlp_text_preprocessing.download_nltk_packages()."
            ) from e
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)
    _nlp_cache[model_name] = nlp
    return nlp


def _run_spacy(text, disable=(), model_name=DEFAULT_SPACY_MODEL):
    """Runs spaCy on text, raising the length limit instead of crashing on long documents."""
    nlp = get_spacy_nlp(model_name)
    if len(text) >= nlp.max_length:
        nlp.max_length = len(text) + 1
    if disable:
        present = [c for c in disable if c in nlp.pipe_names]
        with nlp.select_pipes(disable=present):
            return nlp(text)
    return nlp(text)


# Stop words (spaCy's English list), loaded on first use.
# Negations are kept by remove_stop_words() by default because dropping them
# flips meaning ("I do not like it" -> "like").
NEGATION_WORDS = frozenset({
    'no', 'not', 'nor', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither',
    "n't", 'cannot', 'without', 'n‘t', 'n’t',
})
_stop_words_cache = None


def get_stop_words():
    """Returns the English stop-word set (spaCy's list, loaded lazily)."""
    global _stop_words_cache
    if _stop_words_cache is None:
        try:
            from spacy.lang.en.stop_words import STOP_WORDS
        except ImportError as e:
            raise _missing_dependency('spacy', purpose='the stop-word list') from e
        _stop_words_cache = frozenset(STOP_WORDS)
    return _stop_words_cache


# Lazy-loaded NaiveBayesAnalyzer instance (legacy sentiment method)
_naive_bayes_analyzer = None


def _get_naive_bayes_analyzer():
    global _naive_bayes_analyzer
    if _naive_bayes_analyzer is None:
        _require_nltk('movie_reviews')
        _import_textblob()
        from textblob.sentiments import NaiveBayesAnalyzer
        _naive_bayes_analyzer = NaiveBayesAnalyzer()
    return _naive_bayes_analyzer


def _clean_input(x):
    """
    Helper function to safely handle non-string, None, or NaN inputs.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode('utf-8', 'replace')
    if isinstance(x, float) and math.isnan(x):
        return ""
    # Only consult pandas if the caller already imported it (no per-call import cost).
    pd = sys.modules.get('pandas')
    if pd is not None:
        try:
            r = pd.isna(x)
            if getattr(r, 'ndim', 0) == 0 and r:
                return ""
        except Exception:
            pass
    return str(x)

# EMOJI REGEX PATTERN & MAPPING
# Kept for backwards compatibility only: the emoji functions below use the
# `emoji` package, which knows the real Unicode emoji set (flags, skin tones,
# ZWJ sequences, keycaps) instead of this coarse code-point range.
EMOJI_PATTERN = r'[\U00010000-\U0010ffff☀-⟿⌀-⏿⬀-⯿←-⇿]'
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


# Compiled patterns shared by the cleaning / counting helpers
_EMAIL_RE = re.compile(
    r'(?<![\w.+%-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b')
# A URL must start with a scheme or "www." (so "httpd" or "http" in prose is left alone)
# and may not end in sentence punctuation.
_URL_RE = re.compile(
    r'(?:https?://|ftp://|www\.)[^\s<>"\']*[^\s<>"\'.,;:!?)\]}]', re.IGNORECASE)
_RT_RE = re.compile(r'\bRT\s+@\w+:?')
_MENTION_RE = re.compile(r'(?<![\w@])@\w+')
_HASHTAG_RE = re.compile(r'(?<![\w#])#\w+')
_PHONE_RE = re.compile(
    r'(?<![\w.])\+?(?:\(\d{1,4}\)|\d{1,4})(?:[\s.-]?(?:\(\d{1,4}\)|\d{1,6})){1,6}(?![\w:])')
_CARD_RE = re.compile(r'(?<![\w.])\d(?:[ -]?\d){12,18}(?![\w])')
_SSN_RE = re.compile(r'(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)')
_NUMBER_RE = re.compile(r'\b\d+(?:[.,]\d+)*\b')
_ISO_DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
_BLOCK_TAGS = ('p', 'div', 'br', 'li', 'ul', 'ol', 'tr', 'table', 'h1', 'h2', 'h3',
               'h4', 'h5', 'h6', 'blockquote', 'pre', 'section', 'article', 'hr')


def _is_phone(candidate):
    digits = re.sub(r'\D', '', candidate)
    if not 10 <= len(digits) <= 15:
        return False
    if _ISO_DATE_RE.search(candidate):
        return False
    if re.fullmatch(r'\d+\.\d+', candidate.strip()):   # a decimal number, not a phone
        return False
    return True


def _luhn_ok(digits):
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        flip = not flip
    return total % 10 == 0


def _is_card(candidate):
    digits = re.sub(r'\D', '', candidate)
    if not 13 <= len(digits) <= 19:
        return False
    if re.fullmatch(r'\d{4}([ -]?)\d{4}\1\d{4}\1\d{4}', candidate):
        return True            # classic 4-4-4-4 layout
    return _luhn_ok(digits)


def _import_emoji():
    try:
        import emoji
        return emoji
    except ImportError as e:
        raise _missing_dependency('emoji', purpose='emoji handling') from e


def _syllables(word):
    w = re.sub(r'[^a-z]', '', word.lower())
    if not w:
        return 1
    groups = len(re.findall(r'[aeiouy]+', w))
    if w.endswith('e') and not w.endswith('le') and groups > 1:
        groups -= 1                                      # silent final "e"
    return max(1, groups)


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

def _is_stop_word(token, stop_words):
    core = token.lower().strip('.,;:!?"“”()[]{}*')
    return core in stop_words

def stop_words_count(x):
    """Returns the count of stopwords in text x (case-insensitive, ignores attached punctuation)."""
    text = _clean_input(x)
    stop_words = get_stop_words()
    return sum(1 for word in text.split() if _is_stop_word(word, stop_words))

def hashtags_count(x):
    """Returns count of hashtags (#tag) in text x."""
    return len(_HASHTAG_RE.findall(_clean_input(x)))

def mentions_count(x):
    """Returns count of user mentions (@user) in text x (email addresses are not mentions)."""
    return len(_MENTION_RE.findall(_clean_input(x)))

def numerics_count(x):
    """Returns count of standalone numeric tokens in text x."""
    text = _clean_input(x)
    return len(re.findall(r'\b\d+\b', text))

def upper_case_count(x):
    """Returns count of fully upper-case words (2+ characters, so "I" and "A" are ignored)."""
    text = _clean_input(x)
    return len([word for word in text.split() if len(word) > 1 and word.isupper()])

def sentence_count(x):
    """Returns the number of sentences in text x (a '.' inside 3.14 or e.g.com does not split)."""
    text = _clean_input(x)
    if not text.strip():
        return 0
    sentences = re.split(r'[.!?]+(?=\s|$)', text)
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
    """Returns proportion of upper-case words relative to total words."""
    wc = word_count(x)
    return upper_case_count(x) / wc if wc > 0 else 0.0

def unique_word_ratio(x):
    """Returns proportion of unique words relative to total words."""
    text = _clean_input(x)
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0.0

def readability_score(x):
    """Calculates the Flesch Reading Ease score (syllables estimated per word)."""
    text = _clean_input(x)
    words = text.split()
    wc = len(words)
    sc = sentence_count(text)
    if wc == 0 or sc == 0:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
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
    try:
        import pandas as pd
    except ImportError as e:
        raise _missing_dependency('pandas', extra='pandas', purpose='text_stats_batch') from e
    return df[column].apply(text_stats).apply(pd.Series)

# Preprocessing and Cleaning

def to_lower_case(x):
    """Converts text x to lowercase."""
    text = _clean_input(x)
    return text.lower()

def contraction_to_expansion(x):
    """
    Expands contractions / slang in text x using the contractions dictionary.

    Whitespace and surrounding punctuation are preserved ("can't." -> "cannot."),
    curly apostrophes are understood, and a leading capital is kept.
    """
    text = _clean_input(x)

    def _expand(match):
        token = match.group(0)
        lead = re.match(r'^[^\w]*', token).group(0)
        trail = re.search(r'[^\w]*$', token).group(0)
        core = token[len(lead):len(token) - len(trail)] if trail else token[len(lead):]
        if not core:
            return token
        key = core.replace('’', "'").replace('‘', "'").lower()
        expansion = contractions.get(key)
        if expansion is None:
            return token
        if core[0].isupper() and not core.isupper():
            expansion = expansion[:1].upper() + expansion[1:]
        return lead + expansion + trail

    return re.sub(r'\S+', _expand, text)

def remove_emails(x):
    """Removes email addresses from text x."""
    return _EMAIL_RE.sub('', _clean_input(x))

def count_emails(x):
    """Returns count of email addresses in text x."""
    return len(_EMAIL_RE.findall(_clean_input(x)))

def count_urls(x):
    """Returns count of URLs (http://, https://, ftp://, www.) in text x."""
    return len(_URL_RE.findall(_clean_input(x)))

def remove_urls(x):
    """Removes URLs from text x (trailing sentence punctuation is kept)."""
    return _URL_RE.sub('', _clean_input(x))

def count_rt(x):
    """Returns count of retweets (RT @user) in text x."""
    return len(_RT_RE.findall(_clean_input(x)))

def remove_rt(x):
    """Removes retweet prefixes (RT @user:) from text x."""
    return _RT_RE.sub('', _clean_input(x))

def remove_html_tag(x):
    """Strips HTML tags from text x using BeautifulSoup (script/style content is dropped)."""
    text = _clean_input(x)
    if not text:
        return ""
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise _missing_dependency('beautifulsoup4', purpose='remove_html_tag') from e
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # bs4 warns when text merely looks like a URL/filename
        soup = BeautifulSoup(text, 'lxml')
        for tag in soup(['script', 'style']):
            tag.decompose()
        # Block-level tags separate words ("<p>a</p><p>b</p>" -> "a b", not "ab").
        for tag in soup.find_all(_BLOCK_TAGS):
            tag.insert_before(' ')
            tag.insert_after(' ')
        return soup.get_text().strip()

def remove_accented_chars(x):
    """
    Removes accents (é -> e) while keeping non-Latin scripts intact.

    Only combining marks are dropped; Hindi, Chinese, Arabic, etc. are not erased.
    """
    text = _clean_input(x)
    decomposed = unicodedata.normalize('NFKD', text)
    # Only Latin-style diacritics (U+0300-U+036F); Indic vowel signs / viramas etc. must stay.
    stripped = ''.join(ch for ch in decomposed if not '̀' <= ch <= 'ͯ')
    return unicodedata.normalize('NFC', stripped)

def remove_mentions(x):
    """Removes user mentions (@user) from text x; email addresses are left alone."""
    return _MENTION_RE.sub('', _clean_input(x)).strip()

def remove_special_chars(x):
    """
    Removes special/punctuation characters from text x.

    Hyphens and slashes between words become spaces ("well-known" -> "well known")
    instead of gluing the words together; apostrophes are dropped ("don't" -> "dont").
    """
    text = _clean_input(x)
    text = re.sub(r'(?<=\w)[-/](?=\w)', ' ', text)
    return re.sub(r'[^\w\s]', '', text)

def remove_repeated_chars(x):
    """
    Shortens elongated words: 3+ repeats of a letter become 2 ("soooo" -> "soo").

    Digits, spaces and punctuation are untouched ("1000000" and "..." stay as they are).
    """
    text = _clean_input(x)
    return re.sub(r'([^\W\d_])\1{2,}', r'\1\1', text)

def remove_stop_words(x, keep_negations=True):
    """
    Removes stop words from text x (case-insensitive, punctuation-aware).

    Negations ("not", "no", "never", ...) are kept by default because removing
    them flips the meaning ("I do not like it" must not become "like").
    Pass ``keep_negations=False`` to drop them too.
    """
    text = _clean_input(x)
    stop_words = get_stop_words()
    kept = []
    for word in text.split():
        core = word.lower().strip('.,;:!?"“”()[]{}*')
        if core in stop_words and not (keep_negations and core in NEGATION_WORDS):
            continue
        kept.append(word)
    return " ".join(kept)

def remove_emojis(x):
    """Removes emoji (incl. flags, skin tones and ZWJ sequences) from text x."""
    def _drop(chars, data):
        # (c), (r) and (tm) are in the emoji set but are ordinary text symbols.
        return chars if chars.replace('️', '') in ('©', '®', '™') else ''
    return _import_emoji().replace_emoji(_clean_input(x), replace=_drop)

def extract_emojis(x):
    """Returns a list of emojis present in text x."""
    return [m['emoji'] for m in _import_emoji().emoji_list(_clean_input(x))]

def emoji_to_text(x):
    """Converts emojis in text x to their names ("😂" -> "face_with_tears_of_joy")."""
    text = _import_emoji().demojize(_clean_input(x), delimiters=(' ', ' '))
    return remove_extra_whitespace(text)

def _phone_sub(text, repl):
    return _PHONE_RE.sub(lambda m: repl if _is_phone(m.group(0)) else m.group(0), text)

def remove_phone_numbers(x):
    """Removes phone numbers (international formats included) from text x."""
    return _phone_sub(_clean_input(x), '')

def count_phone_numbers(x):
    """Counts phone numbers in text x."""
    return sum(1 for m in _PHONE_RE.finditer(_clean_input(x)) if _is_phone(m.group(0)))

def mask_pii(x, token='[REDACTED]'):
    """Redacts emails, card numbers, SSNs and phone numbers, replacing each with ``token``."""
    text = _clean_input(x)
    text = _EMAIL_RE.sub(token, text)
    text = _CARD_RE.sub(lambda m: token if _is_card(m.group(0)) else m.group(0), text)
    text = _SSN_RE.sub(token, text)
    return _phone_sub(text, token)

def remove_extra_whitespace(x):
    """Collapses consecutive spaces, tabs, and newlines into a single space."""
    text = _clean_input(x)
    return re.sub(r'\s+', ' ', text).strip()

def expand_hashtags(x):
    """Splits hashtags into words (#MachineLearning2024 -> Machine Learning 2024, #NLPIsFun -> NLP Is Fun)."""
    text = _clean_input(x)
    def _split_tag(match):
        tag = match.group(1).replace('_', ' ')
        tag = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        tag = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', tag)
        tag = re.sub(r'([A-Za-z])(\d)', r'\1 \2', tag)
        tag = re.sub(r'(\d)([A-Za-z])', r'\1 \2', tag)
        return tag
    return re.sub(r'(?<![\w#])#(\w+)', _split_tag, text)

def normalize_unicode_punctuation(x):
    """Replaces smart quotes, dashes, ellipsis and non-breaking/zero-width spaces with ASCII equivalents."""
    text = _clean_input(x)
    text = text.replace('“', '"').replace('”', '"').replace('„', '"')
    text = text.replace('‘', "'").replace('’', "'").replace('‚', "'")
    text = text.replace('—', '-').replace('–', '-').replace('−', '-').replace('…', '...')
    text = text.replace(' ', ' ').replace(' ', ' ').replace(' ', ' ')
    return re.sub('[​‌‍⁠﻿]', '', text)

def remove_numbers(x):
    """
    Removes standalone numbers ("2026", "3.14", "1,000") from text x.

    Digits inside words are kept, so "3rd", "H2O" and "COVID19" survive.
    """
    return _NUMBER_RE.sub('', _clean_input(x))

def remove_currency_symbols(x):
    """Strips currency symbols ($, €, ₹, £, ¥, ₿, ...) from text x."""
    text = _clean_input(x)
    return ''.join(ch for ch in text if unicodedata.category(ch) != 'Sc')

def normalize_whitespace_and_casing(x):
    """Canonical combination of lowercasing and extra whitespace removal."""
    text = _clean_input(x)
    return remove_extra_whitespace(to_lower_case(text))

# Linguistic and Morphological Processing

def _TextBlob(*args, **kwargs):
    return _import_textblob().TextBlob(*args, **kwargs)


def _Word(*args, **kwargs):
    return _import_textblob().Word(*args, **kwargs)


def convert_to_base(x):
    """Lemmatizes nouns and verbs in text x using spaCy while leaving other POS tags as-is."""
    text = _clean_input(x)
    if not text:
        return ""
    doc = _run_spacy(text, disable=('parser', 'ner'))
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
    doc = _run_spacy(text, disable=('parser', 'ner'))
    return ' '.join([token.lemma_ for token in doc])

def extract_entities(x):
    """Extracts named entities (text, label) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    doc = _run_spacy(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_pos_tags(x):
    """Extracts POS tag breakdown (text, pos, tag) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    doc = _run_spacy(text)
    return [(token.text, token.pos_, token.tag_) for token in doc]

def get_dependency_tree(x):
    """Extracts dependency parse tree (token, dep, head) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    doc = _run_spacy(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

def detect_language(x):
    """Detects language code of text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return "en"
    try:
        return str(_TextBlob(text).detect_language())
    except Exception:
        return "en"

def keyword_extraction(x, top_n=5):
    """Extracts top_n keyword tokens based on term frequency."""
    text = _clean_input(x)
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text) if w.lower() not in get_stop_words()]
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
    _require_nltk('wordnet')   # raises a helpful LookupError if missing
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
    _require_nltk('wordnet')   # raises a helpful LookupError if missing
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
    blob = _TextBlob(text)
    res = []
    for np_item in blob.noun_phrases:
        start = text.find(np_item)
        end = start + len(np_item) if start != -1 else len(text)
        res.append({'phrase': np_item, 'start': start, 'end': end})
    return res

def get_wordcloud(x, save_path=None, show=True):
    """Generates and optionally displays or saves a WordCloud for text x."""
    text = _clean_input(x)
    if not text.strip():
        warnings.warn("get_wordcloud received empty text; nothing to draw.")
        return None
    try:
        from wordcloud import WordCloud
    except ImportError as e:
        raise _missing_dependency('wordcloud', extra='viz', purpose='get_wordcloud') from e
    cloud = WordCloud(width=800, height=500).generate(text)
    if save_path:
        cloud.to_file(save_path)
    if show:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise _missing_dependency('matplotlib', extra='viz', purpose='get_wordcloud(show=True)') from e
        fig = plt.figure(figsize=(15, 7), dpi=100)
        plt.imshow(cloud.to_image(), interpolation='bilinear')
        plt.axis('off')
        plt.show()
        plt.close(fig)
    return cloud

def correct_spelling(x):
    """Corrects spelling of words in text x using TextBlob."""
    text = _clean_input(x)
    words = []
    for word in text.split():
        w = _Word(word)
        words.append(str(w.correct()))
    return ' '.join(words)

def get_noun_phrase(x):
    """Extracts noun phrases from text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return []
    blob = _TextBlob(text)
    return list(blob.noun_phrases)

def n_grams(x, n=2):
    """Generates n-grams from text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return []
    return list(_TextBlob(text).ngrams(n))

def singularize_words(x):
    """Converts plural nouns (NNS) to singular in text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return ""
    blob = _TextBlob(text)
    return ' '.join([word.singularize() if tag in ['NNS'] else word for word, tag in blob.tags])

def pluralize_words(x):
    """Converts singular nouns (NN) to plural in text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return ""
    blob = _TextBlob(text)
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
    return _TextBlob(text, analyzer=analyzer).sentiment.classification

def polarity_subjectivity(x):
    """Returns pattern-based sentiment polarity (-1.0 to 1.0) and subjectivity (0.0 to 1.0)."""
    text = _clean_input(x)
    blob = _TextBlob(text)
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