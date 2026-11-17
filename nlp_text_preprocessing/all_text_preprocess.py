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

# --- Word lists for the moderation helpers -------------------------------------------
# These are small, transparent English baselines, NOT a moderation system. Extend them
# with the ``extra_words`` argument, and use a trained model for anything high-stakes.
# Words are listed in their base form; plurals and endings (-s, -ed, -ing, -er, -y) and
# simple leetspeak ("sh1t", "@ss") are matched automatically.

# Strong words are always flagged. Mild words are everyday speech and can be switched
# off with ``include_mild=False``.
PROFANITY_STRONG = {
    'fuck', 'motherfucker', 'shit', 'bullshit', 'shithead', 'dipshit', 'bitch', 'bastard',
    'asshole', 'arsehole', 'asshat', 'dick', 'dickhead', 'prick', 'cunt', 'twat', 'wanker',
    'bollocks', 'slut', 'whore', 'douche', 'douchebag', 'dumbass', 'jackass',
}
PROFANITY_MILD = {'damn', 'goddamn', 'hell', 'crap', 'piss', 'ass', 'arse', 'bloody'}
PROFANITY_WORDS = PROFANITY_STRONG | PROFANITY_MILD   # kept for backward compatibility

# Phrases that contain a profane-looking word but are not profanity.
_PROFANITY_SAFE_PHRASES = re.compile(r"\bhell['’]?s\s+kitchen\b|\bhell\s+on\s+earth\b", re.IGNORECASE)

EMOTION_LEXICON = {
    'joy': {'happy', 'joy', 'joyful', 'love', 'delight', 'delighted', 'wonderful', 'fantastic', 'great',
            'glad', 'excited', 'blessed', 'cheerful', 'amazing', 'awesome', 'thrilled', 'grateful',
            'pleased', 'proud', 'fun', 'enjoy', 'celebrate', 'smile', 'laugh', 'brilliant', 'lovely'},
    'anger': {'angry', 'rage', 'hate', 'furious', 'annoyed', 'mad', 'hostile', 'disgusted', 'outraged',
              'irritated', 'annoying', 'resent', 'bitter', 'livid', 'fume', 'infuriating', 'enraged'},
    'sadness': {'sad', 'depressed', 'unhappy', 'sorrow', 'grief', 'miserable', 'heartbroken', 'gloomy',
                'disappointed', 'lonely', 'hopeless', 'cry', 'weep', 'mourn', 'upset', 'hurt', 'regret'},
    'fear': {'fear', 'scared', 'afraid', 'terrified', 'anxious', 'panic', 'nervous', 'frightened', 'dread',
             'worried', 'worry', 'terror', 'alarmed', 'uneasy', 'horrified', 'petrified'},
}

# Toxicity: insults score on their own at half weight and at full weight when they are
# aimed at a person; threat verbs only count when aimed at a person ("kill the process"
# is fine); a negation just before a word cancels it ("I do not hate you").
TOXICITY_INSULTS = {
    'idiot', 'stupid', 'moron', 'dumb', 'dumbass', 'loser', 'scum', 'fool', 'imbecile', 'cretin',
    'jerk', 'creep', 'lowlife', 'incompetent', 'clueless', 'brainless',
}
TOXICITY_THREATS = {'kill', 'murder', 'die', 'hurt', 'punch', 'stab', 'hate', 'harm'}
# Words that are only an insult when aimed at someone ("take out the trash" is fine).
TOXICITY_DESCRIPTIVE = {'ugly', 'nasty', 'horrible', 'disgusting', 'pathetic', 'worthless', 'trash', 'garbage'}
TOXICITY_WORDS = TOXICITY_INSULTS | TOXICITY_THREATS | TOXICITY_DESCRIPTIVE   # backward compatibility
_TOXICITY_TARGETS = frozenset({
    'you', 'your', 'yours', 'yourself', "you're", 'youre', 'u', 'ur', 'him', 'her', 'them', 'he', 'she',
    'they', 'his', 'hers', 'their', 'everyone', 'all', 'people', 'guys',
})

_SPAM_PHRASES = (
    'buy now', 'click here', 'free money', 'act now', 'limited time', 'you have won', 'you won',
    'winner', 'congratulations', '100% free', 'risk free', 'no obligation', 'order now', 'cash prize',
    'make money fast', 'work from home', 'double your', 'claim your',
)


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


def _join_tokens(doc, attr='lemma_', only_pos=None):
    """Rebuilds text from a spaCy doc keeping the original spacing (no "word ." artefacts)."""
    parts = []
    for token in doc:
        if only_pos is None or token.pos_ in only_pos:
            parts.append(getattr(token, attr) + token.whitespace_)
        else:
            parts.append(token.text_with_ws)
    return ''.join(parts)


def convert_to_base(x):
    """Lemmatizes only nouns and verbs in text x using spaCy; other words are left as-is."""
    text = _clean_input(x)
    if not text:
        return ""
    doc = _run_spacy(text, disable=('parser', 'ner'))
    return _join_tokens(doc, only_pos=('NOUN', 'VERB'))

def lemmatize(x):
    """Lemmatizes all tokens in text x using spaCy (original spacing is preserved)."""
    text = _clean_input(x)
    if not text:
        return ""
    doc = _run_spacy(text, disable=('parser', 'ner'))
    return _join_tokens(doc)

def extract_entities(x):
    """Extracts named entities (text, label) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    doc = _run_spacy(text, disable=('parser', 'lemmatizer'))
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_pos_tags(x):
    """Extracts POS tag breakdown (text, pos, tag) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    doc = _run_spacy(text, disable=('parser', 'ner', 'lemmatizer'))
    return [(token.text, token.pos_, token.tag_) for token in doc]

def get_dependency_tree(x):
    """Extracts dependency parse tree (token, dep, head) from text x using spaCy."""
    text = _clean_input(x)
    if not text:
        return []
    doc = _run_spacy(text, disable=('ner', 'lemmatizer'))
    return [(token.text, token.dep_, token.head.text) for token in doc]

def detect_language(x):
    """
    Detects the language of text x and returns its ISO 639-1 code ("en", "fr", "hi", ...).

    Uses the ``langdetect`` package (deterministic). Returns ``"unknown"`` for
    empty or undetectable text (digits only, emoji only, ...), not a made-up "en".
    """
    text = _clean_input(x)
    if not re.search(r'[^\W\d_]', text):
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError as e:
        raise _missing_dependency('langdetect', purpose='detect_language') from e
    DetectorFactory.seed = 0          # same input -> same answer on every call
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def keyword_extraction(x, top_n=5):
    """Extracts the top_n most frequent non-stop-word tokens (any alphabet, 3+ letters)."""
    text = _clean_input(x)
    stop_words = get_stop_words()
    counts = {}
    for w in re.findall(r'[^\W\d_]{3,}', text.lower()):
        if w not in stop_words:
            counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)   # stable: ties keep first-seen order
    return [w for w, _ in ranked[:top_n]]

def text_similarity(x1, x2):
    """Jaccard similarity of the word sets of x1 and x2 (case- and punctuation-insensitive)."""
    words1 = set(re.findall(r'\w+', _clean_input(x1).lower()))
    words2 = set(re.findall(r'\w+', _clean_input(x2).lower()))
    union = words1 | words2
    return round(len(words1 & words2) / len(union), 4) if union else 0.0

def get_synonyms(word):
    """Returns a sorted list of WordNet synonyms for a word (the word itself excluded)."""
    _require_nltk('wordnet')   # raises a helpful LookupError if missing
    from nltk.corpus import wordnet
    query = _clean_input(word).strip().lower()
    synonyms = set()
    for syn in wordnet.synsets(query.replace(' ', '_')):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    synonyms.discard(query)
    return sorted(synonyms)

def get_antonyms(word):
    """Returns a sorted list of WordNet antonyms for a word."""
    _require_nltk('wordnet')   # raises a helpful LookupError if missing
    from nltk.corpus import wordnet
    query = _clean_input(word).strip().lower()
    antonyms = set()
    for syn in wordnet.synsets(query.replace(' ', '_')):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace('_', ' '))
    return sorted(antonyms)

def chunk_noun_phrases(x):
    """Extracts noun phrases with exact character start/end offsets (spaCy noun chunks)."""
    text = _clean_input(x)
    if not text:
        return []
    doc = _run_spacy(text, disable=('ner', 'lemmatizer'))
    return [{'phrase': c.text, 'start': c.start_char, 'end': c.end_char} for c in doc.noun_chunks]

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

_spelling_cache = {}


def _correct_word(word):
    """Spell-corrects one lowercase alphabetic word (memoised: TextBlob is slow)."""
    if word not in _spelling_cache:
        _spelling_cache[word] = str(_Word(word).correct())
    return _spelling_cache[word]


def correct_spelling(x):
    """
    Corrects the spelling of the words in text x using TextBlob.

    Punctuation, spacing, numbers, URLs and ALL-CAPS acronyms are left untouched, and
    the original capitalisation of each word is kept.
    """
    text = _clean_input(x)

    def _fix(match):
        word = match.group(1)
        if word is None:
            return match.group(0)             # URL / email / @handle / #tag: untouched
        if len(word) < 2 or word.isupper():
            return word                       # single letters and ACRONYMS
        fixed = _correct_word(word.lower())
        return fixed.capitalize() if word[0].isupper() else fixed

    # Only whole alphabetic tokens are corrected: "3rd", "H2O" and "don't" are skipped.
    pattern = r"(?:https?://|www\.)\S+|\S+@\S+|[@#]\w+|(?<![\w'’])([^\W\d_]+)(?![\w'’])"
    return re.sub(pattern, _fix, text)

def get_noun_phrase(x):
    """Extracts noun phrases from text x using TextBlob."""
    text = _clean_input(x)
    if not text:
        return []
    blob = _TextBlob(text)
    return list(blob.noun_phrases)

def n_grams(x, n=2):
    """Returns the n-grams of text x as a list of word tuples (punctuation ignored)."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    words = re.findall(r"\w+(?:['’]\w+)*", _clean_input(x))
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

def _inflect(text, source_tag, method):
    """Applies a TextBlob inflection to words with the given spaCy tag, keeping spacing and case."""
    if not text:
        return ""
    doc = _run_spacy(text, disable=('parser', 'ner', 'lemmatizer'))
    parts = []
    for token in doc:
        if token.tag_ == source_tag and token.is_alpha:
            new = str(getattr(_Word(token.text.lower()), method)())
            if token.text.isupper() and len(token.text) > 1:
                new = new.upper()
            elif token.text[0].isupper():
                new = new[:1].upper() + new[1:]
            parts.append(new + token.whitespace_)
        else:
            parts.append(token.text_with_ws)
    return ''.join(parts)

def singularize_words(x):
    """Converts plural nouns (tag NNS) in text x to singular; spacing and punctuation are kept."""
    return _inflect(_clean_input(x), 'NNS', 'singularize')

def pluralize_words(x):
    """
    Converts singular common nouns (tag NN) in text x to plural; spacing and punctuation are kept.

    Note: this is a purely word-level transformation and does not fix agreement
    ("the dog is barking" -> "the dogs is barking").
    """
    return _inflect(_clean_input(x), 'NN', 'pluralize')

# Safety and Moderation

_LEET = str.maketrans({'0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's'})
_MOD_TOKEN_RE = re.compile(r"[\w@$]+(?:['’]\w+)*")


def _normalize_token(token):
    """Lower-case a token, drop a possessive 's and undo simple leetspeak ("sh1t" -> "shit")."""
    tok = token.replace('’', "'").lower()
    if tok.endswith("'s"):
        tok = tok[:-2]
    if any(c.isalpha() for c in tok) and any(c in '013457@$' for c in tok):
        tok = tok.translate(_LEET)
    return tok


def _word_forms(word):
    """The word plus the base forms it could come from ("loved" -> love, "shitty" -> shit)."""
    forms = {word}
    for suffix in ('ing', 'ed', 'er', 'ers', 'est', 'es', 's', 'd', 'ly', 'y'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[:-len(suffix)]
            forms.add(stem)
            forms.add(stem + 'e')
            if stem[-1] == stem[-2]:                 # shitting -> shitt -> shit
                forms.add(stem[:-1])
            if stem.endswith('i'):                   # happier -> happi -> happy
                forms.add(stem[:-1] + 'y')
    return forms


def _in_lexicon(token, lexicon):
    return not _word_forms(_normalize_token(token)).isdisjoint(lexicon)


def _is_negated(tokens, index, window=3):
    """True if a negation word appears in the ``window`` tokens before ``index``."""
    for prev in tokens[max(0, index - window):index]:
        low = prev.replace('’', "'").lower()
        if low in NEGATION_WORDS or low.endswith("n't"):
            return True
    return False


def _profanity_lexicon(include_mild, extra_words, allow_words):
    lexicon = set(PROFANITY_STRONG)
    if include_mild:
        lexicon |= PROFANITY_MILD
    lexicon |= {w.lower() for w in extra_words}
    return lexicon - {w.lower() for w in allow_words}


def _profane_spans(text, include_mild, extra_words, allow_words):
    lexicon = _profanity_lexicon(include_mild, extra_words, allow_words)
    safe = [m.span() for m in _PROFANITY_SAFE_PHRASES.finditer(text)]
    spans = []
    for m in _MOD_TOKEN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in safe):
            continue
        if _in_lexicon(m.group(0), lexicon):
            spans.append(m.span())
    return spans


def detect_profanity(x, include_mild=True, extra_words=(), allow_words=()):
    """
    Returns True if text x contains profanity.

    Plurals/endings ("shitty", "bitches") and simple leetspeak ("sh1t", "@sshole") are
    matched. ``include_mild=False`` ignores everyday words such as "damn" and "hell".
    ``extra_words`` adds your own words (for example slurs relevant to your community)
    and ``allow_words`` removes some. "Hell's Kitchen" is not flagged.
    """
    return bool(_profane_spans(_clean_input(x), include_mild, extra_words, allow_words))


def censor_profanity(x, include_mild=True, extra_words=(), allow_words=(), mask_char='*'):
    """Replaces each profane word with ``mask_char``; everything else is untouched."""
    text = _clean_input(x)
    for start, end in reversed(_profane_spans(text, include_mild, extra_words, allow_words)):
        text = text[:start] + mask_char * (end - start) + text[end:]
    return text


def toxicity_report(x):
    """
    Explains the toxicity score: ``{'score': float, 'matches': [...], 'ignored': [...]}``.

    Each match is ``{'word', 'weight', 'targeted'}``. Insults and strong profanity weigh
    0.5, or 1.0 when aimed at a person (you / him / them ... within 3 words). Threat words
    ("kill", "hurt", "hate") only count when aimed at a person. A negation in the 3 words
    before a match cancels it and lists it under ``ignored``.
    """
    text = _clean_input(x)
    tokens = _MOD_TOKEN_RE.findall(text)
    matches, ignored = [], []
    for i, tok in enumerate(tokens):
        in_insults = _in_lexicon(tok, TOXICITY_INSULTS)
        targeted_only = _in_lexicon(tok, TOXICITY_THREATS) or _in_lexicon(tok, TOXICITY_DESCRIPTIVE)
        in_profane = _in_lexicon(tok, PROFANITY_STRONG)
        if not (in_insults or targeted_only or in_profane):
            continue
        neighbours = tokens[max(0, i - 3):i] + tokens[i + 1:i + 4]
        targeted = any(n.replace('’', "'").lower() in _TOXICITY_TARGETS for n in neighbours)
        if targeted_only and not (in_insults or in_profane) and not targeted:
            continue
        entry = {'word': tok.lower(), 'weight': 1.0 if targeted else 0.5, 'targeted': targeted}
        (ignored if _is_negated(tokens, i) else matches).append(entry)
    total = sum(m['weight'] for m in matches)
    return {'score': round(1 - 0.5 ** total, 4), 'matches': matches, 'ignored': ignored}


def detect_toxicity(x):
    """
    Baseline toxicity score from 0.0 to 1.0 from a small lexicon (see :func:`toxicity_report`).

    One insult aimed at someone scores 0.5, and the score approaches 1.0 as hits add up.
    "kill the process" and "I do not hate you" score 0. This is a keyword baseline that
    cannot understand sarcasm or context; use a trained model for real moderation.
    """
    return toxicity_report(x)['score']


def detect_spam_signals(x):
    """
    Detects spam signals: shouting (long all-caps text), runs of punctuation, links and
    typical spam phrases ("buy now", "click here").

    Returns the individual signals plus ``spam_phrases`` (the phrases found) and
    ``is_spam_candidate``, which needs at least two signals or three links. Short
    messages such as "OK" or "Hi!" are never flagged for caps or punctuation alone.
    """
    text = _clean_input(x)
    lowered = text.lower()
    caps = word_count(text) >= 3 and capital_ratio(text) > 0.5
    punct = bool(re.search(r'[!?]{3,}', text)) or (len(text) >= 20 and punctuation_density(text) > 0.2)
    urls = count_urls(text)
    phrases = [p for p in _SPAM_PHRASES if p in lowered]
    signals = int(caps) + int(punct) + int(bool(phrases)) + int(urls > 0)
    return {
        'excessive_caps': caps,
        'excessive_punctuation': punct,
        'has_url': urls > 0,
        'spam_phrases': phrases,
        'is_spam_candidate': signals >= 2 or urls > 2 or len(phrases) >= 2,
    }

# Sentiment & Classification

def sentiment_analysis(x, method='pattern', threshold=0.05):
    """
    Classifies the sentiment of text x as ``'pos'``, ``'neg'`` or ``'neu'``.

    method='pattern'      (default) TextBlob's lexicon polarity: instant, no extra
                          downloads. Scores within +/-``threshold`` count as 'neu'.
    method='naive_bayes'  legacy TextBlob NaiveBayes model trained on movie reviews.
                          Needs the ``movie_reviews`` corpus, takes ~30 s to train on
                          first use, and only ever answers 'pos' / 'neg'.
    Empty text is 'neu' (it used to be reported as 'pos').
    """
    text = _clean_input(x)
    if method not in ('pattern', 'naive_bayes'):
        raise ValueError("method must be 'pattern' or 'naive_bayes'")
    if not text.strip():
        return "neu"
    if method == 'naive_bayes':
        analyzer = _get_naive_bayes_analyzer()
        return _TextBlob(text, analyzer=analyzer).sentiment.classification
    polarity = _TextBlob(text).sentiment.polarity
    if polarity > threshold:
        return "pos"
    if polarity < -threshold:
        return "neg"
    return "neu"

def polarity_subjectivity(x):
    """Returns pattern-based sentiment polarity (-1.0 to 1.0) and subjectivity (0.0 to 1.0)."""
    text = _clean_input(x)
    if not text.strip():
        return {'polarity': 0.0, 'subjectivity': 0.0}
    blob = _TextBlob(text)
    return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}

def emotion_detection(x):
    """
    Calculates lexicon-based emotion counts (joy, anger, sadness, fear).

    Word endings are handled ("loved", "happier", "worrying"), and an emotion word is
    ignored when a negation comes within the 2 words before it ("not happy" is not joy).
    """
    tokens = _MOD_TOKEN_RE.findall(_clean_input(x))
    res = {emotion: 0 for emotion in EMOTION_LEXICON}
    for i, tok in enumerate(tokens):
        if _is_negated(tokens, i, window=2):
            continue
        forms = _word_forms(tok.replace('’', "'").lower())
        for emotion, word_set in EMOTION_LEXICON.items():
            if not forms.isdisjoint(word_set):
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

def _resolve_step(step):
    if isinstance(step, str):
        if step in _REGISTERED_STEPS:
            return _REGISTERED_STEPS[step]
        func = globals().get(step)
        # Only public functions can be used by name (not helpers like _clean_input or classes).
        if not step.startswith('_') and callable(func) and not isinstance(func, type):
            return func
        raise ValueError(f"Pipeline step '{step}' is not available.")
    if callable(step):
        return step
    raise TypeError(f"Invalid pipeline step type: {type(step)}")

def _normalize_step(step):
    """
    Turn a step description into ``(function, kwargs, label)``.

    Accepted forms: ``'name'``, ``callable``, ``('name', {kwargs})`` and
    ``(callable, {kwargs})``. Extra keyword arguments are checked against the
    function's signature so mistakes surface when the pipeline is built.
    """
    kwargs = {}
    target = step
    if isinstance(step, (tuple, list)):
        if len(step) != 2 or not isinstance(step[1], dict):
            raise TypeError("A step with arguments must be (name_or_callable, {argument: value}).")
        target, kwargs = step[0], dict(step[1])
    func = _resolve_step(target)
    if kwargs:
        import inspect
        try:
            inspect.signature(func).bind(None, **kwargs)
        except TypeError as exc:
            raise TypeError(f"Bad arguments for pipeline step "
                            f"'{getattr(func, '__name__', func)}': {exc}") from None
    label = target if isinstance(target, str) else getattr(func, '__name__', repr(func))
    return func, kwargs, label


class Pipeline:
    """
    Pipeline engine to chain preprocessing and cleaning steps.

    A step is the name of a built-in function (``'remove_urls'``), the name of a
    function registered with :func:`register_step`, or any callable. To pass
    arguments use ``('remove_stop_words', {'keep_negations': False})``.

    Every step except the last must return a string; the last step may return
    anything (for example ``'sentiment_analysis'`` or ``'text_stats'``).

    The class follows the scikit-learn transformer convention (``fit``,
    ``transform``, ``fit_transform``, ``get_params``, ``set_params``) without
    importing scikit-learn. For use inside ``sklearn.pipeline.Pipeline`` see
    :class:`nlp_text_preprocessing.TextPreprocessor`.
    """
    def __init__(self, steps=None):
        self.steps = steps               # stored as given, so sklearn's clone() works
        self._compiled()                 # fail early on a typo, not in the middle of a run

    def _step_list(self):
        return list(self.steps) if self.steps else []

    def _compiled(self):
        return [_normalize_step(s) for s in self._step_list()]

    def __repr__(self):
        parts = []
        for func, kwargs, label in self._compiled():
            parts.append(f"{label}{kwargs!r}" if kwargs else label)
        return f"Pipeline({parts})"

    def __len__(self):
        return len(self._step_list())

    def add_step(self, step):
        _normalize_step(step)
        self.steps = self._step_list() + [step]
        return self

    @staticmethod
    def _apply(compiled, text):
        res = text
        for i, (func, kwargs, label) in enumerate(compiled):
            res = func(res, **kwargs)
            if i < len(compiled) - 1 and not isinstance(res, str):
                raise TypeError(f"Pipeline step '{label}' returned {type(res).__name__}, but only "
                                f"the last step may return a non-string.")
        return res

    def run(self, text):
        return self._apply(self._compiled(), text)

    def run_batch(self, texts):
        compiled = self._compiled()      # validate once, reuse for every text
        return [self._apply(compiled, t) for t in texts]

    def run_series(self, series):
        compiled = self._compiled()
        return series.apply(lambda t: self._apply(compiled, t))

    # scikit-learn style API
    def fit(self, X=None, y=None):
        self._compiled()
        return self

    def transform(self, X):
        return self.run_batch(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {'steps': self.steps}

    def set_params(self, **params):
        for key, value in params.items():
            if key != 'steps':
                raise ValueError(f"Invalid parameter '{key}' for Pipeline.")
            self.steps = value
        self._compiled()
        return self

    # saving and loading
    def to_config(self):
        """Return a JSON-serializable list describing the steps."""
        config = []
        for func, kwargs, label in self._compiled():
            name = label
            if not isinstance(label, str) or (label not in _REGISTERED_STEPS
                                              and globals().get(label) is not func):
                raise ValueError(f"Step '{label}' is a custom callable and cannot be saved. "
                                 f"Register it with @register_step('name') and use that name.")
            config.append({'step': name, 'args': kwargs} if kwargs else name)
        return config

    @classmethod
    def from_config(cls, config):
        steps = []
        for item in config:
            if isinstance(item, str):
                steps.append(item)
            elif isinstance(item, dict) and 'step' in item:
                steps.append((item['step'], dict(item.get('args', {}))))
            else:
                raise ValueError(f"Invalid pipeline config entry: {item!r}")
        return cls(steps)

    def save(self, path):
        import json
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'version': 1, 'steps': self.to_config()}, fh, indent=2)

    @classmethod
    def load(cls, path):
        import json
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        return cls.from_config(data['steps'])

    @classmethod
    def default(cls):
        return cls(['remove_html_tag', 'remove_urls', 'remove_emails', 'to_lower_case',
                    'contraction_to_expansion', 'remove_special_chars', 'remove_extra_whitespace'])

def diff_report(before_list, after_list):
    """
    Generates a summary diff report comparing text before and after pipeline runs.
    """
    before_list, after_list = list(before_list), list(after_list)
    if len(before_list) != len(after_list):
        raise ValueError("before_list and after_list must have the same length")
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
