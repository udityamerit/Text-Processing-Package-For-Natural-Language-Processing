# v0.3.0: bug-fix and packaging release

This release fixes a large batch of correctness bugs found in an audit, makes `import nlp_text_preprocessing` fast and side-effect free, and cleans up packaging. A few functions now behave differently, so please read **Breaking / behaviour changes** before upgrading.

## Install

```bash
pip install -U nlp-text-preprocessing            # core
pip install -U "nlp-text-preprocessing[all]"     # + pandas, wordcloud, matplotlib
```

Then run the one-time setup (downloads NLTK data and the spaCy model `en_core_web_sm`):

```python
import nlp_text_preprocessing as tp
tp.download_nltk_packages()
tp.check_resources(verbose=True)
```

## Highlights

- **Fast, quiet import.** Nothing heavy is loaded and nothing is downloaded on import. spaCy, TextBlob and NLTK data are loaded on first use and cached.
- **Optional extras.** `pandas`, `wordcloud` and `matplotlib` moved to `[pandas]`, `[viz]` and `[all]`. Dependencies that were never imported (pyaudio, SpeechRecognition, scikit-learn, numpy and others) are removed.
- **Clear errors.** Missing NLTK data or a missing spaCy model now raises an actionable message instead of a confusing failure. New helpers: `check_resources()`, `get_spacy_nlp()`, `get_stop_words()`.
- **Real emoji and language support.** Emoji functions use the `emoji` package (flags, skin tones, ZWJ sequences). `detect_language` uses `langdetect`, since TextBlob has no such method.
- **Sentiment is instant by default.** `sentiment_analysis` uses a fast `pattern` method; the old ~30 second NaiveBayes model is opt-in.
- **Safer PII masking.** `mask_pii` emits `[REDACTED]` and also masks Luhn-valid card numbers and US SSNs.
- **Many text-cleaning fixes** for URLs, emails, retweets, contractions, accents, phone numbers, numbers, hashtags and currency symbols.
- **Tests and CI.** 98 regression tests, plus a GitHub Actions workflow on Python 3.10 to 3.12.

## Breaking / behaviour changes

| Function | Change |
|---|---|
| `sentiment_analysis` | Default method is now `pattern`; returns `'pos'`, `'neg'` or `'neu'`. Use `method='naive_bayes'` for the old behaviour (needs `download_nltk_packages(include_sentiment_corpus=True)`). |
| `detect_language` | Uses `langdetect`; returns `'unknown'` when it cannot tell. |
| `remove_stop_words` | Keeps negations (`not`, `no`, ...) by default. Pass `keep_negations=False` for the old behaviour. |
| `remove_urls` / `count_urls` | Match only `http(s)://`, `ftp://` and `www.` URLs; trailing punctuation is kept. |
| `remove_numbers` | Removes standalone numbers only; `3rd` and `H2O` stay. |
| `remove_repeated_chars` | Shortens letters only; digits are untouched. |
| `remove_special_chars` | `well-known` becomes `well known`. |
| `n_grams` | Returns tuples instead of strings/lists. |
| `download_nltk_packages` | Also installs the spaCy model; no longer downloads unused `stopwords` / `brown`. |
| `clean_text` | Strips HTML first; accepts `remove_stopwords=False`. |

## Known limitations

- `sentence_count` can still split on abbreviations such as "Dr."
- `pluralize_words` does not fix grammatical agreement.
- `lemmatize` maps "me" to "I".

The full list of changes is in [CHANGELOG.md](CHANGELOG.md).
