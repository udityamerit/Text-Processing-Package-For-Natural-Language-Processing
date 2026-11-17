# Changelog

## 0.5.1

### Fixed
- `mask_pii(..., names=True)` with a placeholder such as `<PII>` could produce `<<PII>`, because spaCy read the inserted placeholder as part of an entity. Names are now found on the original text before the other masking.

### Changed
- README: documents the 0.5.0 features (readability scores, name masking, batching, type hints, development tooling) with examples.

## 0.5.0

### Added
- `readability_scores()`: Flesch Reading Ease, Flesch-Kincaid grade, Gunning Fog, Automated Readability Index and Coleman-Liau in one dictionary.
- `mask_pii(..., names=True)`: also redacts people's names using spaCy's named-entity recogniser (opt-in; it needs the spaCy model).
- Type hints on the public API and a `py.typed` marker, so type checkers and IDEs use the annotations.
- Development tooling: `ruff` and `mypy` configuration, coverage reporting, Python 3.13 in the CI matrix and Dependabot for the GitHub Actions versions.
- `RELEASING.md` with the release steps and how to set up PyPI Trusted Publishing.

### Changed
- `Pipeline.run_batch`, `Pipeline.transform` and `Pipeline.run_series` process the spaCy steps (`lemmatize`, `convert_to_base`, `singularize_words`, `pluralize_words`) with `nlp.pipe`, about twice as fast on a list of texts. Results are identical. `run_series` keeps the Series index and name.
- The publish workflow only runs its upload job when the repository variable `TRUSTED_PUBLISHING` is `true` (see `RELEASING.md`).

## 0.4.1

### Added
- New optional extra `[transformers]` (installs `transformers`, for tokenizers used with `truncate_text` / `chunk_text`).

### Fixed
- `sentence_count` / `avg_sentence_len` no longer split after common abbreviations (`Dr.`, `Mr.`, `vs.`, `e.g.`) or single-letter initials (`J. K. Rowling`).
- `lemmatize` keeps pronouns as written (`me` is no longer turned into `I`).

## 0.4.0

### Added
- `Pipeline` steps can take arguments: `('remove_stop_words', {'keep_negations': False})`. Arguments are checked when the pipeline is built.
- `Pipeline.save(path)` / `Pipeline.load(path)` (JSON), `to_config()` / `from_config()`. Custom callables must be registered with `@register_step`.
- scikit-learn style methods on `Pipeline` (`fit`, `transform`, `fit_transform`, `get_params`, `set_params`) and a new `TextPreprocessor` transformer for `sklearn.pipeline` (optional `[sklearn]` extra).
- `Pipeline.__len__`, and a `__repr__` that shows step arguments.
- `light_clean()`: minimal, transformer-friendly cleaning (NFKC, zero-width and control characters, HTML and entities, URL/email/mention handling, optional emoji handling; case and punctuation are kept).
- `truncate_text()` and `chunk_text()`: enforce a token limit with whitespace words or any HuggingFace-style tokenizer, with overlapping chunks.
- Command line tool `nlp-prep` (also `python -m nlp_text_preprocessing`): `clean` for text files, CSV columns and stdin/stdout (`--mode light|standard`, `--steps`, `--pipeline`) and `resources` to check or download the NLTK data and spaCy model.

### Changed (moderation helpers)
- `detect_profanity` / `censor_profanity`: a much larger word list in two tiers (strong and mild), matching of word endings and simple leetspeak, `Hell's Kitchen` is no longer flagged, and new options `include_mild`, `extra_words`, `allow_words` (and `mask_char` for censoring).
- `detect_toxicity`: no longer the share of toxic words. Insults and threats aimed at a person count, `kill the process` / `take out the trash` do not, and a negation cancels a match (`I do not hate you` is 0). New `toxicity_report()` explains each score.
- `detect_spam_signals`: short texts such as `OK` or `Hi!` are no longer spam; spam phrases (`buy now`, `click here`) are detected and returned in `spam_phrases`; `is_spam_candidate` needs at least two signals.
- `emotion_detection`: bigger lexicon, handles word endings (`loved`, `happier`, `worrying`) and ignores negated words (`not happy`).
- The word lists are documented as baselines, and the README no longer presents them as production moderation.

### Changed
- A non-final `Pipeline` step that returns a non-string (for example `text_stats`) now raises a clear `TypeError` instead of silently breaking the chain.

## 0.3.0

### Packaging
- Moved to `pyproject.toml` (setuptools >= 77, SPDX license); `setup.py` removed.
- Removed dependencies that were never imported (pyaudio, SpeechRecognition, openpyxl, PrettyTable, scikit-learn, numpy).
- `pandas`, `wordcloud` and `matplotlib` are now optional extras (`[pandas]`, `[viz]`, `[all]`); `emoji` and `langdetect` are new core dependencies.
- Build artefacts and `__pycache__` are no longer tracked; `nlp_text_preprocessing.__version__` added.

### Import / resources
- `import nlp_text_preprocessing` is light and silent: spaCy, TextBlob, NLTK, matplotlib, wordcloud, bs4 and pandas are imported lazily and nothing is downloaded on import.
- New `check_resources()`, `get_spacy_nlp()`, `get_stop_words()`.
- `download_nltk_packages()` now also installs the spaCy model and no longer downloads the unused `stopwords`/`brown` data; `movie_reviews` is opt-in.
- NLTK zip-only corpora (e.g. `wordnet.zip`) are detected correctly; missing resources raise actionable errors.
- spaCy models are cached per model name; very long texts no longer hit spaCy's `max_length` error.

### Behaviour changes (bug fixes)
- `remove_urls` / `count_urls`: need `http(s)://`, `ftp://` or `www.`; trailing punctuation kept (`httpd` is no longer treated as a URL).
- `remove_rt` removes the trailing colon; `mentions_count` / `remove_mentions` ignore the `@` in e-mail addresses.
- `contraction_to_expansion` preserves whitespace and punctuation and understands curly apostrophes.
- `remove_accented_chars` no longer erases non-Latin scripts.
- `remove_repeated_chars` only shortens letters (`1000000` is no longer turned into `100`).
- `remove_stop_words` keeps negations by default (`keep_negations=False` to opt out); stop-word counting ignores attached punctuation.
- Emoji functions use the `emoji` package (flags, skin tones, ZWJ sequences). `(c)`, `(r)`, `(tm)` are kept by `remove_emojis`.
- Phone detection supports international numbers and no longer matches dates or decimals; `mask_pii` really emits `[REDACTED]` and also masks Luhn-valid card numbers and US SSNs.
- `remove_numbers` removes standalone numbers only (`3rd`, `H2O` are kept); `expand_hashtags` handles acronyms and digits; `remove_currency_symbols` covers all Unicode currency symbols.
- `remove_special_chars` turns `well-known` into `well known`; `remove_html_tag` separates block elements and drops `<script>`/`<style>`.
- `sentence_count` ignores decimal points; `readability_score` estimates syllables per word; `upper_case_count` ignores single letters.
- `lemmatize`, `convert_to_base`, `singularize_words`, `pluralize_words` keep the original spacing.
- `detect_language` uses `langdetect` (TextBlob has no `detect_language`) and returns `"unknown"` when it cannot tell.
- `chunk_noun_phrases` returns exact spaCy offsets; `n_grams` returns tuples; `correct_spelling` preserves punctuation/case/URLs; `get_synonyms` / `get_antonyms` return sorted, complete lists.
- `sentiment_analysis` defaults to the instant `pattern` method and returns `'pos'`/`'neg'`/`'neu'` (empty text is `'neu'`); the 30-second NaiveBayes model is opt-in via `method='naive_bayes'`.
- `censor_profanity` preserves layout; `Pipeline` validates steps up front and only exposes public functions; `diff_report` validates list lengths; `clean_text` reordered (HTML first).
