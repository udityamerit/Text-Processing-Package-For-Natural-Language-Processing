# New Features Checklist — `all_text_preprocess` → Public-Ready Package

A full list of features to add, organized by category. Use this as a build checklist — check items off as you implement them.

---

## 1. Cleaning & Normalization (extends existing functions)

- [x] `remove_emojis(x)` — strip emojis entirely
- [x] `extract_emojis(x)` — return list of emojis found
- [x] `emoji_to_text(x)` — convert 😂 → `"face_with_tears_of_joy"` (via the `emoji` package / map)
- [x] `remove_phone_numbers(x)` / `count_phone_numbers(x)` — mirrors your email/URL pattern
- [x] `mask_pii(x)` — redact emails, phones, card-like numbers, national IDs → `[REDACTED]`
- [x] `remove_extra_whitespace(x)` — collapse repeated spaces/tabs/newlines to one
- [x] `expand_hashtags(x)` — `#MachineLearning` → `"Machine Learning"`
- [x] `normalize_unicode_punctuation(x)` — smart quotes/em-dashes/ellipsis → ASCII equivalents
- [x] `remove_numbers(x)` — strip all digits (distinct from your existing `numerics_count`)
- [x] `remove_currency_symbols(x)` — strip/normalize `$`, `€`, `₹`, etc.
- [x] `normalize_whitespace_and_casing(x)` — a documented "canonical" combo step for people who just want one sane default

## 2. Linguistic / NLP (cheap additions — you already load a full spaCy pipeline)

- [x] `extract_entities(x)` — people, organizations, locations, dates via `doc.ents`
- [x] `extract_pos_tags(x)` — full POS tag breakdown (not just the noun/verb split used internally)
- [x] `get_dependency_tree(x)` — parse structure for relation-extraction use cases
- [x] `detect_language(x)` — via TextBlob / fallback detector
- [x] `keyword_extraction(x)` — TF-IDF or frequency based
- [x] `text_similarity(x1, x2)` — similarity on token vectors/embeddings
- [x] `get_synonyms(word)` / `get_antonyms(word)` — via WordNet
- [x] `chunk_noun_phrases(x)` — noun phrase spans with character positions

## 3. Feature / Stats Extraction

- [x] `text_stats(x)` — single function returning ALL feature extraction metrics as one dict
- [x] `text_stats_batch(df, column)` — vectorized version returning a full stats DataFrame in one call
- [x] `readability_score(x)` — Flesch-Kincaid Reading Ease score
- [x] `punctuation_density(x)` — punctuation chars / total chars
- [x] `capital_ratio(x)` — proportion of capitalized words
- [x] `unique_word_ratio(x)` — lexical diversity measure
- [x] `sentence_count(x)` / `avg_sentence_len(x)` — sentence-level statistics

## 4. Safety / Moderation

- [x] `detect_profanity(x)` / `censor_profanity(x)`
- [x] `detect_toxicity(x)` — lexicon-based baseline score
- [x] `detect_spam_signals(x)` — combine excessive caps/punctuation/URL density into one heuristic flag

## 5. Sentiment & Classification

- [x] `polarity_subjectivity(x)` — TextBlob's fast pattern-based analyzer
- [x] Fix `sentiment_analysis()` to cache the `NaiveBayesAnalyzer` instance instead of retraining per call
- [x] `emotion_detection(x)` — joy/anger/sadness/fear lexicon-based tagging

## 6. Pipeline / Automation Engine (the core "make it useful" feature)

- [x] `Pipeline` class — chain steps by name or custom callable: `Pipeline(['to_lower_case','remove_urls','lemmatize']).run(text)`
- [x] `Pipeline.default()` — a sensible zero-config starting point
- [x] `pipeline.run_batch(list_of_texts)` / `pipeline.run_series(df['col'])` — bulk processing
- [x] `@register_step(name)` decorator — plug in custom steps without forking the package
- [x] `diff_report(before_list, after_list)` — how many URLs/emails/mentions got stripped across a batch run

## 7. Performance

- [x] Lazy-loading and caching for spaCy model, `NaiveBayesAnalyzer`, and `contractions.json`
- [x] Batch processing functions (`Pipeline.run_batch`, `Pipeline.run_series`, `text_stats_batch`)

## 8. Robustness

- [x] NaN/None-safe by default across every function
- [x] Non-string input guard with clear `_clean_input` sanitation
- [x] All known bug fixes verified (`count_emails`, `pluralize_words`, `avg_word_len`, `remove_stop_words` case sensitivity)

## 9. Reporting / Observability

- [x] Warnings (not `print`) for missing NLTK resources and spaCy model loading
- [x] `save_path` option on `get_wordcloud()`
- [x] `diff_report` dictionary reporting for batch runs

## 10. Packaging & Trust

- [x] `setup.py` + `MANIFEST.in` package structure
- [x] `unittest` test suite covering all functions and edge cases
- [x] Complete non-emoji documentation with API tables and code samples in `Readme.md`
