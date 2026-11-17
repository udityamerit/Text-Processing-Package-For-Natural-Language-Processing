# v0.4.0: transformer-friendly cleaning, a command line tool and a safer moderation toolkit

This release adds tools for preparing text for BERT/GPT-style models, a `nlp-prep` command, and a more flexible `Pipeline`. The moderation helpers were rewritten to produce fewer false positives, so please read **Behaviour changes** before upgrading.

## Install

```bash
pip install -U nlp-text-preprocessing            # core
pip install -U "nlp-text-preprocessing[all]"     # + pandas, wordcloud, matplotlib, scikit-learn
```

## What's new

- **`light_clean(text)`**: minimal cleaning for transformer models. It normalises Unicode, removes zero-width and control characters, strips HTML and entities, and removes or replaces URLs, emails and mentions. Case and punctuation are kept. Emojis can be kept, removed or converted to text.
- **`truncate_text()` and `chunk_text()`**: enforce a token limit using words or any HuggingFace-style tokenizer, with overlapping chunks for long documents.
- **`nlp-prep` command** (also `python -m nlp_text_preprocessing`): clean text files, CSV columns or stdin/stdout, with `--mode light|standard`, `--steps` and `--pipeline`. `nlp-prep resources --download` installs the NLTK data and the spaCy model.
- **Pipeline arguments**: `('remove_stop_words', {'keep_negations': False})`, checked when the pipeline is built.
- **Pipeline save/load**: `Pipeline.save(path)` and `Pipeline.load(path)` (JSON). Custom functions must be registered with `@register_step`.
- **scikit-learn support**: `Pipeline` has `fit`/`transform`/`get_params`/`set_params`, and the new `TextPreprocessor` works inside `sklearn.pipeline` (install the `[sklearn]` extra).
- **`toxicity_report()`**: explains every toxicity score (matched words, weights, and negated words that were ignored).

## Behaviour changes

- **`detect_toxicity`** is no longer the share of toxic words. Insults and threats aimed at a person count, `kill the process` and `take out the trash` do not, and a negation cancels a match (`I do not hate you` scores 0). Scores from 0.3.0 are not comparable.
- **`detect_profanity` / `censor_profanity`** use a much larger two-tier word list, match word endings and simple leetspeak, and no longer flag `Hell's Kitchen`. New options: `include_mild`, `extra_words`, `allow_words`, `mask_char`.
- **`detect_spam_signals`**: short texts such as `OK` or `Hi!` are no longer spam, spam phrases are returned in `spam_phrases`, and `is_spam_candidate` needs at least two signals.
- **`emotion_detection`**: bigger lexicon, handles word endings and ignores negated words (`not happy`).
- A non-final `Pipeline` step that returns a non-string now raises a clear `TypeError`.

## Known limits

- Masked profanity such as `f***` is not detected.
- Racial and ethnic slurs are intentionally not in the built-in list. Add them with `extra_words` if you need them.
- The word lists are baselines, not a production moderation system.

Full details are in [CHANGELOG.md](CHANGELOG.md).
