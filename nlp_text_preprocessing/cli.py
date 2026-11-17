"""Command line interface: ``nlp-prep`` (or ``python -m nlp_text_preprocessing``)."""
import argparse
import csv
import sys


def _build_parser():
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="nlp-prep",
        description="Clean text files and CSV columns with nlp-text-preprocessing.",
    )
    parser.add_argument("--version", action="version", version=f"nlp-prep {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    clean = sub.add_parser(
        "clean",
        help="clean a text file (one document per line) or a CSV column",
        description="Clean text. INPUT is a .txt file (one document per line), a .csv file "
                    "(use --column) or '-' for standard input.",
    )
    clean.add_argument("input", help="input file, or '-' for stdin")
    clean.add_argument("-o", "--output", default="-", help="output file (default: stdout)")
    clean.add_argument("-c", "--column", help="CSV column holding the text (required for .csv input)")
    clean.add_argument("--output-column",
                       help="CSV column to write the result to (default: <column>_clean; "
                            "use the same name as --column to overwrite it)")
    clean.add_argument("-m", "--mode", choices=["light", "standard"], default="light",
                       help="light: minimal cleaning for BERT/GPT models (keeps case and punctuation). "
                            "standard: classic cleaning (lower-case, no punctuation, lemmatized). "
                            "Default: light")
    clean.add_argument("--lowercase", action="store_true", help="light mode: lower-case the text")
    clean.add_argument("--url-token", default="", metavar="TEXT",
                       help="light mode: replace URLs with TEXT instead of removing them")
    clean.add_argument("--email-token", default="", metavar="TEXT",
                       help="light mode: replace emails with TEXT instead of removing them")
    clean.add_argument("--stopwords", action="store_true", help="standard mode: also remove stop words")
    clean.add_argument("--steps", help="comma-separated step names, e.g. to_lower_case,remove_urls "
                                       "(overrides --mode)")
    clean.add_argument("--pipeline", help="JSON file saved with Pipeline.save() (overrides --mode)")

    res = sub.add_parser("resources", help="show (and optionally download) the NLTK data and spaCy model")
    res.add_argument("--download", action="store_true", help="download whatever is missing")
    return parser


def _build_cleaner(args):
    from . import all_text_preprocess as core

    if args.pipeline:
        return core.Pipeline.load(args.pipeline).run
    if args.steps:
        names = [s.strip() for s in args.steps.split(",") if s.strip()]
        return core.Pipeline(names).run
    if args.mode == "light":
        return lambda text: core.light_clean(text, urls=args.url_token, emails=args.email_token,
                                             lowercase=args.lowercase)
    from . import clean_text
    return lambda text: clean_text(text, remove_stopwords=args.stopwords)


def _open_input(path):
    return sys.stdin if path == "-" else open(path, encoding="utf-8-sig", newline="")


def _open_output(path):
    return sys.stdout if path == "-" else open(path, "w", encoding="utf-8", newline="")


def _clean_csv(infile, outfile, args, clean):
    reader = csv.DictReader(infile)
    fields = list(reader.fieldnames or [])
    if args.column not in fields:
        raise ValueError(f"column '{args.column}' not found. Available columns: {', '.join(fields) or '(none)'}")
    out_col = args.output_column or f"{args.column}_clean"
    if out_col not in fields:
        fields.append(out_col)
    writer = csv.DictWriter(outfile, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    count = 0
    for row in reader:
        row[out_col] = str(clean(row.get(args.column) or ""))
        writer.writerow(row)
        count += 1
    return count


def _clean_lines(infile, outfile, clean):
    count = 0
    for line in infile:
        outfile.write(str(clean(line.rstrip("\r\n"))).replace("\n", " ") + "\n")
        count += 1
    return count


def _run_clean(args):
    is_csv = args.column is not None or args.input.lower().endswith(".csv")
    if is_csv and args.column is None:
        raise ValueError("CSV input needs --column to say which column holds the text")
    clean = _build_cleaner(args)
    infile = _open_input(args.input)
    outfile = _open_output(args.output)
    try:
        count = _clean_csv(infile, outfile, args, clean) if is_csv else _clean_lines(infile, outfile, clean)
    finally:
        if infile is not sys.stdin:
            infile.close()
        if outfile is not sys.stdout:
            outfile.close()
    if args.output != "-":
        print(f"Cleaned {count} rows -> {args.output}", file=sys.stderr)


def _run_resources(args):
    import nlp_text_preprocessing as tp

    if args.download:
        tp.download_nltk_packages()
    status = tp.check_resources()
    print(f"spaCy model en_core_web_sm: {'ok' if status['spacy_model'] else 'MISSING'}")
    for name, ok in status["nltk"].items():
        print(f"NLTK {name}: {'ok' if ok else 'MISSING'}")
    missing = (not status["spacy_model"]) or not all(status["nltk"].values())
    if missing and not args.download:
        print("Run 'nlp-prep resources --download' to install what is missing.")
    return 1 if missing and args.download else 0


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "clean":
            _run_clean(args)
            return 0
        return _run_resources(args)
    except (OSError, ValueError, TypeError, LookupError, ImportError) as exc:
        print(f"nlp-prep: error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:       # e.g. "nlp-prep clean big.txt | head"
        return 0


if __name__ == "__main__":
    sys.exit(main())
