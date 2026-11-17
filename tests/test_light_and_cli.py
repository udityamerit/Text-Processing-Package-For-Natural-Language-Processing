import csv
import io
import sys

import pytest

import nlp_text_preprocessing as tp
from nlp_text_preprocessing import cli


# ---------------------------------------------------------------- light_clean
def test_light_clean_exported():
    for name in ("light_clean", "truncate_text", "chunk_text"):
        assert hasattr(tp, name)
        assert name in tp.__all__


def test_light_clean_html_entities_and_zero_width():
    assert tp.light_clean("<p>Hello &amp; welcome,​  World!</p>") == "Hello & welcome, World!"


def test_light_clean_keeps_case_and_punctuation():
    assert tp.light_clean("Keep   Case. And punctuation!!") == "Keep Case. And punctuation!!"


def test_light_clean_does_not_eat_math_symbols():
    assert tp.light_clean("a < b and c > d") == "a < b and c > d"


def test_light_clean_nfkc_and_control_chars():
    assert tp.light_clean("café ﬁnal Ａ") == "café final A"
    assert tp.light_clean("x\x00y\x07z") == "x y z"


def test_light_clean_url_email_mention_modes():
    text = "Visit https://x.com/a now or mail me@x.com @bob"
    assert tp.light_clean(text) == "Visit now or mail @bob"
    assert tp.light_clean(text, urls="[URL]", emails="[EMAIL]", mentions="@user") == \
        "Visit [URL] now or mail [EMAIL] @user"
    assert tp.light_clean(text, urls=None, emails=None).count("x.com") == 2


def test_light_clean_newlines_and_lowercase():
    text = "line1\n\n  line2   \nline3"
    assert tp.light_clean(text) == "line1 line2 line3"
    assert tp.light_clean(text, keep_newlines=True) == "line1\nline2\nline3"
    assert tp.light_clean("Hi There", lowercase=True) == "hi there"


def test_light_clean_emoji_modes():
    text = "I love it \U0001F600 so much"
    assert "\U0001F600" in tp.light_clean(text)
    assert tp.light_clean(text, emojis="remove") == "I love it so much"
    assert "\U0001F600" not in tp.light_clean(text, emojis="text")
    with pytest.raises(ValueError):
        tp.light_clean("x", emojis="bad")


# ----------------------------------------------------------- truncate / chunk
class FakeHFTokenizer:
    """Mimics the two methods of a HuggingFace tokenizer that the helpers use."""

    def tokenize(self, text):
        return [text[i:i + 2] for i in range(0, len(text), 2)]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


def test_truncate_text_words():
    assert tp.truncate_text("one two three four five", 3) == "one two three"
    assert tp.truncate_text("short text", 10) == "short text"


def test_chunk_text_words_with_and_without_overlap():
    text = "one two three four five six seven"
    assert tp.chunk_text(text, 3) == ["one two three", "four five six", "seven"]
    assert tp.chunk_text(text, 3, overlap=1) == ["one two three", "three four five", "five six seven"]


def test_truncate_and_chunk_with_hf_style_tokenizer():
    tok = FakeHFTokenizer()
    assert tp.truncate_text("abcdefghij", 3, tokenizer=tok) == "abcdef"
    assert tp.chunk_text("abcdefghij", 2, overlap=1, tokenizer=tok) == ["abcd", "cdef", "efgh", "ghij"]


def test_chunk_text_with_callable_tokenizer():
    assert tp.chunk_text("a b c d", 2, tokenizer=lambda t: t.split()) == ["a b", "c d"]


def test_chunk_text_validation():
    with pytest.raises(ValueError):
        tp.chunk_text("a b c", 0)
    with pytest.raises(ValueError):
        tp.chunk_text("a b c", 3, overlap=3)
    with pytest.raises(TypeError):
        tp.chunk_text("a b c", 2, tokenizer=42)


# ------------------------------------------------------------------------ CLI
def test_cli_clean_text_file(tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text("<b>Hello</b> https://x.com world\nSecond   line\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    assert cli.main(["clean", str(src), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["Hello world", "Second line"]
    assert "Cleaned 2 rows" in capsys.readouterr().err


def test_cli_clean_stdin_stdout(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Hi   there\n"))
    assert cli.main(["clean", "-", "--lowercase"]) == 0
    assert capsys.readouterr().out == "hi there\n"


def test_cli_light_options(tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text("see https://x.com\n", encoding="utf-8")
    assert cli.main(["clean", str(src), "--url-token", "[URL]"]) == 0
    assert capsys.readouterr().out == "see [URL]\n"


def test_cli_clean_csv_column(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("id,text\n1,\"<i>Hello</i>,  world\"\n2,Plain\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert cli.main(["clean", str(src), "-c", "text", "-o", str(out)]) == 0
    with open(out, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["text_clean"] == "Hello, world"
    assert rows[0]["text"] == "<i>Hello</i>,  world"
    assert rows[1]["text_clean"] == "Plain"


def test_cli_csv_overwrite_column(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("text\n  A   B \n", encoding="utf-8")
    out = tmp_path / "out.csv"
    assert cli.main(["clean", str(src), "-c", "text", "--output-column", "text", "-o", str(out)]) == 0
    with open(out, encoding="utf-8", newline="") as f:
        assert list(csv.DictReader(f)) == [{"text": "A B"}]


def test_cli_csv_errors(tmp_path, capsys):
    src = tmp_path / "in.csv"
    src.write_text("text\nA\n", encoding="utf-8")
    assert cli.main(["clean", str(src)]) == 1
    assert "--column" in capsys.readouterr().err
    assert cli.main(["clean", str(src), "-c", "nope"]) == 1
    assert "Available columns: text" in capsys.readouterr().err


def test_cli_missing_file(tmp_path, capsys):
    assert cli.main(["clean", str(tmp_path / "missing.txt")]) == 1
    assert "nlp-prep: error" in capsys.readouterr().err


def test_cli_steps_option(tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text("Hello WORLD\n", encoding="utf-8")
    assert cli.main(["clean", str(src), "--steps", "to_lower_case"]) == 0
    assert capsys.readouterr().out == "hello world\n"


def test_cli_pipeline_file(tmp_path, capsys):
    cfg = tmp_path / "p.json"
    tp.Pipeline(["to_lower_case", "remove_urls"]).save(str(cfg))
    src = tmp_path / "in.txt"
    src.write_text("Go TO https://x.com\n", encoding="utf-8")
    assert cli.main(["clean", str(src), "--pipeline", str(cfg)]) == 0
    assert capsys.readouterr().out.strip() == "go to"


def test_cli_unknown_step(tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text("x\n", encoding="utf-8")
    assert cli.main(["clean", str(src), "--steps", "no_such_step"]) == 1
    assert "nlp-prep: error" in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert tp.__version__ in capsys.readouterr().out


def test_cli_resources(monkeypatch, capsys):
    monkeypatch.setattr(tp, "check_resources", lambda: {"spacy_model": True, "nltk": {"punkt": True}})
    assert cli.main(["resources"]) == 0
    out = capsys.readouterr().out
    assert "en_core_web_sm: ok" in out and "NLTK punkt: ok" in out
    monkeypatch.setattr(tp, "check_resources", lambda: {"spacy_model": False, "nltk": {"punkt": True}})
    assert cli.main(["resources"]) == 0
    assert "MISSING" in capsys.readouterr().out
