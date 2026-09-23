"""
utils.py — DATA VORTEX A'26, Round 2
Shared utilities for text cleaning and label encoding.

Importable by any phase script and by the Phase 8 inference pipeline.
The cleaning function must be the single source of truth — do not redefine
or re-implement it in downstream scripts.

Regex patterns are intentionally kept consistent with 01_audit.py so that
Phase 1's detection counts and Phase 2's cleaning are provably matched.
"""

import re
import html
import json
import codecs
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns  (kept 1-to-1 with 01_audit.py for traceability)
# ---------------------------------------------------------------------------
PAT_UNICODE_ESCAPE = re.compile(r'\\u[0-9A-Fa-f]{4}')
PAT_HTML_ENTITY    = re.compile(r'&(?:amp|quot|#39|lt|gt);', re.IGNORECASE)
PAT_AT_ALL         = re.compile(r'@[A-Za-z0-9_]+')   # @user AND other mentions
PAT_URL            = re.compile(r'https?://\S+|www\.\S+')
PAT_MULTI_SPACE    = re.compile(r'[ \t]+')


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(raw: str) -> str:
    """
    Apply the full cleaning pipeline to a single raw post_text string.

    Steps (in order — order matters):
      1. Decode literal \\uXXXX Unicode escape sequences into real characters.
      2. Decode HTML entities (&amp; → &, &quot; → ", &#39; → ', etc.).
      3. Normalise all @mentions (including @user) to the single token "@user".
      4. Replace URLs with the placeholder token "<url>".
      5. Collapse repeated horizontal whitespace; strip leading/trailing space.

    Deliberately NOT done here (see Phase 2 design notes):
      - Lowercasing  — vectorizer config decision (Phase 4); ablatable later.
      - Hashtag removal / # stripping — same reason.
      - Stemming / lemmatisation — same reason.
    """
    # ── Step 1: decode literal Unicode escapes ─────────────────────────────
    # e.g. the 6 chars \u2019 become the single char '
    # Use chr(int(hex, 16)) — the unicode_escape codec misreads UTF-8 as latin-1 on Py3.
    text = PAT_UNICODE_ESCAPE.sub(
        lambda m: chr(int(m.group(0)[2:], 16)),
        raw
    )

    # ── Step 2: decode HTML entities (loop to handle double-encoded e.g. &amp;quot;) ─
    _prev = None
    _passes = 0
    while text != _prev and _passes < 3:
        _prev = text
        text = html.unescape(text)
        _passes += 1

    # ── Step 3: normalise @mentions → @user ────────────────────────────────
    text = PAT_AT_ALL.sub('@user', text)

    # ── Step 4: replace URLs ────────────────────────────────────────────────
    text = PAT_URL.sub('<url>', text)

    # ── Step 5: collapse whitespace ─────────────────────────────────────────
    text = PAT_MULTI_SPACE.sub(' ', text).strip()

    return text


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------
# Fixed, explicit mappings — order is alphabetical to be deterministic but
# the mapping is SAVED to disk and must be the sole reference at inference
# time; do not recompute from the data.

SENTIMENT_LABEL_MAP: dict = {
    "Negative": 0,
    "Neutral":  1,
    "Positive": 2,
}

TOPIC_LABEL_MAP: dict = {
    "Account_Security":    0,
    "Community_Discussion": 1,
    "Feature_Feedback":    2,
    "Technical_Issues":    3,
}

LABEL_ENCODERS = {
    "sentiment_label": SENTIMENT_LABEL_MAP,
    "topic_category":  TOPIC_LABEL_MAP,
}


def encode_labels(series, mapping: dict):
    """Map a pandas Series of string labels to integer ids using `mapping`."""
    unknown = set(series.unique()) - set(mapping.keys())
    if unknown:
        raise ValueError("Unknown label(s) not in mapping: " + str(unknown))
    return series.map(mapping)


def save_label_encoders(path: Path):
    """Serialise LABEL_ENCODERS to JSON so inference can reload it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(LABEL_ENCODERS, f, indent=2)
    print("[utils] Label encoders saved to " + str(path))


def load_label_encoders(path: Path) -> dict:
    """Reload label encoder mappings from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Smoke-test (runs when this module is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Verify the Unicode-escape decoding on a known example from the audit report
    raw_example  = r"Tune into @user for live coverage of tonight\u2019s NNS race."
    clean_example = clean_text(raw_example)
    assert "\u2019" in clean_example, (
        "Unicode decode failed: \\u2019 not converted to right-single-quote. Got: " + clean_example
    )
    assert r"\u2019" not in clean_example, (
        "Literal \\u2019 still present in cleaned text. Got: " + clean_example
    )
    print("[utils smoke-test] Unicode decode: OK")
    print("  raw   : " + raw_example)
    print("  clean : " + clean_example)

    # Verify HTML entity decoding
    raw_html = "Federer &amp; Murray march on"
    clean_html = clean_text(raw_html)
    assert "&amp;" not in clean_html, "HTML entity decode failed."
    assert "&" in clean_html, "& missing after HTML decode."
    print("[utils smoke-test] HTML entity decode: OK")
    print("  raw   : " + raw_html)
    print("  clean : " + clean_html)

    # Verify @mention normalisation
    raw_mention = "@dhrit sent a reply to @user"
    clean_mention = clean_text(raw_mention)
    assert clean_mention == "@user sent a reply to @user", (
        "Mention normalisation failed. Got: " + clean_mention
    )
    print("[utils smoke-test] @mention normalisation: OK")

    # Verify URL replacement
    raw_url = "Check https://example.com for details"
    clean_url = clean_text(raw_url)
    assert "<url>" in clean_url, "URL replacement failed."
    print("[utils smoke-test] URL replacement: OK")

    print("\n[utils smoke-test] ALL ASSERTIONS PASSED.")
