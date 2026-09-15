"""
Shared utilities for the DATA VORTEX pipeline.
All functions here are pure / deterministic so results are reproducible
across audit -> clean -> validate -> eda -> insights.
"""
import re
import html
import pandas as pd

RAW_USERS_PATH = "data/raw/Social_Engine_Users.csv"
RAW_POSTS_PATH = "data/raw/Social_Engine_Posts_Corrupted.csv"
CLEAN_USERS_PATH = "data/cleaned/users_cleaned.csv"
CLEAN_POSTS_PATH = "data/cleaned/posts_cleaned.csv"

NULL_LITERALS = {"NULL", "NAN", "NONE", "N/A", "NA", ""}

UNIX_RE = re.compile(r"^\d{10}$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
DMY_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")


def load_raw_posts():
    """Load posts with default NA handling disabled so we can see literal
    'NULL' strings vs true empty fields separately during auditing.
    dtype=object (not the pandas StringDtype) is used deliberately: assigning
    Python None into a StringDtype column silently becomes the *float* NaN,
    and str(nan) == 'nan', which would corrupt cleaned text with the literal
    word "nan". object dtype preserves None exactly as None."""
    return pd.read_csv(RAW_POSTS_PATH, dtype=object, keep_default_na=False)


def load_raw_users():
    return pd.read_csv(RAW_USERS_PATH, dtype=object, keep_default_na=False)


def classify_timestamp_format(ts: str) -> str:
    ts = str(ts).strip()
    if UNIX_RE.match(ts):
        return "unix_epoch"
    if ISO_RE.match(ts):
        return "iso_8601"
    if DMY_RE.match(ts):
        return "dd_mm_yyyy"
    return "unknown"


def parse_timestamp(ts: str):
    """Parse one of the three known timestamp formats into a pandas Timestamp.
    Returns pd.NaT for anything that doesn't match a known format (none were
    found in this dataset, but the guard is kept for safety/reproducibility).
    """
    ts = str(ts).strip()
    fmt = classify_timestamp_format(ts)
    if fmt == "unix_epoch":
        return pd.to_datetime(int(ts), unit="s")
    if fmt == "iso_8601":
        return pd.to_datetime(ts)
    if fmt == "dd_mm_yyyy":
        return pd.to_datetime(ts, format="%d-%m-%Y")
    return pd.NaT


def normalize_missing(value):
    """Convert literal placeholder strings ('NULL', '', 'NaN', etc.) to real
    missing values (None). Leaves genuine content untouched."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    v = str(value).strip()
    if v.upper() in NULL_LITERALS:
        return None
    return v


def clean_text(value):
    """Decode HTML entities and strip stray whitespace from free text.
    Does not alter wording, punctuation choice, or meaning."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    v = html.unescape(str(value)).strip()
    if v == "":
        return None
    return v
