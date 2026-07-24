import re
import emoji
from mfs_sentiment.dictionaries import BANGLISH_KEYWORDS


# def is_banglish(text: str) -> bool:
#     """Return True when a review contains known Bangla-English mixed keywords."""
#     if not isinstance(text, str):
#         return False
#     words = set(text.lower().split())
#     return bool(words & BANGLISH_KEYWORDS)


def apply_custom_dict(text: str, translation_dict: dict) -> str:
    """Replace known tokens in text using a custom translation lookup."""
    if not isinstance(text, str):
        return ""
    for bangla, english in translation_dict.items():
        text = text.replace(bangla, english)
    return text


def convert_emojis(text: str) -> str:
    """Replace emojis with their textual descriptions for downstream normalization."""
    if not isinstance(text, str):
        return ""
    return emoji.demojize(text, delimiters=(" ", " "))


def clean_text(text: str) -> str:
    """Normalize review text by lowercasing, removing URLs, and collapsing whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # Collapse repeated characters
    text = re.sub(r"[^a-z0-9\s:_.,!?&'()-]", "", text)  # Keep letters, digits, emoji tags
    text = re.sub(r"\s+", " ", text).strip()
    return text