import re
import emoji


def apply_custom_dict(text: str, mapping: dict) -> str:
    """Apply a token-level replacement dictionary to normalize noisy review text."""
    if not isinstance(text, str):
        return ""
    for key, value in mapping.items():
        text = text.replace(key, value)
    # Alternative regex approach with word boundaries (commented):
    # for key, value in mapping.items():
    #     text = re.sub(rf"\b{re.escape(key)}\b", value, text)
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
    text = re.sub(r"[^a-z0-9\s:_.,!?&'()-]", "", text)  # Keep letters, digits, emoji tags, and punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text